// Scenes of data-first tiles (content/wgtScene.ts): a level-0 tile's trees carry entity ids and a
// record provider; a coarser level's thinned trees and building prisms (format 2.5) carry none,
// sit on its own terrain as the compiler places them, and are marked with userData.level.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { BoxGeometry, InstancedMesh, Matrix4, Mesh, MeshStandardMaterial, Quaternion, Vector3 } from 'three';
import type { Group, Object3D } from 'three';
import type { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { RECORD_PROVIDER, buildWgtScene } from '../src/content/wgtScene.ts';
import type { WgtWorld } from '../src/content/wgtScene.ts';
import type { ArchetypeLibrary } from '../src/content/archetypes.ts';

const v = JSON.parse(readFileSync(new URL('../../tests/conformance/tree_tiles.json', import.meta.url), 'utf8'));
const ARCHETYPES = ['broadleaf', 'conifer', 'palm', 'shrub', 'grass'];

function bytes(b64: string): ArrayBuffer {
  const buf = Buffer.from(b64, 'base64');
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

function world(): WgtWorld {
  const lib: ArchetypeLibrary = new Map(ARCHETYPES.map((a) => [a, [{ name: `${a}_foliage`, geometry: new BoxGeometry(), material: new MeshStandardMaterial() }]]));
  return {
    tileSize: v.tile_size,
    archetypes: Promise.resolve(lib),
    trees: { cellSize: v.cell_size, seed: v.seed, archetypes: ARCHETYPES, categories: ['none', 'dense_forest'] },
    buildingMaterials: [],
  };
}

// Tiles without a GLB section never touch the loader.
const loader = null as unknown as GLTFLoader;

function layer(scene: Group, name: string): Object3D {
  let found: Object3D | undefined;
  scene.traverse((o) => {
    if (!found && (o.userData as { layer?: string }).layer === name && o.name === name) found = o;
  });
  assert.ok(found, `no ${name} layer`);
  return found;
}

test('level-0 trees have entity ids and records', async () => {
  const scene = await buildWgtScene(bytes(v.wgt_base64), loader, '', world());
  const ids = layer(scene, 'trees').children.flatMap((g) => (g.userData as { instance_ids: string[] }).instance_ids);
  assert.deepEqual([...ids].sort(), [...v.expected.ids].sort());
  assert.equal(typeof scene.userData[RECORD_PROVIDER], 'function');
});

test('coarse trees and prisms have no ids and sit where the compiler puts them', async () => {
  const c = v.coarse;
  const scene = await buildWgtScene(bytes(c.wgt_base64), loader, '', world());
  const root = scene.getObjectByName(`tile_${c.level}/${c.i}_${c.j}`);
  assert.ok(root);
  assert.equal(scene.userData[RECORD_PROVIDER], undefined);

  // Trees: one InstancedMesh per archetype, in TRE order within it.
  const e = c.expected;
  const byArch = new Map<number, number[]>();
  e.archetype_codes.forEach((code: number, n: number) => byArch.set(code, [...(byArch.get(code) ?? []), n]));
  const trees = layer(scene, 'trees');
  assert.equal(trees.children.length, byArch.size);
  const m = new Matrix4();
  const p = new Vector3();
  const q = new Quaternion();
  const s = new Vector3();
  for (const group of trees.children) {
    assert.deepEqual((group.userData as { instance_ids: string[] }).instance_ids, []);
    const rows = byArch.get(ARCHETYPES.indexOf(group.name))!;
    const im = group.children[0] as InstancedMesh;
    assert.equal(im.count, rows.length);
    assert.equal((im.userData as { level?: number }).level, 1);
    rows.forEach((n, k) => {
      im.getMatrixAt(k, m);
      m.decompose(p, q, s);
      for (let a = 0; a < 3; a++) {
        assert.ok(Math.abs(p.getComponent(a) - e.translation[3 * n + a]) < 1e-4, `translation of tree ${n}`);
        assert.ok(Math.abs(s.getComponent(a) - e.scale[3 * n + a]) < 1e-4, `scale of tree ${n}`);
      }
    });
  }

  // Buildings: flat prisms between each building's floor and roof, without feature ids.
  const buildings = layer(scene, 'buildings');
  assert.deepEqual((buildings.userData as { feature_ids: string[] }).feature_ids, []);
  const heights = new Set<number>();
  for (const mesh of buildings.children as Mesh[]) {
    assert.equal((mesh.userData as { level?: number }).level, 1);
    const pos = mesh.geometry.getAttribute('position');
    for (let k = 0; k < pos.count; k++) heights.add(Math.round(pos.getY(k) * 1000));
  }
  const want = new Set([...e.bottom, ...e.top].map((z: number) => Math.round(Math.fround(z) * 1000)));
  assert.deepEqual([...heights].sort(), [...want].sort());
});
