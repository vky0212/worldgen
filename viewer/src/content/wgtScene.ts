/**
 * Scene of a data-first tile (`.wgt`): the embedded GLB's layers plus a terrain
 * mesh generated from the heightmap, arranged exactly like a baked tile (root
 * node `tile_<i>_<j>` translated to the tile corner, layer children with
 * `userData.layer`, terrain feature id `terrain:<i>_<j>`), so the world model,
 * picking, colour modes and photoreal mode treat both formats alike.
 */
import { BufferAttribute, BufferGeometry, Group, InstancedMesh, Matrix4, Mesh, MeshStandardMaterial, Quaternion, Vector3 } from 'three';
import type { Object3D } from 'three';
import type { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { decodeGrid, decodeHeights, readWgt } from './wgt.ts';
import type { HeightGrid } from './wgt.ts';
import { terrainArrays } from '../gen/terrainMesh.ts';
import { decodeTrees, treeIdAt, treeInstances, treeRecordAt } from './trees.ts';
import type { TreeContext } from './trees.ts';
import type { ArchetypeLibrary } from './archetypes.ts';
import type { EntityRecord } from '../types';

/** World-level inputs of data tiles (tileset.json). */
export interface WgtWorld {
  tileSize: number;
  /** Tree archetype meshes (tiles/archetypes.glb), shared by every tile. */
  archetypes?: Promise<ArchetypeLibrary | null>;
  trees?: { cellSize: number; seed: number; archetypes: readonly string[]; categories: readonly string[] };
}

/** userData key of a data tile's scene: lazily synthesises records of its trees by entity id. */
export const RECORD_PROVIDER = 'worldgenRecordProvider';

function tileRootOf(scene: Group, name: string): Object3D | null {
  let found: Object3D | null = null;
  scene.traverse((o) => {
    if (!found && (o.name === name || (o.userData as Record<string, unknown>).tile !== undefined)) found = o;
  });
  return found;
}

export async function buildWgtScene(buffer: ArrayBuffer, loader: GLTFLoader, baseUrl: string, world: WgtWorld): Promise<Group> {
  const tileSize = world.tileSize;
  const c = await readWgt(buffer);
  const id = `${c.i}_${c.j}`;
  const glb = c.sections.get('GLB');
  const scene = glb && glb.byteLength
    ? ((await loader.parseAsync(glb.buffer.slice(glb.byteOffset, glb.byteOffset + glb.byteLength) as ArrayBuffer, baseUrl)).scene as Group)
    : new Group();
  let root = tileRootOf(scene, `tile_${id}`);
  if (!root) {
    const g = new Group();
    g.name = `tile_${id}`;
    g.position.set(c.i * tileSize, 0, -c.j * tileSize);
    g.userData = { tile: id };
    scene.add(g);
    root = g;
  }
  const hgt = c.sections.get('HGT');
  if (!hgt) throw new Error(`tile ${id} has no HGT section`);
  const rgb = c.sections.get('RGB');
  const heights = decodeHeights(hgt);
  const t = terrainArrays(heights, rgb ? decodeGrid(rgb, 3).data : null, c.i * tileSize, c.j * tileSize, tileSize);

  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(t.positions, 3));
  geo.setAttribute('normal', new BufferAttribute(t.normals, 3));
  geo.setAttribute('uv', new BufferAttribute(t.uvs, 2));
  if (t.colors) geo.setAttribute('color', new BufferAttribute(t.colors, 4, true));
  geo.setAttribute('_feature_id_0', new BufferAttribute(new Float32Array(t.positions.length / 3), 1));
  geo.setIndex(new BufferAttribute(t.indices, 1));
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  const featureIds = [`terrain:${id}`];
  geo.userData = { layer: 'terrain', feature_ids: featureIds };
  // The baked terrain's glTF material: white base colour, roughness 0.95, vertex colours.
  const material = new MeshStandardMaterial({ name: 'terrain', color: 0xffffff, roughness: 0.95, metalness: 0, vertexColors: !!t.colors });
  const mesh = new Mesh(geo, material);
  mesh.name = 'terrain';
  mesh.userData = { layer: 'terrain', feature_ids: featureIds };
  root.add(mesh);
  // Layer order as in baked tiles: terrain first.
  root.children.splice(root.children.indexOf(mesh), 1);
  root.children.unshift(mesh);

  const tre = c.sections.get('TRE');
  const library = world.archetypes ? await world.archetypes : null;
  if (tre && world.trees && library) {
    const ctx: TreeContext = { i: c.i, j: c.j, tileSize, ...world.trees };
    addTrees(root, scene, decodeTrees(tre, c.sections.get('TOB')), heights, ctx, library);
  }
  return scene;
}

/**
 * The `trees` layer of a data tile, structured like GLTFLoader's output for a baked
 * tile's EXT_mesh_gpu_instancing nodes: `trees` > one group per archetype (userData
 * `layer`, `archetype`, `instance_ids`) > one InstancedMesh per archetype part.
 */
function addTrees(root: Object3D, scene: Group, records: ReturnType<typeof decodeTrees>, heights: HeightGrid,
  ctx: TreeContext, library: ArchetypeLibrary): void {
  if (!records.count) return;
  const inst = treeInstances(records, heights, ctx);
  const layer = new Group();
  layer.name = 'trees';
  layer.userData = { layer: 'trees' };
  const rowOf = new Map<string, number>();
  const m = new Matrix4();
  const p = new Vector3();
  const q = new Quaternion();
  const s = new Vector3();
  for (const [code, rows] of inst.byArchetype) {
    const arch = ctx.archetypes[code];
    const parts = arch ? library.get(arch) : undefined;
    if (!parts) continue;
    const ids = rows.map((n) => treeIdAt(records, n, ctx));
    ids.forEach((id, k) => rowOf.set(id, rows[k]));
    const group = new Group();
    group.name = arch;
    group.userData = { layer: 'trees', archetype: arch, instance_ids: ids };
    for (const part of parts) {
      const im = new InstancedMesh(part.geometry, part.material.clone(), rows.length);
      im.name = part.name;
      rows.forEach((n, k) => {
        const o = 10 * n;
        p.set(inst.trs[o], inst.trs[o + 1], inst.trs[o + 2]);
        q.set(inst.trs[o + 3], inst.trs[o + 4], inst.trs[o + 5], inst.trs[o + 6]);
        s.set(inst.trs[o + 7], inst.trs[o + 8], inst.trs[o + 9]);
        im.setMatrixAt(k, m.compose(p, q, s));
      });
      im.instanceMatrix.needsUpdate = true;
      im.computeBoundingSphere();
      im.userData = { layer: 'trees', archetype: arch, instance_ids: ids };
      group.add(im);
    }
    layer.add(group);
  }
  root.add(layer);
  scene.userData[RECORD_PROVIDER] = (id: string): EntityRecord | undefined => {
    const n = rowOf.get(id);
    return n === undefined ? undefined : treeRecordAt(records, n, ctx, heights);
  };
}
