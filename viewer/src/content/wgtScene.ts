/**
 * Scene of a data-first tile (`.wgt`): the embedded GLB's layers plus a terrain
 * mesh generated from the heightmap, arranged exactly like a baked tile (root
 * node `tile_<i>_<j>` translated to the tile corner, layer children with
 * `userData.layer`, terrain feature id `terrain:<i>_<j>`), so the world model,
 * picking, colour modes and photoreal mode treat both formats alike.
 */
import { BufferAttribute, BufferGeometry, Group, Mesh, MeshStandardMaterial } from 'three';
import type { Object3D } from 'three';
import type { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { decodeGrid, decodeHeights, readWgt } from './wgt.ts';
import { terrainArrays } from '../gen/terrainMesh.ts';

function tileRootOf(scene: Group, name: string): Object3D | null {
  let found: Object3D | null = null;
  scene.traverse((o) => {
    if (!found && (o.name === name || (o.userData as Record<string, unknown>).tile !== undefined)) found = o;
  });
  return found;
}

export async function buildWgtScene(buffer: ArrayBuffer, loader: GLTFLoader, baseUrl: string, tileSize: number): Promise<Group> {
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
  const t = terrainArrays(decodeHeights(hgt), rgb ? decodeGrid(rgb, 3).data : null, c.i * tileSize, c.j * tileSize, tileSize);

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
  return scene;
}
