/**
 * Scene of a data-first tile (`.wgt`): the embedded GLB's layers plus a terrain
 * mesh generated from the heightmap, arranged exactly like a baked tile (root
 * node `tile_<i>_<j>` translated to the tile corner, layer children with
 * `userData.layer`, terrain feature id `terrain:<i>_<j>`), so the world model,
 * picking, colour modes and photoreal mode treat both formats alike.
 *
 * Tiles of a coarser level (format 2.4) are `2^level` times the size: root
 * `tile_<level>/<i>_<j>`, a terrain mesh with optional skirts and cells limited
 * to what it covers, and (2.5) simplified buildings and trees. Their content has
 * no entities: no feature or instance ids, and `userData.level` on every mesh.
 */
import { BufferAttribute, BufferGeometry, Group, InstancedMesh, LinearSRGBColorSpace, Matrix4, Mesh, MeshStandardMaterial, Quaternion, Vector3 } from 'three';
import type { Object3D } from 'three';
import type { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { decodeGrid, decodeHeights, readWgt } from './wgt.ts';
import type { HeightGrid } from './wgt.ts';
import { terrainArrays } from '../gen/terrainMesh.ts';
import type { TerrainOptions } from '../gen/terrainMesh.ts';
import { decodeTrees, treeIdAt, treeInstances, treeRecordAt } from './trees.ts';
import type { TreeContext } from './trees.ts';
import type { ArchetypeLibrary } from './archetypes.ts';
import { decodeBuildings } from './buildings.ts';
import { buildingMeshes } from '../gen/buildingMesh.ts';
import { LAYER_NAMES } from '../types.ts';
import type { BuildingMaterialDoc, EntityRecord } from '../types.ts';

/** World-level inputs of data tiles (tileset.json). */
export interface WgtWorld {
  tileSize: number;
  /** Tree archetype meshes (tiles/archetypes.glb), shared by every tile. */
  archetypes?: Promise<ArchetypeLibrary | null>;
  trees?: { cellSize: number; seed: number; archetypes: readonly string[]; categories: readonly string[] };
  /** Building material palette (tileset.json `buildings.materials`), indexed by BLD material code. */
  buildingMaterials?: readonly BuildingMaterialDoc[];
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

export async function buildWgtScene(buffer: ArrayBuffer, loader: GLTFLoader, baseUrl: string, world: WgtWorld,
  terrain: TerrainOptions = {}): Promise<Group> {
  const c = await readWgt(buffer);
  // Level-0 tiles keep the 1.x ids; a coarser tile is 2^level times the size.
  const tileSize = world.tileSize * 2 ** c.level;
  const id = c.level ? `${c.level}/${c.i}_${c.j}` : `${c.i}_${c.j}`;
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
  const t = terrainArrays(heights, rgb ? decodeGrid(rgb, 3).data : null, c.i * tileSize, c.j * tileSize, tileSize, terrain);

  const geo = new BufferGeometry();
  geo.setAttribute('position', new BufferAttribute(t.positions, 3));
  geo.setAttribute('normal', new BufferAttribute(t.normals, 3));
  geo.setAttribute('uv', new BufferAttribute(t.uvs, 2));
  if (t.colors) geo.setAttribute('color', new BufferAttribute(t.colors, 4, true));
  geo.setAttribute('_feature_id_0', new BufferAttribute(new Float32Array(t.positions.length / 3), 1));
  geo.setIndex(new BufferAttribute(t.indices, 1));
  geo.computeBoundingBox();
  geo.computeBoundingSphere();
  // A coarser tile stands for many level-0 tiles: no entity of its own.
  const featureIds = c.level ? [] : [`terrain:${id}`];
  geo.userData = { layer: 'terrain', feature_ids: featureIds };
  // The baked terrain's glTF material: white base colour, roughness 0.95, vertex colours.
  const material = new MeshStandardMaterial({ name: 'terrain', color: 0xffffff, roughness: 0.95, metalness: 0, vertexColors: !!t.colors });
  const mesh = new Mesh(geo, material);
  mesh.name = 'terrain';
  mesh.userData = c.level ? { layer: 'terrain', feature_ids: featureIds, level: c.level } : { layer: 'terrain', feature_ids: featureIds };
  root.add(mesh);
  // Layer order as in baked tiles: terrain first.
  root.children.splice(root.children.indexOf(mesh), 1);
  root.children.unshift(mesh);

  const bld = c.sections.get('BLD');
  if (bld) addBuildings(root, bld, c.sections.get('IDS'), c.i * tileSize, c.j * tileSize, world.buildingMaterials ?? [], c.level);

  const tre = c.sections.get('TRE');
  const library = world.archetypes ? await world.archetypes : null;
  if (tre && world.trees && library) {
    const ctx: TreeContext = { i: c.i, j: c.j, tileSize, ...world.trees };
    addTrees(root, scene, decodeTrees(tre, c.sections.get('TOB')), heights, ctx, library, c.level);
  }
  return scene;
}

/**
 * The `buildings` layer of a data tile, generated from its footprint records and structured
 * like GLTFLoader's output for a baked tile: a `buildings` group (userData `layer`,
 * `feature_ids`) holding one mesh per material whose `_feature_id_0` indexes those ids.
 */
function addBuildings(root: Object3D, bld: Uint8Array, ids: Uint8Array | undefined, x0: number, y0: number,
  palette: readonly BuildingMaterialDoc[], level: number): void {
  const rec = decodeBuildings(bld, ids);
  if (!rec.count) return;
  // Coarser levels: prisms standing for buildings, without entities.
  const featureIds = level ? [] : rec.ids;
  const layer = new Group();
  layer.name = 'buildings';
  layer.userData = { layer: 'buildings', feature_ids: featureIds };
  for (const [code, a] of buildingMeshes(rec, x0, y0)) {
    const doc = palette[code] ?? { name: 'building_default', color: [0.78, 0.76, 0.72, 1], roughness: 0.85, metallic: 0 };
    const geo = new BufferGeometry();
    geo.setAttribute('position', new BufferAttribute(a.positions, 3));
    geo.setAttribute('normal', new BufferAttribute(a.normals, 3));
    geo.setAttribute('uv', new BufferAttribute(a.uvs, 2));
    geo.setAttribute('_feature_id_0', new BufferAttribute(a.features, 1));
    geo.setIndex(new BufferAttribute(a.indices, 1));
    geo.computeBoundingBox();
    geo.computeBoundingSphere();
    geo.userData = { layer: 'buildings', feature_ids: featureIds };
    // As GLTFLoader builds it from the glTF material: linear base colour factor.
    const material = new MeshStandardMaterial({ name: doc.name, roughness: doc.roughness, metalness: doc.metallic });
    material.color.setRGB(doc.color[0], doc.color[1], doc.color[2], LinearSRGBColorSpace);
    material.opacity = doc.color[3];
    const mesh = new Mesh(geo, material);
    mesh.name = 'buildings';
    mesh.userData = level ? { layer: 'buildings', feature_ids: featureIds, level } : { layer: 'buildings', feature_ids: featureIds };
    layer.add(mesh);
  }
  // Keep the baked layer order (terrain, roads, bridges, tunnels, buildings, water, ...).
  const rank = (o: Object3D) => LAYER_NAMES.indexOf((o.userData as { layer?: string }).layer as (typeof LAYER_NAMES)[number]);
  const at = root.children.findIndex((o) => rank(o) > LAYER_NAMES.indexOf('buildings'));
  root.add(layer);
  if (at >= 0) {
    root.children.splice(root.children.indexOf(layer), 1);
    root.children.splice(at, 0, layer);
  }
}

/**
 * The `trees` layer of a data tile, structured like GLTFLoader's output for a baked
 * tile's EXT_mesh_gpu_instancing nodes: `trees` > one group per archetype (userData
 * `layer`, `archetype`, `instance_ids`) > one InstancedMesh per archetype part.
 */
function addTrees(root: Object3D, scene: Group, records: ReturnType<typeof decodeTrees>, heights: HeightGrid,
  ctx: TreeContext, library: ArchetypeLibrary, level: number): void {
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
    // Coarser levels: a thinned stand-in for the trees, without entities.
    const ids = level ? [] : rows.map((n) => treeIdAt(records, n, ctx));
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
      im.userData = level ? { layer: 'trees', archetype: arch, instance_ids: ids, level } : { layer: 'trees', archetype: arch, instance_ids: ids };
      group.add(im);
    }
    layer.add(group);
  }
  root.add(layer);
  if (level) return;
  scene.userData[RECORD_PROVIDER] = (id: string): EntityRecord | undefined => {
    const n = rowOf.get(id);
    return n === undefined ? undefined : treeRecordAt(records, n, ctx, heights);
  };
}
