/**
 * The tree archetype meshes of a data-first world (`tiles/archetypes.glb`, one root
 * node per archetype), loaded once and instanced by every tile. Geometries are
 * shared between tiles (never disposed with a tile); materials are cloned per
 * tile, because colour modes and photoreal mode adjust them per tile.
 */
import type { BufferGeometry, Material, Object3D } from 'three';
import { Mesh } from 'three';
import type { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import { markShared } from '../stream/content';

export interface ArchetypePart {
  name: string;
  geometry: BufferGeometry;
  material: Material;
}

export type ArchetypeLibrary = Map<string, ArchetypePart[]>;

export async function loadArchetypes(loader: GLTFLoader, url: string): Promise<ArchetypeLibrary> {
  const gltf = await loader.loadAsync(url);
  const lib: ArchetypeLibrary = new Map();
  for (const node of gltf.scene.children as Object3D[]) {
    const arch = String((node.userData as Record<string, unknown>).archetype ?? node.name);
    const parts: ArchetypePart[] = [];
    node.traverse((o) => {
      if (!(o instanceof Mesh)) return;
      const material = Array.isArray(o.material) ? o.material[0] : o.material;
      parts.push({ name: material?.name ?? arch, geometry: markShared(o.geometry as BufferGeometry), material });
    });
    if (parts.length) lib.set(arch, parts);
  }
  return lib;
}
