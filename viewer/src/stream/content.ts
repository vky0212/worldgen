/**
 * Tile content on the GPU: what a loaded tile owns, how much GPU memory it
 * takes and how to free it.
 *
 * Resources shared between tiles (tree impostor geometries, colour-mode and
 * highlight materials) carry `userData.worldgenShared = true` and are never
 * disposed here.
 */
import { InstancedMesh } from 'three';
import type { BufferAttribute, BufferGeometry, Group, InterleavedBufferAttribute, Material, Object3D, Texture } from 'three';

/** Drawable content of one tile. */
export interface TileContent {
  root: Group;
  pickables: readonly unknown[];
  /** Estimated GPU memory (bytes): vertex/index/instance buffers and textures. */
  gpuBytes: number;
  /** Frees every GPU resource the tile owns. The root must be detached first. */
  dispose(): void;
}

/** Marks a geometry, material or texture as shared: tile disposal skips it. */
export function markShared<T extends { userData: Record<string, unknown> }>(resource: T): T {
  resource.userData.worldgenShared = true;
  return resource;
}

export function isShared(resource: { userData?: Record<string, unknown> } | null | undefined): boolean {
  return resource?.userData?.worldgenShared === true;
}

interface Drawable extends Object3D {
  geometry?: BufferGeometry;
  material?: Material | Material[];
}

function materialsOf(o: Drawable): Material[] {
  const m = o.material;
  if (!m) return [];
  return Array.isArray(m) ? m : [m];
}

function texturesOf(m: Material): Texture[] {
  const out: Texture[] = [];
  for (const key of Object.keys(m)) {
    const v = (m as unknown as Record<string, unknown>)[key];
    if (v && typeof v === 'object' && (v as Texture).isTexture) out.push(v as Texture);
  }
  return out;
}

function textureBytes(t: Texture): number {
  const img = t.image as { width?: number; height?: number; data?: ArrayBufferView } | undefined;
  if (!img) return 0;
  if (img.data && typeof img.data.byteLength === 'number') return img.data.byteLength;
  const w = Number(img.width) || 0;
  const h = Number(img.height) || 0;
  return Math.round(w * h * 4 * (t.generateMipmaps ? 4 / 3 : 1));
}

/** Sum of the buffers and textures under `root` (each counted once). */
export function estimateGpuBytes(root: Object3D): number {
  const arrays = new Set<ArrayBufferLike | ArrayBufferView>();
  const textures = new Set<Texture>();
  let bytes = 0;
  const addAttr = (a: BufferAttribute | InterleavedBufferAttribute | null | undefined): void => {
    if (!a) return;
    const arr = (a as InterleavedBufferAttribute).isInterleavedBufferAttribute
      ? (a as InterleavedBufferAttribute).data.array
      : (a as BufferAttribute).array;
    if (!arr || arrays.has(arr)) return;
    arrays.add(arr);
    bytes += arr.byteLength;
  };
  root.traverse((obj) => {
    const o = obj as Drawable;
    const geo = o.geometry;
    if (geo && geo.isBufferGeometry && !isShared(geo)) {
      for (const name of Object.keys(geo.attributes)) addAttr(geo.attributes[name] as BufferAttribute);
      addAttr(geo.index);
    }
    if (o instanceof InstancedMesh) {
      addAttr(o.instanceMatrix);
      addAttr(o.instanceColor);
    }
    for (const m of materialsOf(o)) {
      if (isShared(m)) continue;
      for (const t of texturesOf(m)) {
        if (textures.has(t) || isShared(t)) continue;
        textures.add(t);
        bytes += textureBytes(t);
      }
    }
  });
  return bytes;
}

/**
 * Disposes every geometry, material and texture under `root` that is not
 * shared, and the instance buffers of instanced meshes. Image bitmaps are
 * closed so their decoded pixels are released right away.
 */
export function disposeTree(root: Object3D): void {
  const geometries = new Set<BufferGeometry>();
  const materials = new Set<Material>();
  const instanced: InstancedMesh[] = [];
  root.traverse((obj) => {
    const o = obj as Drawable;
    if (o.geometry && o.geometry.isBufferGeometry && !isShared(o.geometry)) geometries.add(o.geometry);
    for (const m of materialsOf(o)) if (!isShared(m)) materials.add(m);
    if (o instanceof InstancedMesh) instanced.push(o);
  });
  const textures = new Set<Texture>();
  for (const m of materials) for (const t of texturesOf(m)) if (!isShared(t)) textures.add(t);
  // Instance buffers first (the renderer frees them on the mesh's dispose event).
  for (const m of instanced) m.dispose();
  for (const g of geometries) g.dispose();
  for (const m of materials) m.dispose();
  for (const t of textures) {
    t.dispose();
    const img = t.image as { close?: () => void } | undefined;
    if (img && typeof img.close === 'function') img.close();
  }
}
