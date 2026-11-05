/**
 * `BLD` / `IDS` sections of data-first tiles: building footprints with the compiler's resolved
 * roofs (layout in docs/output-format.md and src/worldgen/export/footprints.py).
 */

/** Decoded `BLD` columns plus the first polygon / ring / vertex of each row. */
export interface BuildingRecords {
  count: number;
  /** Floor z per building (metres, local frame). */
  bottom: Float32Array;
  /** Roof top z per building. */
  top: Float32Array;
  polygons: Uint16Array;
  /** Index into tileset.json `buildings.materials`. */
  material: Uint8Array;
  /** Per polygon: 0 flat, 1 gabled, 2 skillion, 3 pyramidal. */
  shape: Uint8Array;
  rings: Uint16Array;
  eave: Float32Array;
  /** Ridge centre (gabled, skillion) or apex (pyramidal), metres from the tile corner. */
  cx: Float32Array;
  cy: Float32Array;
  /** Ridge direction (unit). */
  ux: Float32Array;
  uy: Float32Array;
  /** Half width across the ridge. */
  half: Float32Array;
  vertices: Uint32Array;
  /** Vertex (x, y) pairs in millimetres from the tile corner. */
  xy: Int32Array;
  polyStart: Uint32Array;
  ringStart: Uint32Array;
  vertStart: Uint32Array;
  /** Entity id per building (`IDS`). */
  ids: string[];
}

export const ROOF_FLAT = 0;
export const ROOF_GABLED = 1;
export const ROOF_SKILLION = 2;
export const ROOF_PYRAMIDAL = 3;

type Kind = 'f32' | 'u32' | 'i32' | 'u16' | 'u8';
const SIZE: Record<Kind, number> = { f32: 4, u32: 4, i32: 4, u16: 2, u8: 1 };

function starts(counts: ArrayLike<number>): Uint32Array {
  const out = new Uint32Array(counts.length + 1);
  for (let k = 0; k < counts.length; k++) out[k + 1] = out[k] + counts[k];
  return out;
}

export function decodeBuildings(bld: Uint8Array, ids: Uint8Array | undefined): BuildingRecords {
  const view = new DataView(bld.buffer, bld.byteOffset, bld.byteLength);
  if (bld.byteLength < 16) throw new Error('BLD section is truncated');
  const n = { B: view.getUint32(0, true), P: view.getUint32(4, true), R: view.getUint32(8, true), V: view.getUint32(12, true) };
  let off = 16;
  // Columns in file order (each starts 4-byte aligned).
  const read = (kind: Kind, len: number) => {
    const bytes = len * SIZE[kind];
    if (off + bytes > bld.byteLength) throw new Error('BLD section is truncated');
    let out: Float32Array | Uint32Array | Int32Array | Uint16Array | Uint8Array;
    if (kind === 'f32') {
      const a = new Float32Array(len);
      for (let k = 0; k < len; k++) a[k] = view.getFloat32(off + 4 * k, true);
      out = a;
    } else if (kind === 'u32') {
      const a = new Uint32Array(len);
      for (let k = 0; k < len; k++) a[k] = view.getUint32(off + 4 * k, true);
      out = a;
    } else if (kind === 'i32') {
      const a = new Int32Array(len);
      for (let k = 0; k < len; k++) a[k] = view.getInt32(off + 4 * k, true);
      out = a;
    } else if (kind === 'u16') {
      const a = new Uint16Array(len);
      for (let k = 0; k < len; k++) a[k] = view.getUint16(off + 2 * k, true);
      out = a;
    } else {
      out = bld.slice(off, off + len);
    }
    off += bytes;
    off += (4 - (off % 4)) % 4;
    return out;
  };
  const bottom = read('f32', n.B) as Float32Array;
  const top = read('f32', n.B) as Float32Array;
  const polygons = read('u16', n.B) as Uint16Array;
  const material = read('u8', n.B) as Uint8Array;
  const shape = read('u8', n.P) as Uint8Array;
  const rings = read('u16', n.P) as Uint16Array;
  const eave = read('f32', n.P) as Float32Array;
  const cx = read('f32', n.P) as Float32Array;
  const cy = read('f32', n.P) as Float32Array;
  const ux = read('f32', n.P) as Float32Array;
  const uy = read('f32', n.P) as Float32Array;
  const half = read('f32', n.P) as Float32Array;
  const vertices = read('u32', n.R) as Uint32Array;
  const xy = read('i32', 2 * n.V) as Int32Array;
  const idList = ids && ids.byteLength ? ((JSON.parse(new TextDecoder().decode(ids)) as { BLD?: string[] }).BLD ?? []) : [];
  return {
    count: n.B, bottom, top, polygons, material, shape, rings, eave, cx, cy, ux, uy, half, vertices, xy,
    polyStart: starts(polygons), ringStart: starts(rings), vertStart: starts(vertices),
    ids: Array.from({ length: n.B }, (_, k) => idList[k] ?? `building:#${k}`),
  };
}
