/**
 * Quadtree tile addressing shared by every tileset level.
 *
 * Level 0 is the finest level (the 256 m tiles of a 1.x world). A tile at
 * level L covers `tileSize * 2^L` metres; tile (i, j) spans
 * x ∈ [i·s, (i+1)·s), y ∈ [j·s, (j+1)·s) in local ENU metres, so indices may
 * be negative. Parents are found with floor shifts (`i >> 1` rounds towards
 * −∞), which keeps the tree aligned across the origin.
 *
 * Pure module: no three.js, no DOM (unit-tested with node --test).
 */

export interface TileKey {
  /** 0 = finest level; each level up doubles the tile size. */
  level: number;
  i: number;
  j: number;
}

/** Axis-aligned bounds in local ENU metres: [xmin, ymin, xmax, ymax]. */
export type Bounds2 = [number, number, number, number];

/** Stable string id of a key, e.g. "0/-16/-17". */
export function keyId(k: TileKey): string {
  return `${k.level}/${k.i}/${k.j}`;
}

export function parseKeyId(id: string): TileKey | null {
  const m = /^(\d+)\/(-?\d+)\/(-?\d+)$/.exec(id);
  if (!m) return null;
  return { level: Number(m[1]), i: Number(m[2]), j: Number(m[3]) };
}

/** Floor division by 2^n for (32-bit) integers, correct for negative values. */
export function floorShift(v: number, n: number): number {
  return v >> n;
}

/** Parent key one level up (coarser). */
export function parentKey(k: TileKey): TileKey {
  return { level: k.level + 1, i: floorShift(k.i, 1), j: floorShift(k.j, 1) };
}

/** Ancestor of `k` at `level` (>= k.level). */
export function ancestorKey(k: TileKey, level: number): TileKey {
  if (level < k.level) throw new Error(`ancestor level ${level} below key level ${k.level}`);
  const n = level - k.level;
  return { level, i: floorShift(k.i, n), j: floorShift(k.j, n) };
}

/** The four children one level down (finer), in (i, j) row-major order. */
export function childKeys(k: TileKey): TileKey[] {
  if (k.level <= 0) return [];
  const level = k.level - 1;
  const i = k.i * 2;
  const j = k.j * 2;
  return [
    { level, i, j },
    { level, i: i + 1, j },
    { level, i, j: j + 1 },
    { level, i: i + 1, j: j + 1 },
  ];
}

/** True when `a` is a strict ancestor of `b`. */
export function isAncestor(a: TileKey, b: TileKey): boolean {
  if (a.level <= b.level) return false;
  const anc = ancestorKey(b, a.level);
  return anc.i === a.i && anc.j === a.j;
}

/** Edge length (m) of a tile at `level`. */
export function levelSize(tileSize: number, level: number): number {
  return tileSize * 2 ** level;
}

export function keyBounds(k: TileKey, tileSize: number): Bounds2 {
  const s = levelSize(tileSize, k.level);
  return [k.i * s, k.j * s, (k.i + 1) * s, (k.j + 1) * s];
}

/** Key of the tile at `level` containing local point (x, y). */
export function keyAt(x: number, y: number, tileSize: number, level = 0): TileKey {
  const s = levelSize(tileSize, level);
  return { level, i: Math.floor(x / s), j: Math.floor(y / s) };
}
