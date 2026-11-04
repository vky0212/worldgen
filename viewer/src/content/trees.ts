/**
 * Trees of a data-first tile: the `TRE ` records (16 bytes per rendered tree) and
 * `TOB ` (full records of trees whose id cannot be derived, i.e. mapped trees).
 * Layout and semantics: src/worldgen/export/instances.py; checked against
 * tests/conformance/tree_tiles.json. Pure module (no three.js).
 */
import { hashU64 } from '../gen/hash64.ts';
import { heightAt } from '../gen/heightQuery.ts';
import type { HeightGrid } from './wgt.ts';
import type { EntityRecord } from '../types';

export const FLAG_OBSERVED = 1;
export const FLAG_TOB = 2;
export const FLAG_GROUND = 4;
const RECORD = 16;

export interface TreeRecords {
  count: number;
  view: DataView;
  /** Mapped trees' full records (TOB), referenced by index. */
  tob: { id: string; record: EntityRecord }[];
}

/** World-level parameters (tileset.json "trees"). */
export interface TreeContext {
  i: number;
  j: number;
  tileSize: number;
  cellSize: number;
  seed: number;
  archetypes: readonly string[];
  categories: readonly string[];
}

export function decodeTrees(tre: Uint8Array, tob: Uint8Array | undefined): TreeRecords {
  const view = new DataView(tre.buffer, tre.byteOffset, tre.byteLength);
  const count = view.getUint32(0, true);
  const size = view.getUint32(4, true);
  if (size !== RECORD) throw new Error(`unsupported TRE record size ${size}`);
  if (8 + count * RECORD > tre.byteLength) throw new Error('TRE section is truncated');
  const body = new DataView(tre.buffer, tre.byteOffset + 8, count * RECORD);
  const parsed = tob && tob.byteLength ? (JSON.parse(new TextDecoder().decode(tob)) as { id: string; record: EntityRecord }[]) : [];
  return { count, view: body, tob: parsed };
}

/** Raw fields of record `n`. */
export function treeFields(t: TreeRecords, n: number) {
  const v = t.view;
  const o = n * RECORD;
  return {
    x: v.getUint16(o, true),
    y: v.getUint16(o + 2, true),
    heightCm: v.getUint16(o + 4, true),
    k: v.getUint16(o + 6, true),
    crownDm: v.getUint8(o + 8),
    trunkCm: v.getUint8(o + 9),
    yaw: v.getUint8(o + 10),
    archCat: v.getUint8(o + 11),
    confidence: v.getUint8(o + 12),
    flags: v.getUint8(o + 13),
    dx: v.getUint8(o + 14),
    dy: v.getUint8(o + 15),
  };
}

function cellOrigin(ctx: TreeContext): [number, number] {
  return [Math.floor((ctx.i * ctx.tileSize) / ctx.cellSize), Math.floor((ctx.j * ctx.tileSize) / ctx.cellSize)];
}

/** Entity id of record `n`: derived (`tree:gen:<i>_<j>_[g]<k>`) or from TOB. */
export function treeIdAt(t: TreeRecords, n: number, ctx: TreeContext): string {
  const f = treeFields(t, n);
  if (f.flags & FLAG_TOB) return t.tob[f.k]?.id ?? `tree:unknown:${n}`;
  const [cx, cy] = cellOrigin(ctx);
  return `tree:gen:${cx + f.dx}_${cy + f.dy}_${f.flags & FLAG_GROUND ? 'g' : ''}${f.k}`;
}

/** Local ENU position (x, y) of record `n`. */
export function treePosition(t: TreeRecords, n: number, ctx: TreeContext): [number, number] {
  const f = treeFields(t, n);
  return [ctx.i * ctx.tileSize + (f.x / 65535) * ctx.tileSize, ctx.j * ctx.tileSize + (f.y / 65535) * ctx.tileSize];
}

export interface TreeInstances {
  /** Archetype code (index into ctx.archetypes) -> record numbers, in TRE order. */
  byArchetype: Map<number, number[]>;
  /** Per record: translation (3, glTF axes, relative to the tile corner), rotation (4), scale (3). */
  trs: Float32Array;
}

/**
 * Instance transforms of every record, exactly as the compiler places its archetypes:
 * translation at the trunk base sunk by 0.15 m + trunk radius, rotation `yaw` about the
 * vertical, scale (crown, height + sink, crown). Ground height from the tile's heightmap.
 */
export function treeInstances(t: TreeRecords, g: HeightGrid, ctx: TreeContext): TreeInstances {
  const byArchetype = new Map<number, number[]>();
  const trs = new Float32Array(10 * t.count);
  const x0 = ctx.i * ctx.tileSize;
  const y0 = ctx.j * ctx.tileSize;
  for (let n = 0; n < t.count; n++) {
    const f = treeFields(t, n);
    const arch = f.archCat & 15;
    let list = byArchetype.get(arch);
    if (!list) byArchetype.set(arch, (list = []));
    list.push(n);
    const x = x0 + (f.x / 65535) * ctx.tileSize;
    const y = y0 + (f.y / 65535) * ctx.tileSize;
    const sink = 0.15 + f.trunkCm / 100;
    const ground = heightAt(g, x, y);
    const yaw = (f.yaw / 256) * 2 * Math.PI;
    const crown = f.crownDm / 10;
    const o = 10 * n;
    trs[o] = x - x0;
    trs[o + 1] = ground - sink;
    trs[o + 2] = -(y - y0);
    trs[o + 3] = 0;
    trs[o + 4] = Math.sin(yaw / 2);
    trs[o + 5] = 0;
    trs[o + 6] = Math.cos(yaw / 2);
    trs[o + 7] = crown;
    trs[o + 8] = f.heightCm / 100 + sink;
    trs[o + 9] = crown;
  }
  return { byArchetype, trs };
}

/** Published per-tree seed of a generated tree (hash of seed, salt, cell, k), as a decimal string. */
export function treeSeed(t: TreeRecords, n: number, ctx: TreeContext): string | null {
  const f = treeFields(t, n);
  if (f.flags & FLAG_TOB) return null;
  const [cx, cy] = cellOrigin(ctx);
  const salt = f.flags & FLAG_GROUND ? 6 : 3;
  return (hashU64(ctx.seed, salt, cx + f.dx, cy + f.dy, f.k) & 0x7fffffffffffffffn).toString();
}

const round = (v: number, d: number): number => Math.round(v * 10 ** d) / 10 ** d;

/**
 * The inspect record of record `n` (what a baked tile's meta.json held for the tree), with
 * values at the record's precision (cm heights, dm crowns).
 */
export function treeRecordAt(t: TreeRecords, n: number, ctx: TreeContext, g: HeightGrid): EntityRecord {
  const f = treeFields(t, n);
  if (f.flags & FLAG_TOB) {
    const r = t.tob[f.k];
    if (r) return r.record;
  }
  const [x, y] = treePosition(t, n, ctx);
  const ground = heightAt(g, x, y);
  const seed = treeSeed(t, n, ctx);
  const est = (value: number) => ({ value: round(value, 2), method: 'procedural', confidence: 0.3 });
  return {
    kind: 'tree',
    layer: 'trees',
    source: 'procedural',
    source_id: `seed:${ctx.seed}`,
    confidence: round(f.confidence / 255, 3),
    properties: {
      archetype: ctx.archetypes[f.archCat & 15] ?? 'unknown',
      category: ctx.categories[f.archCat >> 4] ?? 'none',
      ground_elevation: round(ground, 2),
      observed: false,
      seed,
    },
    inferred: { height: est(f.heightCm / 100), crown_radius: est(f.crownDm / 10), trunk_radius: est(f.trunkCm / 100) },
    missing: ['species'],
    uncertain: true,
    centroid: [round(x, 2), round(y, 2), round(ground, 2)],
  } as EntityRecord;
}
