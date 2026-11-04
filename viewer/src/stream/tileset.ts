/**
 * Tileset adapter: turns a world's tile index into a quadtree of TileNodes
 * that the traversal walks.
 *
 * Schema 1.x worlds have a flat `tiles/index.json` with one level of 256 m
 * tiles. They become level-0 nodes with content, under virtual (content-less)
 * ancestors up to a few roots, so the traversal can cull whole blocks of the
 * world at once. Content-less nodes always refine: a 1.x world has nothing
 * coarser to show, so the finest tiles are drawn wherever they are visible
 * (limited by the view range and the GPU budget). Later formats (2.x) plug in
 * coarser levels with content and real geometric errors.
 *
 * Pure module: no three.js, no DOM (unit-tested with node --test).
 */
import type { SubtreeDoc, TileEntry, TileIndex, TilesetDoc } from '../types';
import { keyBounds, keyId, parentKey } from './quadtree.ts';
import type { Bounds2, TileKey } from './quadtree.ts';

/** Loadable content of a node (one 1.x tile). */
export interface ContentRef {
  /** Content id; for 1.x the tile id from the index ("i_j"). */
  id: string;
  entry: TileEntry;
}

export interface TileNode {
  key: TileKey;
  /** Key id ("level/i/j"). */
  id: string;
  bounds: Bounds2;
  zmin: number;
  zmax: number;
  /**
   * Geometric error (m) of drawing this node instead of its children: the
   * traversal refines when it projects to more than tau pixels. 0 for leaves.
   */
  geometricError: number;
  content: ContentRef | null;
  parent: TileNode | null;
  children: TileNode[];
}

export interface Tileset {
  tileSize: number;
  roots: TileNode[];
  /** Every node by key id. */
  nodes: Map<string, TileNode>;
  /** Nodes with content by content id. */
  byContentId: Map<string, TileNode>;
  contentNodes: TileNode[];
  /** Index entries that could not be placed (duplicate keys). */
  skipped: number;
}

const DEFAULT_TILE_SIZE = 256;
/** Stop adding virtual levels once at most this many roots remain. */
const MAX_ROOTS = 4;
const MAX_VIRTUAL_LEVELS = 12;

/** Key of a 1.x index entry: its i/j, or the cell of its bounds' centre. */
export function entryKey(entry: TileEntry, tileSize: number): TileKey {
  if (Number.isInteger(entry.i) && Number.isInteger(entry.j)) return { level: 0, i: entry.i, j: entry.j };
  const b = entry.bounds;
  const cx = (b[0] + b[2]) / 2;
  const cy = (b[1] + b[3]) / 2;
  return { level: 0, i: Math.floor(cx / tileSize), j: Math.floor(cy / tileSize) };
}

/**
 * Builds the single-level quadtree of a schema 1.x tile index. `defaultZ` is
 * used for entries without a zrange.
 */
export function tilesetFromIndex(index: TileIndex, defaultZ: [number, number] = [0, 50]): Tileset {
  const tileSize = Number(index.tile_size) > 0 ? Number(index.tile_size) : DEFAULT_TILE_SIZE;
  const nodes = new Map<string, TileNode>();
  const byContentId = new Map<string, TileNode>();
  const contentNodes: TileNode[] = [];
  let skipped = 0;

  for (const entry of index.tiles) {
    if (!entry || typeof entry.id !== 'string') {
      skipped++;
      continue;
    }
    const key = entryKey(entry, tileSize);
    const id = keyId(key);
    if (nodes.has(id) || byContentId.has(entry.id)) {
      skipped++;
      continue;
    }
    const z = Array.isArray(entry.zrange) && entry.zrange.length >= 2 ? entry.zrange : defaultZ;
    const bounds: Bounds2 = Array.isArray(entry.bounds) && entry.bounds.length >= 4
      ? [entry.bounds[0], entry.bounds[1], entry.bounds[2], entry.bounds[3]]
      : keyBounds(key, tileSize);
    const node: TileNode = {
      key,
      id,
      bounds,
      zmin: Number(z[0]),
      zmax: Number(z[1]),
      geometricError: 0,
      content: { id: entry.id, entry },
      parent: null,
      children: [],
    };
    nodes.set(id, node);
    byContentId.set(entry.id, node);
    contentNodes.push(node);
  }

  // Virtual ancestors, level by level, until only a few roots remain.
  let level: TileNode[] = contentNodes;
  for (let depth = 0; depth < MAX_VIRTUAL_LEVELS && level.length > MAX_ROOTS; depth++) {
    const next = new Map<string, TileNode>();
    for (const child of level) {
      const pk = parentKey(child.key);
      const pid = keyId(pk);
      let parent = next.get(pid);
      if (!parent) {
        parent = {
          key: pk,
          id: pid,
          bounds: keyBounds(pk, tileSize),
          zmin: child.zmin,
          zmax: child.zmax,
          geometricError: Infinity,
          content: null,
          parent: null,
          children: [],
        };
        next.set(pid, parent);
        nodes.set(pid, parent);
      }
      parent.children.push(child);
      child.parent = parent;
      parent.zmin = Math.min(parent.zmin, child.zmin);
      parent.zmax = Math.max(parent.zmax, child.zmax);
      // Children may extend past the parent's nominal square (non-grid bounds).
      parent.bounds = unionBounds(parent.bounds, child.bounds);
    }
    if (next.size >= level.length) {
      // No reduction (cannot happen with floor shifts, but never loop forever).
      for (const n of next.values()) {
        for (const c of n.children) c.parent = null;
        nodes.delete(n.id);
      }
      break;
    }
    level = [...next.values()];
  }

  return { tileSize, roots: level, nodes, byContentId, contentNodes, skipped };
}

/** Fills a tileset URI template ("tiles/{level}/{si}_{sj}/{i}_{j}.wgt"). */
export function fillUri(template: string, keys: Record<string, number>): string {
  return template.replace(/\{(\w+)\}/g, (m, k: string) => (k in keys ? String(keys[k]) : m));
}

/**
 * The level-0 tiles of a data-first world (format 2.x: tiles/tileset.json and its
 * subtree files) as index entries whose content is a `.wgt` container, so the
 * quadtree and streamer treat them like 1.x tiles.
 */
export function indexFromTileset(ts: TilesetDoc, subtrees: readonly SubtreeDoc[]): TileIndex {
  const size = Number(ts.tile_size) > 0 ? Number(ts.tile_size) : DEFAULT_TILE_SIZE;
  const tiles: TileEntry[] = [];
  for (const sub of subtrees) {
    for (const lvl of sub.levels ?? []) {
      if (lvl.level !== 0) continue; // coarser levels arrive with the LOD pyramid
      for (let k = 0; k < lvl.i.length; k++) {
        const i = lvl.i[k];
        const j = lvl.j[k];
        const keys = { level: 0, si: sub.si, sj: sub.sj, i, j };
        const flags = lvl.flags?.[k] ?? 1;
        const zmin = lvl.zmin?.[k];
        const zmax = lvl.zmax?.[k];
        const counts: Record<string, number> = {};
        for (const [name, arr] of Object.entries(lvl.counts ?? {})) counts[name] = arr[k] ?? 0;
        tiles.push({
          id: `${i}_${j}`,
          i,
          j,
          bounds: [i * size, j * size, (i + 1) * size, (j + 1) * size],
          zrange: zmin != null && zmax != null ? [zmin, zmax] : undefined,
          glb: fillUri(ts.uri.content, keys),
          meta: flags & 1 && ts.uri.meta ? fillUri(ts.uri.meta, keys) : null,
          ortho: flags & 2 && ts.uri.ortho ? fillUri(ts.uri.ortho, keys) : null,
          collision: flags & 4 && ts.uri.collision ? fillUri(ts.uri.collision, keys) : null,
          hash: lvl.hash?.[k],
          counts,
          format: 'wgt',
        });
      }
    }
  }
  return { schema: 'worldgen.tile_index', schema_version: ts.schema_version, tile_size: size, ortho: ts.ortho ?? null, tiles };
}

function unionBounds(a: Bounds2, b: Bounds2): Bounds2 {
  return [Math.min(a[0], b[0]), Math.min(a[1], b[1]), Math.max(a[2], b[2]), Math.max(a[3], b[3])];
}

/** Ancestors of a node, nearest first. */
export function ancestorsOf(node: TileNode): TileNode[] {
  const out: TileNode[] = [];
  for (let p = node.parent; p; p = p.parent) out.push(p);
  return out;
}
