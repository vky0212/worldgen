/**
 * Tileset adapter: turns a world's tile index into a quadtree of TileNodes
 * that the traversal walks.
 *
 * Schema 1.x worlds have a flat `tiles/index.json` with one level of 256 m
 * tiles. They become level-0 nodes with content, under virtual (content-less)
 * ancestors up to a few roots, so the traversal can cull whole blocks of the
 * world at once. Content-less nodes always refine: a 1.x world has nothing
 * coarser to show, so the finest tiles are drawn wherever they are visible
 * (limited by the view range and the GPU budget).
 *
 * Data-first worlds of format 2.4 add coarser levels (1 to 4) with content:
 * terrain-only tiles that stand for 2^L x 2^L level-0 tiles, with a geometric
 * error the traversal compares against the screen-space error. Tiles of
 * different levels side by side do not share all their edge vertices, so every
 * tile of such a world gets skirts: strips hanging down from its edges, as
 * deep as the largest crack a neighbour can leave (see `skirtDepths`).
 *
 * Pure module: no three.js, no DOM (unit-tested with node --test).
 */
import type { SubtreeDoc, TileEntry, TileIndex, TilesetDoc } from '../types';
import { keyBounds, keyId, parentKey } from './quadtree.ts';
import type { Bounds2, TileKey } from './quadtree.ts';

/** Loadable content of a node (one tile). */
export interface ContentRef {
  /** Content id: the tile id from the index ("i_j"; "L/i_j" for coarser levels). */
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
  /** Largest difference (m) between this node's terrain and the level-0 terrain (0 at level 0). */
  terrainError: number;
  /** Skirt depth (m) under the west, east, south and north edge; null = no skirts. */
  skirts: [number, number, number, number] | null;
  /** A coarser tile some of whose level-0 tiles do not exist: it covers only the others. */
  partial: boolean;
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
  /** Highest level with content (0 for single-level worlds). */
  maxLevel: number;
  /** True where a level-0 tile exists at local (x, y): what a partial coarser tile covers. */
  leafAt(x: number, y: number): boolean;
}

const DEFAULT_TILE_SIZE = 256;
/** Stop adding virtual levels once at most this many roots remain. */
const MAX_ROOTS = 4;
const MAX_VIRTUAL_LEVELS = 12;
/** Added to every skirt (m): covers rounding and the deepest crack exactly. */
export const SKIRT_MARGIN = 1;

/** Key of an index entry: its level and i/j, or the level-0 cell of its bounds' centre. */
export function entryKey(entry: TileEntry, tileSize: number): TileKey {
  const level = Number.isInteger(entry.level) && (entry.level as number) > 0 ? (entry.level as number) : 0;
  if (Number.isInteger(entry.i) && Number.isInteger(entry.j)) return { level, i: entry.i, j: entry.j };
  const b = entry.bounds;
  const cx = (b[0] + b[2]) / 2;
  const cy = (b[1] + b[3]) / 2;
  return { level: 0, i: Math.floor(cx / tileSize), j: Math.floor(cy / tileSize) };
}

function makeNode(key: TileKey, bounds: Bounds2, zmin: number, zmax: number, content: ContentRef | null): TileNode {
  return {
    key,
    id: keyId(key),
    bounds,
    zmin,
    zmax,
    geometricError: content ? 0 : Infinity,
    terrainError: 0,
    skirts: null,
    partial: false,
    content,
    parent: null,
    children: [],
  };
}

/**
 * Builds the quadtree of a tile index: its tiles of every level linked to their parents, and
 * virtual ancestors above the highest level. `defaultZ` is used for entries without a zrange.
 */
export function tilesetFromIndex(index: TileIndex, defaultZ: [number, number] = [0, 50]): Tileset {
  const tileSize = Number(index.tile_size) > 0 ? Number(index.tile_size) : DEFAULT_TILE_SIZE;
  const nodes = new Map<string, TileNode>();
  const byContentId = new Map<string, TileNode>();
  const contentNodes: TileNode[] = [];
  const byLevel = new Map<number, TileNode[]>();
  let skipped = 0;
  let maxLevel = 0;

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
    const node = makeNode(key, bounds, Number(z[0]), Number(z[1]), { id: entry.id, entry });
    if (key.level > 0) {
      node.geometricError = Math.max(Number(entry.geometricError) || 0, 0);
      node.terrainError = Math.max(Number(entry.terrainError) || 0, 0);
    }
    nodes.set(id, node);
    byContentId.set(entry.id, node);
    contentNodes.push(node);
    const list = byLevel.get(key.level);
    if (list) list.push(node);
    else byLevel.set(key.level, [node]);
    maxLevel = Math.max(maxLevel, key.level);
  }

  // Link level by level, bottom up: a node's parent is the index's tile one level up when there
  // is one, else a virtual node. Above the highest level, virtual levels until few roots remain.
  let frontier: TileNode[] = byLevel.get(0) ?? [];
  for (let level = 0; level < maxLevel + MAX_VIRTUAL_LEVELS; level++) {
    if (level >= maxLevel && frontier.length <= MAX_ROOTS) break;
    const created: TileNode[] = [];
    for (const child of frontier) {
      const pk = parentKey(child.key);
      let parent = nodes.get(keyId(pk));
      if (!parent) {
        parent = makeNode(pk, keyBounds(pk, tileSize), child.zmin, child.zmax, null);
        nodes.set(parent.id, parent);
        created.push(parent);
      }
      parent.children.push(child);
      child.parent = parent;
      parent.zmin = Math.min(parent.zmin, child.zmin);
      parent.zmax = Math.max(parent.zmax, child.zmax);
      // Children may extend past the parent's nominal square (non-grid bounds).
      parent.bounds = unionBounds(parent.bounds, child.bounds);
    }
    const next = [...(byLevel.get(level + 1) ?? []), ...created];
    if (level >= maxLevel && next.length >= frontier.length) {
      // No reduction (cannot happen with floor shifts, but never loop forever).
      for (const n of created) {
        for (const c of n.children) c.parent = null;
        nodes.delete(n.id);
      }
      break;
    }
    frontier = next;
  }
  const roots = frontier;
  // A content node no level below linked to (none of its children exist) is a root as well.
  for (const n of contentNodes) if (!n.parent && !roots.includes(n)) roots.push(n);

  const leaves = new Set((byLevel.get(0) ?? []).map((n) => `${n.key.i}_${n.key.j}`));
  const leafAt = (x: number, y: number): boolean => leaves.has(`${Math.floor(x / tileSize)}_${Math.floor(y / tileSize)}`);
  if (maxLevel > 0) {
    for (const n of contentNodes) {
      if (n.key.level > 0) n.partial = leafCount(n) < 4 ** n.key.level;
      n.skirts = skirtDepths(n, nodes);
    }
  }
  return { tileSize, roots, nodes, byContentId, contentNodes, skipped, maxLevel, leafAt };
}

/** Level-0 tiles under a node. */
function leafCount(node: TileNode): number {
  if (node.key.level === 0) return node.content ? 1 : 0;
  let n = 0;
  for (const c of node.children) n += leafCount(c);
  return n;
}

/**
 * Skirt depth (m) under each edge of a node: west, east, south, north. Across an edge, the
 * world is drawn by the same-level neighbour or by a coarser tile containing it (one of its
 * ancestors). Edges of two levels differ by at most the sum of their terrain errors, so the
 * skirt reaches that far below the edge, plus SKIRT_MARGIN. No neighbour (the world's edge,
 * or a missing tile): no skirt.
 */
export function skirtDepths(node: TileNode, nodes: ReadonlyMap<string, TileNode>): [number, number, number, number] {
  const k = node.key;
  const depth = (di: number, dj: number): number => {
    const nb = nodes.get(keyId({ level: k.level, i: k.i + di, j: k.j + dj }));
    if (!nb || !nb.content) return 0;
    let worst = nb.terrainError;
    for (let p = nb.parent; p; p = p.parent) if (p.content) worst = Math.max(worst, p.terrainError);
    return node.terrainError + worst + SKIRT_MARGIN;
  };
  return [depth(-1, 0), depth(1, 0), depth(0, -1), depth(0, 1)];
}

/** Fills a tileset URI template ("tiles/{level}/{si}_{sj}/{i}_{j}.wgt"). */
export function fillUri(template: string, keys: Record<string, number>): string {
  return template.replace(/\{(\w+)\}/g, (m, k: string) => (k in keys ? String(keys[k]) : m));
}

/**
 * The tiles of a data-first world (format 2.x: tiles/tileset.json and its subtree files) as
 * index entries whose content is a `.wgt` container, so the quadtree and streamer treat them
 * like 1.x tiles. Level 0 keeps the 1.x ids ("i_j"); coarser levels (format 2.4) are "L/i_j",
 * with their geometric and terrain errors and no meta.json or collision mesh.
 */
export function indexFromTileset(ts: TilesetDoc, subtrees: readonly SubtreeDoc[]): TileIndex {
  const size = Number(ts.tile_size) > 0 ? Number(ts.tile_size) : DEFAULT_TILE_SIZE;
  const sourceMpp = Number(ts.ortho?.effective_source_m_per_px ?? ts.ortho?.source_m_per_px ?? ts.ortho?.m_per_px) || 0;
  const tiles: TileEntry[] = [];
  for (const sub of subtrees) {
    for (const lvl of sub.levels ?? []) {
      const level = lvl.level;
      if (!Number.isInteger(level) || level < 0) continue;
      const s = size * 2 ** level;
      const levelMpp = Number(ts.levels?.find((l) => l.level === level)?.ortho?.m_per_px) || 0;
      for (let k = 0; k < lvl.i.length; k++) {
        const i = lvl.i[k];
        const j = lvl.j[k];
        const keys = { level, si: sub.si, sj: sub.sj, i, j };
        const flags = lvl.flags?.[k] ?? (level ? 0 : 1);
        const zmin = lvl.zmin?.[k];
        const zmax = lvl.zmax?.[k];
        const entry: TileEntry = {
          id: level ? `${level}/${i}_${j}` : `${i}_${j}`,
          i,
          j,
          bounds: [i * s, j * s, (i + 1) * s, (j + 1) * s],
          zrange: zmin != null && zmax != null ? [zmin, zmax] : undefined,
          glb: fillUri(ts.uri.content, keys),
          meta: level === 0 && flags & 1 && ts.uri.meta ? fillUri(ts.uri.meta, keys) : null,
          ortho: flags & 2 && ts.uri.ortho ? fillUri(ts.uri.ortho, keys) : null,
          collision: level === 0 && flags & 4 && ts.uri.collision ? fillUri(ts.uri.collision, keys) : null,
          hash: lvl.hash?.[k],
          format: 'wgt',
        };
        if (level === 0) {
          const counts: Record<string, number> = {};
          for (const [name, arr] of Object.entries(lvl.counts ?? {})) counts[name] = arr[k] ?? 0;
          entry.counts = counts;
        } else {
          entry.level = level;
          entry.geometricError = lvl.err?.[k] ?? 0;
          entry.terrainError = lvl.terr?.[k] ?? 0;
          if (levelMpp) entry.orthoMpp = Math.max(sourceMpp, levelMpp);
        }
        tiles.push(entry);
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
