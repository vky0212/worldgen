/**
 * Tile selection: which nodes to draw and which to load for a camera.
 *
 * Screen-space error (SSE) refinement with hysteresis over a TileNode
 * quadtree:
 *  - a node is refined when its geometric error projects to more than `tau`
 *    pixels, and a previously refined node only collapses below
 *    `tau / hysteresis` (no flicker at the threshold);
 *  - content-less (virtual) nodes always refine;
 *  - replacement refinement without holes or overlaps: a refined node keeps
 *    being drawn until every visible child subtree can be drawn completely;
 *    a node that is not ready yet is replaced by its already loaded
 *    descendants (zooming out) or by its nearest loaded ancestor;
 *  - nodes outside the frustum but within `prefetchRadius` are requested at
 *    low priority so turning the camera does not show empty ground.
 *
 * Coordinates are local ENU metres (x east, y north, z up).
 * Pure module: no three.js, no DOM (unit-tested with node --test).
 */
import type { TileNode } from './tileset.ts';

/** Plane n·p + d = 0; points with n·p + d >= 0 are inside. */
export type Plane = [number, number, number, number];

export interface ViewState {
  /** Camera position, local ENU. */
  position: [number, number, number];
  /** Frustum planes in ENU (inside = non-negative side). Empty = everything visible. */
  planes: Plane[];
  /** Pixels per unit error at unit distance: screenHeightPx / (2·tan(fovY/2)). */
  sseFactor: number;
}

export interface TraversalOptions {
  /** Refine when a node's screen-space error exceeds this many pixels. */
  tau: number;
  /** Collapse a refined node only below tau / hysteresis. */
  hysteresis: number;
  /** Nodes outside the frustum but this close (m) are prefetched. */
  prefetchRadius: number;
  /** How many levels below a not-yet-loaded node to look for loaded stand-ins. */
  descendantFallbackDepth: number;
}

export const DEFAULT_TRAVERSAL: TraversalOptions = {
  tau: 16,
  hysteresis: 1.5,
  prefetchRadius: 384,
  descendantFallbackDepth: 2,
};

/** Load state of a node's content as seen by the traversal. */
export type ContentStatus = 'ready' | 'failed' | 'pending';
export type StatusLookup = (node: TileNode) => ContentStatus;

/**
 * Tier of a wanted node: 0 = ancestor needed to fill a hole, 1 = selected and
 * visible, 2 = prefetch outside the frustum.
 */
export type WantTier = 0 | 1 | 2;

export interface Want {
  node: TileNode;
  tier: WantTier;
  /** Distance (m) from the camera to the node's box. */
  distance: number;
  /** Smaller = more urgent. */
  priority: number;
  ready: boolean;
}

export interface TraversalResult {
  /** Nodes to draw: no two overlap (none is an ancestor of another). */
  render: TileNode[];
  /** Content nodes the selection needs (ready or not), one entry per node. */
  wanted: Want[];
  /** Nodes refined this time (feed back as `prevRefined` for hysteresis). */
  refined: Set<string>;
  /** Visible selected nodes with nothing drawn over their area. */
  holes: number;
  /** Visible selected content nodes. */
  selected: number;
  /** Nodes visited (cost metric). */
  visited: number;
}

const TIER_STRIDE = 1e9;

export function priorityOf(tier: WantTier, distance: number): number {
  return tier * TIER_STRIDE + Math.min(distance, TIER_STRIDE - 1);
}

/** Distance from a point to a node's 3D box (0 inside). */
export function boxDistance(node: TileNode, p: readonly [number, number, number]): number {
  const [xmin, ymin, xmax, ymax] = node.bounds;
  const dx = Math.max(xmin - p[0], 0, p[0] - xmax);
  const dy = Math.max(ymin - p[1], 0, p[1] - ymax);
  const dz = Math.max(node.zmin - p[2], 0, p[2] - node.zmax);
  return Math.hypot(dx, dy, dz);
}

/** False when the node's box is entirely outside one of the planes. */
export function boxVisible(node: TileNode, planes: readonly Plane[]): boolean {
  const [xmin, ymin, xmax, ymax] = node.bounds;
  for (const [nx, ny, nz, d] of planes) {
    // Corner furthest along the normal (the "positive vertex").
    const x = nx >= 0 ? xmax : xmin;
    const y = ny >= 0 ? ymax : ymin;
    const z = nz >= 0 ? node.zmax : node.zmin;
    if (nx * x + ny * y + nz * z + d < 0) return false;
  }
  return true;
}

export function screenSpaceError(node: TileNode, distance: number, sseFactor: number): number {
  if (node.geometricError === 0) return 0;
  return (node.geometricError * sseFactor) / Math.max(distance, 1e-6);
}

interface SubResult {
  /** Every visible selected area of the subtree is drawn by `render`. */
  complete: boolean;
  render: TileNode[];
  holes: number;
}

const EMPTY_COMPLETE: SubResult = { complete: true, render: [], holes: 0 };

export function traverse(
  roots: readonly TileNode[],
  view: ViewState,
  status: StatusLookup,
  prevRefined: ReadonlySet<string>,
  options: Partial<TraversalOptions> = {},
): TraversalResult {
  const opts: TraversalOptions = { ...DEFAULT_TRAVERSAL, ...options };
  const refined = new Set<string>();
  const wantedById = new Map<string, Want>();
  let selected = 0;
  let visited = 0;
  const cam = view.position;

  const want = (node: TileNode, tier: WantTier, distance: number, ready: boolean): void => {
    const prev = wantedById.get(node.id);
    if (prev && prev.tier <= tier) return;
    wantedById.set(node.id, { node, tier, distance, priority: priorityOf(tier, distance), ready });
  };

  const shouldRefine = (node: TileNode, distance: number): boolean => {
    if (node.children.length === 0) return false;
    if (!node.content) return true;
    const sse = screenSpaceError(node, distance, view.sseFactor);
    const threshold = prevRefined.has(node.id) ? opts.tau / opts.hysteresis : opts.tau;
    return sse > threshold;
  };

  /** Already loaded descendants standing in for a node that is not ready (zooming out). */
  const readyDescendants = (node: TileNode, depth: number): SubResult => {
    if (depth <= 0 || node.children.length === 0) return { complete: false, render: [], holes: 0 };
    const render: TileNode[] = [];
    let complete = true;
    for (const child of node.children) {
      if (!boxVisible(child, view.planes)) continue;
      const st = child.content ? status(child) : 'pending';
      if (st === 'ready') {
        render.push(child);
        continue;
      }
      if (st === 'failed') continue;
      const sub = readyDescendants(child, depth - 1);
      render.push(...sub.render);
      if (!sub.complete) complete = false;
    }
    return { complete, render, holes: 0 };
  };

  /** Outside the frustum: request what the selection would be, draw nothing. */
  const prefetch = (node: TileNode): void => {
    visited++;
    const distance = boxDistance(node, cam);
    if (distance > opts.prefetchRadius) return;
    if (shouldRefine(node, distance)) {
      for (const child of node.children) prefetch(child);
      return;
    }
    if (node.content) want(node, 2, distance, status(node) === 'ready');
  };

  const visit = (node: TileNode): SubResult => {
    if (!boxVisible(node, view.planes)) {
      prefetch(node);
      return EMPTY_COMPLETE;
    }
    visited++;
    const distance = boxDistance(node, cam);

    if (!shouldRefine(node, distance)) {
      if (!node.content) return EMPTY_COMPLETE;
      selected++;
      const st = status(node);
      want(node, 1, distance, st === 'ready');
      if (st === 'ready') return { complete: true, render: [node], holes: 0 };
      if (st === 'failed') return EMPTY_COMPLETE;
      const down = readyDescendants(node, opts.descendantFallbackDepth);
      if (down.complete && down.render.length) return down;
      return { complete: false, render: down.render, holes: 1 };
    }

    if (node.content) refined.add(node.id);
    const render: TileNode[] = [];
    let complete = true;
    let holes = 0;
    for (const child of node.children) {
      const sub = visit(child);
      if (sub.render.length) render.push(...sub.render);
      if (!sub.complete) complete = false;
      holes += sub.holes;
    }
    if (complete || !node.content) return { complete, render, holes };

    // Some visible part of the subtree cannot be drawn yet: this node stands in.
    const st = status(node);
    want(node, 0, distance, st === 'ready');
    if (st === 'ready') return { complete: true, render: [node], holes: 0 };
    return { complete: false, render, holes };
  };

  const render: TileNode[] = [];
  let holes = 0;
  for (const root of roots) {
    const sub = visit(root);
    render.push(...sub.render);
    holes += sub.holes;
  }
  const wanted = [...wantedById.values()].sort((a, b) => a.priority - b.priority);
  return { render, wanted, refined, holes, selected, visited };
}
