/**
 * Pure routing over the WorldGen road graph (no three.js, no DOM).
 *
 * Semantics follow docs/output-format.md:
 *  - edge `direction`: `both` = either way, `forward` = from→to only,
 *    `backward` = to→from only;
 *  - `modes[mode] === false` excludes an edge for that travel mode;
 *  - enforced turn restrictions: a prohibitory restriction forbids
 *    from_edge → via_node → to_edge; a mandatory (`only_*`) restriction forbids
 *    every continuation from from_edge at via_node except to_edge.
 *
 * Dijkstra runs over edge-based states (node reached + incoming edge), which is
 * what makes turn restrictions enforceable. Via-way restrictions (non-empty
 * `via_edges`) cannot be expressed with single-edge states and are skipped
 * (reported in `Router.stats`).
 *
 * This module only uses erasable TypeScript syntax so that it runs directly
 * under `node --test` (type stripping) as well as in the Vite bundle.
 */

export type TravelMode = 'motor_vehicle' | 'bicycle' | 'foot';

export interface RouterNode {
  id: string;
  x: number;
  y: number;
  z?: number;
}

export interface RouterEdge {
  id: string;
  from: string;
  to: string;
  direction?: string;
  length_m?: number;
  modes?: Partial<Record<TravelMode, boolean>> | null;
}

export interface RouterRestriction {
  id?: string;
  restriction?: string;
  kind?: string;
  from_edge: string;
  via_node: string;
  to_edge: string;
  via_edges?: readonly string[] | null;
  enforced?: boolean;
}

export interface RouterGraph {
  nodes: readonly RouterNode[];
  edges: readonly RouterEdge[];
  restrictions?: readonly RouterRestriction[] | null;
}

export interface RouteOptions {
  /** Travel mode used to filter edges. Default `motor_vehicle`. */
  mode?: TravelMode;
  /** Traverse one-way edges in both directions (e.g. pedestrians). Default false. */
  ignoreOneway?: boolean;
  /** Allow reversing along the edge just travelled at non-dead-end nodes. Default false. */
  allowUTurns?: boolean;
  /** Apply enforced turn restrictions. Default true. */
  applyRestrictions?: boolean;
}

export interface RouteStep {
  edgeId: string;
  edgeIndex: number;
  /** Node id the step starts at. */
  from: string;
  /** Node id the step ends at. */
  to: string;
  /** True when the edge is traversed to→from. */
  reversed: boolean;
  length: number;
}

export interface RouteResult {
  found: boolean;
  /** Total length in metres (0 when not found). */
  length: number;
  steps: RouteStep[];
  /** Node ids visited in order (start … end). */
  nodeIds: string[];
  /** Human readable reason when `found` is false. */
  reason?: string;
}

export interface RouterStats {
  nodes: number;
  edges: number;
  restrictionsEnforced: number;
  restrictionsNotEnforced: number;
  restrictionsSkipped: number;
  skippedReasons: string[];
}

interface TurnRule {
  banned?: Set<number>;
  only?: Set<number>;
}

interface Adjacency {
  arcEdge: Int32Array;
  arcFrom: Int32Array;
  arcTo: Int32Array;
  arcLen: Float64Array;
  arcReversed: Uint8Array;
  outStart: Int32Array;
  outArcs: Int32Array;
  /** Number of distinct usable edges incident to each node. */
  incident: Int32Array;
}

const MODES: readonly TravelMode[] = ['motor_vehicle', 'bicycle', 'foot'];

/** Is `edge` usable for `mode`? Missing information means "allowed". */
export function edgeAllowsMode(edge: RouterEdge, mode: TravelMode): boolean {
  const modes = edge.modes;
  if (!modes) return true;
  return modes[mode] !== false;
}

export class Router {
  readonly stats: RouterStats;

  private readonly nodeIds: string[];
  private readonly nodeX: Float64Array;
  private readonly nodeY: Float64Array;
  private readonly nodeIndex = new Map<string, number>();
  private readonly edges: RouterEdge[];
  private readonly edgeFrom: Int32Array;
  private readonly edgeTo: Int32Array;
  private readonly edgeLen: Float64Array;
  private readonly edgeIndex = new Map<string, number>();
  private readonly turnRules = new Map<number, TurnRule>();
  private readonly adjacencyCache = new Map<string, Adjacency>();

  constructor(graph: RouterGraph) {
    const nodes = graph.nodes ?? [];
    this.nodeIds = nodes.map((n) => n.id);
    this.nodeX = new Float64Array(nodes.length);
    this.nodeY = new Float64Array(nodes.length);
    nodes.forEach((n, i) => {
      this.nodeIndex.set(n.id, i);
      this.nodeX[i] = Number.isFinite(n.x) ? n.x : 0;
      this.nodeY[i] = Number.isFinite(n.y) ? n.y : 0;
    });

    // Keep only edges whose endpoints exist.
    const skippedReasons: string[] = [];
    this.edges = [];
    for (const e of graph.edges ?? []) {
      if (!this.nodeIndex.has(e.from) || !this.nodeIndex.has(e.to)) {
        skippedReasons.push(`edge ${e.id}: unknown endpoint`);
        continue;
      }
      this.edgeIndex.set(e.id, this.edges.length);
      this.edges.push(e);
    }
    this.edgeFrom = new Int32Array(this.edges.length);
    this.edgeTo = new Int32Array(this.edges.length);
    this.edgeLen = new Float64Array(this.edges.length);
    this.edges.forEach((e, i) => {
      const a = this.nodeIndex.get(e.from)!;
      const b = this.nodeIndex.get(e.to)!;
      this.edgeFrom[i] = a;
      this.edgeTo[i] = b;
      const len = e.length_m;
      this.edgeLen[i] =
        typeof len === 'number' && Number.isFinite(len) && len >= 0
          ? len
          : Math.hypot(this.nodeX[b] - this.nodeX[a], this.nodeY[b] - this.nodeY[a]);
    });

    let enforced = 0;
    let notEnforced = 0;
    let skipped = 0;
    const n = this.nodeIds.length;
    for (const r of graph.restrictions ?? []) {
      const label = r.id ?? `${r.from_edge}->${r.to_edge}`;
      if (r.enforced === false) {
        notEnforced++;
        continue;
      }
      if (r.via_edges && r.via_edges.length > 0) {
        skipped++;
        skippedReasons.push(`restriction ${label}: via-way restrictions are not supported`);
        continue;
      }
      const fe = this.edgeIndex.get(r.from_edge);
      const te = this.edgeIndex.get(r.to_edge);
      const vn = this.nodeIndex.get(r.via_node);
      if (fe === undefined || te === undefined || vn === undefined) {
        skipped++;
        skippedReasons.push(`restriction ${label}: references unknown edge or node`);
        continue;
      }
      if (!this.touches(fe, vn) || !this.touches(te, vn)) {
        skipped++;
        skippedReasons.push(`restriction ${label}: via node is not shared by from/to edges`);
        continue;
      }
      const kind = r.kind ?? (r.restriction?.startsWith('only_') ? 'mandatory' : r.restriction?.startsWith('no_') ? 'prohibitory' : undefined);
      if (kind !== 'prohibitory' && kind !== 'mandatory') {
        skipped++;
        skippedReasons.push(`restriction ${label}: unknown kind`);
        continue;
      }
      const key = fe * n + vn;
      let rule = this.turnRules.get(key);
      if (!rule) {
        rule = {};
        this.turnRules.set(key, rule);
      }
      if (kind === 'prohibitory') (rule.banned ??= new Set()).add(te);
      else (rule.only ??= new Set()).add(te);
      enforced++;
    }

    this.stats = {
      nodes: this.nodeIds.length,
      edges: this.edges.length,
      restrictionsEnforced: enforced,
      restrictionsNotEnforced: notEnforced,
      restrictionsSkipped: skipped,
      skippedReasons,
    };
  }

  hasNode(id: string): boolean {
    return this.nodeIndex.has(id);
  }

  /**
   * Nearest node to local (x, y). With `mode`, only nodes that have at least
   * one edge usable by that mode are considered.
   */
  nearestNode(x: number, y: number, mode?: TravelMode): string | null {
    const adj = mode ? this.adjacency(mode, false) : null;
    let best = -1;
    let bestD = Infinity;
    for (let i = 0; i < this.nodeIds.length; i++) {
      if (adj && adj.incident[i] === 0) continue;
      const dx = this.nodeX[i] - x;
      const dy = this.nodeY[i] - y;
      const d = dx * dx + dy * dy;
      if (d < bestD) {
        bestD = d;
        best = i;
      }
    }
    return best >= 0 ? this.nodeIds[best] : null;
  }

  route(fromId: string, toId: string, options: RouteOptions = {}): RouteResult {
    const mode = options.mode ?? 'motor_vehicle';
    if (!MODES.includes(mode)) return notFound(`unknown travel mode "${mode}"`);
    const s = this.nodeIndex.get(fromId);
    const t = this.nodeIndex.get(toId);
    if (s === undefined) return notFound(`start node ${fromId} is not in the graph`);
    if (t === undefined) return notFound(`end node ${toId} is not in the graph`);
    if (s === t) return { found: true, length: 0, steps: [], nodeIds: [fromId] };

    // Pedestrians may walk against one-way traffic (same rule as the Python router).
    const adj = this.adjacency(mode, options.ignoreOneway === true || mode === 'foot');
    const applyRestrictions = options.applyRestrictions !== false;
    const allowUTurns = options.allowUTurns === true;
    const nArcs = adj.arcEdge.length;
    const nNodes = this.nodeIds.length;
    const dist = new Float64Array(nArcs).fill(Infinity);
    const prev = new Int32Array(nArcs).fill(-1);
    const done = new Uint8Array(nArcs);
    const heap = new MinHeap();

    for (let k = adj.outStart[s]; k < adj.outStart[s + 1]; k++) {
      const b = adj.outArcs[k];
      if (adj.arcLen[b] < dist[b]) {
        dist[b] = adj.arcLen[b];
        heap.push(dist[b], b);
      }
    }

    let goal = -1;
    while (heap.size > 0) {
      const a = heap.pop();
      if (done[a]) continue;
      done[a] = 1;
      const v = adj.arcTo[a];
      if (v === t) {
        goal = a;
        break;
      }
      const e = adj.arcEdge[a];
      const rule = applyRestrictions ? this.turnRules.get(e * nNodes + v) : undefined;
      const da = dist[a];
      for (let k = adj.outStart[v]; k < adj.outStart[v + 1]; k++) {
        const b = adj.outArcs[k];
        if (done[b]) continue;
        const f = adj.arcEdge[b];
        if (f === e && !allowUTurns && adj.incident[v] > 1) continue;
        if (rule) {
          if (rule.banned && rule.banned.has(f)) continue;
          if (rule.only && !rule.only.has(f)) continue;
        }
        const nd = da + adj.arcLen[b];
        if (nd < dist[b]) {
          dist[b] = nd;
          prev[b] = a;
          heap.push(nd, b);
        }
      }
    }

    if (goal < 0) {
      return notFound(`no ${mode.replace('_', ' ')} route between ${fromId} and ${toId}`);
    }

    const arcs: number[] = [];
    for (let a = goal; a >= 0; a = prev[a]) arcs.push(a);
    arcs.reverse();
    const steps: RouteStep[] = arcs.map((a) => {
      const ei = adj.arcEdge[a];
      return {
        edgeId: this.edges[ei].id,
        edgeIndex: ei,
        from: this.nodeIds[adj.arcFrom[a]],
        to: this.nodeIds[adj.arcTo[a]],
        reversed: adj.arcReversed[a] === 1,
        length: adj.arcLen[a],
      };
    });
    const nodeIds = [steps[0].from, ...steps.map((st) => st.to)];
    return { found: true, length: dist[goal], steps, nodeIds };
  }

  private touches(edge: number, node: number): boolean {
    return this.edgeFrom[edge] === node || this.edgeTo[edge] === node;
  }

  /** Directed arcs + CSR adjacency for a (mode, oneway handling) combination. */
  private adjacency(mode: TravelMode, ignoreOneway: boolean): Adjacency {
    const key = `${mode}|${ignoreOneway ? 1 : 0}`;
    const cached = this.adjacencyCache.get(key);
    if (cached) return cached;

    const nNodes = this.nodeIds.length;
    const arcEdge: number[] = [];
    const arcFrom: number[] = [];
    const arcTo: number[] = [];
    const arcReversed: number[] = [];
    const incident = new Int32Array(nNodes);

    this.edges.forEach((e, i) => {
      if (!edgeAllowsMode(e, mode)) return;
      const a = this.edgeFrom[i];
      const b = this.edgeTo[i];
      incident[a]++;
      if (b !== a) incident[b]++;
      const dir = ignoreOneway ? 'both' : (e.direction ?? 'both');
      const fwd = dir === 'both' || dir === 'forward';
      const bwd = dir === 'both' || dir === 'backward';
      if (fwd) {
        arcEdge.push(i);
        arcFrom.push(a);
        arcTo.push(b);
        arcReversed.push(0);
      }
      if (bwd) {
        arcEdge.push(i);
        arcFrom.push(b);
        arcTo.push(a);
        arcReversed.push(1);
      }
    });

    const nArcs = arcEdge.length;
    const outStart = new Int32Array(nNodes + 1);
    for (let k = 0; k < nArcs; k++) outStart[arcFrom[k] + 1]++;
    for (let v = 0; v < nNodes; v++) outStart[v + 1] += outStart[v];
    const fill = outStart.slice(0, nNodes);
    const outArcs = new Int32Array(nArcs);
    for (let k = 0; k < nArcs; k++) outArcs[fill[arcFrom[k]]++] = k;

    const adj: Adjacency = {
      arcEdge: Int32Array.from(arcEdge),
      arcFrom: Int32Array.from(arcFrom),
      arcTo: Int32Array.from(arcTo),
      arcLen: Float64Array.from(arcEdge, (ei) => this.edgeLen[ei]),
      arcReversed: Uint8Array.from(arcReversed),
      outStart,
      outArcs,
      incident,
    };
    this.adjacencyCache.set(key, adj);
    return adj;
  }
}

function notFound(reason: string): RouteResult {
  return { found: false, length: 0, steps: [], nodeIds: [], reason };
}

/** Binary min-heap of (priority, value) pairs. */
class MinHeap {
  private keys: number[] = [];
  private vals: number[] = [];

  get size(): number {
    return this.keys.length;
  }

  push(key: number, val: number): void {
    const keys = this.keys;
    const vals = this.vals;
    let i = keys.length;
    keys.push(key);
    vals.push(val);
    while (i > 0) {
      const p = (i - 1) >> 1;
      if (keys[p] <= key) break;
      keys[i] = keys[p];
      vals[i] = vals[p];
      i = p;
    }
    keys[i] = key;
    vals[i] = val;
  }

  /** Removes and returns the value with the smallest key. */
  pop(): number {
    const keys = this.keys;
    const vals = this.vals;
    const top = vals[0];
    const lastKey = keys.pop()!;
    const lastVal = vals.pop()!;
    const n = keys.length;
    if (n > 0) {
      let i = 0;
      for (;;) {
        const l = 2 * i + 1;
        if (l >= n) break;
        const r = l + 1;
        const c = r < n && keys[r] < keys[l] ? r : l;
        if (keys[c] >= lastKey) break;
        keys[i] = keys[c];
        vals[i] = vals[c];
        i = c;
      }
      keys[i] = lastKey;
      vals[i] = lastVal;
    }
    return top;
  }
}
