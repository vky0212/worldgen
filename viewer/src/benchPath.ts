/**
 * Fly-through benchmark paths and statistics (`?bench=path.json` or
 * `?bench=auto`).
 *
 * A path is a list of camera positions in local ENU metres, flown through at
 * constant speed on a Catmull-Rom spline, with optional look-at points (one
 * per camera point, interpolated the same way). The flight is driven by wall
 * time, so camera poses at a given time are the same on every run.
 *
 * Pure module: no three.js, no DOM (unit-tested with node --test).
 */

export type Vec3 = [number, number, number];

export interface BenchPath {
  name?: string;
  /** Flight time (s) along the whole path. */
  duration_s?: number;
  /** Camera positions (local ENU metres). At least two. */
  points: Vec3[];
  /** Look-at points, one per camera point (default: look along the path). */
  targets?: Vec3[];
  /** Longest wait (s) for the first view to load before flying. */
  warmup_s?: number;
  /** Longest wait (s) for loading to finish after the flight. */
  settle_timeout_s?: number;
}

export const BENCH_DEFAULTS = { duration_s: 90, warmup_s: 30, settle_timeout_s: 30 };

function isVec3(v: unknown): v is Vec3 {
  return Array.isArray(v) && v.length === 3 && v.every((n) => typeof n === 'number' && Number.isFinite(n));
}

/** Checks a parsed path document; throws with a readable message. */
export function validatePath(raw: unknown): BenchPath {
  if (!raw || typeof raw !== 'object') throw new Error('bench path is not a JSON object');
  const p = raw as Record<string, unknown>;
  if (!Array.isArray(p.points) || p.points.length < 2 || !p.points.every(isVec3)) {
    throw new Error('bench path needs "points": [[x, y, z], ...] with at least two points (local ENU metres)');
  }
  if (p.targets !== undefined && (!Array.isArray(p.targets) || p.targets.length !== p.points.length || !p.targets.every(isVec3))) {
    throw new Error('bench path "targets" must have one [x, y, z] per point');
  }
  for (const k of ['duration_s', 'warmup_s', 'settle_timeout_s']) {
    if (p[k] !== undefined && !(typeof p[k] === 'number' && (p[k] as number) > 0)) throw new Error(`bench path "${k}" must be a positive number`);
  }
  return p as unknown as BenchPath;
}

function catmull(p0: Vec3, p1: Vec3, p2: Vec3, p3: Vec3, t: number): Vec3 {
  const t2 = t * t;
  const t3 = t2 * t;
  const out: Vec3 = [0, 0, 0];
  for (let k = 0; k < 3; k++) {
    out[k] = 0.5 * (2 * p1[k] + (-p0[k] + p2[k]) * t + (2 * p0[k] - 5 * p1[k] + 4 * p2[k] - p3[k]) * t2 + (-p0[k] + 3 * p1[k] - 3 * p2[k] + p3[k]) * t3);
  }
  return out;
}

/**
 * Samples camera position (and look-at point) at u ∈ [0, 1] of the path, at
 * constant speed along the control polygon (chord-length parameterisation).
 */
export class SplinePath {
  readonly points: Vec3[];
  readonly targets: Vec3[] | null;
  readonly length: number;
  private readonly cumulative: number[];

  constructor(points: Vec3[], targets: Vec3[] | null = null) {
    if (points.length < 2) throw new Error('a path needs at least two points');
    this.points = points;
    this.targets = targets;
    this.cumulative = [0];
    for (let k = 1; k < points.length; k++) {
      const a = points[k - 1];
      const b = points[k];
      this.cumulative.push(this.cumulative[k - 1] + Math.hypot(b[0] - a[0], b[1] - a[1], b[2] - a[2]));
    }
    this.length = this.cumulative[this.cumulative.length - 1];
  }

  /** Segment index and local parameter for u ∈ [0, 1]. */
  private locate(u: number): [number, number] {
    const n = this.points.length;
    const s = Math.min(1, Math.max(0, u)) * this.length;
    let i = 0;
    while (i < n - 2 && this.cumulative[i + 1] < s) i++;
    const segLen = this.cumulative[i + 1] - this.cumulative[i];
    return [i, segLen > 0 ? (s - this.cumulative[i]) / segLen : 0];
  }

  private evalOn(list: Vec3[], i: number, t: number): Vec3 {
    const n = list.length;
    const at = (k: number): Vec3 => list[Math.min(n - 1, Math.max(0, k))];
    return catmull(at(i - 1), at(i), at(i + 1), at(i + 2), t);
  }

  position(u: number): Vec3 {
    const [i, t] = this.locate(u);
    return this.evalOn(this.points, i, t);
  }

  /** Look-at point: the interpolated target, or a point ahead on the path. */
  target(u: number, aheadMetres = 300): Vec3 {
    const [i, t] = this.locate(u);
    if (this.targets) return this.evalOn(this.targets, i, t);
    const p = this.position(u);
    const q = this.position(Math.min(1, u + aheadMetres / Math.max(this.length, 1)));
    if (Math.hypot(q[0] - p[0], q[1] - p[1]) < 1) return [p[0], p[1] + 1, p[2] - 1];
    return [q[0], q[1], q[2] - 60];
  }
}

export interface Summary {
  n: number;
  mean: number;
  p50: number;
  p95: number;
  p99: number;
  max: number;
}

/** Nearest-rank percentile of an ascending array. */
export function percentile(sorted: readonly number[], p: number): number {
  if (!sorted.length) return Number.NaN;
  const rank = Math.ceil((p / 100) * sorted.length);
  return sorted[Math.min(sorted.length - 1, Math.max(0, rank - 1))];
}

export function summarize(values: readonly number[]): Summary | null {
  const v = values.filter((x) => Number.isFinite(x));
  if (!v.length) return null;
  const sorted = [...v].sort((a, b) => a - b);
  const round = (x: number): number => Math.round(x * 100) / 100;
  return {
    n: v.length,
    mean: round(v.reduce((a, b) => a + b, 0) / v.length),
    p50: round(percentile(sorted, 50)),
    p95: round(percentile(sorted, 95)),
    p99: round(percentile(sorted, 99)),
    max: round(sorted[sorted.length - 1]),
  };
}

export interface AutoPathWorld {
  /** [xmin, ymin, xmax, ymax] of the world (local ENU). */
  bounds: [number, number, number, number];
  /** Highest surface (terrain, buildings, trees) near (x, y). */
  topAt(x: number, y: number): number;
  /** Ground elevation near (x, y). */
  groundAt(x: number, y: number): number;
}

/**
 * A deterministic loop over the world at `agl` metres above the highest
 * surface, looking ahead and down, ending with a climb to an overview of
 * the whole world.
 */
export function autoPath(world: AutoPathWorld, durationS = BENCH_DEFAULTS.duration_s, agl = 120): BenchPath {
  const [xmin, ymin, xmax, ymax] = world.bounds;
  const w = xmax - xmin;
  const h = ymax - ymin;
  const at = (fx: number, fy: number): [number, number] => [xmin + fx * w, ymin + fy * h];
  const loop: [number, number][] = [
    [0.2, 0.2], [0.5, 0.3], [0.8, 0.2], [0.8, 0.5], [0.7, 0.8], [0.4, 0.75], [0.2, 0.6], [0.3, 0.42], [0.5, 0.45],
  ];
  const points: Vec3[] = [];
  const targets: Vec3[] = [];
  loop.forEach(([fx, fy], k) => {
    const [x, y] = at(fx, fy);
    const [nx, ny] = k + 1 < loop.length ? at(...loop[k + 1]) : at(0.5, 0.5);
    const d = Math.hypot(nx - x, ny - y) || 1;
    const tx = x + ((nx - x) / d) * 400;
    const ty = y + ((ny - y) / d) * 400;
    points.push([x, y, world.topAt(x, y) + agl]);
    targets.push([tx, ty, world.groundAt(tx, ty)]);
  });
  // Climb out to an overview from the south.
  const [cx, cy] = at(0.5, 0.5);
  const ground = world.groundAt(cx, cy);
  const span = Math.max(w, h, 500);
  points.push([cx, cy - 0.35 * span, ground + 0.35 * span]);
  targets.push([cx, cy, ground]);
  points.push([cx, cy - 0.7 * span, ground + 0.8 * span]);
  targets.push([cx, cy, ground]);
  return { name: 'auto', duration_s: durationS, points, targets };
}
