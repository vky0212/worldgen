/**
 * Pure job-progress reducer: build events (docs/studio-api.md) → UI state.
 *
 * No DOM, no clock, no Leaflet. `reduceEvents(state, events)` returns a new
 * state and never mutates its input. Events whose `seq` is not greater than
 * the last applied one are ignored, so replays (SSE reconnects, the events of
 * `GET /api/jobs/<id>`) are idempotent. Events without a finite `seq` (only
 * synthesised client-side) are applied without moving `lastSeq`.
 *
 * This module only uses erasable TypeScript syntax and type-only imports so
 * that it runs directly under `node --test` as well as in the Vite bundle.
 */
import type {
  BuildSummary,
  JobEvent,
  JobStateName,
  LatLonPair,
  LogLevel,
  PlanTile,
  RegionInfo,
  StageDef,
} from '../apiTypes';

export type Phase = 'waiting' | JobStateName;
export type StageStatus = 'pending' | 'running' | 'done' | 'failed' | 'skipped' | 'cancelled';
export type SourceStatus = 'pending' | 'start' | 'ok' | 'partial' | 'missing' | 'failed' | 'disabled';
export type TileStatus = 'pending' | 'building' | 'built' | 'unchanged';

export interface StageView {
  name: string;
  label: string;
  weight: number;
  status: StageStatus;
  /** `t` of the start event (seconds since the job started). */
  startT: number | null;
  endT: number | null;
  duration_s: number | null;
  /** Latest `progress` event of this stage. */
  progress: { done: number; total: number; detail: string | null } | null;
}

export interface SourceView {
  name: string;
  status: SourceStatus;
  detail: string | null;
}

export interface LogLine {
  seq: number;
  t: number;
  level: LogLevel;
  message: string;
}

export interface ProgressState {
  /** Highest `seq` applied (0 = none): resume the stream after this. */
  lastSeq: number;
  phase: Phase;
  /** The `end` event arrived: the stream is complete. */
  ended: boolean;
  /** Latest `t` seen. */
  t: number;
  /** Overall progress 0..1 (monotonic while the job runs). */
  overall: number;
  hasPlan: boolean;
  region: RegionInfo | null;
  stages: StageView[];
  currentStage: string | null;
  sources: SourceView[];
  /** Tiles of the plan, in plan order (shared, never mutated). */
  planTiles: PlanTile[];
  tiles: Record<string, TileStatus>;
  tileCounts: Record<TileStatus, number>;
  /** done/total of the `export` stage (from `progress`, else from tile events). */
  exportProgress: { done: number; total: number } | null;
  /** Last MAX_LOG `log` lines (and `error` messages). */
  log: LogLine[];
  logTotal: number;
  warnings: number;
  errors: number;
  summary: BuildSummary | null;
  worldUrl: string | null;
  error: string | null;
}

export const MAX_LOG = 300;

/** The documented pipeline; `plan.stages` replaces it when it arrives. */
export const DEFAULT_STAGES: readonly Required<StageDef>[] = [
  { name: 'acquire', label: 'Download sources', weight: 0.25 },
  { name: 'normalize', label: 'Normalize', weight: 0.02 },
  { name: 'terrain', label: 'Terrain', weight: 0.03 },
  { name: 'topology', label: 'Road topology', weight: 0.03 },
  { name: 'water_levels', label: 'Water levels', weight: 0.01 },
  { name: 'road_elevation', label: 'Road elevation', weight: 0.05 },
  { name: 'terrain_shaping', label: 'Terrain shaping', weight: 0.03 },
  { name: 'buildings', label: 'Buildings', weight: 0.02 },
  { name: 'landcover', label: 'Land cover', weight: 0.02 },
  { name: 'smoothing', label: 'Smooth borders', weight: 0.02 },
  { name: 'intersections', label: 'Intersections', weight: 0.03 },
  { name: 'vegetation', label: 'Vegetation', weight: 0.1 },
  { name: 'validate', label: 'Validate', weight: 0.08 },
  { name: 'export', label: 'Export tiles', weight: 0.28 },
  { name: 'manifest', label: 'Write manifest', weight: 0.03 },
];

export const DEFAULT_SOURCES: readonly string[] = ['osm', 'dem', 'worldcover', 'canopy_height', 'overture'];

function stageView(def: StageDef): StageView {
  return {
    name: def.name,
    label: def.label || humanize(def.name),
    weight: typeof def.weight === 'number' && def.weight > 0 ? def.weight : 0,
    status: 'pending',
    startT: null,
    endT: null,
    duration_s: null,
    progress: null,
  };
}

export function initialProgressState(): ProgressState {
  return {
    lastSeq: 0,
    phase: 'waiting',
    ended: false,
    t: 0,
    overall: 0,
    hasPlan: false,
    region: null,
    stages: DEFAULT_STAGES.map(stageView),
    currentStage: null,
    sources: DEFAULT_SOURCES.map((name) => ({ name, status: 'pending', detail: null })),
    planTiles: [],
    tiles: {},
    tileCounts: { pending: 0, building: 0, built: 0, unchanged: 0 },
    exportProgress: null,
    log: [],
    logTotal: 0,
    warnings: 0,
    errors: 0,
    summary: null,
    worldUrl: null,
    error: null,
  };
}

/** Applies one event; returns a new state (or the same object when ignored). */
export function reduceProgress(state: ProgressState, event: JobEvent): ProgressState {
  return reduceEvents(state, [event]);
}

/** Applies events in order; returns a new state (or `state` when all were ignored). */
export function reduceEvents(state: ProgressState, events: readonly JobEvent[]): ProgressState {
  let draft: ProgressState | null = null;
  for (const ev of events) {
    if (!ev || typeof ev !== 'object') continue;
    const current: ProgressState = draft ?? state;
    if (Number.isFinite(ev.seq) && ev.seq <= current.lastSeq) continue;
    if (!draft) draft = cloneState(state);
    apply(draft, ev);
  }
  return draft ?? state;
}

function cloneState(s: ProgressState): ProgressState {
  return {
    ...s,
    stages: s.stages.map((st) => ({ ...st, progress: st.progress ? { ...st.progress } : null })),
    sources: s.sources.map((src) => ({ ...src })),
    tiles: { ...s.tiles },
    tileCounts: { ...s.tileCounts },
    exportProgress: s.exportProgress ? { ...s.exportProgress } : null,
    log: s.log.slice(),
  };
}

/** Mutates the private draft. */
function apply(s: ProgressState, ev: JobEvent): void {
  if (Number.isFinite(ev.seq)) s.lastSeq = ev.seq;
  if (typeof ev.t === 'number' && Number.isFinite(ev.t)) s.t = Math.max(s.t, ev.t);
  const t = typeof ev.t === 'number' && Number.isFinite(ev.t) ? ev.t : s.t;

  switch (ev.type) {
    case 'job':
      if (ev.state === 'queued' || ev.state === 'running') {
        if (!isTerminal(s.phase)) s.phase = ev.state;
      } else if (ev.state) {
        s.phase = ev.state;
      }
      break;

    case 'plan':
      applyPlan(s, ev.stages, ev.tiles, ev.sources);
      if (ev.region) s.region = ev.region;
      if (s.phase === 'waiting' || s.phase === 'queued') s.phase = 'running';
      break;

    case 'stage': {
      if (!ev.name) break;
      const st = findOrAddStage(s, ev.name);
      if (s.phase === 'waiting' || s.phase === 'queued') s.phase = 'running';
      if (ev.state === 'start') {
        // A stage that never reported its end is over once the next one starts.
        for (const other of s.stages) {
          if (other !== st && other.status === 'running') finishStage(other, t, null);
        }
        st.status = 'running';
        st.startT = t;
        st.endT = null;
        st.duration_s = null;
        s.currentStage = st.name;
      } else if (ev.state === 'end') {
        if (st.startT === null && typeof ev.duration_s === 'number') st.startT = t - ev.duration_s;
        finishStage(st, t, typeof ev.duration_s === 'number' ? ev.duration_s : null);
        if (s.currentStage === st.name) s.currentStage = null;
      } else if (ev.state === 'skipped') {
        st.status = 'skipped';
        if (s.currentStage === st.name) s.currentStage = null;
      }
      break;
    }

    case 'source': {
      if (!ev.name) break;
      let src = s.sources.find((x) => x.name === ev.name);
      if (!src) {
        src = { name: ev.name, status: 'pending', detail: null };
        s.sources.push(src);
      }
      src.status = ev.state;
      if (ev.detail !== undefined) src.detail = ev.detail ?? null;
      break;
    }

    case 'progress': {
      if (!ev.stage) break;
      const st = findOrAddStage(s, ev.stage);
      const done = finiteOr(ev.done, 0);
      const total = finiteOr(ev.total, 0);
      st.progress = { done, total, detail: ev.detail ?? null };
      if (st.status === 'pending') {
        st.status = 'running';
        st.startT = t;
        s.currentStage = st.name;
      }
      if (st.name === 'export') s.exportProgress = { done, total };
      break;
    }

    case 'tile': {
      if (!ev.id) break;
      const prev = s.tiles[ev.id];
      if (prev) s.tileCounts[prev]--;
      s.tiles[ev.id] = ev.state;
      s.tileCounts[ev.state] = (s.tileCounts[ev.state] ?? 0) + 1;
      break;
    }

    case 'log':
      pushLog(s, { seq: ev.seq, t, level: normalizeLevel(ev.level), message: String(ev.message ?? '') });
      break;

    case 'done':
      s.summary = ev.summary ?? {};
      if (ev.world_url) s.worldUrl = ev.world_url;
      s.phase = 'succeeded';
      s.overall = 1;
      settleStages(s, t, 'succeeded');
      break;

    case 'error':
      s.error = ev.message || 'The build failed.';
      pushLog(s, { seq: ev.seq, t, level: 'error', message: s.error });
      s.phase = 'failed';
      settleStages(s, t, 'failed');
      break;

    case 'end':
      s.ended = true;
      if (ev.state) s.phase = ev.state;
      if (ev.state === 'succeeded') s.overall = 1;
      settleStages(s, t, ev.state ?? 'failed');
      // Unfinished downloads and tiles are no longer in progress.
      for (const src of s.sources) if (src.status === 'start') src.status = 'pending';
      for (const [id, st] of Object.entries(s.tiles)) {
        if (st === 'building') {
          s.tiles[id] = 'pending';
          s.tileCounts.building--;
          s.tileCounts.pending++;
        }
      }
      break;

    default:
      // Unknown event type: only `seq` / `t` / `overall` are used.
      break;
  }

  if (typeof ev.overall === 'number' && Number.isFinite(ev.overall)) {
    s.overall = Math.max(s.overall, clamp01(ev.overall));
  } else if (ev.type === 'stage' || ev.type === 'progress') {
    s.overall = Math.max(s.overall, derivedOverall(s.stages));
  }
}

function applyPlan(s: ProgressState, stages: StageDef[] | undefined, tiles: PlanTile[] | undefined, sources: string[] | undefined): void {
  s.hasPlan = true;
  if (Array.isArray(stages) && stages.length) {
    const old = new Map(s.stages.map((st) => [st.name, st]));
    s.stages = stages.filter((d) => d && d.name).map((d) => {
      const fresh = stageView(d);
      const prev = old.get(d.name);
      // Keep progress already seen (a plan arriving late must not reset it).
      return prev && prev.status !== 'pending' ? { ...prev, label: fresh.label, weight: fresh.weight } : fresh;
    });
  }
  if (Array.isArray(sources)) {
    const old = new Map(s.sources.map((x) => [x.name, x]));
    s.sources = sources.map((name) => old.get(name) ?? { name, status: 'pending', detail: null });
  }
  if (Array.isArray(tiles)) {
    s.planTiles = tiles.filter((tl) => tl && typeof tl.id === 'string');
    const next: Record<string, TileStatus> = {};
    const counts: Record<TileStatus, number> = { pending: 0, building: 0, built: 0, unchanged: 0 };
    for (const tl of s.planTiles) {
      const st = s.tiles[tl.id] ?? 'pending';
      next[tl.id] = st;
      counts[st]++;
    }
    // Tiles reported before the plan (unusual) are kept.
    for (const [id, st] of Object.entries(s.tiles)) {
      if (!(id in next)) {
        next[id] = st;
        counts[st]++;
      }
    }
    s.tiles = next;
    s.tileCounts = counts;
  }
}

function findOrAddStage(s: ProgressState, name: string): StageView {
  let st = s.stages.find((x) => x.name === name);
  if (!st) {
    st = stageView({ name, weight: 0 });
    s.stages.push(st);
  }
  return st;
}

function finishStage(st: StageView, t: number, duration: number | null): void {
  st.status = 'done';
  st.endT = t;
  st.duration_s = duration ?? (st.startT !== null ? Math.max(0, t - st.startT) : null);
  if (st.progress && st.progress.total > 0) st.progress = { ...st.progress, done: st.progress.total };
}

/** Terminal bookkeeping: what happens to stages still running / pending. */
function settleStages(s: ProgressState, t: number, outcome: JobStateName): void {
  for (const st of s.stages) {
    if (st.status === 'running') {
      if (outcome === 'succeeded') finishStage(st, t, null);
      else {
        st.status = outcome === 'cancelled' ? 'cancelled' : 'failed';
        st.endT = t;
        st.duration_s = st.startT !== null ? Math.max(0, t - st.startT) : null;
      }
    } else if (st.status === 'pending' && outcome === 'succeeded') {
      // The build finished without reporting this stage.
      st.status = 'skipped';
    }
  }
  s.currentStage = null;
}

function pushLog(s: ProgressState, line: LogLine): void {
  s.log.push(line);
  if (s.log.length > MAX_LOG) s.log.splice(0, s.log.length - MAX_LOG);
  s.logTotal++;
  if (line.level === 'warning') s.warnings++;
  else if (line.level === 'error') s.errors++;
}

/** Overall progress from stage weights (used when events carry no `overall`). */
export function derivedOverall(stages: readonly StageView[]): number {
  const total = stages.reduce((a, st) => a + st.weight, 0);
  if (total <= 0) return 0;
  let done = 0;
  for (const st of stages) {
    if (st.status === 'done' || st.status === 'skipped') done += st.weight;
    else if (st.status === 'running' && st.progress && st.progress.total > 0) {
      done += st.weight * Math.min(1, st.progress.done / st.progress.total);
    }
  }
  return clamp01(done / total);
}

/** Tiles done/total for the export mini bar. */
export function exportCounts(s: ProgressState): { done: number; total: number } {
  if (s.exportProgress && s.exportProgress.total > 0) return s.exportProgress;
  const total = Object.keys(s.tiles).length;
  return { done: s.tileCounts.built + s.tileCounts.unchanged, total };
}

export function isTerminal(phase: Phase): boolean {
  return phase === 'succeeded' || phase === 'failed' || phase === 'cancelled';
}

export function stageLabel(s: ProgressState, name: string | null): string | null {
  if (!name) return null;
  return s.stages.find((st) => st.name === name)?.label ?? humanize(name);
}

/** Tile ring lookup for the map (plan order). */
export function tileRings(s: ProgressState): { id: string; ring: LatLonPair[] }[] {
  return s.planTiles.filter((tl) => Array.isArray(tl.ring) && tl.ring.length >= 3).map((tl) => ({ id: tl.id, ring: tl.ring! }));
}

function normalizeLevel(level: unknown): LogLevel {
  return level === 'warning' || level === 'error' ? level : level === 'warn' ? 'warning' : 'info';
}

function humanize(name: string): string {
  const s = name.replace(/_/g, ' ');
  return s.charAt(0).toUpperCase() + s.slice(1);
}

function clamp01(v: number): number {
  return Math.min(1, Math.max(0, v));
}

function finiteOr(v: unknown, fallback: number): number {
  return typeof v === 'number' && Number.isFinite(v) ? v : fallback;
}
