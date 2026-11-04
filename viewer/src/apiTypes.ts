/**
 * Types mirroring docs/studio-api.md (the `worldgen serve` JSON + SSE API).
 *
 * Types only: no DOM, no runtime code, so pure modules (and `node --test`)
 * can import them with `import type`. Everything the UI does not strictly
 * need is optional so that a newer or partial server never crashes the page.
 */

/** `[lat, lon]` in WGS84 degrees (the API's ring order). */
export type LatLonPair = [number, number];

export interface Health {
  ok: boolean;
  version?: string;
  output_root?: string;
  default_world?: string | null;
  busy?: boolean;
}

export interface Profile {
  name: string;
  description?: string;
}

export interface RegionInfo {
  name?: string;
  center?: { lat: number; lon: number } | null;
  radius_m?: number | null;
  /** [west, south, east, north] */
  bbox_wgs84?: [number, number, number, number] | null;
}

export interface PlanTile {
  id: string;
  ring?: LatLonPair[];
}

export interface PlanLimits {
  max_radius_m?: number;
  max_tiles?: number;
}

export interface Plan {
  region: RegionInfo;
  footprint?: LatLonPair[];
  tile_size?: number;
  tiles: PlanTile[];
  tile_count?: number;
  area_km2?: number;
  limits?: PlanLimits;
}

export interface ValidationCounts {
  errors?: number;
  warnings?: number;
  info?: number;
}

export interface WorldInfo {
  name: string;
  url: string;
  center?: { lat: number; lon: number } | null;
  radius_m?: number | null;
  bbox_wgs84?: [number, number, number, number] | null;
  footprint?: LatLonPair[] | null;
  created_utc?: string | null;
  profile?: string | null;
  tile_count?: number | null;
  statistics?: Record<string, number> | null;
  validation?: ValidationCounts | null;
}

export type JobStateName = 'queued' | 'running' | 'succeeded' | 'failed' | 'cancelled';

/** POST /api/jobs body (also `Job.request`, normalised by the server). */
export interface JobRequest {
  name?: string;
  center?: { lat: number; lon: number };
  radius_m?: number;
  /** [south, west, north, east] */
  bbox?: [number, number, number, number];
  profile?: string;
  offline?: boolean;
  force?: boolean;
  /** Instance budgets: above them, smaller buildings / trees are dropped at random. */
  budget?: BudgetOptions;
  /** "drape": roads follow the terrain triangles; "ribbon": flat cross-sections. */
  road_surface?: RoadSurface;
}

export type RoadSurface = 'drape' | 'ribbon';

export interface BudgetOptions {
  enabled: boolean;
  buildings_per_tile?: number;
  trees_per_tile?: number;
}

export interface BuildSummary {
  statistics?: Record<string, number>;
  validation?: ValidationCounts;
  tiles_built?: number;
  tiles_unchanged?: number;
  duration_s?: number;
}

export interface Job {
  id: string;
  name: string;
  state: JobStateName;
  created_utc?: string | null;
  started_utc?: string | null;
  finished_utc?: string | null;
  request?: JobRequest | null;
  world_url?: string | null;
  progress?: number | null;
  stage?: string | null;
  message?: string | null;
  summary?: BuildSummary | null;
}

// ---------------------------------------------------------------- events

export interface StageDef {
  name: string;
  label?: string;
  weight?: number;
}

export type SourceStateName = 'start' | 'ok' | 'partial' | 'missing' | 'failed' | 'disabled';
export type TileStateName = 'building' | 'built' | 'unchanged';
export type LogLevel = 'info' | 'warning' | 'error';

interface EventBase {
  /** 1, 2, … per job. */
  seq: number;
  /** Seconds since the job started. */
  t?: number;
  /** Overall progress 0..1 (on events that change it). */
  overall?: number;
}

export interface JobStateEvent extends EventBase {
  type: 'job';
  state: JobStateName;
}

export interface PlanEvent extends EventBase {
  type: 'plan';
  stages?: StageDef[];
  tiles?: PlanTile[];
  region?: RegionInfo;
  sources?: string[];
}

export interface StageEvent extends EventBase {
  type: 'stage';
  name: string;
  state: 'start' | 'end' | 'skipped';
  duration_s?: number;
}

export interface SourceEvent extends EventBase {
  type: 'source';
  name: string;
  state: SourceStateName;
  detail?: string | null;
}

export interface StageProgressEvent extends EventBase {
  type: 'progress';
  stage: string;
  done: number;
  total: number;
  detail?: string | null;
}

export interface TileEvent extends EventBase {
  type: 'tile';
  id: string;
  state: TileStateName;
}

export interface LogEvent extends EventBase {
  type: 'log';
  level: LogLevel;
  message: string;
}

export interface DoneEvent extends EventBase {
  type: 'done';
  summary?: BuildSummary;
  world_url?: string;
}

export interface JobErrorEvent extends EventBase {
  type: 'error';
  message?: string;
}

export interface EndEvent extends EventBase {
  type: 'end';
  state: 'succeeded' | 'failed' | 'cancelled';
}

export type JobEvent =
  | JobStateEvent
  | PlanEvent
  | StageEvent
  | SourceEvent
  | StageProgressEvent
  | TileEvent
  | LogEvent
  | DoneEvent
  | JobErrorEvent
  | EndEvent;

export type JobEventType = JobEvent['type'];
