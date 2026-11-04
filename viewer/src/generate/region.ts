/**
 * Pure region helpers for the generate page: region model, API query/body
 * building, name suggestion and validation, presets.
 *
 * No DOM and no Leaflet; only erasable TypeScript syntax and type-only
 * imports, so it runs under `node --test`.
 */
import type { BudgetOptions, JobRequest, RoadSurface, WorldInfo } from '../apiTypes';

export interface CircleRegion {
  kind: 'circle';
  lat: number;
  lon: number;
  radius_m: number;
}

export interface BboxRegion {
  kind: 'bbox';
  south: number;
  west: number;
  north: number;
  east: number;
}

export type Region = CircleRegion | BboxRegion;

export const NAME_RE = /^[a-z0-9_-]{1,48}$/;
export const MIN_RADIUS_M = 100;
export const DEFAULT_RADIUS_M = 1000;
/** Used until `/api/plan` reports `limits.max_radius_m`. */
export const FALLBACK_MAX_RADIUS_M = 5000;
/** Tile count above which the page warns about a large build. */
export const LARGE_TILE_COUNT = 200;

export interface Preset {
  id: string;
  label: string;
  lat: number;
  lon: number;
}

export const PRESETS: readonly Preset[] = [
  { id: 'taipei_101', label: 'Taipei 101', lat: 25.033, lon: 121.5654 },
  { id: 'lower_manhattan', label: 'Lower Manhattan', lat: 40.7075, lon: -74.0113 },
  { id: 'paris_ile_de_la_cite', label: 'Paris – Île de la Cité', lat: 48.8546, lon: 2.3477 },
  { id: 'tokyo_shibuya', label: 'Tokyo – Shibuya', lat: 35.6595, lon: 139.7005 },
  { id: 'zurich', label: 'Zurich', lat: 47.3769, lon: 8.5417 },
];

const round = (v: number, digits: number): number => {
  const f = 10 ** digits;
  return Math.round(v * f) / f;
};

export function regionCenter(r: Region): { lat: number; lon: number } {
  if (r.kind === 'circle') return { lat: r.lat, lon: r.lon };
  return { lat: (r.south + r.north) / 2, lon: (r.west + r.east) / 2 };
}

/** Orders the edges and clamps to valid WGS84 ranges. */
export function normalizeBbox(b: { south: number; west: number; north: number; east: number }): BboxRegion {
  const south = clamp(Math.min(b.south, b.north), -85, 85);
  const north = clamp(Math.max(b.south, b.north), -85, 85);
  const west = clamp(Math.min(b.west, b.east), -180, 180);
  const east = clamp(Math.max(b.west, b.east), -180, 180);
  return { kind: 'bbox', south, west, north, east };
}

/** Returns a problem description, or null when the region is usable. */
export function regionProblem(r: Region): string | null {
  const finite = (...v: number[]): boolean => v.every((x) => typeof x === 'number' && Number.isFinite(x));
  if (r.kind === 'circle') {
    if (!finite(r.lat, r.lon, r.radius_m)) return 'Enter a valid latitude, longitude and radius.';
    if (Math.abs(r.lat) > 85 || Math.abs(r.lon) > 180) return 'Latitude must be within ±85°, longitude within ±180°.';
    if (r.radius_m < MIN_RADIUS_M) return `The radius must be at least ${MIN_RADIUS_M} m.`;
    return null;
  }
  if (!finite(r.south, r.west, r.north, r.east)) return 'Enter valid south, west, north and east coordinates.';
  if (r.south >= r.north || r.west >= r.east) return 'The rectangle is empty (south < north and west < east are required).';
  if (Math.abs(r.south) > 85 || Math.abs(r.north) > 85 || Math.abs(r.west) > 180 || Math.abs(r.east) > 180) {
    return 'Latitude must be within ±85°, longitude within ±180°.';
  }
  return null;
}

/** Query parameters of `GET /api/plan`. */
export function planParams(r: Region, profile?: string | null): Record<string, string | number> {
  const p: Record<string, string | number> =
    r.kind === 'circle'
      ? { lat: round(r.lat, 6), lon: round(r.lon, 6), radius: Math.round(r.radius_m) }
      : { bbox: [r.south, r.west, r.north, r.east].map((v) => round(v, 6)).join(',') };
  if (profile) p.profile = profile;
  return p;
}

export interface JobOptions {
  name?: string;
  profile?: string | null;
  offline?: boolean;
  force?: boolean;
  budget?: BudgetOptions;
  roadSurface?: RoadSurface;
}

/** Server defaults (configs: buildings.budget / vegetation.budget). */
export const DEFAULT_BUDGET: Required<BudgetOptions> = { enabled: true, buildings_per_tile: 250, trees_per_tile: 2500 };

/** Body of `POST /api/jobs`. */
export function jobBody(r: Region, opts: JobOptions = {}): JobRequest {
  const body: JobRequest =
    r.kind === 'circle'
      ? { center: { lat: round(r.lat, 6), lon: round(r.lon, 6) }, radius_m: Math.round(r.radius_m) }
      : { bbox: [round(r.south, 6), round(r.west, 6), round(r.north, 6), round(r.east, 6)] };
  if (opts.name) body.name = opts.name;
  if (opts.profile) body.profile = opts.profile;
  body.offline = !!opts.offline;
  body.force = !!opts.force;
  if (opts.budget) {
    const b: BudgetOptions = { enabled: opts.budget.enabled };
    if (opts.budget.enabled) {
      for (const key of ['buildings_per_tile', 'trees_per_tile'] as const) {
        const v = opts.budget[key];
        if (v !== undefined && Number.isFinite(v) && v >= 1) b[key] = Math.round(v);
      }
    }
    body.budget = b;
  }
  if (opts.roadSurface) body.road_surface = opts.roadSurface;
  return body;
}

/** Lower-case slug limited to [a-z0-9_-]{1,48}; "" when nothing is left. */
export function slugify(text: string): string {
  return text
    .normalize('NFKD')
    .replace(/[̀-ͯ]/g, '')
    .toLowerCase()
    .replace(/[^a-z0-9_-]+/g, '_')
    .replace(/_+/g, '_')
    .replace(/^[_-]+|[_-]+$/g, '')
    .slice(0, 48)
    .replace(/[_-]+$/g, '');
}

/** Name suggestion: the place (preset/world) when known, else the centre, e.g. "n25_033_e121_565". */
export function suggestName(r: Region, place?: string | null): string {
  const fromPlace = place ? slugify(place) : '';
  if (fromPlace) return fromPlace;
  const c = regionCenter(r);
  const part = (v: number, pos: string, neg: string): string => `${v >= 0 ? pos : neg}${Math.abs(v).toFixed(3).replace('.', '_')}`;
  return `${part(c.lat, 'n', 's')}_${part(c.lon, 'e', 'w')}`;
}

export function nameProblem(name: string): string | null {
  if (!name) return 'Enter a name.';
  if (name.length > 48) return 'The name is longer than 48 characters.';
  if (!NAME_RE.test(name)) return 'Use only a–z, 0–9, "_" and "-" (lower case).';
  return null;
}

/** Region of an existing world (circle when it has centre + radius, else its bbox). */
export function regionFromWorld(w: Pick<WorldInfo, 'center' | 'radius_m' | 'bbox_wgs84'>): Region | null {
  if (w.center && typeof w.radius_m === 'number' && w.radius_m > 0) {
    return { kind: 'circle', lat: w.center.lat, lon: w.center.lon, radius_m: w.radius_m };
  }
  const b = w.bbox_wgs84;
  if (Array.isArray(b) && b.length === 4) return normalizeBbox({ west: b[0], south: b[1], east: b[2], north: b[3] });
  return null;
}

/** Region of a job request (`Job.request`). */
export function regionFromRequest(req: JobRequest | null | undefined): Region | null {
  if (!req) return null;
  if (req.center && typeof req.radius_m === 'number') {
    return { kind: 'circle', lat: req.center.lat, lon: req.center.lon, radius_m: req.radius_m };
  }
  const b = req.bbox;
  if (Array.isArray(b) && b.length === 4) return normalizeBbox({ south: b[0], west: b[1], north: b[2], east: b[3] });
  return null;
}

/** Stable key of a region (≈1 cm / 1 m precision) to detect real changes. */
export function regionKey(r: Region): string {
  return r.kind === 'circle'
    ? `c:${r.lat.toFixed(7)},${r.lon.toFixed(7)},${Math.round(r.radius_m)}`
    : `b:${[r.south, r.west, r.north, r.east].map((v) => v.toFixed(7)).join(',')}`;
}

/** Circle → bounding square; bbox → inscribed circle (radius clamped to [min, max]). */
export function convertRegion(r: Region, to: Region['kind'], maxRadius = FALLBACK_MAX_RADIUS_M): Region {
  if (r.kind === to) return r;
  const M_PER_DEG = 111320;
  if (r.kind === 'circle') {
    const dLat = r.radius_m / M_PER_DEG;
    const dLon = r.radius_m / (M_PER_DEG * Math.max(0.01, Math.cos((r.lat * Math.PI) / 180)));
    return normalizeBbox({ south: r.lat - dLat, north: r.lat + dLat, west: r.lon - dLon, east: r.lon + dLon });
  }
  const c = regionCenter(r);
  const halfH = ((r.north - r.south) / 2) * M_PER_DEG;
  const halfW = ((r.east - r.west) / 2) * M_PER_DEG * Math.cos((c.lat * Math.PI) / 180);
  const radius = clamp(Math.round(Math.min(halfH, halfW)), MIN_RADIUS_M, maxRadius);
  return { kind: 'circle', lat: c.lat, lon: c.lon, radius_m: radius };
}

function clamp(v: number, lo: number, hi: number): number {
  return Math.min(hi, Math.max(lo, v));
}
