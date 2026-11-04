/**
 * Geographic helpers: WGS84 <-> local ENU via proj4, and ENU <-> three.js axes.
 *
 * Local ENU: x = east, y = north, z = up (metres).
 * glTF / three.js: +Y up, so ENU (x, y, z) -> three (x, z, -y).
 */
import proj4 from 'proj4';
import type { Converter } from 'proj4';
import { Vector3 } from 'three';
import type { Manifest } from './types';

export interface LatLon {
  lat: number;
  lon: number;
}

export interface LocalXY {
  x: number;
  y: number;
}

const EARTH_RADIUS = 6371008.8;
const DEG = Math.PI / 180;

/**
 * Converts between WGS84 and the world's local frame. Uses
 * `manifest.crs.local.proj` when proj4 can parse it, otherwise falls back to a
 * local equirectangular approximation around `manifest.origin` (flagged by
 * `exact === false`). `available === false` when neither is possible.
 */
export class GeoFrame {
  readonly available: boolean;
  readonly exact: boolean;
  readonly warning: string | null;
  private readonly converter: Converter | null;
  private readonly origin: LatLon | null;

  constructor(manifest: Manifest | null) {
    const projString = manifest?.crs?.local?.proj;
    const origin = manifest?.origin ?? manifest?.region?.center ?? null;
    let converter: Converter | null = null;
    let warning: string | null = null;
    if (projString) {
      try {
        converter = proj4('EPSG:4326', projString);
        // Smoke test: catches PROJ strings proj4 accepts but cannot evaluate.
        const p = converter.forward([origin?.lon ?? 0, origin?.lat ?? 0]);
        if (!Number.isFinite(p[0]) || !Number.isFinite(p[1])) throw new Error('non-finite result');
      } catch (e) {
        converter = null;
        warning = `proj4 could not use manifest.crs.local.proj (${errorText(e)}); using an approximate local projection.`;
      }
    } else if (origin) {
      warning = 'manifest.crs.local.proj is missing; using an approximate local projection.';
    }
    this.converter = converter;
    this.origin = origin && Number.isFinite(origin.lat) && Number.isFinite(origin.lon) ? origin : null;
    this.exact = converter !== null;
    this.available = converter !== null || this.origin !== null;
    if (!this.available) warning = 'No CRS information in the manifest: geographic coordinates are unavailable.';
    this.warning = warning;
  }

  /** WGS84 -> local ENU x/y. Returns null when unavailable. */
  toLocal(lat: number, lon: number): LocalXY | null {
    if (this.converter) {
      const [x, y] = this.converter.forward([lon, lat]);
      return Number.isFinite(x) && Number.isFinite(y) ? { x, y } : null;
    }
    if (this.origin) {
      const o = this.origin;
      return {
        x: (lon - o.lon) * DEG * EARTH_RADIUS * Math.cos(o.lat * DEG),
        y: (lat - o.lat) * DEG * EARTH_RADIUS,
      };
    }
    return null;
  }

  /** Local ENU x/y -> WGS84. Returns null when unavailable. */
  toLatLon(x: number, y: number): LatLon | null {
    if (this.converter) {
      const [lon, lat] = this.converter.inverse([x, y]);
      return Number.isFinite(lat) && Number.isFinite(lon) ? { lat, lon } : null;
    }
    if (this.origin) {
      const o = this.origin;
      return {
        lat: o.lat + y / EARTH_RADIUS / DEG,
        lon: o.lon + x / (EARTH_RADIUS * Math.cos(o.lat * DEG)) / DEG,
      };
    }
    return null;
  }
}

/** ENU (x east, y north, z up) -> three.js world position. */
export function enuToThree(x: number, y: number, z: number, target = new Vector3()): Vector3 {
  return target.set(x, z, -y);
}

/** three.js world position -> ENU. */
export function threeToEnu(v: Vector3): { x: number; y: number; z: number } {
  return { x: v.x, y: -v.z, z: v.y };
}

/**
 * Parses "lat, lon" (comma, semicolon or whitespace separated, optional N/S/E/W
 * suffixes). Returns null when the text is not a valid coordinate pair.
 */
export function parseLatLon(text: string): LatLon | null {
  const cleaned = text.trim().replace(/°/g, ' ');
  if (!cleaned) return null;
  const parts = cleaned.split(/[,;]\s*|\s+/).filter(Boolean);
  if (parts.length !== 2) return null;
  const parsePart = (s: string, pos: string, neg: string): number => {
    const m = /^([+-]?\d+(?:\.\d+)?)([A-Za-z]?)$/.exec(s);
    if (!m) return NaN;
    let v = Number(m[1]);
    const hemi = m[2].toUpperCase();
    if (hemi === neg) v = -Math.abs(v);
    else if (hemi && hemi !== pos) return NaN;
    return v;
  };
  const lat = parsePart(parts[0], 'N', 'S');
  const lon = parsePart(parts[1], 'E', 'W');
  if (!Number.isFinite(lat) || !Number.isFinite(lon)) return null;
  if (Math.abs(lat) > 90 || Math.abs(lon) > 180) return null;
  return { lat, lon };
}

export function formatLatLon(p: LatLon | null, digits = 6): string {
  if (!p) return '—';
  return `${p.lat.toFixed(digits)}, ${p.lon.toFixed(digits)}`;
}

export function formatMetres(m: number): string {
  if (!Number.isFinite(m)) return '—';
  return Math.abs(m) >= 1000 ? `${(m / 1000).toFixed(2)} km` : `${m.toFixed(1)} m`;
}

function errorText(e: unknown): string {
  return e instanceof Error ? e.message : String(e);
}
