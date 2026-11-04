/**
 * Fetching of world documents and progressive tile loading.
 */
import type { Group } from 'three';
import { GLTFLoader } from 'three/addons/loaders/GLTFLoader.js';
import type { SchemaDoc, TileEntry, TileMeta } from './types';

/** Major schema version this viewer understands. */
export const SUPPORTED_MAJOR = 1;

export class HttpError extends Error {
  readonly status: number;
  readonly url: string;

  constructor(status: number, url: string, message: string) {
    super(message);
    this.name = 'HttpError';
    this.status = status;
    this.url = url;
  }
}

/**
 * Base URL of the compiled world: `?world=<url>` (absolute, or relative to the
 * page) or `world/` relative to the page by default. Always ends with "/".
 */
export function resolveWorldBase(location: Location = window.location): URL {
  const param = new URLSearchParams(location.search).get('world');
  let raw = param && param.trim() ? param.trim() : 'world/';
  if (!raw.endsWith('/')) raw += '/';
  return new URL(raw, location.href);
}

/** Resolves a world-relative path (e.g. "tiles/0_0.glb") against the base URL. */
export function worldUrl(base: URL, path: string): string {
  return new URL(path.replace(/^\.?\//, ''), base).href;
}

/** Fetches and parses JSON; throws HttpError on HTTP errors and Error on bad JSON. */
export async function fetchJson<T>(url: string, signal?: AbortSignal): Promise<T> {
  let res: Response;
  try {
    res = await fetch(url, { signal, cache: 'no-cache' });
  } catch (e) {
    throw new HttpError(0, url, `Network error while fetching ${url}: ${errorText(e)}`);
  }
  if (!res.ok) throw new HttpError(res.status, url, `HTTP ${res.status} for ${url}`);
  const text = await res.text();
  try {
    return JSON.parse(text) as T;
  } catch {
    const hint = /^\s*</.test(text) ? ' (got HTML — is the world directory being served?)' : '';
    throw new Error(`${url} is not valid JSON${hint}`);
  }
}

/** Like fetchJson but resolves to null when the document does not exist (404/410). */
export async function fetchOptionalJson<T>(url: string, signal?: AbortSignal): Promise<T | null> {
  try {
    return await fetchJson<T>(url, signal);
  } catch (e) {
    if (e instanceof HttpError && (e.status === 404 || e.status === 410)) return null;
    throw e;
  }
}

/**
 * Validates `schema` / `schema_version`. Throws for an unsupported major version
 * (the contract requires consumers to reject those); returns warnings for
 * softer problems.
 */
export function checkSchema(doc: unknown, expectedSchema: string, label: string): string[] {
  const warnings: string[] = [];
  if (!doc || typeof doc !== 'object') throw new Error(`${label} is not a JSON object`);
  const d = doc as SchemaDoc;
  if (d.schema !== undefined && d.schema !== expectedSchema) {
    warnings.push(`${label}: unexpected schema "${d.schema}" (expected "${expectedSchema}")`);
  }
  if (d.schema_version === undefined) {
    warnings.push(`${label}: no schema_version`);
  } else {
    const major = Number.parseInt(String(d.schema_version).split('.')[0], 10);
    if (!Number.isFinite(major)) {
      warnings.push(`${label}: unparseable schema_version "${d.schema_version}"`);
    } else if (major !== SUPPORTED_MAJOR) {
      throw new Error(
        `${label} has schema_version ${d.schema_version}; this viewer only understands ${SUPPORTED_MAJOR}.x`,
      );
    }
  }
  return warnings;
}

export interface TileLoadCallbacks {
  /** Called as soon as a tile's GLB has been parsed. */
  onTile(entry: TileEntry, scene: Group): void;
  /** Called when a tile's meta.json has loaded (null = not available). */
  onMeta(entry: TileEntry, meta: TileMeta | null, error: string | null): void;
  onProgress(state: TileProgress): void;
  onError(entry: TileEntry, message: string): void;
}

export interface TileProgress {
  total: number;
  loaded: number;
  failed: number;
  done: boolean;
}

/**
 * Loads every tile GLB (+ meta.json alongside), nearest to `focus` first, with
 * a small concurrency limit. Failures are reported per tile and never abort the
 * whole load.
 */
export class TileLoader {
  private readonly gltf = new GLTFLoader();
  private readonly base: URL;
  private readonly concurrency: number;

  constructor(base: URL, concurrency = 4) {
    this.base = base;
    this.concurrency = Math.max(1, concurrency);
  }

  async loadAll(entries: TileEntry[], focus: { x: number; y: number }, cb: TileLoadCallbacks): Promise<void> {
    const queue = [...entries].sort((a, b) => tileDistance(a, focus) - tileDistance(b, focus));
    const state: TileProgress = { total: queue.length, loaded: 0, failed: 0, done: queue.length === 0 };
    cb.onProgress({ ...state });
    let next = 0;
    const metaPromises: Promise<void>[] = [];

    const worker = async (): Promise<void> => {
      while (next < queue.length) {
        const entry = queue[next++];
        metaPromises.push(this.loadMeta(entry, cb));
        try {
          if (!entry.glb) throw new Error('tile index entry has no "glb" path');
          const gltf = await this.gltf.loadAsync(worldUrl(this.base, entry.glb));
          cb.onTile(entry, gltf.scene);
          state.loaded++;
        } catch (e) {
          state.failed++;
          cb.onError(entry, `Tile ${entry.id}: ${describeLoadError(e)}`);
        }
        cb.onProgress({ ...state });
      }
    };

    await Promise.all(Array.from({ length: Math.min(this.concurrency, queue.length) }, worker));
    await Promise.all(metaPromises);
    state.done = true;
    cb.onProgress({ ...state });
  }

  private async loadMeta(entry: TileEntry, cb: TileLoadCallbacks): Promise<void> {
    if (!entry.meta) {
      cb.onMeta(entry, null, 'tile index entry has no "meta" path');
      return;
    }
    try {
      const meta = await fetchOptionalJson<TileMeta>(worldUrl(this.base, entry.meta));
      if (!meta) {
        cb.onMeta(entry, null, `${entry.meta} not found`);
        return;
      }
      checkSchema(meta, 'worldgen.tile_meta', entry.meta);
      if (!meta.entities || typeof meta.entities !== 'object') meta.entities = {};
      cb.onMeta(entry, meta, null);
    } catch (e) {
      cb.onMeta(entry, null, errorText(e));
    }
  }
}

function tileDistance(t: TileEntry, focus: { x: number; y: number }): number {
  const b = t.bounds;
  if (!b || b.length < 4) return Number.MAX_VALUE;
  const cx = (b[0] + b[2]) / 2;
  const cy = (b[1] + b[3]) / 2;
  return Math.hypot(cx - focus.x, cy - focus.y);
}

function describeLoadError(e: unknown): string {
  // GLTFLoader reports HTTP failures as "fetch for "<url>" responded with 404: Not Found".
  const msg = errorText(e);
  if (/responded with 404/.test(msg)) return 'GLB not found (404)';
  return msg;
}

export function errorText(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (e && typeof e === 'object' && 'message' in e) return String((e as { message: unknown }).message);
  return String(e);
}
