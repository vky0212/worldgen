/**
 * Fetching of world documents (tiles stream through stream/TileStreamer).
 */
import type { SchemaDoc } from './types';

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

export function errorText(e: unknown): string {
  if (e instanceof Error) return e.message;
  if (e && typeof e === 'object' && 'message' in e) return String((e as { message: unknown }).message);
  return String(e);
}
