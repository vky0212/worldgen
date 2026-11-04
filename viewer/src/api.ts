/**
 * Small client for the `worldgen serve` JSON API (docs/studio-api.md).
 *
 * URLs are resolved relative to the page (`api/...`), which is `/api/...`
 * when the viewer is served at `/` (by `worldgen serve` or the Vite dev
 * server's proxy).
 */
import type { Health, Job, JobEvent, JobRequest, Plan, Profile, WorldInfo } from './apiTypes';

export class ApiError extends Error {
  /** HTTP status; 0 = network error / timeout (API unreachable). */
  readonly status: number;

  constructor(status: number, message: string) {
    super(message);
    this.name = 'ApiError';
    this.status = status;
  }
}

type Params = Record<string, string | number | boolean | null | undefined>;

export function apiUrl(path: string, params?: Params): string {
  const url = new URL(`api/${path.replace(/^\/+/, '')}`, window.location.href);
  for (const [k, v] of Object.entries(params ?? {})) {
    if (v !== undefined && v !== null && v !== '') url.searchParams.set(k, String(v));
  }
  return url.href;
}

interface RequestOptions {
  params?: Params;
  body?: unknown;
  signal?: AbortSignal;
  timeoutMs?: number;
}

async function request<T>(method: 'GET' | 'POST', path: string, opts: RequestOptions = {}): Promise<T> {
  const signals: AbortSignal[] = [];
  if (opts.signal) signals.push(opts.signal);
  if (opts.timeoutMs) signals.push(AbortSignal.timeout(opts.timeoutMs));
  const signal = signals.length > 1 ? AbortSignal.any(signals) : signals[0];
  let res: Response;
  try {
    res = await fetch(apiUrl(path, opts.params), {
      method,
      signal,
      cache: 'no-store',
      headers: opts.body !== undefined ? { 'Content-Type': 'application/json' } : undefined,
      body: opts.body !== undefined ? JSON.stringify(opts.body) : undefined,
    });
  } catch (e) {
    // Let callers distinguish their own aborts from "server unreachable".
    if (opts.signal?.aborted) throw e;
    throw new ApiError(0, `The WorldGen API is unreachable (${e instanceof Error ? e.message : String(e)}).`);
  }
  let text: string;
  try {
    text = await res.text();
  } catch (e) {
    if (opts.signal?.aborted) throw e;
    throw new ApiError(0, `The connection to the WorldGen API broke (${e instanceof Error ? e.message : String(e)}).`);
  }
  let data: unknown = null;
  try {
    data = text ? JSON.parse(text) : null;
  } catch {
    // Non-JSON: e.g. a static server answering index.html for /api/*.
    if (res.ok) throw new ApiError(res.status, `api/${path} did not return JSON (is the WorldGen API running?)`);
  }
  if (!res.ok) {
    const msg = data && typeof data === 'object' && 'error' in data ? String((data as { error: unknown }).error) : `HTTP ${res.status}`;
    throw new ApiError(res.status, msg);
  }
  return data as T;
}

export const api = {
  health: (timeoutMs = 3000): Promise<Health> => request<Health>('GET', 'health', { timeoutMs }),
  profiles: (): Promise<{ profiles: Profile[] }> => request('GET', 'profiles', { timeoutMs: 10000 }),
  plan: (params: Params, signal?: AbortSignal): Promise<Plan> => request('GET', 'plan', { params, signal, timeoutMs: 30000 }),
  worlds: (): Promise<{ worlds: WorldInfo[] }> => request('GET', 'worlds', { timeoutMs: 10000 }),
  jobs: (): Promise<{ jobs: Job[] }> => request('GET', 'jobs', { timeoutMs: 10000 }),
  job: (id: string): Promise<{ job: Job; events?: JobEvent[] }> => request('GET', `jobs/${encodeURIComponent(id)}`, { timeoutMs: 15000 }),
  createJob: (body: JobRequest): Promise<{ job: Job }> => request('POST', 'jobs', { body, timeoutMs: 15000 }),
  cancelJob: (id: string): Promise<{ job: Job }> => request('POST', `jobs/${encodeURIComponent(id)}/cancel`, { body: {}, timeoutMs: 15000 }),
  eventsUrl: (id: string, after: number): string => apiUrl(`jobs/${encodeURIComponent(id)}/events`, { after }),
};

/** Viewer link for a world URL (`index.html?world=/worlds/x/`), relative to the page. */
export function viewerHref(worldUrl: string): string {
  return `index.html?world=${encodeURIComponent(worldUrl).replace(/%2F/gi, '/')}`;
}

/** Network error, timeout, or a gateway error from a proxy (e.g. the Vite dev server). */
export function isUnreachable(e: unknown): boolean {
  return e instanceof ApiError && (e.status === 0 || e.status === 502 || e.status === 503 || e.status === 504);
}

export function errorMessage(e: unknown): string {
  if (e instanceof Error) return e.message;
  return String(e);
}
