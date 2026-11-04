/**
 * Lazily loaded tile meta.json documents, at most `capacity` at a time.
 *
 * meta.json files are large (tree records dominate) and only needed to inspect
 * features, style buildings and colour by metadata. They are fetched for tiles
 * near the camera (`setNear`) and on demand (`require`, e.g. after a click);
 * the least recently used ones beyond the capacity are dropped, except pinned
 * ones (the selection) and the current near set.
 *
 * Pure module apart from the injected loader (unit-tested with node --test).
 */
import { LruCache } from './cache.ts';
import { RequestQueue } from './requestQueue.ts';
import type { TileEntry, TileMeta } from '../types';

export interface MetaCacheOptions {
  /** Maximum number of documents kept. */
  capacity?: number;
  /** Concurrent fetches. */
  maxActive?: number;
  /** Fetches one document; null = the tile has none. */
  load(entry: TileEntry, signal: AbortSignal): Promise<TileMeta | null>;
  onLoaded(tileId: string, meta: TileMeta | null, error: string | null): void;
  onEvicted(tileId: string): void;
}

export interface NearTile {
  entry: TileEntry;
  /** Distance (m) from the camera: nearer tiles load first. */
  distance: number;
}

/** Urgent (required) requests start before every proximity request. */
const URGENT = -1;

export class MetaCache {
  private readonly opts: MetaCacheOptions;
  private readonly lru: LruCache<TileMeta | null>;
  private readonly queue: RequestQueue;
  private readonly entries = new Map<string, TileEntry>();
  private near = new Map<string, number>();
  private readonly urgent = new Set<string>();
  private readonly pinned = new Set<string>();
  private readonly waiters = new Map<string, ((meta: TileMeta | null) => void)[]>();

  constructor(opts: MetaCacheOptions) {
    this.opts = opts;
    this.lru = new LruCache<TileMeta | null>(Math.max(1, opts.capacity ?? 64), (id) => this.opts.onEvicted(id));
    this.queue = new RequestQueue({
      maxActive: opts.maxActive ?? 2,
      run: (id, signal) => this.fetch(id, signal),
    });
  }

  get capacity(): number {
    return this.lru.budget;
  }

  get size(): number {
    return this.lru.size;
  }

  get loadingCount(): number {
    return this.queue.activeCount + this.queue.queuedCount;
  }

  has(tileId: string): boolean {
    return this.lru.has(tileId);
  }

  get(tileId: string): TileMeta | null | undefined {
    return this.lru.get(tileId);
  }

  /** Ids of the cached documents, least recently used first. */
  keys(): string[] {
    return this.lru.keys();
  }

  /**
   * Replaces the proximity set (tiles near the camera or otherwise wanted
   * soon). Requests for tiles no longer near are dropped or aborted.
   */
  setNear(tiles: readonly NearTile[]): void {
    const near = new Map<string, number>();
    for (const t of tiles) {
      this.entries.set(t.entry.id, t.entry);
      near.set(t.entry.id, t.distance);
      this.lru.touch(t.entry.id);
    }
    this.near = near;
    this.sync();
  }

  /** Keeps these documents (e.g. the selected feature's tiles) whatever the camera does. */
  setPinned(tileIds: Iterable<string>): void {
    this.pinned.clear();
    for (const id of tileIds) this.pinned.add(id);
  }

  /** Loads a tile's document as soon as possible; resolves once it is available. */
  require(entry: TileEntry): Promise<TileMeta | null> {
    this.entries.set(entry.id, entry);
    if (this.lru.has(entry.id)) {
      this.lru.touch(entry.id);
      return Promise.resolve(this.lru.get(entry.id) ?? null);
    }
    const p = new Promise<TileMeta | null>((resolve) => {
      const list = this.waiters.get(entry.id) ?? [];
      list.push(resolve);
      this.waiters.set(entry.id, list);
    });
    this.urgent.add(entry.id);
    this.sync();
    return p;
  }

  /** Aborts everything in flight. */
  clear(): void {
    this.queue.clear();
    this.urgent.clear();
    for (const [, list] of this.waiters) for (const w of list) w(null);
    this.waiters.clear();
  }

  private sync(): void {
    const wanted: { id: string; priority: number }[] = [];
    for (const id of this.urgent) if (!this.lru.has(id)) wanted.push({ id, priority: URGENT });
    for (const [id, d] of this.near) if (!this.lru.has(id) && !this.urgent.has(id)) wanted.push({ id, priority: d });
    this.queue.update(wanted);
  }

  private async fetch(id: string, signal: AbortSignal): Promise<void> {
    const entry = this.entries.get(id);
    let meta: TileMeta | null = null;
    let error: string | null = null;
    if (!entry) {
      error = 'unknown tile';
    } else if (!entry.meta) {
      error = 'tile index entry has no "meta" path';
    } else {
      try {
        meta = await this.opts.load(entry, signal);
        if (!meta) error = `${entry.meta} not found`;
      } catch (e) {
        if (signal.aborted) throw e;
        error = e instanceof Error ? e.message : String(e);
      }
    }
    if (signal.aborted) throw new DOMException('aborted', 'AbortError');
    // Failures are cached too (as null): they are not retried on every camera move.
    this.lru.set(id, meta, 1);
    this.urgent.delete(id);
    this.opts.onLoaded(id, meta, error);
    for (const w of this.waiters.get(id) ?? []) w(meta);
    this.waiters.delete(id);
    this.lru.trim((k) => this.pinned.has(k) || this.near.has(k) || this.urgent.has(k) || k === id);
  }
}
