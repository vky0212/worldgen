/**
 * Prioritised fetch queue with a concurrency limit and cancellation.
 *
 * `update()` replaces the set of wanted requests: queued entries that are no
 * longer wanted are dropped and running ones are aborted (their AbortSignal
 * fires), so the network always works on what the camera needs now. Lower
 * priority numbers start first. An optional `canStart` gate (e.g. the GPU
 * budget) holds back further starts without dropping anything.
 *
 * Pure module: no three.js, no DOM (unit-tested with node --test).
 */

export type RequestOutcome = 'done' | 'error' | 'aborted';

export interface RequestQueueOptions {
  /** Maximum number of requests running at once. */
  maxActive: number;
  /** Performs one request; must reject (or resolve) once `signal` aborts. */
  run: (id: string, signal: AbortSignal) => Promise<void>;
  /** Gate for starting the next request (checked in priority order). */
  canStart?: (id: string) => boolean;
  /** Called once per started request when it settles or is aborted. */
  onSettled?: (id: string, outcome: RequestOutcome, error?: unknown) => void;
}

export interface WantedRequest {
  id: string;
  priority: number;
}

interface Active {
  controller: AbortController;
  priority: number;
}

export class RequestQueue {
  private readonly opts: RequestQueueOptions;
  private readonly queued = new Map<string, number>();
  private readonly active = new Map<string, Active>();
  /** Requests aborted because they were no longer wanted (lifetime total). */
  abortedCount = 0;
  startedCount = 0;

  constructor(opts: RequestQueueOptions) {
    this.opts = { ...opts, maxActive: Math.max(1, Math.floor(opts.maxActive)) };
  }

  get activeCount(): number {
    return this.active.size;
  }

  get queuedCount(): number {
    return this.queued.size;
  }

  get idle(): boolean {
    return this.active.size === 0 && this.queued.size === 0;
  }

  isActive(id: string): boolean {
    return this.active.has(id);
  }

  isQueued(id: string): boolean {
    return this.queued.has(id);
  }

  /** Ids of the running requests. */
  activeIds(): string[] {
    return [...this.active.keys()];
  }

  /**
   * Makes `wanted` the complete set of wanted requests: new ones are queued,
   * priorities updated, unwanted queued ones dropped, unwanted running ones
   * aborted. Then starts what fits.
   */
  update(wanted: Iterable<WantedRequest>): void {
    const next = new Map<string, number>();
    for (const w of wanted) {
      const prev = next.get(w.id);
      if (prev === undefined || w.priority < prev) next.set(w.id, w.priority);
    }
    for (const [id, a] of [...this.active]) {
      const p = next.get(id);
      if (p === undefined) this.abort(id);
      else a.priority = p;
    }
    this.queued.clear();
    for (const [id, p] of next) if (!this.active.has(id)) this.queued.set(id, p);
    this.pump();
  }

  /** Adds (or re-prioritises) one request without touching the others. */
  request(id: string, priority: number): void {
    const a = this.active.get(id);
    if (a) {
      a.priority = Math.min(a.priority, priority);
      return;
    }
    const prev = this.queued.get(id);
    this.queued.set(id, prev === undefined ? priority : Math.min(prev, priority));
    this.pump();
  }

  /** Drops a queued request or aborts a running one. */
  cancel(id: string): void {
    this.queued.delete(id);
    if (this.active.has(id)) this.abort(id);
  }

  /** Aborts everything. */
  clear(): void {
    this.queued.clear();
    for (const id of [...this.active.keys()]) this.abort(id);
  }

  /** Starts queued requests in priority order while slots and the gate allow. */
  pump(): void {
    if (this.active.size >= this.opts.maxActive || this.queued.size === 0) return;
    const order = [...this.queued].sort((a, b) => a[1] - b[1]);
    for (const [id, priority] of order) {
      if (this.active.size >= this.opts.maxActive) break;
      if (this.opts.canStart && !this.opts.canStart(id)) break;
      this.queued.delete(id);
      this.start(id, priority);
    }
  }

  private start(id: string, priority: number): void {
    const controller = new AbortController();
    const entry: Active = { controller, priority };
    this.active.set(id, entry);
    this.startedCount++;
    let promise: Promise<void>;
    try {
      promise = this.opts.run(id, controller.signal);
    } catch (e) {
      promise = Promise.reject(e);
    }
    promise.then(
      () => this.settle(id, entry, 'done'),
      (e: unknown) => this.settle(id, entry, controller.signal.aborted ? 'aborted' : 'error', e),
    );
  }

  private settle(id: string, entry: Active, outcome: RequestOutcome, error?: unknown): void {
    // An aborted request was already removed (and reported) by abort().
    if (this.active.get(id) !== entry) return;
    this.active.delete(id);
    this.opts.onSettled?.(id, outcome, error);
    this.pump();
  }

  private abort(id: string): void {
    const a = this.active.get(id);
    if (!a) return;
    this.active.delete(id);
    this.abortedCount++;
    a.controller.abort();
    this.opts.onSettled?.(id, 'aborted');
  }
}
