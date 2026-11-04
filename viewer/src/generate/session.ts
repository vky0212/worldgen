/**
 * Follows one build job: replays its events, streams new ones over SSE,
 * reconnects (resuming after the last seen `seq`) and feeds the pure reducer.
 *
 * Connection handling:
 *  - `GET /api/jobs/<id>` first (job record + events so far) unless the job
 *    was just created, then `EventSource(/api/jobs/<id>/events?after=<lastSeq>)`.
 *  - On a connection error the EventSource is closed (its built-in retry would
 *    reuse the original `after`) and reopened with the current `lastSeq`, with
 *    exponential backoff. From the second failure on, the JSON endpoint is
 *    polled first: it detects a vanished job (404) and catches up on a job that
 *    finished while disconnected.
 *  - The stream is closed for good when `end` arrives.
 *  - Events are batched per animation frame; duplicates are dropped by `seq`
 *    here and again by the reducer.
 */
import { ApiError, api, errorMessage } from '../api';
import type { Job, JobEvent, JobEventType } from '../apiTypes';
import { initialProgressState, isTerminal, reduceEvents } from './progress';
import type { ProgressState } from './progress';

/** Every event `type` of the contract (the SSE `event:` names). */
const EVENT_TYPES: readonly JobEventType[] = ['job', 'plan', 'stage', 'source', 'progress', 'tile', 'log', 'done', 'error', 'end'];

export type ConnectionStatus = 'connecting' | 'live' | 'reconnecting' | 'closed' | 'lost';

export interface SessionCallbacks {
  onState(state: ProgressState, prev: ProgressState): void;
  onJob(job: Job): void;
  onConnection(status: ConnectionStatus, detail?: string): void;
}

export class JobSession {
  readonly id: string;
  state: ProgressState = initialProgressState();
  job: Job | null;
  private readonly cb: SessionCallbacks;
  private es: EventSource | null = null;
  private queue: JobEvent[] = [];
  /** Highest seq received (queued or applied). */
  private seenSeq = 0;
  private flushHandle: { raf: number; timer: number } | null = null;
  private retryTimer = 0;
  private failures = 0;
  private disposed = false;

  constructor(id: string, cb: SessionCallbacks, job: Job | null = null) {
    this.id = id;
    this.cb = cb;
    this.job = job;
  }

  async start(): Promise<void> {
    if (!this.job) {
      const ok = await this.resync();
      if (!ok) return;
    }
    this.connect();
  }

  dispose(): void {
    this.disposed = true;
    this.closeSource();
    window.clearTimeout(this.retryTimer);
    this.cancelFlush();
    this.queue = [];
  }

  get disposedOrEnded(): boolean {
    return this.disposed || this.state.ended;
  }

  /** Re-reads the job record (e.g. for `started_utc` / `finished_utc`). */
  async refreshJob(): Promise<void> {
    try {
      const { job } = await api.job(this.id);
      if (this.disposed) return;
      this.job = job;
      this.cb.onJob(job);
    } catch {
      // Not critical: the event stream is the source of truth.
    }
  }

  // ------------------------------------------------------------ stream

  private connect(): void {
    if (this.disposed || this.state.ended) return;
    this.closeSource();
    this.cb.onConnection(this.failures ? 'reconnecting' : 'connecting');
    const es = new EventSource(api.eventsUrl(this.id, this.seenSeq));
    this.es = es;
    es.onopen = () => {
      if (es !== this.es) return;
      this.failures = 0;
      this.cb.onConnection('live');
    };
    const onMessage = (e: Event): void => {
      if (es === this.es) this.onMessage(e as MessageEvent<string>);
    };
    for (const type of EVENT_TYPES) {
      if (type !== 'error') es.addEventListener(type, onMessage);
    }
    es.onmessage = onMessage; // messages without an `event:` line
    // "error" is both the EventSource connection-error event and the job's
    // `event: error` message; only the latter carries data.
    es.addEventListener('error', (e) => {
      if (es !== this.es) return;
      if (e instanceof MessageEvent && typeof e.data === 'string' && e.data) this.onMessage(e);
      else this.onConnectionError();
    });
  }

  private onMessage(e: MessageEvent<string>): void {
    let ev: JobEvent;
    try {
      ev = JSON.parse(e.data) as JobEvent;
    } catch {
      return;
    }
    if (!ev || typeof ev !== 'object') return;
    if (!ev.type && e.type !== 'message') (ev as { type: string }).type = e.type;
    if (typeof ev.seq !== 'number') {
      const id = Number(e.lastEventId);
      if (Number.isFinite(id) && id > 0) ev.seq = id;
    }
    this.enqueue([ev]);
    if (ev.type === 'end') {
      // The server closes the stream after `end`; do not let EventSource retry.
      this.closeSource();
      this.cb.onConnection('closed');
      this.flush();
    }
  }

  private onConnectionError(): void {
    this.closeSource();
    if (this.disposed || this.state.ended || this.queueHasEnd()) return;
    this.failures++;
    this.cb.onConnection('reconnecting', this.failures > 1 ? `connection lost, retry ${this.failures}` : 'connection lost');
    const delay = Math.min(10000, 500 * 2 ** (this.failures - 1));
    window.clearTimeout(this.retryTimer);
    this.retryTimer = window.setTimeout(() => void this.reconnect(), delay);
  }

  private async reconnect(): Promise<void> {
    if (this.disposed || this.state.ended) return;
    if (this.failures >= 2) {
      const ok = await this.resync();
      if (!ok || this.disposed || this.state.ended) return;
    }
    this.connect();
  }

  /**
   * Loads the job record and its events. Returns false when following the
   * job must stop (it no longer exists); network errors return true so that
   * the caller keeps retrying.
   */
  private async resync(): Promise<boolean> {
    try {
      const { job, events } = await api.job(this.id);
      if (this.disposed) return false;
      this.job = job;
      this.cb.onJob(job);
      if (Array.isArray(events)) this.enqueue(events);
      this.flush();
      // Fallback for a finished job whose event list was not returned.
      if (isTerminal(job.state) && !this.state.ended) this.synthesizeEnd(job);
      return true;
    } catch (e) {
      if (e instanceof ApiError && e.status === 404) {
        this.dispose();
        this.cb.onConnection('lost', `Job ${this.id} no longer exists on the server (was it restarted?).`);
        return false;
      }
      if (!this.job) {
        // First load failed: retry like a dropped connection.
        this.cb.onConnection('reconnecting', errorMessage(e));
        this.failures++;
        window.clearTimeout(this.retryTimer);
        this.retryTimer = window.setTimeout(() => void this.start(), Math.min(10000, 1000 * this.failures));
        return false;
      }
      return true;
    }
  }

  private synthesizeEnd(job: Job): void {
    const extra: JobEvent[] = [];
    if (job.state === 'succeeded' && !this.state.summary) {
      extra.push({ type: 'done', seq: Number.NaN, summary: job.summary ?? {}, world_url: job.world_url ?? undefined });
    } else if (job.state === 'failed' && !this.state.error) {
      extra.push({ type: 'error', seq: Number.NaN, message: job.message ?? 'The build failed.' });
    }
    extra.push({ type: 'end', seq: Number.NaN, state: job.state as 'succeeded' | 'failed' | 'cancelled' });
    // NaN seq: applied by the reducer without moving lastSeq.
    const prev = this.state;
    this.state = reduceEvents(prev, extra);
    if (this.state !== prev) this.cb.onState(this.state, prev);
    this.cb.onConnection('closed');
  }

  private queueHasEnd(): boolean {
    return this.queue.some((e) => e.type === 'end');
  }

  private closeSource(): void {
    if (this.es) {
      this.es.close();
      this.es = null;
    }
  }

  // ------------------------------------------------------------ batching

  private enqueue(events: readonly JobEvent[]): void {
    for (const ev of events) {
      if (!ev || typeof ev !== 'object') continue;
      if (Number.isFinite(ev.seq)) {
        if (ev.seq <= this.seenSeq) continue;
        this.seenSeq = ev.seq;
      }
      this.queue.push(ev);
    }
    if (this.queue.length) this.scheduleFlush();
  }

  private scheduleFlush(): void {
    if (this.flushHandle) return;
    const run = (): void => this.flush();
    // rAF does not run in background tabs; the timer keeps the state current there.
    this.flushHandle = { raf: window.requestAnimationFrame(run), timer: window.setTimeout(run, 250) };
  }

  private cancelFlush(): void {
    if (!this.flushHandle) return;
    window.cancelAnimationFrame(this.flushHandle.raf);
    window.clearTimeout(this.flushHandle.timer);
    this.flushHandle = null;
  }

  private flush(): void {
    this.cancelFlush();
    if (this.disposed || !this.queue.length) return;
    const events = this.queue;
    this.queue = [];
    const prev = this.state;
    this.state = reduceEvents(prev, events);
    if (this.state === prev) return;
    this.cb.onState(this.state, prev);
    if (prev.phase !== this.state.phase) void this.refreshJob();
  }
}
