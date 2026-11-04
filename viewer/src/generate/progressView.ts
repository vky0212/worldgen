/**
 * DOM rendering of a build's progress (right-hand panel of the generate
 * page): overall bar, stage timeline with source chips and the export bar,
 * result card and the log console. All state comes from the reducer.
 */
import { viewerHref } from '../api';
import type { Job } from '../apiTypes';
import { appendChildren, clear, h } from '../dom';
import { MAX_LOG, exportCounts, isTerminal, stageLabel } from './progress';
import type { LogLine, Phase, ProgressState, SourceStatus, StageStatus, StageView } from './progress';
import type { ConnectionStatus } from './session';

export interface ProgressViewHandlers {
  onCancel(): void;
  onClose(): void;
}

interface StageRow {
  row: HTMLElement;
  icon: HTMLElement;
  dur: HTMLElement;
  detail: HTMLElement;
}

const PHASE_TEXT: Record<Phase, string> = {
  waiting: 'Connecting…',
  queued: 'Queued',
  running: 'Running',
  succeeded: 'Done',
  failed: 'Failed',
  cancelled: 'Cancelled',
};

const SOURCE_TEXT: Record<SourceStatus, string> = {
  pending: 'pending',
  start: 'downloading',
  ok: 'ok',
  partial: 'partial',
  missing: 'missing',
  failed: 'failed',
  disabled: 'disabled',
};

const STAGE_ICON: Record<StageStatus, string> = {
  pending: '',
  running: '',
  done: '✓',
  failed: '✕',
  skipped: '–',
  cancelled: '■',
};

export class ProgressView {
  private readonly root: HTMLElement;
  private readonly handlers: ProgressViewHandlers;

  private title!: HTMLElement;
  private subtitle!: HTMLElement;
  private phaseBadge!: HTMLElement;
  private connDot!: HTMLElement;
  private closeBtn!: HTMLButtonElement;
  private pct!: HTMLElement;
  private barFill!: HTMLElement;
  private bar!: HTMLElement;
  private elapsed!: HTMLElement;
  private current!: HTMLElement;
  private eta!: HTMLElement;
  private strip!: HTMLElement;
  private cancelBtn!: HTMLButtonElement;
  private result!: HTMLElement;
  private stageList!: HTMLElement;
  private logBox!: HTMLElement;
  private logStats!: HTMLElement;

  private stageRows = new Map<string, StageRow>();
  private stripSegs = new Map<string, HTMLElement>();
  private stageKey = '';
  private sourceChips = new Map<string, HTMLElement>();
  private sourcesKey = '';
  private sourcesRow: HTMLElement | null = null;
  private exportRow: { fill: HTMLElement; text: HTMLElement; wrap: HTMLElement } | null = null;
  private lastLogSeq = 0;
  private resultKey = '';

  private state: ProgressState | null = null;
  private job: Job | null = null;
  private jobId = '';
  private lastTWall = 0;
  private lastT = -1;
  private connection: ConnectionStatus = 'connecting';
  private connectionDetail = '';
  private tickTimer = 0;

  constructor(root: HTMLElement, handlers: ProgressViewHandlers) {
    this.root = root;
    this.handlers = handlers;
  }

  /** Starts showing a (new) job; resets everything. */
  open(jobId: string, job: Job | null): void {
    this.jobId = jobId;
    this.job = job;
    this.state = null;
    this.lastLogSeq = 0;
    this.lastT = -1;
    this.resultKey = '';
    this.stageKey = '';
    this.sourcesKey = '';
    this.connection = 'connecting';
    this.build();
    this.root.hidden = false;
    window.clearInterval(this.tickTimer);
    this.tickTimer = window.setInterval(() => this.tick(), 500);
    this.renderHeader();
  }

  close(): void {
    window.clearInterval(this.tickTimer);
    this.root.hidden = true;
    clear(this.root);
    this.state = null;
    this.job = null;
  }

  get isOpen(): boolean {
    return !this.root.hidden;
  }

  setJob(job: Job): void {
    this.job = job;
    this.renderHeader();
    this.tick();
  }

  setConnection(status: ConnectionStatus, detail = ''): void {
    this.connection = status;
    this.connectionDetail = detail;
    this.renderHeader();
    if (this.state) this.renderResult(this.state);
  }

  /** Cancel request finished (ok or not). */
  setCancelling(cancelling: boolean): void {
    this.cancelBtn.disabled = cancelling;
    this.cancelBtn.textContent = cancelling ? 'Cancelling…' : 'Cancel build';
  }

  update(state: ProgressState): void {
    const prev = this.state;
    this.state = state;
    if (state.t !== this.lastT) {
      this.lastT = state.t;
      this.lastTWall = performance.now();
    }
    this.renderStages(state);
    this.renderHeader();
    this.renderOverall(state);
    if (!prev || prev.log !== state.log) this.renderLog(state);
    this.renderResult(state);
  }

  // ------------------------------------------------------------ structure

  private build(): void {
    clear(this.root);
    this.title = h('div', { class: 'jp-title' });
    this.subtitle = h('div', { class: 'jp-subtitle muted small' });
    this.phaseBadge = h('span', { class: 'phase-badge' });
    this.connDot = h('span', { class: 'conn-dot' });
    this.closeBtn = h('button', { class: 'icon-btn', 'aria-label': 'Close' }, '×');
    this.closeBtn.addEventListener('click', () => this.handlers.onClose());

    this.pct = h('div', { class: 'jp-pct' }, '0%');
    this.elapsed = h('div', { class: 'jp-elapsed', title: 'Elapsed time' }, '0:00');
    this.barFill = h('div', { class: 'jp-bar-fill' });
    this.bar = h('div', { class: 'jp-bar', role: 'progressbar', 'aria-valuemin': '0', 'aria-valuemax': '100' }, this.barFill);
    this.current = h('div', { class: 'jp-current' });
    this.eta = h('div', { class: 'jp-eta muted small' });
    this.strip = h('div', { class: 'jp-strip', 'aria-hidden': 'true' });
    this.cancelBtn = h('button', { class: 'btn btn-danger' }, 'Cancel build');
    this.cancelBtn.addEventListener('click', () => this.handlers.onCancel());

    this.result = h('div', { class: 'jp-result' });
    this.stageList = h('ol', { class: 'stage-list' });
    this.stageRows.clear();
    this.stripSegs.clear();
    this.sourceChips.clear();
    this.sourcesRow = null;
    this.exportRow = null;

    this.logBox = h('div', { class: 'log', role: 'log', 'aria-live': 'off' });
    this.logStats = h('span', { class: 'log-stats' });
    const onlyWarnings = h('input', { type: 'checkbox' });
    onlyWarnings.addEventListener('change', () => {
      this.logBox.classList.toggle('only-warnings', onlyWarnings.checked);
      this.logBox.scrollTop = this.logBox.scrollHeight;
    });

    this.root.append(
      h(
        'header',
        { class: 'jp-header' },
        h('div', { class: 'jp-heading' }, h('div', { class: 'jp-title-row' }, this.title, this.phaseBadge, this.connDot), this.subtitle),
        this.closeBtn,
      ),
      h(
        'div',
        { class: 'jp-overall' },
        h('div', { class: 'jp-overall-top' }, this.pct, h('div', { class: 'jp-overall-meta' }, this.current, this.eta), this.elapsed),
        this.bar,
        this.strip,
        h('div', { class: 'jp-actions' }, this.cancelBtn),
      ),
      this.result,
      h('div', { class: 'jp-section-title' }, 'Stages'),
      this.stageList,
      h(
        'div',
        { class: 'jp-section-title log-head' },
        h('span', null, 'Log'),
        this.logStats,
        h('label', { class: 'check small log-filter' }, onlyWarnings, h('span', null, 'Warnings only')),
      ),
      this.logBox,
    );
  }

  // ------------------------------------------------------------ header + overall

  private renderHeader(): void {
    if (!this.title) return;
    const s = this.state;
    const name = this.job?.name ?? s?.region?.name ?? this.jobId;
    const phase: Phase = s?.phase ?? (this.job?.state as Phase | undefined) ?? 'waiting';
    this.title.textContent = `${isTerminal(phase) ? 'Build' : 'Building'} “${name}”`;
    this.phaseBadge.textContent = PHASE_TEXT[phase];
    this.phaseBadge.className = `phase-badge phase-${phase}`;
    const bits = [`job ${this.jobId}`];
    const req = this.job?.request;
    if (req?.profile) bits.push(`profile ${req.profile}`);
    if (req?.offline) bits.push('offline');
    if (req?.force) bits.push('force');
    this.subtitle.textContent = bits.join(' · ');
    const conn = this.connection;
    this.connDot.className = `conn-dot conn-${conn}`;
    const connText: Record<ConnectionStatus, string> = {
      connecting: 'Connecting to the event stream…',
      live: 'Live: receiving events',
      reconnecting: 'Reconnecting…',
      closed: 'Stream complete',
      lost: 'Lost the job',
    };
    this.connDot.title = `${connText[conn]}${this.connectionDetail ? ` (${this.connectionDetail})` : ''}`;
    const terminal = s ? isTerminal(s.phase) || s.ended : false;
    this.cancelBtn.hidden = terminal || conn === 'lost';
    this.closeBtn.title = terminal ? 'Close' : 'Close the panel (the build keeps running; reopen it from Jobs)';
  }

  private renderOverall(s: ProgressState): void {
    const pct = Math.round(s.overall * 1000) / 10;
    this.pct.textContent = `${pct >= 100 ? 100 : pct > 0 && pct < 10 ? pct.toFixed(1) : Math.floor(pct)}%`;
    this.barFill.style.width = `${Math.min(100, s.overall * 100)}%`;
    this.bar.setAttribute('aria-valuenow', String(Math.round(s.overall * 100)));
    this.bar.className = `jp-bar phase-${s.phase}`;
    const label = stageLabel(s, s.currentStage);
    this.current.textContent =
      s.phase === 'queued'
        ? 'Waiting in the queue'
        : s.phase === 'succeeded'
          ? 'Completed'
          : s.phase === 'failed'
            ? 'Failed'
            : s.phase === 'cancelled'
              ? 'Cancelled'
              : label ?? (s.hasPlan ? 'Starting…' : 'Waiting for the plan…');
    for (const st of s.stages) {
      const seg = this.stripSegs.get(st.name);
      if (seg) seg.className = `seg seg-${st.status}`;
    }
    this.tick();
  }

  /** Elapsed time and running-stage durations (called every 500 ms). */
  private tick(): void {
    const s = this.state;
    if (!s || !this.elapsed) return;
    const now = this.nowT();
    this.elapsed.textContent = s.phase === 'queued' || s.phase === 'waiting' ? '—' : clock(now);
    for (const st of s.stages) {
      const row = this.stageRows.get(st.name);
      if (row) row.dur.textContent = stageDuration(st, now);
    }
    // ETA from the overall rate; only once there is enough signal.
    if (s.phase === 'running' && s.overall > 0.05 && s.overall < 0.995 && now > 5) {
      const remaining = (now * (1 - s.overall)) / s.overall;
      this.eta.textContent = `≈ ${roughDuration(remaining)} left`;
    } else this.eta.textContent = '';
  }

  /** Seconds since the job started, as well as it can be known. */
  private nowT(): number {
    const s = this.state;
    if (!s) return 0;
    const job = this.job;
    const started = job?.started_utc ? Date.parse(job.started_utc) : NaN;
    if (isTerminal(s.phase) || s.ended) {
      if (typeof s.summary?.duration_s === 'number') return s.summary.duration_s;
      const finished = job?.finished_utc ? Date.parse(job.finished_utc) : NaN;
      if (Number.isFinite(started) && Number.isFinite(finished)) return Math.max(s.t, (finished - started) / 1000);
      return s.t;
    }
    if (Number.isFinite(started)) return Math.max(s.t, (Date.now() - started) / 1000);
    return s.t + (performance.now() - this.lastTWall) / 1000;
  }

  // ------------------------------------------------------------ stages

  private renderStages(s: ProgressState): void {
    const key = s.stages.map((st) => `${st.name}:${st.weight}:${st.label}`).join('|');
    if (key !== this.stageKey) this.buildStages(s);
    for (const st of s.stages) {
      const row = this.stageRows.get(st.name);
      if (!row) continue;
      row.row.className = `stage stage-${st.status}`;
      row.icon.className = `st-icon st-${st.status}`;
      row.icon.textContent = STAGE_ICON[st.status];
      row.icon.title = st.status;
      const p = st.progress;
      if (st.name !== 'export' && p && st.status === 'running' && (p.total > 0 || p.detail)) {
        row.detail.hidden = false;
        row.detail.textContent = [p.detail, p.total > 0 ? `${p.done}/${p.total}` : ''].filter(Boolean).join(' · ');
      } else row.detail.hidden = true;
    }
    this.renderSources(s);
    this.renderExport(s);
  }

  private buildStages(s: ProgressState): void {
    this.stageKey = s.stages.map((st) => `${st.name}:${st.weight}:${st.label}`).join('|');
    clear(this.stageList);
    clear(this.strip);
    this.stageRows.clear();
    this.stripSegs.clear();
    this.sourcesRow = null;
    this.sourcesKey = '';
    this.exportRow = null;
    const totalWeight = s.stages.reduce((a, st) => a + st.weight, 0) || 1;
    for (const st of s.stages) {
      const icon = h('span', { class: 'st-icon' });
      const dur = h('span', { class: 'st-dur' });
      const detail = h('div', { class: 'st-detail small muted' });
      detail.hidden = true;
      const li = h(
        'li',
        { class: 'stage', dataset: { stage: st.name } },
        h('div', { class: 'st-main' }, icon, h('span', { class: 'st-label', title: st.name }, st.label), dur),
        detail,
      );
      if (st.name === 'acquire') {
        this.sourcesRow = h('div', { class: 'source-chips' });
        li.append(this.sourcesRow);
      }
      if (st.name === 'export') {
        const fill = h('div', { class: 'mini-fill' });
        const text = h('span', { class: 'mini-text' });
        const wrap = h('div', { class: 'mini-bar-wrap' }, h('div', { class: 'mini-bar' }, fill), text);
        this.exportRow = { fill, text, wrap };
        li.append(wrap);
      }
      this.stageList.append(li);
      this.stageRows.set(st.name, { row: li, icon, dur, detail });
      if (st.weight > 0) {
        const seg = h('span', { class: 'seg seg-pending', title: st.label, style: `flex-grow:${st.weight / totalWeight}` });
        this.strip.append(seg);
        this.stripSegs.set(st.name, seg);
      }
    }
  }

  private renderSources(s: ProgressState): void {
    const row = this.sourcesRow;
    if (!row) return;
    const key = s.sources.map((x) => x.name).join('|');
    if (key !== this.sourcesKey) {
      this.sourcesKey = key;
      clear(row);
      this.sourceChips.clear();
      for (const src of s.sources) {
        const chip = h('span', { class: 'src-chip' }, h('span', { class: 'src-dot' }), src.name);
        row.append(chip);
        this.sourceChips.set(src.name, chip);
      }
    }
    for (const src of s.sources) {
      const chip = this.sourceChips.get(src.name);
      if (!chip) continue;
      chip.className = `src-chip src-${src.status}`;
      chip.title = `${src.name}: ${SOURCE_TEXT[src.status] ?? src.status}${src.detail ? ` — ${src.detail}` : ''}`;
    }
  }

  private renderExport(s: ProgressState): void {
    const ex = this.exportRow;
    if (!ex) return;
    const { done, total } = exportCounts(s);
    const st = s.stages.find((x) => x.name === 'export');
    ex.wrap.hidden = total === 0;
    ex.fill.style.width = total ? `${Math.min(100, (done / total) * 100)}%` : '0';
    const parts = [`${done} / ${total} tiles`];
    if (s.tileCounts.unchanged) parts.push(`${s.tileCounts.unchanged} unchanged`);
    if (s.tileCounts.building) parts.push(`${s.tileCounts.building} building`);
    ex.text.textContent = parts.join(' · ');
    ex.wrap.classList.toggle('active', st?.status === 'running');
  }

  // ------------------------------------------------------------ result card

  private renderResult(s: ProgressState): void {
    const key = `${s.phase}|${s.ended}|${s.error ?? ''}|${s.worldUrl ?? ''}|${this.connection === 'lost' ? this.connectionDetail : ''}`;
    if (key === this.resultKey) return;
    this.resultKey = key;
    clear(this.result);
    if (this.connection === 'lost') {
      this.result.append(h('div', { class: 'card card-warn' }, h('b', null, 'Lost track of this build. '), this.connectionDetail));
      return;
    }
    switch (s.phase) {
      case 'queued':
        this.result.append(h('div', { class: 'card card-info' }, 'Queued: another build is running. This one starts when it finishes.'));
        break;
      case 'succeeded':
        this.result.append(this.summaryCard(s));
        break;
      case 'failed':
        this.result.append(
          h(
            'div',
            { class: 'card card-error' },
            h('div', { class: 'card-title' }, 'The build failed'),
            h('div', { class: 'card-msg mono wrap' }, s.error ?? this.job?.message ?? 'No error message was reported.'),
            h('div', { class: 'small muted' }, 'Check the log below, adjust the settings and generate again.'),
          ),
        );
        break;
      case 'cancelled':
        this.result.append(
          h('div', { class: 'card card-muted' }, h('div', { class: 'card-title' }, 'Build cancelled'), h('div', { class: 'small muted' }, 'Adjust the settings if needed and generate again.')),
        );
        break;
      default:
        break;
    }
  }

  private summaryCard(s: ProgressState): HTMLElement {
    const sum = s.summary ?? {};
    const stats = sum.statistics ?? {};
    const val = sum.validation ?? {};
    const worldUrl = s.worldUrl ?? this.job?.world_url ?? null;
    const stat = (label: string, value: number | undefined | null, cls = ''): HTMLElement =>
      h('div', { class: `stat ${cls}`.trim() }, h('div', { class: 'stat-value' }, fmtInt(value)), h('div', { class: 'stat-label' }, label));
    const duration = typeof sum.duration_s === 'number' ? sum.duration_s : this.nowT();
    const errors = val.errors ?? 0;
    const warnings = val.warnings ?? 0;
    return h(
      'div',
      { class: 'card card-success' },
      h('div', { class: 'card-title' }, h('span', { class: 'ok-mark' }, '✓'), ` World “${this.job?.name ?? 'world'}” is ready`),
      h('div', { class: 'small muted' }, `Built in ${clock(duration)}`),
      h(
        'div',
        { class: 'stat-grid' },
        stat('roads', stats.roads),
        stat('buildings', stats.buildings),
        stat('trees', stats.trees),
        stat('bridges', stats.bridges),
        stat('tunnels', stats.tunnels),
        stat('tiles built', sum.tiles_built),
      ),
      h(
        'div',
        { class: 'summary-line' },
        h('span', { class: `sev ${errors ? 'sev-error' : 'sev-ok'}` }, `${fmtInt(errors)} errors`),
        h('span', { class: `sev ${warnings ? 'sev-warning' : 'sev-ok'}` }, `${fmtInt(warnings)} warnings`),
        h('span', { class: 'sev' }, `${fmtInt(sum.tiles_unchanged ?? 0)} tiles unchanged`),
      ),
      worldUrl
        ? h('a', { class: 'btn btn-primary btn-big', href: viewerHref(worldUrl) }, 'Open in 3D viewer →')
        : h('div', { class: 'small warn-text' }, 'The server did not report a world URL.'),
    );
  }

  // ------------------------------------------------------------ log

  private renderLog(s: ProgressState): void {
    const box = this.logBox;
    const atBottom = box.scrollHeight - box.scrollTop - box.clientHeight < 24;
    const fresh: LogLine[] = s.log.filter((l) => l.seq > this.lastLogSeq);
    for (const l of fresh) {
      box.append(
        h(
          'div',
          { class: `log-line log-${l.level}` },
          h('span', { class: 'log-t' }, `+${l.t.toFixed(1)}s`),
          h('span', { class: 'log-msg' }, l.message),
        ),
      );
      this.lastLogSeq = l.seq;
    }
    while (box.childElementCount > MAX_LOG) box.firstElementChild?.remove();
    if (!s.log.length && !box.childElementCount) box.append(h('div', { class: 'log-empty muted' }, 'No log lines yet.'));
    else box.querySelector('.log-empty')?.remove();
    if (atBottom || fresh.length === box.childElementCount) box.scrollTop = box.scrollHeight;
    const dropped = s.logTotal - s.log.length;
    clear(this.logStats);
    appendChildren(this.logStats, [
      h('span', { class: 'muted' }, `${s.logTotal} lines${dropped > 0 ? ` (last ${s.log.length} shown)` : ''}`),
      s.warnings ? h('span', { class: 'sev sev-warning' }, `${s.warnings} warnings`) : null,
      s.errors ? h('span', { class: 'sev sev-error' }, `${s.errors} errors`) : null,
    ]);
  }
}

// ---------------------------------------------------------------- formatting

export function clock(seconds: number): string {
  if (!Number.isFinite(seconds) || seconds < 0) return '—';
  const s = Math.floor(seconds);
  const hh = Math.floor(s / 3600);
  const mm = Math.floor((s % 3600) / 60);
  const ss = s % 60;
  return hh ? `${hh}:${String(mm).padStart(2, '0')}:${String(ss).padStart(2, '0')}` : `${mm}:${String(ss).padStart(2, '0')}`;
}

function stageDuration(st: StageView, now: number): string {
  switch (st.status) {
    case 'running':
      return st.startT !== null ? shortDuration(Math.max(0, now - st.startT)) : '';
    case 'done':
    case 'failed':
    case 'cancelled':
      return st.duration_s !== null ? shortDuration(st.duration_s) : '';
    case 'skipped':
      return 'skipped';
    default:
      return '';
  }
}

function shortDuration(s: number): string {
  if (s < 0.05) return '<0.1 s';
  if (s < 10) return `${s.toFixed(1)} s`;
  if (s < 60) return `${Math.round(s)} s`;
  return clock(s);
}

function roughDuration(s: number): string {
  if (s < 60) return `${Math.max(5, Math.round(s / 5) * 5)} s`;
  if (s < 3600) return `${Math.round(s / 60)} min`;
  return `${(s / 3600).toFixed(1)} h`;
}

function fmtInt(v: number | undefined | null): string {
  return typeof v === 'number' && Number.isFinite(v) ? v.toLocaleString('en-US') : '—';
}
