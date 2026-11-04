// Unit tests for the pure job-progress reducer (src/generate/progress.ts).
// Run with `npm test` (Node's built-in test runner + native TypeScript type stripping).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  DEFAULT_STAGES,
  MAX_LOG,
  derivedOverall,
  exportCounts,
  initialProgressState,
  reduceEvents,
  reduceProgress,
  stageLabel,
  tileRings,
} from '../src/generate/progress.ts';
import type { ProgressState } from '../src/generate/progress.ts';
import type { JobEvent, LatLonPair } from '../src/apiTypes.ts';

// ---------------------------------------------------------------- fixtures

const ring = (lat: number, lon: number): LatLonPair[] => [
  [lat, lon],
  [lat + 0.002, lon],
  [lat + 0.002, lon + 0.002],
  [lat, lon + 0.002],
];
const TILES = [
  { id: '0_0', ring: ring(25.03, 121.56) },
  { id: '0_1', ring: ring(25.03, 121.562) },
  { id: '1_0', ring: ring(25.032, 121.56) },
  { id: '1_1', ring: ring(25.032, 121.562) },
];
const SOURCES = ['osm', 'dem', 'worldcover', 'canopy_height', 'overture'];

/** Builds events with increasing seq and t, like the server does. */
function recorder() {
  let seq = 0;
  let t = 0;
  const events: JobEvent[] = [];
  const add = (e: Record<string, unknown>, dt = 0.5): void => {
    t += dt;
    events.push({ seq: ++seq, t: Math.round(t * 10) / 10, ...e } as unknown as JobEvent);
  };
  return { events, add };
}

/** Start of every build: queued → running → plan → acquire (with sources and logs). */
function prologue(add: ReturnType<typeof recorder>['add']): void {
  add({ type: 'job', state: 'queued' }, 0);
  add({ type: 'job', state: 'running' }, 0.1);
  add({
    type: 'plan',
    stages: DEFAULT_STAGES.map((s) => ({ ...s })),
    tiles: TILES,
    region: { center: { lat: 25.033, lon: 121.5654 }, radius_m: 300, bbox_wgs84: [121.56, 25.03, 121.564, 25.034] },
    sources: SOURCES,
  });
  add({ type: 'stage', name: 'acquire', state: 'start', overall: 0 });
  add({ type: 'source', name: 'osm', state: 'start', detail: 'Overpass query' });
  add({ type: 'log', level: 'info', message: 'Downloading OSM extract' });
  add({ type: 'source', name: 'osm', state: 'ok', detail: '2475 ways' }, 2);
  add({ type: 'source', name: 'dem', state: 'start' });
  add({ type: 'source', name: 'dem', state: 'partial', detail: '1 of 2 DEM tiles' }, 1);
  add({ type: 'log', level: 'warning', message: 'DEM tile N25E121 missing; filling with lower resolution data' });
  add({ type: 'source', name: 'worldcover', state: 'ok' });
  add({ type: 'source', name: 'canopy_height', state: 'missing', detail: 'no coverage' });
  add({ type: 'source', name: 'overture', state: 'disabled', detail: 'disabled by profile' });
  add({ type: 'stage', name: 'acquire', state: 'end', duration_s: 5.2, overall: 0.25 });
}

function successRun(): JobEvent[] {
  const { events, add } = recorder();
  prologue(add);
  let overall = 0.25;
  for (const s of DEFAULT_STAGES.slice(1, -2)) {
    if (s.name === 'landcover') {
      add({ type: 'stage', name: s.name, state: 'skipped', overall: (overall += s.weight) });
      continue;
    }
    add({ type: 'stage', name: s.name, state: 'start', overall });
    add({ type: 'stage', name: s.name, state: 'end', duration_s: 0.4, overall: (overall += s.weight) }, 0.4);
  }
  add({ type: 'stage', name: 'export', state: 'start', overall });
  add({ type: 'progress', stage: 'export', done: 0, total: 4, detail: 'tiles', overall });
  add({ type: 'tile', id: '0_0', state: 'building' });
  add({ type: 'tile', id: '0_0', state: 'built' });
  add({ type: 'progress', stage: 'export', done: 1, total: 4, overall: overall + 0.07 });
  add({ type: 'tile', id: '0_1', state: 'building' });
  add({ type: 'tile', id: '0_1', state: 'built' });
  add({ type: 'tile', id: '1_0', state: 'unchanged' });
  add({ type: 'tile', id: '1_1', state: 'building' });
  add({ type: 'tile', id: '1_1', state: 'built' });
  add({ type: 'progress', stage: 'export', done: 4, total: 4, overall: overall + 0.28 });
  add({ type: 'stage', name: 'export', state: 'end', duration_s: 3.1, overall: overall + 0.28 });
  add({ type: 'stage', name: 'manifest', state: 'start', overall: 0.97 });
  add({ type: 'stage', name: 'manifest', state: 'end', duration_s: 0.1, overall: 1 });
  add({
    type: 'done',
    summary: {
      statistics: { roads: 2475, buildings: 4038, trees: 55704, bridges: 13, tunnels: 56 },
      validation: { errors: 0, warnings: 140, info: 28 },
      tiles_built: 3,
      tiles_unchanged: 1,
      duration_s: 21.4,
    },
    world_url: '/worlds/taipei/',
  });
  add({ type: 'end', state: 'succeeded' });
  return events;
}

const fold = (events: readonly JobEvent[]): ProgressState => reduceEvents(initialProgressState(), events);
const stage = (s: ProgressState, name: string) => s.stages.find((x) => x.name === name)!;

// ---------------------------------------------------------------- tests

test('initial state lists the documented stages and sources', () => {
  const s = initialProgressState();
  assert.equal(s.phase, 'waiting');
  assert.equal(s.lastSeq, 0);
  assert.deepEqual(
    s.stages.map((x) => x.name),
    DEFAULT_STAGES.map((x) => x.name),
  );
  assert.ok(s.stages.every((x) => x.status === 'pending'));
  assert.deepEqual(
    s.sources.map((x) => x.name),
    SOURCES,
  );
  assert.ok(Math.abs(DEFAULT_STAGES.reduce((a, x) => a + x.weight, 0) - 1) < 1e-9);
});

test('plan → tiles pending, stages and sources from the plan', () => {
  const { events, add } = recorder();
  prologue(add);
  const afterPlan = fold(events.slice(0, 3));
  assert.equal(afterPlan.phase, 'running');
  assert.equal(afterPlan.hasPlan, true);
  assert.equal(afterPlan.planTiles.length, 4);
  assert.deepEqual(afterPlan.tileCounts, { pending: 4, building: 0, built: 0, unchanged: 0 });
  assert.equal(tileRings(afterPlan).length, 4);
  assert.equal(stageLabel(afterPlan, 'acquire'), 'Download sources');
  assert.equal(afterPlan.region?.radius_m, 300);
});

test('acquire: running stage, source states with details, log counts', () => {
  const { events, add } = recorder();
  prologue(add);
  const mid = fold(events.slice(0, 10)); // up to the DEM warning, acquire still running
  assert.equal(mid.currentStage, 'acquire');
  assert.equal(stage(mid, 'acquire').status, 'running');
  const src = Object.fromEntries(mid.sources.map((x) => [x.name, x]));
  assert.equal(src.osm.status, 'ok');
  assert.equal(src.osm.detail, '2475 ways');
  assert.equal(src.dem.status, 'partial');
  assert.equal(src.worldcover.status, 'pending');
  assert.equal(mid.warnings, 1);
  assert.equal(mid.log.length, 2);

  const end = fold(events);
  const acq = stage(end, 'acquire');
  assert.equal(acq.status, 'done');
  assert.equal(acq.duration_s, 5.2);
  assert.equal(end.currentStage, null);
  assert.equal(end.overall, 0.25);
  const src2 = Object.fromEntries(end.sources.map((x) => [x.name, x.status]));
  assert.deepEqual(src2, { osm: 'ok', dem: 'partial', worldcover: 'ok', canopy_height: 'missing', overture: 'disabled' });
});

test('a full successful build', () => {
  const events = successRun();
  const s = fold(events);
  assert.equal(s.phase, 'succeeded');
  assert.equal(s.ended, true);
  assert.equal(s.overall, 1);
  assert.equal(s.lastSeq, events.length);
  assert.equal(s.worldUrl, '/worlds/taipei/');
  assert.equal(s.summary?.statistics?.roads, 2475);
  assert.equal(s.summary?.tiles_unchanged, 1);
  assert.equal(stage(s, 'landcover').status, 'skipped');
  assert.ok(s.stages.filter((x) => x.name !== 'landcover').every((x) => x.status === 'done'));
  assert.deepEqual(s.tileCounts, { pending: 0, building: 0, built: 3, unchanged: 1 });
  assert.deepEqual(s.tiles, { '0_0': 'built', '0_1': 'built', '1_0': 'unchanged', '1_1': 'built' });
  assert.deepEqual(exportCounts(s), { done: 4, total: 4 });
  assert.equal(s.error, null);
});

test('export: building tiles, progress and the export bar mid-way', () => {
  const events = successRun();
  const firstBuilding = events.findIndex((e) => e.type === 'tile' && e.state === 'building');
  const s = fold(events.slice(0, firstBuilding + 1));
  assert.equal(s.currentStage, 'export');
  assert.equal(s.tiles['0_0'], 'building');
  assert.deepEqual(s.tileCounts, { pending: 3, building: 1, built: 0, unchanged: 0 });
  assert.deepEqual(exportCounts(s), { done: 0, total: 4 });
  assert.deepEqual(stage(s, 'export').progress, { done: 0, total: 4, detail: 'tiles' });
});

test('overall progress is monotonic and the input state is never mutated', () => {
  const events = successRun();
  let s = initialProgressState();
  let last = 0;
  for (const ev of events) {
    const before = JSON.stringify(s);
    const next = reduceProgress(s, ev);
    assert.equal(JSON.stringify(s), before, `event ${ev.seq} mutated its input`);
    assert.ok(next.overall >= last, `overall went backwards at seq ${ev.seq}`);
    last = next.overall;
    s = next;
  }
  assert.deepEqual(s, fold(events), 'one-by-one equals batched');
});

test('replays are idempotent by seq', () => {
  const events = successRun();
  const s = fold(events);
  // Full replay (e.g. GET /api/jobs/<id> after the stream): nothing changes.
  assert.equal(reduceEvents(s, events), s);
  // Overlapping replay after a reconnect: resume mid-way with some duplicates.
  const half = Math.floor(events.length / 2);
  const partial = fold(events.slice(0, half));
  const resumed = reduceEvents(partial, events.slice(half - 5));
  assert.deepEqual(resumed, s);
  // Duplicated log lines are not counted twice.
  const dup = reduceEvents(partial, [...events.slice(half - 3, half), ...events.slice(half - 3, half)]);
  assert.equal(dup, partial);
  // Out-of-order stale event after newer ones: ignored.
  assert.equal(reduceProgress(s, events[3]), s);
});

test('a failing build', () => {
  const { events, add } = recorder();
  prologue(add);
  add({ type: 'stage', name: 'normalize', state: 'start', overall: 0.25 });
  add({ type: 'stage', name: 'normalize', state: 'end', duration_s: 0.3, overall: 0.27 });
  add({ type: 'stage', name: 'terrain', state: 'start', overall: 0.27 });
  add({ type: 'log', level: 'error', message: 'Traceback (most recent call last): …' }, 2);
  add({ type: 'error', message: 'MemoryError: DEM grid too large' });
  add({ type: 'end', state: 'failed' });
  const s = fold(events);
  assert.equal(s.phase, 'failed');
  assert.equal(s.ended, true);
  assert.equal(s.error, 'MemoryError: DEM grid too large');
  assert.equal(stage(s, 'terrain').status, 'failed');
  assert.ok((stage(s, 'terrain').duration_s ?? -1) >= 0);
  assert.equal(stage(s, 'topology').status, 'pending', 'later stages stay pending');
  assert.equal(s.currentStage, null);
  assert.ok(s.overall < 1);
  assert.equal(s.errors, 2, 'log error + error event');
  assert.equal(s.log.at(-1)?.message, 'MemoryError: DEM grid too large');
  assert.equal(s.summary, null);
});

test('a cancelled build', () => {
  const events = successRun();
  const cut = events.findIndex((e) => e.type === 'tile' && e.id === '0_1' && e.state === 'building');
  const kept = events.slice(0, cut + 1);
  const lastSeq = kept.at(-1)!.seq;
  const t = (kept.at(-1)!.t ?? 0) + 1;
  const cancelled: JobEvent[] = [
    ...kept,
    { seq: lastSeq + 1, t, type: 'log', level: 'warning', message: 'Cancellation requested' },
    { seq: lastSeq + 2, t, type: 'end', state: 'cancelled' },
  ];
  const s = fold(cancelled);
  assert.equal(s.phase, 'cancelled');
  assert.ok(s.sources.every((x) => x.status !== 'start'), 'no source left "downloading"');
  assert.equal(s.ended, true);
  assert.equal(stage(s, 'export').status, 'cancelled');
  assert.equal(stage(s, 'manifest').status, 'pending');
  assert.equal(s.tiles['0_0'], 'built');
  assert.equal(s.tiles['0_1'], 'pending', 'an unfinished tile is no longer "building"');
  assert.deepEqual(s.tileCounts, { pending: 3, building: 0, built: 1, unchanged: 0 });
  assert.ok(s.overall < 1);
  // A late job-state event cannot revive an ended job.
  const late = reduceProgress(s, { seq: lastSeq + 3, t, type: 'job', state: 'running' });
  assert.equal(late.phase, 'cancelled');
});

test('the log keeps the last MAX_LOG lines', () => {
  const events: JobEvent[] = [];
  for (let i = 1; i <= MAX_LOG + 50; i++) {
    events.push({ seq: i, t: i / 10, type: 'log', level: i % 100 === 0 ? 'warning' : 'info', message: `line ${i}` });
  }
  const s = fold(events);
  assert.equal(s.log.length, MAX_LOG);
  assert.equal(s.logTotal, MAX_LOG + 50);
  assert.equal(s.log[0].seq, 51);
  assert.equal(s.warnings, 3);
});

test('overall is derived from stage weights when events carry none', () => {
  const s = fold([
    { seq: 1, t: 0, type: 'stage', name: 'acquire', state: 'start' },
    { seq: 2, t: 5, type: 'stage', name: 'acquire', state: 'end' },
    { seq: 3, t: 5, type: 'stage', name: 'export', state: 'start' },
    { seq: 4, t: 6, type: 'progress', stage: 'export', done: 1, total: 2 },
  ]);
  assert.equal(stage(s, 'acquire').duration_s, 5, 'duration from t when duration_s is absent');
  assert.ok(Math.abs(s.overall - (0.25 + 0.14)) < 1e-9);
  assert.ok(Math.abs(derivedOverall(s.stages) - s.overall) < 1e-9);
});

test('unknown stages and event types are tolerated', () => {
  const s = fold([
    { seq: 1, t: 0, type: 'stage', name: 'postprocess', state: 'start' },
    { seq: 2, t: 1, type: 'telemetry' } as unknown as JobEvent,
    { seq: 3, t: 2, type: 'source', name: 'lidar', state: 'ok' },
  ]);
  assert.equal(s.lastSeq, 3);
  assert.equal(stage(s, 'postprocess').status, 'running');
  assert.equal(stage(s, 'postprocess').label, 'Postprocess');
  assert.equal(s.sources.at(-1)?.name, 'lidar');
});
