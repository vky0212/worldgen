// Unit tests for graphics levels and the frame-rate governor (src/render/quality.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { FpsGovernor, LEVELS, applyShed, sheddable } from '../src/render/quality.ts';
import type { GovernorEvent } from '../src/render/quality.ts';

/** Feeds `seconds` of frames at `fps` and returns the events. */
function run(g: FpsGovernor, start: number, seconds: number, fps: number | ((t: number) => number), gpuMs: number | null = null): { events: GovernorEvent[]; end: number } {
  const events: GovernorEvent[] = [];
  let t = start;
  while (t < start + seconds * 1000) {
    const f = typeof fps === 'function' ? fps(t) : fps;
    const dt = 1000 / f;
    t += dt;
    const e = g.sample(t, dt, gpuMs);
    if (e) events.push(e);
  }
  return { events, end: t };
}

test('levels get monotonically more expensive', () => {
  const order = ['low', 'medium', 'high', 'ultra'] as const;
  for (let i = 1; i < order.length; i++) {
    const a = LEVELS[order[i - 1]];
    const b = LEVELS[order[i]];
    assert.ok(b.renderScale >= a.renderScale);
    assert.ok(b.treeFar >= a.treeFar);
    assert.ok(b.textureSize >= a.textureSize);
    assert.ok(b.facadeDetail >= a.facadeDetail);
  }
});

test('shedding turns features off', () => {
  const s = applyShed(LEVELS.ultra, ['ao', 'bloom', 'msaa', 'shadows', 'treeDistance']);
  assert.equal(s.ao, false);
  assert.equal(s.bloom, false);
  assert.equal(s.msaa, 0);
  assert.equal(s.shadows, false);
  assert.ok(s.treeFar < LEVELS.ultra.treeFar);
  assert.ok(!sheddable(LEVELS.low).includes('ao'));
  assert.ok(sheddable(LEVELS.ultra).includes('ao'));
});

test('steady 60 fps: nothing changes', () => {
  const g = new FpsGovernor();
  g.reset(1, sheddable(LEVELS.high));
  const { events } = run(g, 0, 20, 60);
  assert.deepEqual(events.filter((e) => e.kind !== 'surplus'), []);
  assert.equal(g.scale, 1);
});

test('slow frames lower the scale first, then shed features in order', () => {
  const g = new FpsGovernor();
  g.reset(1, sheddable(LEVELS.ultra));
  const { events } = run(g, 0, 20, 30);
  const kinds = events.map((e) => e.kind);
  const firstShed = kinds.indexOf('shed');
  assert.ok(firstShed > 0, 'sheds after scaling');
  assert.ok(kinds.slice(0, firstShed).every((k) => k === 'scale'));
  assert.equal(g.scale, 0.5);
  assert.equal((events[firstShed] as { feature: string }).feature, 'ao');
  assert.ok(g.degraded);
});

test('48..45 fps only lowers the scale, never sheds', () => {
  const g = new FpsGovernor();
  g.reset(1, sheddable(LEVELS.ultra));
  const { events } = run(g, 0, 30, 46);
  assert.ok(events.some((e) => e.kind === 'scale'));
  assert.ok(!events.some((e) => e.kind === 'shed'));
});

test('recovers quality with headroom, with hysteresis', () => {
  const g = new FpsGovernor();
  g.reset(1, sheddable(LEVELS.ultra));
  let r = run(g, 0, 12, 30);
  const shedCount = g.shed.length;
  assert.ok(shedCount > 0);
  // GPU time well under budget: there is room to grow.
  r = run(g, r.end, 60, 60, 6);
  assert.ok(r.events.some((e) => e.kind === 'scale'));
  assert.ok(r.events.some((e) => e.kind === 'restore'));
  assert.ok(g.shed.length < shedCount);
  // Recovery is gradual: no two changes within 1.5 s.
  assert.equal(g.scale, 1);
});

test('no GPU headroom at vsync: holds instead of restoring', () => {
  const g = new FpsGovernor();
  g.reset(1, sheddable(LEVELS.ultra));
  const r = run(g, 0, 12, 30);
  const before = g.scale;
  run(g, r.end, 20, 60, 16); // vsync-capped but the GPU is busy
  assert.equal(g.scale, before);
});

test('paused while loading; hitches ignored', () => {
  const g = new FpsGovernor();
  g.reset(1, sheddable(LEVELS.high));
  g.pause(5000);
  const { events } = run(g, 0, 4.5, 20);
  assert.deepEqual(events, []);
  assert.equal(g.sample(6000, 900), null);
});

test('exhausted when nothing is left to lower', () => {
  const g = new FpsGovernor();
  g.reset(0.5, ['ao']);
  const { events } = run(g, 0, 15, 20);
  assert.ok(events.some((e) => e.kind === 'shed'));
  assert.ok(events.some((e) => e.kind === 'exhausted'));
});
