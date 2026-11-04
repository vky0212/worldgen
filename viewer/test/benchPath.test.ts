// Unit tests for fly-through benchmark paths and statistics (src/benchPath.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { SplinePath, autoPath, percentile, summarize, validatePath } from '../src/benchPath.ts';
import type { Vec3 } from '../src/benchPath.ts';

const close = (a: Vec3, b: Vec3, eps = 1e-6): boolean => a.every((v, k) => Math.abs(v - b[k]) < eps);

test('the spline passes through its control points at constant speed', () => {
  const pts: Vec3[] = [[0, 0, 100], [1000, 0, 100], [1000, 1000, 150], [0, 1000, 100]];
  const s = new SplinePath(pts);
  assert.equal(s.length, 1000 + Math.hypot(0, 1000, 50) + Math.hypot(1000, 0, 50));
  assert.ok(close(s.position(0), pts[0]));
  assert.ok(close(s.position(1), pts[3]));
  assert.ok(close(s.position(1000 / s.length), pts[1]));
  // Deterministic: same u, same pose.
  assert.deepEqual(s.position(0.37), s.position(0.37));
  // Roughly uniform speed: equal steps of u move similar distances on a straight-ish leg.
  const a = s.position(0.05);
  const b = s.position(0.1);
  const c = s.position(0.15);
  const d1 = Math.hypot(b[0] - a[0], b[1] - a[1]);
  const d2 = Math.hypot(c[0] - b[0], c[1] - b[1]);
  assert.ok(Math.abs(d1 - d2) / d1 < 0.25, `${d1} vs ${d2}`);
});

test('targets follow their points; without targets the camera looks ahead and down', () => {
  const pts: Vec3[] = [[0, 0, 100], [1000, 0, 100]];
  const tg: Vec3[] = [[0, 500, 0], [1000, 500, 0]];
  const s = new SplinePath(pts, tg);
  assert.ok(close(s.target(0), tg[0]));
  assert.ok(close(s.target(1), tg[1]));
  const free = new SplinePath(pts);
  const t = free.target(0.2);
  assert.ok(t[0] > free.position(0.2)[0], 'looks forward');
  assert.ok(t[2] < 100, 'looks down');
});

test('path validation', () => {
  assert.throws(() => validatePath(null));
  assert.throws(() => validatePath({ points: [[0, 0, 0]] }), /at least two/);
  assert.throws(() => validatePath({ points: [[0, 0, 0], [1, 1, 'x']] }));
  assert.throws(() => validatePath({ points: [[0, 0, 0], [1, 1, 1]], targets: [[0, 0, 0]] }), /targets/);
  assert.throws(() => validatePath({ points: [[0, 0, 0], [1, 1, 1]], duration_s: -1 }), /duration_s/);
  const ok = validatePath({ points: [[0, 0, 0], [1, 1, 1]], duration_s: 20 });
  assert.equal(ok.duration_s, 20);
});

test('percentiles and summaries', () => {
  const v = Array.from({ length: 100 }, (_, k) => k + 1);
  assert.equal(percentile(v, 50), 50);
  assert.equal(percentile(v, 95), 95);
  assert.equal(percentile(v, 100), 100);
  const s = summarize([4, 1, 3, 2, Number.NaN])!;
  assert.deepEqual(s, { n: 4, mean: 2.5, p50: 2, p95: 4, p99: 4, max: 4 });
  assert.equal(summarize([]), null);
});

test('auto path stays inside the world, above the surface, and ends high', () => {
  const world = { bounds: [-4096, -4352, 4096, 4352] as [number, number, number, number], topAt: () => 200, groundAt: () => 50 };
  const p = autoPath(world, 60, 120);
  assert.equal(p.duration_s, 60);
  assert.equal(p.points.length, p.targets!.length);
  for (const [x, y, z] of p.points.slice(0, -2)) {
    assert.ok(x > -4096 && x < 4096 && y > -4352 && y < 4352);
    assert.equal(z, 320);
  }
  const last = p.points[p.points.length - 1];
  assert.ok(last[2] > 5000, 'overview at the end');
  validatePath(p);
});
