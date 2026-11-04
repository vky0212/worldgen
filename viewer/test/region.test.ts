// Unit tests for the generate page's pure region helpers (src/generate/region.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  NAME_RE,
  convertRegion,
  jobBody,
  nameProblem,
  planParams,
  regionFromRequest,
  regionFromWorld,
  regionProblem,
  slugify,
  suggestName,
} from '../src/generate/region.ts';
import type { Region } from '../src/generate/region.ts';

const circle: Region = { kind: 'circle', lat: 25.033, lon: 121.5654, radius_m: 1000.4 };
const bbox: Region = { kind: 'bbox', south: 48.85, west: 2.34, north: 48.86, east: 2.36 };

test('plan query parameters', () => {
  assert.deepEqual(planParams(circle, 'basic'), { lat: 25.033, lon: 121.5654, radius: 1000, profile: 'basic' });
  assert.deepEqual(planParams(bbox), { bbox: '48.85,2.34,48.86,2.36' });
});

test('job body (circle and bbox order south,west,north,east)', () => {
  assert.deepEqual(jobBody(circle, { name: 'taipei', profile: 'basic' }), {
    center: { lat: 25.033, lon: 121.5654 },
    radius_m: 1000,
    name: 'taipei',
    profile: 'basic',
    offline: false,
    force: false,
  });
  assert.deepEqual(jobBody(bbox, { offline: true, force: true }), { bbox: [48.85, 2.34, 48.86, 2.36], offline: true, force: true });
});

test('names: suggestion, slugs and validation', () => {
  assert.equal(suggestName(circle), 'n25_033_e121_565');
  assert.equal(suggestName({ kind: 'circle', lat: -33.8688, lon: -151.2093, radius_m: 500 }), 's33_869_w151_209');
  assert.equal(suggestName(circle, 'Paris – Île de la Cité'), 'paris_ile_de_la_cite');
  assert.equal(slugify('  Tokyo / Shibuya!! '), 'tokyo_shibuya');
  assert.equal(slugify('x'.repeat(60)).length, 48);
  for (const n of [suggestName(circle), suggestName(bbox), 'taipei_101', 'a-b']) assert.ok(NAME_RE.test(n), n);
  assert.equal(nameProblem('taipei'), null);
  assert.ok(nameProblem(''));
  assert.ok(nameProblem('Taipei'));
  assert.ok(nameProblem('a'.repeat(49)));
  assert.ok(nameProblem('my world'));
});

test('region validation', () => {
  assert.equal(regionProblem(circle), null);
  assert.ok(regionProblem({ ...circle, radius_m: 50 } as Region));
  assert.ok(regionProblem({ ...circle, lat: Number.NaN } as Region));
  assert.ok(regionProblem({ kind: 'bbox', south: 1, west: 1, north: 1, east: 2 }));
});

test('regions from worlds and job requests', () => {
  assert.deepEqual(regionFromWorld({ center: { lat: 1, lon: 2 }, radius_m: 300, bbox_wgs84: [0, 0, 1, 1] }), {
    kind: 'circle',
    lat: 1,
    lon: 2,
    radius_m: 300,
  });
  // bbox_wgs84 is [west, south, east, north]
  assert.deepEqual(regionFromWorld({ center: null, radius_m: null, bbox_wgs84: [2.34, 48.85, 2.36, 48.86] }), bbox);
  assert.deepEqual(regionFromRequest({ bbox: [48.85, 2.34, 48.86, 2.36] }), bbox);
  assert.equal(regionFromRequest(null), null);
});

test('circle ↔ rectangle conversion', () => {
  const sq = convertRegion(circle, 'bbox');
  assert.equal(sq.kind, 'bbox');
  if (sq.kind !== 'bbox') return;
  assert.ok(Math.abs((sq.north - sq.south) * 111320 - 2000.8) < 1);
  const back = convertRegion(sq, 'circle');
  assert.equal(back.kind, 'circle');
  if (back.kind !== 'circle') return;
  assert.ok(Math.abs(back.radius_m - 1000) <= 2, String(back.radius_m));
  assert.ok(Math.abs(back.lat - circle.lat) < 1e-9);
});
