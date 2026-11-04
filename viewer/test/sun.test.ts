// Unit tests for the solar position helper (src/render/sun.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { dayOfYear, solarPosition, sunDirectionEnu } from '../src/render/sun.ts';

const DEG = Math.PI / 180;

test('day of year', () => {
  assert.equal(dayOfYear(new Date(Date.UTC(2026, 0, 1))), 1);
  assert.equal(dayOfYear(new Date(Date.UTC(2026, 11, 31))), 365);
});

test('equinox noon at the equator: sun near zenith', () => {
  const p = solarPosition(0, 80, 12);
  assert.ok(p.elevation > 88 * DEG);
});

test('northern summer noon at 25 N: high, to the south-ish; morning east, afternoon west', () => {
  const noon = solarPosition(25, 172, 12);
  assert.ok(noon.elevation > 80 * DEG);
  const morning = solarPosition(25, 172, 8);
  const afternoon = solarPosition(25, 172, 16);
  assert.ok(morning.azimuth > 0 && morning.azimuth < Math.PI, 'morning sun in the east');
  assert.ok(afternoon.azimuth > Math.PI, 'afternoon sun in the west');
  assert.ok(Math.abs(morning.elevation - afternoon.elevation) < 1e-6);
});

test('winter at 52 N: low noon sun due south; night below horizon', () => {
  const p = solarPosition(52, 355, 12);
  assert.ok(Math.abs(p.azimuth - Math.PI) < 1e-6);
  assert.ok(p.elevation > 10 * DEG && p.elevation < 18 * DEG);
  assert.ok(solarPosition(52, 355, 0).elevation < 0);
});

test('ENU direction is a unit vector pointing the right way', () => {
  const d = sunDirectionEnu({ azimuth: Math.PI / 2, elevation: 0 });
  assert.ok(Math.abs(d[0] - 1) < 1e-9 && Math.abs(d[1]) < 1e-9 && Math.abs(d[2]) < 1e-9);
  const u = sunDirectionEnu(solarPosition(25, 100, 10));
  assert.ok(Math.abs(Math.hypot(...u) - 1) < 1e-9);
});
