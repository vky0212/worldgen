// Unit tests for the lazily loaded meta.json cache (src/stream/metaCache.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { MetaCache } from '../src/stream/metaCache.ts';
import type { TileEntry, TileMeta } from '../src/types.ts';

const entry = (id: string, meta = true): TileEntry => ({ id, i: 0, j: 0, bounds: [0, 0, 1, 1], glb: `${id}.glb`, meta: meta ? `${id}.meta.json` : null });
const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

function harness(capacity: number) {
  const fetched: string[] = [];
  const loaded: string[] = [];
  const evicted: string[] = [];
  const errors: string[] = [];
  const cache = new MetaCache({
    capacity,
    maxActive: 2,
    load: async (e) => {
      fetched.push(e.id);
      await tick();
      return { entities: { [`thing:${e.id}`]: { kind: 'thing' } } } as TileMeta;
    },
    onLoaded: (id, meta, error) => {
      loaded.push(id);
      if (error) errors.push(`${id}: ${error}`);
      assert.ok(meta || error);
    },
    onEvicted: (id) => evicted.push(id),
  });
  return { cache, fetched, loaded, evicted, errors };
}

async function settle(): Promise<void> {
  for (let k = 0; k < 20; k++) await tick();
}

test('loads near tiles nearest first and keeps at most `capacity` documents', async () => {
  const h = harness(3);
  h.cache.setNear([
    { entry: entry('c'), distance: 30 },
    { entry: entry('a'), distance: 10 },
    { entry: entry('b'), distance: 20 },
  ]);
  await settle();
  assert.deepEqual(h.fetched, ['a', 'b', 'c']);
  assert.equal(h.cache.size, 3);
  // The camera moved on: d and e are near now; the oldest documents make room.
  h.cache.setNear([
    { entry: entry('d'), distance: 5 },
    { entry: entry('e'), distance: 6 },
  ]);
  await settle();
  assert.equal(h.cache.size, 3);
  assert.deepEqual(h.evicted, ['a', 'b']);
  assert.ok(h.cache.has('d') && h.cache.has('e'));
});

test('required documents jump the queue and resolve; pinned ones survive', async () => {
  const h = harness(2);
  h.cache.setPinned(['p']);
  const p = h.cache.require(entry('p'));
  h.cache.setNear([
    { entry: entry('x'), distance: 1 },
    { entry: entry('y'), distance: 2 },
    { entry: entry('z'), distance: 3 },
  ]);
  const meta = await p;
  assert.ok(meta && meta.entities['thing:p']);
  await settle();
  assert.equal(h.fetched[0], 'p');
  assert.ok(h.cache.has('p'), 'the pinned document is kept');
  // Already cached: resolves without fetching again.
  const again = await h.cache.require(entry('p'));
  assert.equal(again, meta);
  assert.equal(h.fetched.filter((id) => id === 'p').length, 1);
});

test('tiles without meta are reported once and not retried', async () => {
  const h = harness(4);
  const res = await h.cache.require(entry('none', false));
  assert.equal(res, null);
  h.cache.setNear([{ entry: entry('none', false), distance: 1 }]);
  await settle();
  assert.equal(h.errors.length, 1);
  assert.equal(h.fetched.length, 0);
});
