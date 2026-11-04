// Unit tests for the LRU byte-budget cache (src/stream/cache.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { DEFAULT_GPU_BUDGET, LruCache, parseBudgetMb } from '../src/stream/cache.ts';

test('tracks total cost and recency', () => {
  const c = new LruCache<string>(100);
  c.set('a', 'A', 30);
  c.set('b', 'B', 30);
  c.set('c', 'C', 30);
  assert.equal(c.total, 90);
  assert.deepEqual(c.keys(), ['a', 'b', 'c']);
  c.touch('a');
  assert.deepEqual(c.keys(), ['b', 'c', 'a']);
  c.set('b', 'B2', 10);
  assert.equal(c.total, 70);
  assert.deepEqual(c.keys(), ['c', 'a', 'b']);
  c.setCost('c', 50);
  assert.equal(c.total, 90);
  assert.equal(c.delete('a'), 'A');
  assert.equal(c.total, 60);
  assert.equal(c.get('zzz'), undefined);
});

test('trim evicts least recently used first until the budget fits', () => {
  const evicted: string[] = [];
  const c = new LruCache<number>(100, (id) => evicted.push(id));
  for (const id of ['a', 'b', 'c', 'd', 'e']) c.set(id, 0, 30);
  c.touch('a');
  assert.deepEqual(c.trim(), ['b', 'c']);
  assert.deepEqual(evicted, ['b', 'c']);
  assert.equal(c.total, 90);
  assert.deepEqual(c.keys(), ['d', 'e', 'a']);
  assert.deepEqual(c.trim(), [], 'under budget: nothing to do');
});

test('pinned entries are never evicted, even over budget', () => {
  const c = new LruCache<number>(50);
  for (const id of ['a', 'b', 'c', 'd']) c.set(id, 0, 20);
  const pinned = new Set(['a', 'b', 'c']);
  assert.deepEqual(c.trim((id) => pinned.has(id)), ['d']);
  assert.equal(c.total, 60, 'still over budget: everything left is pinned');
  assert.deepEqual(c.trim((id) => pinned.has(id)), []);
  pinned.delete('a');
  assert.deepEqual(c.trim((id) => pinned.has(id)), ['a']);
  assert.equal(c.total, 40);
  assert.equal(c.evictedCount, 2);
});

test('trim honours an explicit budget (headroom for loads in flight)', () => {
  const c = new LruCache<number>(100);
  for (const id of ['a', 'b', 'c']) c.set(id, 0, 30);
  assert.deepEqual(c.trim(() => false, 40), ['a', 'b']);
});

test('budget parameter parsing', () => {
  assert.equal(parseBudgetMb('512'), 512 * 1024 * 1024);
  assert.equal(parseBudgetMb('512MB'), 512 * 1024 * 1024);
  assert.equal(parseBudgetMb('1.5gb'), 1536 * 1024 * 1024);
  assert.equal(parseBudgetMb(''), null);
  assert.equal(parseBudgetMb('lots'), null);
  assert.equal(parseBudgetMb('-5'), null);
  assert.equal(parseBudgetMb(null), null);
  assert.equal(DEFAULT_GPU_BUDGET, 1024 ** 3);
});
