// Unit tests for the prioritised fetch queue (src/stream/requestQueue.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { RequestQueue } from '../src/stream/requestQueue.ts';
import type { RequestOutcome } from '../src/stream/requestQueue.ts';

interface Pending {
  resolve: () => void;
  reject: (e: unknown) => void;
  signal: AbortSignal;
}

/** A queue whose requests finish only when the test says so. */
function harness(maxActive: number, canStart?: (id: string) => boolean) {
  const pending = new Map<string, Pending>();
  const started: string[] = [];
  const settled: [string, RequestOutcome][] = [];
  const q = new RequestQueue({
    maxActive,
    canStart,
    run: (id, signal) =>
      new Promise<void>((resolve, reject) => {
        started.push(id);
        pending.set(id, { resolve, reject, signal });
        signal.addEventListener('abort', () => reject(new DOMException('aborted', 'AbortError')));
      }),
    onSettled: (id, outcome) => settled.push([id, outcome]),
  });
  const finish = async (id: string, ok = true): Promise<void> => {
    const p = pending.get(id);
    assert.ok(p, `${id} is running`);
    pending.delete(id);
    if (ok) p.resolve();
    else p.reject(new Error('boom'));
    await new Promise((r) => setTimeout(r, 0));
  };
  return { q, pending, started, settled, finish };
}

const tick = (): Promise<void> => new Promise((r) => setTimeout(r, 0));

test('starts by priority within the concurrency limit', async () => {
  const h = harness(2);
  h.q.update([
    { id: 'c', priority: 30 },
    { id: 'a', priority: 10 },
    { id: 'b', priority: 20 },
    { id: 'd', priority: 40 },
  ]);
  assert.deepEqual(h.started, ['a', 'b']);
  assert.equal(h.q.activeCount, 2);
  assert.equal(h.q.queuedCount, 2);
  await h.finish('b');
  assert.deepEqual(h.started, ['a', 'b', 'c']);
  // A new urgent request jumps the queue.
  h.q.request('z', 1);
  await h.finish('a');
  assert.deepEqual(h.started, ['a', 'b', 'c', 'z']);
  await h.finish('c', false);
  assert.deepEqual(h.settled, [
    ['b', 'done'],
    ['a', 'done'],
    ['c', 'error'],
  ]);
  assert.deepEqual(h.started, ['a', 'b', 'c', 'z', 'd']);
});

test('requests that are no longer wanted are aborted or dropped', async () => {
  const h = harness(2);
  h.q.update([
    { id: 'a', priority: 1 },
    { id: 'b', priority: 2 },
    { id: 'c', priority: 3 },
    { id: 'd', priority: 4 },
  ]);
  const signalA = h.pending.get('a')!.signal;
  // Camera moved: only b (still running) and e are wanted now.
  h.q.update([
    { id: 'e', priority: 5 },
    { id: 'b', priority: 0 },
  ]);
  assert.ok(signalA.aborted, 'running request a was aborted');
  assert.ok(!h.pending.get('b')!.signal.aborted, 'b keeps running');
  assert.deepEqual(h.started, ['a', 'b', 'e'], 'the freed slot went to e; c and d were dropped unstarted');
  assert.equal(h.q.abortedCount, 1);
  assert.ok(!h.q.isQueued('c') && !h.q.isQueued('d'));
  await tick();
  // The aborted request's late rejection is not reported twice.
  assert.deepEqual(h.settled, [['a', 'aborted']]);
  h.q.cancel('e');
  assert.ok(h.pending.get('e')!.signal.aborted);
  h.q.clear();
  assert.ok(h.pending.get('b')!.signal.aborted);
  assert.ok(h.q.idle);
});

test('the start gate holds back requests without dropping them', async () => {
  let budget = 1;
  const h = harness(4, () => budget > 0);
  const gate = h.q;
  gate.update([
    { id: 'a', priority: 1 },
    { id: 'b', priority: 2 },
    { id: 'c', priority: 3 },
  ]);
  assert.deepEqual(h.started, ['a', 'b', 'c']);
  gate.clear();
  const h2 = harness(4, () => budget-- > 0);
  h2.q.update([
    { id: 'a', priority: 1 },
    { id: 'b', priority: 2 },
  ]);
  assert.deepEqual(h2.started, ['a']);
  assert.ok(h2.q.isQueued('b'));
  budget = 1;
  h2.q.pump();
  assert.deepEqual(h2.started, ['a', 'b']);
});
