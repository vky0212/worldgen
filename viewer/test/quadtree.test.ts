// Unit tests for quadtree tile keys (src/stream/quadtree.ts) and the 1.x tileset adapter (src/stream/tileset.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  ancestorKey,
  childKeys,
  floorShift,
  isAncestor,
  keyAt,
  keyBounds,
  keyId,
  parentKey,
  parseKeyId,
} from '../src/stream/quadtree.ts';
import { ancestorsOf, tilesetFromIndex } from '../src/stream/tileset.ts';
import type { TileEntry, TileIndex } from '../src/types.ts';

test('floor shifts round towards minus infinity', () => {
  assert.equal(floorShift(5, 1), 2);
  assert.equal(floorShift(-1, 1), -1);
  assert.equal(floorShift(-2, 1), -1);
  assert.equal(floorShift(-3, 1), -2);
  assert.equal(floorShift(-17, 2), -5);
  for (let v = -40; v <= 40; v++) {
    for (let n = 0; n < 5; n++) assert.equal(floorShift(v, n), Math.floor(v / 2 ** n), `${v} >> ${n}`);
  }
});

test('parent and children are inverse, also for negative indices', () => {
  for (const k of [
    { level: 0, i: 0, j: 0 },
    { level: 0, i: -1, j: -1 },
    { level: 0, i: -16, j: -17 },
    { level: 2, i: 3, j: -5 },
  ]) {
    const p = parentKey(k);
    assert.equal(p.level, k.level + 1);
    const kids = childKeys(p);
    assert.equal(kids.length, 4);
    assert.ok(kids.some((c) => c.i === k.i && c.j === k.j && c.level === k.level), `${keyId(k)} is a child of its parent`);
    for (const c of kids) assert.deepEqual(parentKey(c), p);
    assert.ok(isAncestor(p, k));
    assert.ok(!isAncestor(k, p));
    assert.ok(!isAncestor(k, k));
  }
  assert.deepEqual(childKeys({ level: 0, i: 1, j: 1 }), []);
});

test('ancestor keys skip levels and contain the child bounds', () => {
  const k = { level: 0, i: -16, j: -17 };
  const a = ancestorKey(k, 3);
  assert.deepEqual(a, { level: 3, i: -2, j: -3 });
  assert.ok(isAncestor(a, k));
  const kb = keyBounds(k, 256);
  const ab = keyBounds(a, 256);
  assert.deepEqual(kb, [-4096, -4352, -3840, -4096]);
  assert.ok(ab[0] <= kb[0] && ab[1] <= kb[1] && ab[2] >= kb[2] && ab[3] >= kb[3]);
  assert.throws(() => ancestorKey(a, 1));
});

test('key ids round-trip; keyAt finds the containing tile', () => {
  const k = { level: 1, i: -3, j: 7 };
  assert.equal(keyId(k), '1/-3/7');
  assert.deepEqual(parseKeyId('1/-3/7'), k);
  assert.equal(parseKeyId('x/1/2'), null);
  assert.deepEqual(keyAt(-0.5, 255.9, 256), { level: 0, i: -1, j: 0 });
  assert.deepEqual(keyAt(-0.5, 256, 256, 1), { level: 1, i: -1, j: 0 });
});

function entry(i: number, j: number, size = 256): TileEntry {
  return {
    id: `${i}_${j}`,
    i,
    j,
    bounds: [i * size, j * size, (i + 1) * size, (j + 1) * size],
    zrange: [10 + i, 20 + j],
    glb: `tiles/${i}_${j}.glb`,
    meta: `tiles/${i}_${j}.meta.json`,
  };
}

test('1.x index becomes level-0 content under a few virtual roots', () => {
  const tiles: TileEntry[] = [];
  for (let i = -16; i < 16; i++) for (let j = -17; j < 17; j++) tiles.push(entry(i, j));
  const index: TileIndex = { tile_size: 256, tiles };
  const ts = tilesetFromIndex(index);
  assert.equal(ts.tileSize, 256);
  assert.equal(ts.contentNodes.length, tiles.length);
  assert.equal(ts.skipped, 0);
  assert.ok(ts.roots.length >= 1 && ts.roots.length <= 4, `roots: ${ts.roots.length}`);
  for (const r of ts.roots) {
    assert.equal(r.content, null);
    assert.equal(r.parent, null);
  }
  // Every content node reaches a root; virtual nodes contain their children.
  const roots = new Set(ts.roots);
  for (const n of ts.contentNodes) {
    assert.equal(n.key.level, 0);
    assert.equal(n.geometricError, 0);
    assert.equal(n.children.length, 0);
    const chain = ancestorsOf(n);
    assert.ok(roots.has(chain[chain.length - 1]));
    for (const a of chain) {
      assert.equal(a.content, null);
      assert.ok(a.bounds[0] <= n.bounds[0] && a.bounds[2] >= n.bounds[2]);
      assert.ok(a.bounds[1] <= n.bounds[1] && a.bounds[3] >= n.bounds[3]);
      assert.ok(a.zmin <= n.zmin && a.zmax >= n.zmax);
    }
  }
  const t = ts.byContentId.get('-16_-17')!;
  assert.deepEqual(t.bounds, [-4096, -4352, -3840, -4096]);
  assert.equal(t.zmin, -6);
  assert.equal(t.zmax, 3);
});

test('adapter keeps the first of duplicate keys and derives keys from bounds', () => {
  const a = entry(0, 0);
  const dup = { ...entry(0, 0), id: 'dup' };
  const noIj = { ...entry(2, -1), id: 'free' } as Partial<TileEntry> as TileEntry;
  delete (noIj as Partial<TileEntry>).i;
  delete (noIj as Partial<TileEntry>).j;
  const ts = tilesetFromIndex({ tiles: [a, dup, noIj] });
  assert.equal(ts.contentNodes.length, 2);
  assert.equal(ts.skipped, 1);
  assert.deepEqual(ts.byContentId.get('free')!.key, { level: 0, i: 2, j: -1 });
  // Two tiles: already at most four roots, no virtual level needed.
  assert.equal(ts.roots.length, 2);
});
