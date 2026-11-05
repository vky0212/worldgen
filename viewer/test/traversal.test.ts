// Unit tests for tile selection (src/stream/traversal.ts): SSE refinement,
// hysteresis, frustum culling, prefetch and the no-holes / no-overlap guarantee.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { childKeys, keyBounds, keyId } from '../src/stream/quadtree.ts';
import type { TileKey } from '../src/stream/quadtree.ts';
import { tilesetFromIndex } from '../src/stream/tileset.ts';
import type { TileNode } from '../src/stream/tileset.ts';
import { traverse } from '../src/stream/traversal.ts';
import type { ContentStatus, Plane, TraversalOptions, ViewState } from '../src/stream/traversal.ts';
import type { TileEntry } from '../src/types.ts';

const TILE = 256;
/** Geometric error per level of the synthetic pyramid. */
const ERR = [0, 16, 32, 64];
const OPTS: Partial<TraversalOptions> = { tau: 16, hysteresis: 1.5, prefetchRadius: 0 };
/** Deep enough to reach every loaded descendant of the 4-level pyramid. */
const DEEP: Partial<TraversalOptions> = { ...OPTS, descendantFallbackDepth: 3 };

/** A full quadtree with content at every level, roots at `top`. */
function pyramid(top: number, rootKeys: [number, number][]): { roots: TileNode[]; all: TileNode[] } {
  const all: TileNode[] = [];
  const make = (key: TileKey, parent: TileNode | null): TileNode => {
    const node: TileNode = {
      key,
      id: keyId(key),
      bounds: keyBounds(key, TILE),
      zmin: 0,
      zmax: 10,
      geometricError: ERR[key.level],
      terrainError: 0,
      skirts: null,
      partial: false,
      content: { id: keyId(key), entry: {} as TileEntry },
      parent,
      children: [],
    };
    all.push(node);
    node.children = childKeys(key).map((k) => make(k, node));
    return node;
  };
  return { roots: rootKeys.map(([i, j]) => make({ level: top, i, j }, null)), all };
}

function view(x: number, y: number, z: number, planes: Plane[] = [], sseFactor = 100): ViewState {
  return { position: [x, y, z], planes, sseFactor };
}

const ids = (nodes: readonly TileNode[]): string[] => nodes.map((n) => n.id).sort();

function isAncestorNode(a: TileNode, b: TileNode): boolean {
  for (let p = b.parent; p; p = p.parent) if (p === a) return true;
  return false;
}

test('selection refines near the camera and stays coarse far away', () => {
  const { roots } = pyramid(2, [[0, 0]]);
  const all = new Set<string>();
  const res = traverse(roots, view(128, 128, 50), () => 'ready', new Set(), OPTS);
  for (const n of res.render) all.add(n.id);
  // Root (err 32, d 40) and L1 (0,0) (err 16, d 40) refine; the other L1 nodes are far enough.
  assert.deepEqual(ids(res.render), ['0/0/0', '0/0/1', '0/1/0', '0/1/1', '1/0/1', '1/1/0', '1/1/1']);
  assert.equal(res.holes, 0);
  assert.equal(res.selected, 7);
  assert.ok(res.refined.has('2/0/0') && res.refined.has('1/0/0'));
  assert.equal(res.refined.size, 2);
  // Everything selected is wanted at tier 1 and nearer tiles come first.
  assert.deepEqual(ids(res.wanted.map((w) => w.node)), ids(res.render));
  assert.ok(res.wanted.every((w) => w.tier === 1 && w.ready));
  for (let k = 1; k < res.wanted.length; k++) assert.ok(res.wanted[k - 1].priority <= res.wanted[k].priority);
});

test('hysteresis: refine above tau, collapse only below tau / 1.5', () => {
  const { roots } = pyramid(2, [[0, 0]]);
  // L1 (1,0) spans x 512..1024, y 0..512. Camera at x 512 - d over its edge: SSE = 16 * 100 / d.
  const at = (d: number): ViewState => view(512 - d, 256, 10);
  const l1 = '1/1/0';
  const refinedAt = (d: number, prev: Set<string>): boolean => traverse(roots, at(d), () => 'ready', prev, OPTS).refined.has(l1);
  // SSE 20 > 16: refines from scratch.
  assert.equal(refinedAt(80, new Set()), true);
  // SSE 13.3 (between 10.7 and 16): keeps whatever it was.
  assert.equal(refinedAt(120, new Set()), false);
  assert.equal(refinedAt(120, new Set([l1, '2/0/0'])), true);
  // SSE 8 < 10.7: collapses even if it was refined.
  assert.equal(refinedAt(200, new Set([l1, '2/0/0'])), false);
  // Walking out and back in: no flicker inside the band.
  let prev = new Set<string>();
  const states: boolean[] = [];
  for (const d of [60, 90, 120, 140, 120, 90, 120, 160, 140, 110, 95]) {
    const r = traverse(roots, at(d), () => 'ready', prev, OPTS);
    prev = r.refined;
    states.push(r.refined.has(l1));
  }
  // d < 100 → SSE > 16 refines; d > 150 → SSE < 10.7 collapses; otherwise unchanged.
  assert.deepEqual(states, [true, true, true, true, true, true, true, false, false, false, true]);
});

test('frustum planes cull, and nearby culled tiles are prefetched at low priority', () => {
  const { roots } = pyramid(2, [[0, 0]]);
  // Only x <= 500 is inside (plane: -x + 500 >= 0).
  const planes: Plane[] = [[-1, 0, 0, 500]];
  const far = traverse(roots, view(128, 128, 500, planes), () => 'ready', new Set(), OPTS);
  assert.ok(far.render.every((n) => n.bounds[0] < 500), 'nothing right of the plane is drawn');
  assert.ok(far.wanted.every((w) => w.tier !== 2));
  const near = traverse(roots, view(490, 128, 20, planes), () => 'pending', new Set(), { ...OPTS, prefetchRadius: 100 });
  const prefetched = near.wanted.filter((w) => w.tier === 2).map((w) => w.node.id);
  assert.ok(prefetched.length > 0, 'tiles just outside the frustum are prefetched');
  assert.ok(near.wanted.filter((w) => w.tier === 2).every((w) => w.node.bounds[0] >= 512 && w.distance <= 100));
  const firstPrefetch = near.wanted.findIndex((w) => w.tier === 2);
  assert.ok(near.wanted.slice(0, firstPrefetch).every((w) => w.tier < 2), 'visible tiles come first');
  assert.ok(!near.render.some((n) => prefetched.includes(n.id)));
});

test('a refined node stays drawn until all its visible children are ready', () => {
  const { roots } = pyramid(1, [[0, 0]]);
  const ready = new Set<string>(['1/0/0']);
  const status = (n: TileNode): ContentStatus => (ready.has(n.id) ? 'ready' : 'pending');
  const v = view(256, 256, 20);
  let r = traverse(roots, v, status, new Set(), OPTS);
  assert.deepEqual(ids(r.render), ['1/0/0']);
  assert.equal(r.holes, 0);
  assert.ok(r.wanted.some((w) => w.node.id === '1/0/0' && w.tier === 0), 'the stand-in parent is pinned as tier 0');
  for (const id of ['0/0/0', '0/1/0', '0/0/1']) {
    ready.add(id);
    r = traverse(roots, v, status, r.refined, OPTS);
    assert.deepEqual(ids(r.render), ['1/0/0'], `parent still drawn with ${ready.size - 1} children ready`);
  }
  ready.add('0/1/1');
  r = traverse(roots, v, status, r.refined, OPTS);
  assert.deepEqual(ids(r.render), ['0/0/0', '0/0/1', '0/1/0', '0/1/1']);
  assert.ok(!r.wanted.some((w) => w.node.id === '1/0/0'), 'the parent is no longer needed');
});

test('zooming out keeps loaded children until the parent is ready', () => {
  const { roots } = pyramid(1, [[0, 0]]);
  const ready = new Set<string>(['0/0/0', '0/0/1', '0/1/0', '0/1/1']);
  const status = (n: TileNode): ContentStatus => (ready.has(n.id) ? 'ready' : 'pending');
  const r = traverse(roots, view(256, 256, 5000), status, new Set(), OPTS);
  assert.equal(r.refined.size, 0, 'far away the parent is selected');
  assert.deepEqual(ids(r.render), ['0/0/0', '0/0/1', '0/1/0', '0/1/1']);
  assert.equal(r.holes, 0);
  assert.ok(r.wanted.some((w) => w.node.id === '1/0/0' && !w.ready));
});

/** Deterministic PRNG for the scripted load orders. */
function rng(seed: number): () => number {
  let s = seed >>> 0;
  return () => {
    s = (s + 0x6d2b79f5) >>> 0;
    let t = s;
    t = Math.imul(t ^ (t >>> 15), t | 1);
    t ^= t + Math.imul(t ^ (t >>> 7), t | 61);
    return ((t ^ (t >>> 14)) >>> 0) / 4294967296;
  };
}

/** Rendered set has no overlaps and covers every leaf that has a ready node on its chain. */
function checkCoverage(all: TileNode[], render: TileNode[], isReady: (n: TileNode) => boolean, label: string): void {
  for (const a of render) for (const b of render) assert.ok(a === b || !isAncestorNode(a, b), `${label}: ${a.id} overlaps ${b.id}`);
  const drawn = new Set(render);
  for (const leaf of all.filter((n) => n.children.length === 0)) {
    const chain: TileNode[] = [];
    for (let p: TileNode | null = leaf; p; p = p.parent) chain.push(p);
    const covering = chain.filter((n) => drawn.has(n)).length;
    assert.ok(covering <= 1, `${label}: ${leaf.id} drawn ${covering} times`);
    if (chain.some(isReady)) assert.equal(covering, 1, `${label}: hole over ${leaf.id} although ${chain.find(isReady)!.id} is ready`);
  }
}

test('no holes or overlaps for any set of loaded tiles', () => {
  const { roots, all } = pyramid(3, [[0, 0], [-1, 0]]);
  const rand = rng(7);
  const cams: ViewState[] = [view(100, 100, 30), view(-900, 1500, 200), view(0, 0, 3000), view(1800, 300, 80)];
  for (let trial = 0; trial < 300; trial++) {
    const p = rand();
    const ready = new Set(all.filter(() => rand() < p).map((n) => n.id));
    const isReady = (n: TileNode): boolean => ready.has(n.id);
    const status = (n: TileNode): ContentStatus => (isReady(n) ? 'ready' : 'pending');
    const v = cams[trial % cams.length];
    let prev = new Set<string>();
    for (let k = 0; k < 2; k++) {
      const r = traverse(roots, v, status, prev, DEEP);
      checkCoverage(all, r.render, isReady, `trial ${trial}`);
      prev = r.refined;
    }
  }
});

test('scripted load orders converge to the exact selection with no holes', () => {
  const { roots, all } = pyramid(3, [[0, 0]]);
  const target = traverse(roots, view(300, 200, 40), () => 'ready', new Set(), OPTS);
  for (let seed = 1; seed <= 40; seed++) {
    const rand = rng(seed);
    const ready = new Set<string>();
    const isReady = (n: TileNode): boolean => ready.has(n.id);
    const status = (n: TileNode): ContentStatus => (isReady(n) ? 'ready' : 'pending');
    let prev = new Set<string>();
    let steps = 0;
    let sawRootReady = false;
    for (;;) {
      const r = traverse(roots, view(300, 200, 40), status, prev, DEEP);
      prev = r.refined;
      checkCoverage(all, r.render, isReady, `seed ${seed} step ${steps}`);
      if (sawRootReady) assert.equal(r.holes, 0, `seed ${seed}: hole after the root loaded`);
      const missing = r.wanted.filter((w) => !w.ready);
      if (missing.length === 0) {
        assert.deepEqual(ids(r.render), ids(target.render));
        assert.equal(r.holes, 0);
        break;
      }
      // The loader finishes one wanted request, usually not the most urgent one.
      const pick = missing[Math.floor(rand() * missing.length)];
      ready.add(pick.node.id);
      if (pick.node.parent === null) sawRootReady = true;
      assert.ok(++steps < 200, 'converges');
    }
  }
});

test('flat 1.x tilesets: every visible tile is selected, virtual nodes are never drawn', () => {
  const tiles: TileEntry[] = [];
  for (let i = -6; i < 6; i++) {
    for (let j = -5; j < 7; j++) {
      tiles.push({ id: `${i}_${j}`, i, j, bounds: [i * 256, j * 256, (i + 1) * 256, (j + 1) * 256], zrange: [0, 30], glb: '' });
    }
  }
  const ts = tilesetFromIndex({ tile_size: 256, tiles });
  const all = traverse(ts.roots, view(0, 0, 2000), () => 'ready', new Set(), OPTS);
  assert.equal(all.render.length, tiles.length);
  assert.ok(all.render.every((n) => n.content !== null && n.key.level === 0));
  // Only the half-space x >= 0 is visible.
  const half = traverse(ts.roots, view(0, 0, 2000, [[1, 0, 0, -1]]), () => 'ready', new Set(), OPTS);
  assert.equal(half.render.length, tiles.length / 2);
  // Nothing loaded yet: every visible tile is a hole and wanted, nearest first.
  const cold = traverse(ts.roots, view(10, 10, 50), () => 'pending', new Set(), OPTS);
  assert.equal(cold.render.length, 0);
  assert.equal(cold.holes, tiles.length);
  assert.equal(cold.wanted[0].node.content!.id, '0_0');
  // Failed tiles are not holes.
  const failed = traverse(ts.roots, view(10, 10, 50), () => 'failed', new Set(), OPTS);
  assert.equal(failed.holes, 0);
});
