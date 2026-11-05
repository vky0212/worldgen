// Unit tests for levels of detail in the tileset adapter (src/stream/tileset.ts) and the
// terrain mesher's skirts and coverage (src/gen/terrainMesh.ts), format 2.4.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { indexFromTileset, skirtDepths, SKIRT_MARGIN, tilesetFromIndex } from '../src/stream/tileset.ts';
import type { TileNode } from '../src/stream/tileset.ts';
import { traverse } from '../src/stream/traversal.ts';
import { terrainArrays } from '../src/gen/terrainMesh.ts';
import type { HeightGrid } from '../src/content/wgt.ts';
import type { SubtreeDoc, TilesetDoc } from '../src/types.ts';

const T = 256;

/** A 2.4 world: level-0 tiles i in [0, 4) x j in [0, 3) (row j = 2 only i < 2), levels 1 and 2. */
function world(): { ts: TilesetDoc; subs: SubtreeDoc[] } {
  const l0: [number, number][] = [];
  for (let j = 0; j < 3; j++) for (let i = 0; i < 4; i++) if (j < 2 || i < 2) l0.push([i, j]);
  const l1: [number, number][] = [[0, 0], [1, 0], [0, 1]];
  const ts: TilesetDoc = {
    schema: 'worldgen.tileset', schema_version: '2.4.0', tile_size: T,
    levels: [
      { level: 0, size: 256, spacing: 4, geometric_error: 0 },
      { level: 1, size: 512, spacing: 8, geometric_error: 20, ortho: { pixels: 64, m_per_px: 8 } },
      { level: 2, size: 1024, spacing: 16, geometric_error: 40 },
    ],
    uri: {
      content: 'tiles/{level}/{si}_{sj}/{i}_{j}.wgt', meta: 'tiles/0/{si}_{sj}/{i}_{j}.meta.json',
      ortho: 'tiles/{level}/{si}_{sj}/{i}_{j}.ortho.jpg', collision: 'tiles/0/{si}_{sj}/{i}_{j}.collision.glb',
      subtree: 'tiles/sub/{si}_{sj}.json',
    },
    ortho: { effective_source_m_per_px: 10 },
    subtrees: [{ si: 0, sj: 0 }],
  };
  const sub: SubtreeDoc = {
    schema: 'worldgen.tileset_subtree', schema_version: '2.4.0', si: 0, sj: 0,
    levels: [
      { level: 0, i: l0.map((t) => t[0]), j: l0.map((t) => t[1]), zmin: l0.map(() => 0), zmax: l0.map(() => 30),
        flags: l0.map(() => 3), counts: { roads: l0.map(() => 1) } },
      { level: 1, i: l1.map((t) => t[0]), j: l1.map((t) => t[1]), zmin: [0, 0, 0], zmax: [30, 30, 30],
        err: [12, 20, 8], terr: [1.5, 2.5, 0.5], flags: [2, 2, 2] },
      { level: 2, i: [0], j: [0], zmin: [0], zmax: [30], err: [40], terr: [6], flags: [0] },
    ],
  };
  return { ts, subs: [sub] };
}

test('every level becomes index entries; coarser ones have no meta.json and carry their errors', () => {
  const { ts, subs } = world();
  const index = indexFromTileset(ts, subs);
  assert.equal(index.tiles.length, 10 + 3 + 1);
  const l0 = index.tiles.find((t) => t.id === '1_1')!;
  assert.equal(l0.level, undefined);
  assert.equal(l0.meta, 'tiles/0/0_0/1_1.meta.json');
  assert.equal(l0.ortho, 'tiles/0/0_0/1_1.ortho.jpg');
  assert.deepEqual(l0.counts, { roads: 1 });
  const c = index.tiles.find((t) => t.id === '1/1_0')!;
  assert.equal(c.level, 1);
  assert.deepEqual(c.bounds, [512, 0, 1024, 512]);
  assert.equal(c.glb, 'tiles/1/0_0/1_0.wgt');
  assert.equal(c.ortho, 'tiles/1/0_0/1_0.ortho.jpg');
  assert.equal(c.meta, null);
  assert.equal(c.collision, null);
  assert.equal(c.geometricError, 20);
  assert.equal(c.terrainError, 2.5);
  assert.equal(c.orthoMpp, 10);                       // max(source 10 m, level 8 m)
  const top = index.tiles.find((t) => t.id === '2/0_0')!;
  assert.equal(top.ortho, null);
  assert.equal(top.orthoMpp, undefined);
});

test('levels link into one quadtree with the coarsest level as its roots', () => {
  const { ts, subs } = world();
  const set = tilesetFromIndex(indexFromTileset(ts, subs));
  assert.equal(set.maxLevel, 2);
  assert.deepEqual(set.roots.map((n) => n.id), ['2/0/0']);
  const root = set.roots[0];
  assert.equal(root.geometricError, 40);
  assert.deepEqual(root.children.map((n) => n.id).sort(), ['1/0/0', '1/0/1', '1/1/0']);
  const l1 = set.nodes.get('1/1/0')!;
  assert.equal(l1.content?.id, '1/1_0');
  assert.deepEqual(l1.children.map((n) => n.id).sort(), ['0/2/0', '0/2/1', '0/3/0', '0/3/1']);
  assert.equal(set.nodes.get('0/2/1')!.parent, l1);
  assert.equal(set.nodes.get('0/2/1')!.geometricError, 0);
  // Coverage: 1/0/1 has 2 of its 4 level-0 tiles, the root 10 of 16.
  assert.equal(set.nodes.get('1/0/1')!.partial, true);
  assert.equal(l1.partial, false);
  assert.equal(root.partial, true);
  assert.equal(set.leafAt(300, 600), true);           // tile 1_2
  assert.equal(set.leafAt(600, 600), false);          // tile 2_2 does not exist
  assert.equal(set.contentNodes.length, 14);
});

test('skirts reach below the deepest crack a neighbour of any level can leave', () => {
  const { ts, subs } = world();
  const set = tilesetFromIndex(indexFromTileset(ts, subs));
  const n = (id: string): TileNode => set.nodes.get(id)!;
  // Level-0 tile 1_0: west 0_0 (ancestors 1/0_0 terr 1.5, 2/0_0 terr 6), east 2_0 (1/1_0 2.5, 6),
  // south: none (edge of the world), north 1_1.
  assert.deepEqual(n('0/1/0').skirts, [6 + SKIRT_MARGIN, 6 + SKIRT_MARGIN, 0, 6 + SKIRT_MARGIN]);
  // Level-1 tile 1/0_0 (terr 1.5): east 1/1_0 (terr 2.5, parent 6), north 1/0_1 (0.5, 6), no west/south.
  assert.deepEqual(n('1/0/0').skirts, [0, 1.5 + 6 + SKIRT_MARGIN, 0, 1.5 + 6 + SKIRT_MARGIN]);
  // The top level has no neighbours.
  assert.deepEqual(n('2/0/0').skirts, [0, 0, 0, 0]);
  assert.deepEqual(skirtDepths(n('0/3/1'), set.nodes), [6 + SKIRT_MARGIN, 0, 6 + SKIRT_MARGIN, 0]);
});

test('single-level worlds keep their tileset unchanged: no skirts, virtual roots', () => {
  const { ts, subs } = world();
  subs[0].levels = subs[0].levels.slice(0, 1);
  const set = tilesetFromIndex(indexFromTileset({ ...ts, levels: ts.levels!.slice(0, 1) }, subs));
  assert.equal(set.maxLevel, 0);
  assert.ok(set.contentNodes.every((x) => x.skirts === null && !x.partial));
  assert.ok(set.roots.every((r) => r.content === null));
});

test('the traversal draws coarse tiles far away and level 0 close by, without holes', () => {
  const { ts, subs } = world();
  const set = tilesetFromIndex(indexFromTileset(ts, subs));
  const ready = () => 'ready' as const;
  const far = traverse(set.roots, { position: [512, 400, 20000], planes: [], sseFactor: 1000 }, ready, new Set(), { tau: 16 });
  assert.deepEqual(far.render.map((x) => x.id), ['2/0/0']);
  const near = traverse(set.roots, { position: [100, 100, 40], planes: [], sseFactor: 1000 }, ready, new Set(), { tau: 16 });
  const ids = near.render.map((x) => x.id).sort();
  assert.ok(ids.includes('0/0/0'), ids.join());
  assert.equal(near.holes, 0);
  // No drawn tile is an ancestor of another.
  for (const a of near.render) for (let p = a.parent; p; p = p.parent) assert.ok(!near.render.includes(p));
});

function grid(nx: number, ny: number, spacing: number): HeightGrid {
  const z = new Float64Array(nx * ny);
  for (let r = 0; r < ny; r++) for (let c = 0; c < nx; c++) z[r * nx + c] = 10 + 0.5 * c - 0.25 * r;
  return { nx, ny, apron: 1, spacing, quantum: 0.01, x0: -spacing, y0: -spacing, z };
}

test('skirts and coverage are added after the surface, which stays the reference mesh', () => {
  const g = grid(7, 7, 8);                    // 5 x 5 tile nodes, 4 x 4 cells, 32 m tile
  const plain = terrainArrays(g, null, 0, 0, 32);
  assert.equal(plain.surfaceVertices, 25);
  assert.equal(plain.indices.length, 6 * 16);
  const skirted = terrainArrays(g, null, 0, 0, 32, { skirts: [3, 0, 2, 0] });
  assert.deepEqual(Array.from(skirted.positions.slice(0, 75)), Array.from(plain.positions));
  assert.deepEqual(Array.from(skirted.indices.slice(0, 96)), Array.from(plain.indices));
  assert.equal(skirted.positions.length / 3, 25 + 5 + 5);                    // west + south edges
  assert.equal(skirted.indices.length, 96 + 12 * (4 + 4));
  // West skirt: the west edge's vertices (c = 0) lowered by 3 m.
  for (let r = 0; r < 5; r++) {
    const src = r * 5;
    const v = 25 + r;
    assert.equal(skirted.positions[3 * v], plain.positions[3 * src]);
    assert.equal(skirted.positions[3 * v + 1], plain.positions[3 * src + 1] - 3);
    assert.equal(skirted.positions[3 * v + 2], plain.positions[3 * src + 2]);
  }
  // Coverage: only the west half of the tile (cell centres x < 16).
  const half = terrainArrays(g, null, 0, 0, 32, { covers: (x) => x < 16, skirts: [1, 1, 1, 1] });
  assert.equal(half.surfaceVertices, 25);
  const cells = 8;
  // The east skirt runs along uncovered cells only: none. West: 4 segments; south and north: 2 each.
  assert.equal(half.indices.length, 6 * cells + 12 * (4 + 2 + 2));
  assert.equal(half.positions.length / 3, 25 + 5 + 5 + 5);
});
