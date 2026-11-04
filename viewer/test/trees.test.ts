// Data-first trees (TRE/TOB) and the portable hash, against the compiler's conformance vectors.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { decodeHeights, readWgt } from '../src/content/wgt.ts';
import { decodeTrees, treeIdAt, treeInstances, treeRecordAt, treeSeed } from '../src/content/trees.ts';
import type { TreeContext } from '../src/content/trees.ts';
import { hashU64, hashUnit } from '../src/gen/hash64.ts';

const load = (name: string) => JSON.parse(readFileSync(new URL(`../../tests/conformance/${name}`, import.meta.url), 'utf8'));

test('hash64 matches the compiler', () => {
  const v = load('hash64.json') as { cases: { words: string[]; u64: string; unit: number }[] };
  for (const c of v.cases) {
    const words = c.words.map((w) => BigInt(w));
    assert.equal(`0x${hashU64(...words).toString(16).padStart(16, '0')}`, c.u64, c.words.join(','));
    assert.equal(hashUnit(...words), c.unit);
  }
});

test('tree records: ids, archetypes, transforms and seeds', async () => {
  const v = load('tree_tiles.json');
  const buf = Buffer.from(v.wgt_base64, 'base64');
  const c = await readWgt(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
  const g = decodeHeights(c.sections.get('HGT')!);
  const t = decodeTrees(c.sections.get('TRE')!, c.sections.get('TOB'));
  const ctx: TreeContext = {
    i: v.i, j: v.j, tileSize: v.tile_size, cellSize: v.cell_size, seed: v.seed,
    archetypes: ['broadleaf', 'conifer', 'palm', 'shrub', 'grass'],
    categories: ['none', 'dense_forest', 'sparse_woodland', 'shrubland', 'grassland', 'agricultural', 'urban_trees', 'wetland', 'mangrove'],
  };
  const e = v.expected;
  assert.equal(t.count, e.ids.length);
  const ids = Array.from({ length: t.count }, (_, n) => treeIdAt(t, n, ctx));
  assert.deepEqual(ids, e.ids);
  const inst = treeInstances(t, g, ctx);
  const archOf: number[] = [];
  for (const [arch, rows] of inst.byArchetype) for (const n of rows) archOf[n] = arch;
  assert.deepEqual(archOf, e.archetype_codes);
  for (let n = 0; n < t.count; n++) {
    for (let a = 0; a < 3; a++) {
      assert.ok(Math.abs(inst.trs[10 * n + a] - e.translation[3 * n + a]) < 1e-4, `translation ${n}.${a}`);
      assert.ok(Math.abs(inst.trs[10 * n + 7 + a] - e.scale[3 * n + a]) < 1e-6, `scale ${n}.${a}`);
    }
    for (let a = 0; a < 4; a++) assert.ok(Math.abs(inst.trs[10 * n + 3 + a] - e.rotation[4 * n + a]) < 1e-6, `rotation ${n}.${a}`);
    assert.equal(treeSeed(t, n, ctx), e.seeds[n]);
  }
  // Records: generated trees are synthesised, the mapped tree comes from TOB.
  const mapped = ids.indexOf('tree:osm:node/42');
  assert.equal(treeRecordAt(t, mapped, ctx, g).source, 'osm');
  const gen = treeRecordAt(t, 0, ctx, g);
  assert.equal(gen.kind, 'tree');
  assert.equal(gen.source, 'procedural');
  assert.equal((gen.properties as Record<string, unknown>).seed, e.seeds[0]);
  assert.deepEqual(Object.keys(gen.inferred ?? {}), ['height', 'crown_radius', 'trunk_radius']);
});
