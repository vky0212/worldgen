// Data-first tiles: .wgt container, heightmap decoding and the runtime terrain mesh, checked
// against the compiler's conformance vectors (tests/conformance, `python -m worldgen.conformance`).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { decodeGrid, decodeHeights, readWgt } from '../src/content/wgt.ts';
import { terrainArrays } from '../src/gen/terrainMesh.ts';
import { fillUri, indexFromTileset, tilesetFromIndex } from '../src/stream/tileset.ts';
import type { SubtreeDoc, TilesetDoc } from '../src/types.ts';

interface Expected { positions: number[]; normals: number[]; uvs: number[]; colors: number[]; indices: number[] }
interface VectorTile { level?: number; i: number; j: number; size?: number; bounds: number[]; wgt_base64: string; expected: Expected }
const vectors = JSON.parse(readFileSync(new URL('../../tests/conformance/terrain_tiles.json', import.meta.url), 'utf8')) as {
  tile_size: number;
  tiles: VectorTile[];
};

function bytes(b64: string): ArrayBuffer {
  const buf = Buffer.from(b64, 'base64');
  return buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer;
}

test('container header and sections', async () => {
  const t = vectors.tiles[1];
  const c = await readWgt(bytes(t.wgt_base64));
  assert.equal(c.level, 0);
  assert.deepEqual([c.i, c.j], [t.i, t.j]);
  assert.deepEqual([...c.sections.keys()].sort(), ['HGT', 'LCC', 'RGB']);
  await assert.rejects(readWgt(new ArrayBuffer(30)), /not a WorldGen tile container/);
});

test('runtime terrain mesh equals the baked terrain of the conformance tiles', async () => {
  for (const t of vectors.tiles) {
    const c = await readWgt(bytes(t.wgt_base64));
    const g = decodeHeights(c.sections.get('HGT')!);
    assert.equal(g.apron, 1);
    const rgb = decodeGrid(c.sections.get('RGB')!, 3);
    assert.equal(c.level, t.level ?? 0);
    const size = t.size ?? vectors.tile_size;
    const m = terrainArrays(g, rgb.data, t.i * size, t.j * size, size);
    const e = t.expected;
    assert.deepEqual(Array.from(m.positions), e.positions, `positions of tile ${t.i}_${t.j}`);
    assert.deepEqual(Array.from(m.indices), e.indices, `indices of tile ${t.i}_${t.j}`);
    assert.deepEqual(Array.from(m.uvs), e.uvs, `uvs of tile ${t.i}_${t.j}`);
    assert.deepEqual(Array.from(m.colors!), e.colors, `colours of tile ${t.i}_${t.j}`);
    let worst = 0;
    for (let k = 0; k < e.normals.length; k++) worst = Math.max(worst, Math.abs(m.normals[k] - e.normals[k]));
    assert.ok(worst < 1e-6, `normals of tile ${t.i}_${t.j} differ by ${worst}`);
  }
});

test('tileset and subtrees become index entries with .wgt content', () => {
  const ts: TilesetDoc = {
    schema: 'worldgen.tileset', schema_version: '2.0.0', tile_size: 256,
    uri: {
      content: 'tiles/{level}/{si}_{sj}/{i}_{j}.wgt', meta: 'tiles/0/{si}_{sj}/{i}_{j}.meta.json',
      ortho: 'tiles/0/{si}_{sj}/{i}_{j}.ortho.jpg', collision: 'tiles/0/{si}_{sj}/{i}_{j}.collision.glb',
      subtree: 'tiles/sub/{si}_{sj}.json',
    },
    subtrees: [{ si: -1, sj: 0 }],
  };
  const sub: SubtreeDoc = {
    si: -1, sj: 0,
    levels: [{ level: 0, i: [-1, -16], j: [0, 15], zmin: [1, null], zmax: [9, null], hash: ['ab', 'cd'], flags: [3, 5],
      counts: { roads: [4, 0] } }],
  };
  assert.equal(fillUri(ts.uri.subtree, { si: -1, sj: 0 }), 'tiles/sub/-1_0.json');
  const index = indexFromTileset(ts, [sub]);
  assert.equal(index.tiles.length, 2);
  const [a, b] = index.tiles;
  assert.deepEqual(a, {
    id: '-1_0', i: -1, j: 0, bounds: [-256, 0, 0, 256], zrange: [1, 9], glb: 'tiles/0/-1_0/-1_0.wgt',
    meta: 'tiles/0/-1_0/-1_0.meta.json', ortho: 'tiles/0/-1_0/-1_0.ortho.jpg', collision: null, hash: 'ab',
    counts: { roads: 4 }, format: 'wgt',
  });
  assert.equal(b.zrange, undefined);
  assert.equal(b.ortho, null);
  assert.equal(b.collision, 'tiles/0/-1_0/-16_15.collision.glb');
  const tileset = tilesetFromIndex(index);
  assert.equal(tileset.contentNodes.length, 2);
  assert.equal(tileset.byContentId.get('-16_15')?.content?.entry.format, 'wgt');
});
