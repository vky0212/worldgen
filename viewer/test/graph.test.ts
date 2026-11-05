// Binary road graph (core.wgt + strings.json) and the entity locator, against the compiler's
// conformance vectors.
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { decodeGraphCore } from '../src/content/graphCore.ts';
import type { GraphStrings } from '../src/content/graphCore.ts';
import { Locator, bucketOf, derivedTile, fnv1a32 } from '../src/stream/locate.ts';

const load = (name: string) => JSON.parse(readFileSync(new URL(`../../tests/conformance/${name}`, import.meta.url), 'utf8'));

test('graph core decodes to the road graph document', async () => {
  const v = load('graph_core.json');
  const buf = Buffer.from(v.core_base64, 'base64');
  const g = await decodeGraphCore(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer, v.strings as GraphStrings);
  assert.deepEqual(JSON.parse(JSON.stringify(g)), v.expected);
});

test('graph core rejects strings of another graph', async () => {
  const v = load('graph_core.json');
  const buf = Buffer.from(v.core_base64, 'base64');
  const strings = { ...v.strings, edges: v.strings.edges.slice(1) };
  await assert.rejects(decodeGraphCore(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer, strings), /edges/);
});

test('locator hash and buckets match the compiler', () => {
  const v = load('locate.json') as { cases: { id: string; fnv1a32: string; bucket_64: number }[] };
  for (const c of v.cases) {
    assert.equal(`0x${fnv1a32(c.id).toString(16).padStart(8, '0')}`, c.fnv1a32, c.id);
    assert.equal(bucketOf(c.id, 64), c.bucket_64, c.id);
  }
});

test('locator fetches each bucket once and resolves ids', async () => {
  const fetched: string[] = [];
  const loc = new Locator({ uri: 'tiles/locate/{bucket}.json', buckets: 4 }, async (path) => {
    fetched.push(path);
    const b = Number(/(\d+)\.json$/.exec(path)![1]);
    const entities: Record<string, string[]> = {};
    for (const id of ['road:osm:way/1', 'building:osm:way/2', 'bridge:osm:way/3']) if (bucketOf(id, 4) === b) entities[id] = [`${b}_0`];
    return { entities };
  });
  assert.deepEqual(await loc.tilesOf('road:osm:way/1'), [`${bucketOf('road:osm:way/1', 4)}_0`]);
  assert.deepEqual(await loc.tilesOf('road:osm:way/1'), [`${bucketOf('road:osm:way/1', 4)}_0`]);
  assert.deepEqual(await loc.tilesOf('nothing:here'), []);
  assert.equal(new Set(fetched).size, fetched.length);
  const failing = new Locator({ uri: 'x/{bucket}.json', buckets: 1 }, () => Promise.reject(new Error('404')));
  assert.deepEqual(await failing.tilesOf('a'), []);
});

test('tiles derived from ids', () => {
  assert.equal(derivedTile('terrain:-3_7', 256, 16), '-3_7');
  assert.equal(derivedTile('tree:gen:31_-1_4', 256, 16), '1_-1');
  assert.equal(derivedTile('tree:gen:-17_32_g12', 256, 16), '-2_2');
  assert.equal(derivedTile('tree:gen:31_-1_4', 250, 16), null);   // cells straddle tiles
  assert.equal(derivedTile('building:osm:way/1', 256, 16), null);
});
