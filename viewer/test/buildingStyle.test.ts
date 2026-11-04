// Unit tests for geo data → building style mapping (src/render/buildingStyle.ts).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import {
  LAYER,
  PROFILES,
  STYLE_TEXELS,
  classGroup,
  hash01,
  packStyles,
  parseColour,
  regionIdAt,
  styleFor,
} from '../src/render/buildingStyle.ts';
import type { EntityRecord } from '../src/types.ts';

const taiwan = PROFILES[regionIdAt(25.18, 121.54)];

function building(props: Record<string, unknown>, inferred: Record<string, number> = {}): EntityRecord {
  return {
    kind: 'building',
    confidence: 0.9,
    properties: props,
    inferred: Object.fromEntries(Object.entries(inferred).map(([k, v]) => [k, { value: v, method: 'test', confidence: 0.5 }])),
    centroid: [0, 0, 101],
  };
}

test('region ids from coordinates', () => {
  assert.equal(regionIdAt(25.18, 121.54), 'east_asia_subtropical'); // Taipei
  assert.equal(regionIdAt(35.68, 139.69), 'east_asia_temperate'); // Tokyo
  assert.equal(regionIdAt(1.35, 103.82), 'tropical'); // Singapore
  assert.equal(regionIdAt(41.9, 12.5), 'mediterranean'); // Rome
  assert.equal(regionIdAt(52.52, 13.4), 'temperate_europe'); // Berlin
  assert.equal(regionIdAt(59.33, 18.07), 'nordic'); // Stockholm
  assert.equal(regionIdAt(40.71, -74.0), 'north_america'); // New York
  assert.equal(regionIdAt(24.47, 54.37), 'arid'); // Abu Dhabi
  assert.equal(regionIdAt(-34.6, -58.4), 'generic'); // Buenos Aires
  assert.equal(regionIdAt(Number.NaN, 0), 'generic');
});

test('class groups', () => {
  assert.equal(classGroup('house', null), 'residential');
  assert.equal(classGroup('apartments', null), 'apartments');
  assert.equal(classGroup('yes', 'restaurant'), 'commercial');
  assert.equal(classGroup('yes', 'place_of_worship'), 'religious');
  assert.equal(classGroup('temple', null), 'religious');
  assert.equal(classGroup('shed', null), 'industrial');
  assert.equal(classGroup('yes', 'shelter'), 'small');
  assert.equal(classGroup('yes', null), 'unknown');
});

test('colour parsing: names, hex, junk', () => {
  assert.deepEqual(parseColour('green'), [0.24, 0.48, 0.32]);
  assert.deepEqual(parseColour('Dark Grey'), [0.32, 0.32, 0.33]);
  const hex = parseColour('#ff0000')!;
  assert.ok(hex[0] > 0.85 && hex[0] < 0.95 && hex[1] > 0.05 && hex[1] < 0.1);
  assert.deepEqual(parseColour('#fff'), parseColour('#ffffff'));
  assert.equal(parseColour('not a colour'), null);
  assert.equal(parseColour(42), null);
});

test('styles are deterministic per id', () => {
  const r = building({ class: 'house' }, { height: 6, levels: 2, ground_elevation: 100 });
  assert.deepEqual(styleFor('bld:1', r, taiwan), styleFor('bld:1', r, taiwan));
  assert.equal(hash01('bld:1'), hash01('bld:1'));
  assert.notEqual(hash01('bld:1'), hash01('bld:2'));
  assert.ok(hash01('x') >= 0 && hash01('x') < 1);
});

test('storeys follow levels and height; ground from inferred elevation', () => {
  const s = styleFor('b', building({ class: 'apartments', levels: 5, height: 16 }, { ground_elevation: 654.4 }), taiwan);
  assert.equal(s.floorHeight, 3.2);
  assert.equal(s.groundZ, 654.4);
  // Unrealistic storey heights are clamped.
  assert.equal(styleFor('b', building({ levels: 1, height: 12 }), taiwan).floorHeight, 5);
  assert.equal(styleFor('b', building({ levels: 10, height: 12 }), taiwan).floorHeight, 2.6);
  // No elevation: centroid z - 1.
  assert.equal(styleFor('b', building({ class: 'house' }), taiwan).groundZ, 100);
});

test('tagged material and colours override the regional palette', () => {
  const s = styleFor('b', building({ class: 'yes', material: 'stone', roof_colour: 'green', roof_shape: 'pyramidal' }), taiwan);
  assert.equal(s.wall, LAYER.STONE);
  assert.notEqual(s.roof, LAYER.ROOF_FLAT);
  // green (sRGB 0.24, 0.48, 0.32) in linear
  assert.ok(Math.abs(s.roofTint[1] - 0.196) < 0.01 && s.roofTint[1] > s.roofTint[0]);
});

test('commercial and shop amenities get a shopfront ground floor', () => {
  const s = styleFor('b', building({ class: 'retail', levels: 3, material: 'plaster' }), taiwan);
  assert.equal(s.ground, LAYER.SHOPFRONT);
  const r = styleFor('c', building({ class: 'yes', amenity: 'restaurant', levels: 2 }), taiwan);
  assert.equal(r.ground, LAYER.SHOPFRONT);
  const h = styleFor('d', building({ class: 'house', levels: 1 }), taiwan);
  assert.notEqual(h.ground, LAYER.SHOPFRONT);
});

test('East Asian temples get glazed pitched roofs; industrial gets metal', () => {
  const t = styleFor('t', building({ class: 'temple', roof_shape: 'hipped' }), taiwan);
  assert.equal(t.roof, LAYER.ROOF_CLAY);
  assert.ok(t.roofGloss >= 0.7);
  const i = styleFor('i', building({ class: 'warehouse', roof_shape: 'gabled' }), taiwan);
  assert.equal(i.roof, LAYER.ROOF_METAL);
});

test('tall towers are glass or concrete', () => {
  for (const id of ['a', 'b', 'c', 'd', 'e', 'f']) {
    const s = styleFor(id, building({ class: 'office', levels: 20 }), PROFILES.temperate_europe);
    assert.ok(s.wall === LAYER.CURTAIN || s.wall === LAYER.CONCRETE, `wall ${s.wall}`);
  }
});

test('regional palettes differ: Nordic houses lean to timber, Taiwan to ceramic tile', () => {
  const count = (region: (typeof PROFILES)[keyof typeof PROFILES], layer: number): number => {
    let n = 0;
    for (let i = 0; i < 400; i++) if (styleFor(`h${i}`, building({ class: 'house', levels: 2 }), region).wall === layer) n++;
    return n;
  };
  assert.ok(count(PROFILES.nordic, LAYER.TIMBER) > 150);
  assert.ok(count(taiwan, LAYER.CERAMIC) > 150);
  assert.ok(count(PROFILES.mediterranean, LAYER.PLASTER) > 250);
});

test('no metadata: falls back to the material class group', () => {
  const s = styleFor('x', undefined, taiwan, 'industrial');
  assert.equal(s.group, 'industrial');
  assert.ok(s.floorHeight >= 2.6 && s.floorHeight <= 5);
});

test('packing layout', () => {
  const a = styleFor('a', building({ class: 'house', levels: 2, height: 6 }, { ground_elevation: 10 }), taiwan);
  const b = styleFor('b', building({ class: 'office', levels: 4, height: 14 }, { ground_elevation: 20 }), taiwan);
  const data = packStyles([a, b]);
  assert.equal(data.length, 2 * STYLE_TEXELS * 4);
  const o = STYLE_TEXELS * 4;
  assert.deepEqual([...data.slice(o, o + 4)], [b.wall, b.ground, b.roof, Math.fround(b.floorHeight)]);
  assert.equal(data[o + 12], 20);
  assert.equal(data[7], Math.fround(a.seed));
});
