// Data-first buildings (BLD/IDS): decoding and the runtime mesher, against the compiler's
// conformance vectors (invariants of its baked building meshes).
import { test } from 'node:test';
import assert from 'node:assert/strict';
import { readFileSync } from 'node:fs';
import { readWgt } from '../src/content/wgt.ts';
import { decodeBuildings } from '../src/content/buildings.ts';
import { buildingMeshes, triangulate, withLineVertices } from '../src/gen/buildingMesh.ts';
import type { MeshArrays } from '../src/gen/buildingMesh.ts';

const load = (name: string) => JSON.parse(readFileSync(new URL(`../../tests/conformance/${name}`, import.meta.url), 'utf8'));

interface Expected {
  id: string;
  material: number;
  roofs: string[];
  footprint_area: number;
  volume: number;
  up_area: number;
  down_area: number;
  zmin: number;
  zmax: number;
}

async function records() {
  const v = load('building_tiles.json');
  const buf = Buffer.from(v.wgt_base64, 'base64');
  const c = await readWgt(buf.buffer.slice(buf.byteOffset, buf.byteOffset + buf.byteLength) as ArrayBuffer);
  return { v, rec: decodeBuildings(c.sections.get('BLD')!, c.sections.get('IDS')) };
}

/** Per building row: volume, up/down projected areas and z range (ENU, from glTF axes). */
function invariants(meshes: Map<number, MeshArrays>, count: number) {
  const out = Array.from({ length: count }, () => ({ volume: 0, up: 0, down: 0, zmin: Infinity, zmax: -Infinity }));
  for (const m of meshes.values()) {
    const P = (v: number): [number, number, number] => [m.positions[3 * v], -m.positions[3 * v + 2], m.positions[3 * v + 1]];
    for (let v = 0; v < m.features.length; v++) {
      const o = out[m.features[v]];
      const z = m.positions[3 * v + 1];
      o.zmin = Math.min(o.zmin, z);
      o.zmax = Math.max(o.zmax, z);
    }
    for (let t = 0; t < m.indices.length; t += 3) {
      const ia = m.indices[t], ib = m.indices[t + 1], ic = m.indices[t + 2];
      const f = m.features[ia];
      assert.equal(m.features[ib], f);
      assert.equal(m.features[ic], f);
      const [ax, ay, az] = P(ia), [bx, by, bz] = P(ib), [cx, cy, cz] = P(ic);
      // a . (b x c) / 6
      out[f].volume += (ax * (by * cz - bz * cy) + ay * (bz * cx - bx * cz) + az * (bx * cy - by * cx)) / 6;
      const crossZ = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax);
      if (crossZ > 0) out[f].up += crossZ / 2;
      else out[f].down -= crossZ / 2;
    }
  }
  return out;
}

test('BLD decodes footprints, roofs and ids', async () => {
  const { v, rec } = await records();
  const e = v.expected as Expected[];
  assert.equal(rec.count, e.length);
  assert.deepEqual(rec.ids, e.map((x) => x.id));
  assert.deepEqual(Array.from(rec.material), e.map((x) => x.material));
  for (let k = 0; k < rec.count; k++) {
    const roofs = [];
    for (let p = rec.polyStart[k]; p < rec.polyStart[k + 1]; p++) roofs.push(v.roof_shapes[rec.shape[p]]);
    assert.deepEqual(roofs, e[k].roofs, e[k].id);
  }
  // Footprint area from the rings: exteriors counter-clockwise, holes clockwise.
  for (let k = 0; k < rec.count; k++) {
    let area = 0;
    for (let p = rec.polyStart[k]; p < rec.polyStart[k + 1]; p++) {
      for (let r = rec.ringStart[p]; r < rec.ringStart[p + 1]; r++) {
        const a = rec.vertStart[r], b = rec.vertStart[r + 1];
        let s = 0;
        for (let q = a; q < b; q++) {
          const n = q + 1 < b ? q + 1 : a;
          s += (rec.xy[2 * q] * rec.xy[2 * n + 1] - rec.xy[2 * n] * rec.xy[2 * q + 1]) / 1e6;
        }
        area += s / 2;
        assert.ok(r === rec.ringStart[p] ? s > 0 : s < 0, `${e[k].id}: ring orientation`);
      }
    }
    assert.ok(Math.abs(area - e[k].footprint_area) < 1e-6, `${e[k].id}: area ${area} vs ${e[k].footprint_area}`);
  }
});

test('building meshes enclose the baked solids', async () => {
  const { v, rec } = await records();
  const e = v.expected as Expected[];
  const x0 = v.i * v.tile_size, y0 = v.j * v.tile_size;
  const meshes = buildingMeshes(rec, x0, y0);
  const inv = invariants(meshes, rec.count);
  const close = (a: number, b: number, rel: number, abs: number) => Math.abs(a - b) <= Math.max(abs, rel * Math.abs(b));
  for (let k = 0; k < rec.count; k++) {
    const got = inv[k], want = e[k];
    assert.ok(close(got.volume, want.volume, 1e-4, 1e-3), `${want.id}: volume ${got.volume} vs ${want.volume}`);
    assert.ok(close(got.up, want.up_area, 1e-4, 1e-4), `${want.id}: roof area ${got.up} vs ${want.up_area}`);
    assert.ok(close(got.down, want.down_area, 1e-4, 1e-4), `${want.id}: floor area ${got.down} vs ${want.down_area}`);
    assert.ok(Math.abs(got.zmin - want.zmin) < 1e-3, `${want.id}: zmin ${got.zmin} vs ${want.zmin}`);
    assert.ok(Math.abs(got.zmax - want.zmax) < 1e-3, `${want.id}: zmax ${got.zmax} vs ${want.zmax}`);
  }
  // One mesh per material in use, every vertex attribute present.
  assert.deepEqual([...meshes.keys()], [...new Set(e.map((x) => x.material))].sort((a, b) => a - b));
  for (const m of meshes.values()) {
    const n = m.features.length;
    assert.equal(m.positions.length, 3 * n);
    assert.equal(m.normals.length, 3 * n);
    assert.equal(m.uvs.length, 2 * n);
    for (let q = 0; q < n; q++) {
      const l = Math.hypot(m.normals[3 * q], m.normals[3 * q + 1], m.normals[3 * q + 2]);
      assert.ok(Math.abs(l - 1) < 1e-5);
    }
  }
});

test('rows select buildings; selection keeps feature rows', async () => {
  const { v, rec } = await records();
  const meshes = buildingMeshes(rec, v.i * v.tile_size, v.j * v.tile_size, [4, 8]);
  const rows = new Set<number>();
  for (const m of meshes.values()) for (const f of m.features) rows.add(f);
  assert.deepEqual([...rows].sort((a, b) => a - b), [4, 8]);
});

test('ridge vertices and triangulation helpers', () => {
  // A 10 x 6 rectangle crossed by the line y = 3 along +x: two vertices inserted.
  const ring = Float64Array.from([0, 0, 10, 0, 10, 6, 0, 6]);
  const out = withLineVertices(ring, 5, 3, 1, 0);
  assert.deepEqual(Array.from(out), [0, 0, 10, 0, 10, 3, 10, 6, 0, 6, 0, 3]);
  // A square with a square hole: 8 triangles, counter-clockwise, total area 100 - 36.
  const outer = Float64Array.from([0, 0, 10, 0, 10, 10, 0, 10]);
  const hole = Float64Array.from([2, 2, 2, 8, 8, 8, 8, 2]);
  const { coords, tris } = triangulate([outer, hole]);
  let area = 0;
  for (let t = 0; t < tris.length; t += 3) {
    const [a, b, c] = [tris[t], tris[t + 1], tris[t + 2]];
    const cross = (coords[2 * b] - coords[2 * a]) * (coords[2 * c + 1] - coords[2 * a + 1]) - (coords[2 * b + 1] - coords[2 * a + 1]) * (coords[2 * c] - coords[2 * a]);
    assert.ok(cross > 0);
    area += cross / 2;
  }
  assert.ok(Math.abs(area - 64) < 1e-9);
});
