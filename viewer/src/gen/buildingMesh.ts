/**
 * Building meshes from `BLD` footprint records: walls, roofs and floors, one mesh per material.
 *
 * Port of the compiler's `building_mesh` (src/worldgen/buildings/buildings.py): the same solid
 * (footprint, floor, eaves, ridge, top) with the same attribute conventions as a baked tile's
 * `buildings` layer — glTF axes relative to the tile corner, wall uv (run / 4, z / 4), cap and
 * roof uv (world x / 4, world y / 4) with v flipped as glTF stores it, `_feature_id_0` = the
 * building's row. Triangulation differs (earcut instead of constrained Delaunay), so vertex
 * lists differ while volume, areas and heights match (tests/conformance/building_tiles.json).
 * Pure: typed arrays in, typed arrays out.
 */
import { Earcut } from 'three/src/extras/Earcut.js';
import type { BuildingRecords } from '../content/buildings.ts';
import { ROOF_GABLED, ROOF_PYRAMIDAL, ROOF_SKILLION } from '../content/buildings.ts';

export interface MeshArrays {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  /** Building row per vertex (index into the tile's building ids). */
  features: Float32Array;
  indices: Uint32Array;
}

/** Texture scale of walls, caps and roofs (metres per uv unit), as in the compiler. */
const UV = 4;
/** Distance below which a vertex counts as on the ridge line (metres). */
const ON_LINE = 1e-7;

type Ring = Float64Array; // x0, y0, x1, y1, ... metres from the tile corner (ENU), not closed

class Builder {
  pos: number[] = [];
  nrm: number[] = [];
  uv: number[] = [];
  feat: number[] = [];
  idx: number[] = [];
  readonly x0: number;
  readonly y0: number;
  constructor(x0: number, y0: number) {
    this.x0 = x0;
    this.y0 = y0;
  }

  /** ENU vertex -> glTF axes (x, z, -y); uv with v flipped. Returns its index. */
  vertex(x: number, y: number, z: number, nx: number, ny: number, nz: number, u: number, v: number, f: number): number {
    this.pos.push(x, z, -y);
    this.nrm.push(nx, nz, -ny);
    this.uv.push(u, 1 - v);
    this.feat.push(f);
    return this.feat.length - 1;
  }

  arrays(): MeshArrays {
    return {
      positions: new Float32Array(this.pos), normals: new Float32Array(this.nrm), uvs: new Float32Array(this.uv),
      features: new Float32Array(this.feat), indices: new Uint32Array(this.idx),
    };
  }

  /** Vertical quads along a closed ring, bottom `zb`, top `zt[i]` per ring vertex. */
  walls(ring: Ring, zb: number, zt: ArrayLike<number>, f: number): void {
    const n = ring.length / 2;
    let s = 0;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      const ax = ring[2 * i], ay = ring[2 * i + 1], bx = ring[2 * j], by = ring[2 * j + 1];
      const dx = bx - ax, dy = by - ay;
      const len = Math.hypot(dx, dy);
      if (len < 1e-6) continue;
      const nx = dy / len, ny = -dx / len;
      const k = this.vertex(ax, ay, zb, nx, ny, 0, s / UV, zb / UV, f);
      this.vertex(bx, by, zb, nx, ny, 0, (s + len) / UV, zb / UV, f);
      this.vertex(bx, by, zt[j], nx, ny, 0, (s + len) / UV, zt[j] / UV, f);
      this.vertex(ax, ay, zt[i], nx, ny, 0, s / UV, zt[i] / UV, f);
      this.idx.push(k, k + 1, k + 2, k, k + 2, k + 3);
      s += len;
    }
  }

  /** Horizontal cap at `z` (facing up, or down for floors) over rings (exterior first). */
  cap(rings: Ring[], z: number, down: boolean, f: number): void {
    const { coords, tris } = triangulate(rings);
    const base = this.feat.length;
    const nz = down ? -1 : 1;
    for (let v = 0; v < coords.length / 2; v++) {
      const x = coords[2 * v], y = coords[2 * v + 1];
      this.vertex(x, y, z, 0, 0, nz, (this.x0 + x) / UV, (this.y0 + y) / UV, f);
    }
    for (let t = 0; t < tris.length; t += 3) {
      const a = tris[t], b = tris[t + 1], c = tris[t + 2];
      if (down) this.idx.push(base + a, base + c, base + b);
      else this.idx.push(base + a, base + b, base + c);
    }
  }

  /** A planar triangle (ENU, counter-clockwise from above) with its own flat normal. */
  face(p: number[], withUv: boolean, f: number): void {
    const [ax, ay, az, bx, by, bz, cx, cy, cz] = p;
    const ux = bx - ax, uy = by - ay, uz = bz - az, vx = cx - ax, vy = cy - ay, vz = cz - az;
    let nx = uy * vz - uz * vy, ny = uz * vx - ux * vz, nz = ux * vy - uy * vx;
    const l = Math.hypot(nx, ny, nz);
    if (l < 1e-12) return;
    nx /= l; ny /= l; nz /= l;
    for (let k = 0; k < 9; k += 3) {
      const x = p[k], y = p[k + 1];
      this.idx.push(this.vertex(x, y, p[k + 2], nx, ny, nz, withUv ? (this.x0 + x) / UV : 0, withUv ? (this.y0 + y) / UV : 0, f));
    }
  }
}

/**
 * Earcut triangulation of rings (exterior first, then holes) with counter-clockwise triangles,
 * degenerate ones dropped (as the compiler's `triangulate_polygon`).
 */
export function triangulate(rings: Ring[]): { coords: Float64Array; tris: number[] } {
  let total = 0;
  for (const r of rings) total += r.length;
  const coords = new Float64Array(total);
  const holes: number[] = [];
  let o = 0;
  rings.forEach((r, k) => {
    if (k > 0) holes.push(o / 2);
    coords.set(r, o);
    o += r.length;
  });
  const raw = Earcut.triangulate(Array.from(coords), holes, 2);
  const tris: number[] = [];
  for (let t = 0; t < raw.length; t += 3) {
    const a = raw[t], b = raw[t + 1], c = raw[t + 2];
    const cross = (coords[2 * b] - coords[2 * a]) * (coords[2 * c + 1] - coords[2 * a + 1])
      - (coords[2 * b + 1] - coords[2 * a + 1]) * (coords[2 * c] - coords[2 * a]);
    if (Math.abs(cross) <= 1e-10) continue;
    if (cross > 0) tris.push(a, b, c);
    else tris.push(a, c, b);
  }
  return { coords, tris };
}

/** The ring with a vertex inserted wherever it crosses the line through (px, py) along (ux, uy). */
export function withLineVertices(ring: Ring, px: number, py: number, ux: number, uy: number): Ring {
  const n = ring.length / 2;
  const side = new Float64Array(n);
  for (let i = 0; i < n; i++) side[i] = (ring[2 * i] - px) * -uy + (ring[2 * i + 1] - py) * ux;
  const out: number[] = [];
  for (let i = 0; i < n; i++) {
    const j = (i + 1) % n;
    out.push(ring[2 * i], ring[2 * i + 1]);
    if (side[i] * side[j] < 0) {
      const t = side[i] / (side[i] - side[j]);
      out.push(ring[2 * i] + (ring[2 * j] - ring[2 * i]) * t, ring[2 * i + 1] + (ring[2 * j + 1] - ring[2 * i + 1]) * t);
    }
  }
  return Float64Array.from(out);
}

/** Parts of triangle `tri` (x, y pairs) on each side of the line d(x, y) = 0, as triangles. */
function splitAtLine(tri: number[], d: (x: number, y: number) => number): number[][] {
  const s = [d(tri[0], tri[1]), d(tri[2], tri[3]), d(tri[4], tri[5])].map((v) => (Math.abs(v) < ON_LINE ? 0 : v));
  if (s.every((v) => v >= 0) || s.every((v) => v <= 0)) return [tri];
  const out: number[][] = [];
  for (const sign of [1, -1]) {
    // Sutherland–Hodgman against the half plane sign * d >= 0.
    const poly: number[] = [];
    for (let i = 0; i < 3; i++) {
      const j = (i + 1) % 3;
      const si = sign * s[i], sj = sign * s[j];
      if (si >= 0) poly.push(tri[2 * i], tri[2 * i + 1]);
      if ((si > 0 && sj < 0) || (si < 0 && sj > 0)) {
        const t = si / (si - sj);
        poly.push(tri[2 * i] + (tri[2 * j] - tri[2 * i]) * t, tri[2 * i + 1] + (tri[2 * j + 1] - tri[2 * i + 1]) * t);
      }
    }
    for (let k = 2; k < poly.length / 2; k++) out.push([poly[0], poly[1], poly[2 * k - 2], poly[2 * k - 1], poly[2 * k], poly[2 * k + 1]]);
  }
  return out;
}

/** Rings of polygon `p` in metres from the tile corner. */
function ringsOf(rec: BuildingRecords, p: number): Ring[] {
  const out: Ring[] = [];
  for (let r = rec.ringStart[p]; r < rec.ringStart[p + 1]; r++) {
    const a = rec.vertStart[r], b = rec.vertStart[r + 1];
    const ring = new Float64Array(2 * (b - a));
    for (let k = 0; k < ring.length; k++) ring[k] = rec.xy[2 * a + k] / 1000;
    out.push(ring);
  }
  return out;
}

function polygonMesh(out: Builder, rec: BuildingRecords, p: number, bottom: number, top: number, f: number): void {
  const rings = ringsOf(rec, p);
  const shape = rec.shape[p];
  const eave = rec.eave[p];
  if (shape === ROOF_PYRAMIDAL) {
    const ring = rings[0];
    out.walls(ring, bottom, new Float64Array(ring.length / 2).fill(eave), f);
    const ax = rec.cx[p], ay = rec.cy[p];
    const n = ring.length / 2;
    for (let i = 0; i < n; i++) {
      const j = (i + 1) % n;
      out.face([ring[2 * i], ring[2 * i + 1], eave, ring[2 * j], ring[2 * j + 1], eave, ax, ay, top], false, f);
    }
    out.cap([ring], bottom, true, f);
    return;
  }
  if (shape === ROOF_GABLED || shape === ROOF_SKILLION) {
    const px = rec.cx[p], py = rec.cy[p], ux = rec.ux[p], uy = rec.uy[p], half = rec.half[p];
    const h = top - eave;
    const across = (x: number, y: number) => (x - px) * -uy + (y - py) * ux;
    const roofZ = shape === ROOF_SKILLION
      ? (x: number, y: number) => eave + h * Math.min(Math.max((across(x, y) + half) / (2 * half), 0), 1)
      : (x: number, y: number) => eave + h * Math.min(Math.max(1 - Math.abs(across(x, y)) / half, 0), 1);
    const ring = shape === ROOF_GABLED ? withLineVertices(rings[0], px, py, ux, uy) : rings[0];
    const zt = new Float64Array(ring.length / 2);
    for (let i = 0; i < zt.length; i++) zt[i] = roofZ(ring[2 * i], ring[2 * i + 1]);
    out.walls(ring, bottom, zt, f);
    const { coords, tris } = triangulate([ring]);
    for (let t = 0; t < tris.length; t += 3) {
      const tri = [0, 1, 2].flatMap((k) => [coords[2 * tris[t + k]], coords[2 * tris[t + k] + 1]]);
      const parts = shape === ROOF_GABLED ? splitAtLine(tri, across) : [tri];
      for (const q of parts) {
        out.face([q[0], q[1], roofZ(q[0], q[1]), q[2], q[3], roofZ(q[2], q[3]), q[4], q[5], roofZ(q[4], q[5])], true, f);
      }
    }
    out.cap([ring], bottom, true, f);
    return;
  }
  // Flat: a prism with walls around every ring, top and bottom caps.
  for (const ring of rings) out.walls(ring, bottom, new Float64Array(ring.length / 2).fill(top), f);
  out.cap(rings, top, false, f);
  out.cap(rings, bottom, true, f);
}

/**
 * Meshes of the tile's buildings (all rows, or `rows`), keyed by material index. `x0`, `y0`
 * (the tile corner) only enter the world-anchored texture coordinates.
 */
export function buildingMeshes(rec: BuildingRecords, x0: number, y0: number, rows?: readonly number[]): Map<number, MeshArrays> {
  const builders = new Map<number, Builder>();
  const list = rows ?? Array.from({ length: rec.count }, (_, k) => k);
  for (const k of list) {
    const m = rec.material[k];
    let b = builders.get(m);
    if (!b) builders.set(m, (b = new Builder(x0, y0)));
    for (let p = rec.polyStart[k]; p < rec.polyStart[k + 1]; p++) polygonMesh(b, rec, p, rec.bottom[k], rec.top[k], k);
  }
  const out = new Map<number, MeshArrays>();
  for (const m of [...builders.keys()].sort((a, b) => a - b)) {
    const arrays = builders.get(m)!.arrays();
    if (arrays.indices.length) out.set(m, arrays);
  }
  return out;
}
