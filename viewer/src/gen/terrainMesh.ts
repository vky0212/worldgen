/**
 * Runtime terrain mesh of a data-first tile, from its heightmap: the same
 * vertices, triangles, normals, texture coordinates and colours as the terrain
 * of a baked tile GLB (reference: src/worldgen/export/heightmap.py; checked
 * against tests/conformance/terrain_tiles.json). Tiles of every level of detail
 * mesh alike; `tileSize` is the tile's own edge length.
 *
 * Two runtime additions for worlds with coarser levels, neither of which
 * changes the reference vertices: cells outside `covers` are left out (a
 * coarser tile whose square has missing level-0 tiles), and skirts — vertical
 * strips under the tile's edges, appended after the surface — hide the cracks
 * between neighbouring tiles of different levels.
 *
 * Output is in glTF axes (x, z, -y) relative to the tile's minimum corner, like
 * baked tiles, whose root node carries the corner as its translation.
 * Pure module (no three.js).
 */
import type { HeightGrid } from '../content/wgt.ts';

export interface TerrainArrays {
  positions: Float32Array;
  normals: Float32Array;
  uvs: Float32Array;
  /** RGBA u8 (alpha 255), or null without an RGB section. */
  colors: Uint8Array | null;
  indices: Uint32Array;
  /** Tile nodes per row / column (apron excluded). */
  nx: number;
  ny: number;
  /** Vertices of the surface; skirt vertices follow them. */
  surfaceVertices: number;
}

export interface TerrainOptions {
  /** Skirt depth (m) under the west, east, south and north edge; 0 = none. */
  skirts?: readonly [number, number, number, number] | null;
  /** Cells whose centre (local ENU x, y) this rejects are not meshed. */
  covers?: ((x: number, y: number) => boolean) | null;
}

/**
 * `tileX0`, `tileY0`: the tile's minimum corner (i * tileSize, j * tileSize);
 * `rgb`: the `RGB ` section's nodes (nx * ny * 3), if any.
 */
export function terrainArrays(g: HeightGrid, rgb: Uint8Array | null, tileX0: number, tileY0: number, tileSize: number,
  options: TerrainOptions = {}): TerrainArrays {
  const a = g.apron;
  const nx = g.nx - 2 * a;
  const ny = g.ny - 2 * a;
  const n = nx * ny;

  // Cells kept (all without `covers`), and the skirts' edge segments next to kept cells.
  let keep: Uint8Array | null = null;
  if (options.covers) {
    keep = new Uint8Array((nx - 1) * (ny - 1));
    for (let r = 0; r < ny - 1; r++) {
      const y = g.y0 + (r + a + 0.5) * g.spacing;
      for (let c = 0; c < nx - 1; c++) keep[r * (nx - 1) + c] = options.covers(g.x0 + (c + a + 0.5) * g.spacing, y) ? 1 : 0;
    }
  }
  const kept = (r: number, c: number): boolean => !keep || keep[r * (nx - 1) + c] === 1;
  const edges = skirtEdges(nx, ny, options.skirts ?? null, kept);
  const extra = edges.reduce((s, e) => s + e.nodes.length, 0);

  const total = n + extra;
  const positions = new Float32Array(3 * total);
  const normals = new Float32Array(3 * total);
  const uvs = new Float32Array(2 * total);
  const z = g.z;
  const W = g.nx;
  const sp2 = 2 * g.spacing;
  for (let r = 0; r < ny; r++) {
    const y = g.y0 + (r + a) * g.spacing;
    for (let c = 0; c < nx; c++) {
      const x = g.x0 + (c + a) * g.spacing;
      const k = r * nx + c;
      const at = (r + a) * W + (c + a);
      const h = z[at];
      positions[3 * k] = x - tileX0;
      positions[3 * k + 1] = h;
      positions[3 * k + 2] = -(y - tileY0);
      // Central differences over the apron, normal (-dz/dx, -dz/dy, 1) in ENU.
      const gx = (z[at + 1] - z[at - 1]) / sp2;
      const gy = (z[at + W] - z[at - W]) / sp2;
      const len = Math.sqrt(gx * gx + gy * gy + 1);
      normals[3 * k] = -gx / len;
      normals[3 * k + 1] = 1 / len;
      normals[3 * k + 2] = gy / len;
      uvs[2 * k] = (x - tileX0) / tileSize;
      uvs[2 * k + 1] = 1 - (y - tileY0) / tileSize;
    }
  }
  let colors: Uint8Array | null = null;
  if (rgb) {
    colors = new Uint8Array(4 * total);
    for (let k = 0; k < n; k++) {
      colors[4 * k] = rgb[3 * k];
      colors[4 * k + 1] = rgb[3 * k + 1];
      colors[4 * k + 2] = rgb[3 * k + 2];
      colors[4 * k + 3] = 255;
    }
  }

  // Skirt vertices: copies of the edge vertices (normal, uv, colour too), lowered by the depth.
  let v = n;
  const lowered: number[][] = [];
  for (const e of edges) {
    const ids: number[] = [];
    for (const src of e.nodes) {
      positions[3 * v] = positions[3 * src];
      positions[3 * v + 1] = positions[3 * src + 1] - e.depth;
      positions[3 * v + 2] = positions[3 * src + 2];
      for (let q = 0; q < 3; q++) normals[3 * v + q] = normals[3 * src + q];
      uvs[2 * v] = uvs[2 * src];
      uvs[2 * v + 1] = uvs[2 * src + 1];
      if (colors) for (let q = 0; q < 4; q++) colors[4 * v + q] = colors[4 * src + q];
      ids.push(v++);
    }
    lowered.push(ids);
  }

  // Two triangles per kept cell, split along the (c,r)-(c+1,r+1) diagonal: first every
  // (00, 10, 11) triangle, then every (00, 11, 01) one, as in the baked mesh.
  let cells = 0;
  for (let r = 0; r < ny - 1; r++) for (let c = 0; c < nx - 1; c++) if (kept(r, c)) cells++;
  const quads = edges.reduce((s, e) => s + e.segments.length, 0);
  const indices = new Uint32Array(6 * cells + 12 * quads);
  let t = 0;
  for (let r = 0; r < ny - 1; r++) {
    for (let c = 0; c < nx - 1; c++) {
      if (!kept(r, c)) continue;
      const v00 = r * nx + c;
      indices[t++] = v00;
      indices[t++] = v00 + 1;
      indices[t++] = v00 + nx + 1;
    }
  }
  for (let r = 0; r < ny - 1; r++) {
    for (let c = 0; c < nx - 1; c++) {
      if (!kept(r, c)) continue;
      const v00 = r * nx + c;
      indices[t++] = v00;
      indices[t++] = v00 + nx + 1;
      indices[t++] = v00 + nx;
    }
  }
  // Skirt quads, both windings: they are seen from either side of the edge.
  edges.forEach((e, q) => {
    const low = lowered[q];
    for (const s of e.segments) {
      const p0 = e.nodes[s];
      const p1 = e.nodes[s + 1];
      const l0 = low[s];
      const l1 = low[s + 1];
      indices.set([p0, l0, p1, p1, l0, l1, p0, p1, l0, p1, l1, l0], t);
      t += 12;
    }
  });
  return { positions, normals, uvs, colors, indices, nx, ny, surfaceVertices: n };
}

interface SkirtEdge {
  depth: number;
  /** Surface vertex indices along the edge. */
  nodes: number[];
  /** Segment k joins nodes k and k + 1 (only next to kept cells). */
  segments: number[];
}

function skirtEdges(nx: number, ny: number, depths: readonly number[] | null,
  kept: (r: number, c: number) => boolean): SkirtEdge[] {
  if (!depths) return [];
  const out: SkirtEdge[] = [];
  const add = (depth: number, count: number, node: (k: number) => number, cell: (k: number) => boolean): void => {
    if (!(depth > 0)) return;
    const segments: number[] = [];
    for (let k = 0; k < count - 1; k++) if (cell(k)) segments.push(k);
    if (!segments.length) return;
    const nodes: number[] = [];
    for (let k = 0; k < count; k++) nodes.push(node(k));
    out.push({ depth, nodes, segments });
  };
  add(depths[0], ny, (k) => k * nx, (k) => kept(k, 0));                                  // west
  add(depths[1], ny, (k) => k * nx + nx - 1, (k) => kept(k, nx - 2));                     // east
  add(depths[2], nx, (k) => k, (k) => kept(0, k));                                        // south
  add(depths[3], nx, (k) => (ny - 1) * nx + k, (k) => kept(ny - 2, k));                   // north
  return out;
}
