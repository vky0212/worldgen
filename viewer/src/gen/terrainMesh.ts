/**
 * Runtime terrain mesh of a data-first tile, from its heightmap: the same
 * vertices, triangles, normals, texture coordinates and colours as the terrain
 * of a baked tile GLB (reference: src/worldgen/export/heightmap.py; checked
 * against tests/conformance/terrain_tiles.json).
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
}

/**
 * `tileX0`, `tileY0`: the tile's minimum corner (i * tileSize, j * tileSize);
 * `rgb`: the `RGB ` section's nodes (nx * ny * 3), if any.
 */
export function terrainArrays(g: HeightGrid, rgb: Uint8Array | null, tileX0: number, tileY0: number, tileSize: number): TerrainArrays {
  const a = g.apron;
  const nx = g.nx - 2 * a;
  const ny = g.ny - 2 * a;
  const n = nx * ny;
  const positions = new Float32Array(3 * n);
  const normals = new Float32Array(3 * n);
  const uvs = new Float32Array(2 * n);
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
    colors = new Uint8Array(4 * n);
    for (let k = 0; k < n; k++) {
      colors[4 * k] = rgb[3 * k];
      colors[4 * k + 1] = rgb[3 * k + 1];
      colors[4 * k + 2] = rgb[3 * k + 2];
      colors[4 * k + 3] = 255;
    }
  }
  // Two triangles per cell, split along the (c,r)-(c+1,r+1) diagonal: first every
  // (00, 10, 11) triangle, then every (00, 11, 01) one, as in the baked mesh.
  const cells = (nx - 1) * (ny - 1);
  const indices = new Uint32Array(6 * cells);
  let t = 0;
  for (let r = 0; r < ny - 1; r++) {
    for (let c = 0; c < nx - 1; c++) {
      const v00 = r * nx + c;
      indices[t++] = v00;
      indices[t++] = v00 + 1;
      indices[t++] = v00 + nx + 1;
    }
  }
  for (let r = 0; r < ny - 1; r++) {
    for (let c = 0; c < nx - 1; c++) {
      const v00 = r * nx + c;
      indices[t++] = v00;
      indices[t++] = v00 + nx + 1;
      indices[t++] = v00 + nx;
    }
  }
  return { positions, normals, uvs, colors, indices, nx, ny };
}
