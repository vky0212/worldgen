/**
 * Terrain height of a data-first tile's heightmap at local (x, y), on the mesh
 * triangles (cells split along the (c, r)-(c+1, r+1) diagonal): the compiler's
 * Heightfield.sample (reference: src/worldgen/export/instances.py height_at). Pure module.
 */
import type { HeightGrid } from '../content/wgt.ts';

export function heightAt(g: HeightGrid, x: number, y: number): number {
  const fx = (x - g.x0) / g.spacing;
  const fy = (y - g.y0) / g.spacing;
  const c = Math.min(Math.max(Math.floor(fx), 0), g.nx - 2);
  const r = Math.min(Math.max(Math.floor(fy), 0), g.ny - 2);
  const u = Math.min(Math.max(fx - c, 0), 1);
  const v = Math.min(Math.max(fy - r, 0), 1);
  const at = r * g.nx + c;
  const z00 = g.z[at];
  const z10 = g.z[at + 1];
  const z01 = g.z[at + g.nx];
  const z11 = g.z[at + g.nx + 1];
  return u >= v ? z00 + u * (z10 - z00) + v * (z11 - z10) : z00 + v * (z01 - z00) + u * (z11 - z01);
}
