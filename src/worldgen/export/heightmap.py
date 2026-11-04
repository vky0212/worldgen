"""Terrain of data-first tiles: heightmap, colour and land-cover sections, and the reference mesh.

A tile's terrain ships as data (``HGT ``, ``RGB ``, ``LCC `` sections of its ``.wgt``):
the tile's grid nodes, shared edges included, plus a one-node apron for normals. Viewers
and engines mesh it at runtime; :func:`terrain_mesh_from_sections` is the reference for that
mesh, identical in triangulation and positions to the terrain of a ``baked`` tile.
"""

from __future__ import annotations

import numpy as np

from worldgen.export.container import HeightGrid, decode_grid, decode_heights, encode_grid, encode_heights
from worldgen.geometry.mesh import Mesh
from worldgen.terrain.heightfield import Heightfield
from worldgen.tiling.grid import TileGrid


def tile_node_range(hf: Heightfield, bounds: tuple[float, float, float, float]) -> tuple[int, int, int, int]:
    """(r0, r1, c0, c1) grid nodes of a tile, inclusive, as ``Heightfield.tile_mesh`` picks them."""
    x0, y0, x1, y1 = bounds
    r = hf.resolution
    c0, c1 = int(round((x0 - hf.xmin) / r)), int(round((x1 - hf.xmin) / r))
    r0, r1 = int(round((y0 - hf.ymin) / r)), int(round((y1 - hf.ymin) / r))
    return max(r0, 0), min(r1, hf.rows - 1), max(c0, 0), min(c1, hf.cols - 1)


def _extrapolate_apron(n: np.ndarray, west: bool, east: bool, south: bool, north: bool) -> np.ndarray:
    """Add a linearly extrapolated column/row on the sides where the grid ends, so central
    differences there equal the one-sided differences the whole grid uses at its border."""
    if west:
        n = np.concatenate([2 * n[:, :1] - n[:, 1:2], n], axis=1)
    if east:
        n = np.concatenate([n, 2 * n[:, -1:] - n[:, -2:-1]], axis=1)
    if south:
        n = np.concatenate([2 * n[:1] - n[1:2], n], axis=0)
    if north:
        n = np.concatenate([n, 2 * n[-1:] - n[-2:-1]], axis=0)
    return n


def terrain_sections(hf: Heightfield, bounds: tuple[float, float, float, float], quantum: float,
                     rgba: np.ndarray, codes: np.ndarray | None) -> list[tuple[str, bytes, bool]]:
    """``HGT ``, ``RGB `` and ``LCC `` payloads of one tile (``hf`` already rounded to ``quantum``)."""
    r0, r1, c0, c1 = tile_node_range(hf, bounds)
    wr0, wr1 = max(r0 - 1, 0), min(r1 + 1, hf.rows - 1)
    wc0, wc1 = max(c0 - 1, 0), min(c1 + 1, hf.cols - 1)
    n = np.rint(hf.z[wr0:wr1 + 1, wc0:wc1 + 1] / quantum).astype(np.int64)
    heights = _extrapolate_apron(n, wc0 == c0, wc1 == c1, wr0 == r0, wr1 == r1)
    r = hf.resolution
    x0 = hf.xmin + (c0 - 1) * r
    y0 = hf.ymin + (r0 - 1) * r
    sections = [("HGT", encode_heights(heights, r, quantum, x0, y0), True)]
    col = np.clip(np.round(rgba[r0:r1 + 1, c0:c1 + 1, :3] * 255.0), 0, 255).astype(np.uint8)
    sections.append(("RGB", encode_grid(col), True))
    if codes is not None:
        sections.append(("LCC", encode_grid(codes[r0:r1 + 1, c0:c1 + 1]), True))
    return sections


def heightgrid_normals(g: HeightGrid) -> np.ndarray:
    """Unit normals (ENU) at the tile nodes (apron excluded), central differences."""
    z = g.z
    a = g.apron
    gx = (z[a:-a, 2 * a:] - z[a:-a, :-2 * a]) / (2.0 * g.spacing)
    gy = (z[2 * a:, a:-a] - z[:-2 * a, a:-a]) / (2.0 * g.spacing)
    n = np.dstack([-gx, -gy, np.ones_like(gx)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def terrain_mesh_from_sections(sections: dict[str, bytes], tile_size: float) -> Mesh:
    """Reference terrain mesh of a data-first tile: tile nodes, (i,j)-(i+1,j+1) diagonal,
    positions in local ENU, vertex colours from ``RGB `` (0..1), UVs over the tile square."""
    g = decode_heights(sections["HGT"])
    a = g.apron
    z = g.z[a:-a, a:-a]
    nr, nc = z.shape
    xs = g.x0 + (np.arange(nc) + a) * g.spacing
    ys = g.y0 + (np.arange(nr) + a) * g.spacing
    gx, gy = np.meshgrid(xs, ys)
    pos = np.column_stack([gx.ravel(), gy.ravel(), z.ravel()])
    k = (np.arange(nr - 1)[:, None] * nc + np.arange(nc - 1)[None, :]).ravel()
    v00, v10, v01, v11 = k, k + 1, k + nc, k + nc + 1
    tri = np.concatenate([np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01])]).astype(np.int64)
    tx0, ty0 = xs[0], ys[0]
    uvs = np.column_stack([(gx.ravel() - tx0) / tile_size, (gy.ravel() - ty0) / tile_size])
    cols = None
    if "RGB" in sections:
        rgb = decode_grid(sections["RGB"], 3).reshape(-1, 3).astype(np.float64) / 255.0
        cols = np.column_stack([rgb, np.ones(len(rgb))])
    return Mesh(pos, tri, normals=heightgrid_normals(g).reshape(-1, 3), uvs=uvs, colors=cols)


def tile_terrain_sections(world: object, tid: str, rgba: np.ndarray) -> list[tuple[str, bytes, bool]]:
    """Terrain sections of tile ``tid`` of a compiled world."""
    cfg = world.config  # type: ignore[attr-defined]
    bounds = TileGrid(cfg.tiling.tile_size).bounds(tid)
    return terrain_sections(world.terrain, bounds, cfg.height_quantum(), rgba, world.terrain_codes)  # type: ignore[attr-defined]
