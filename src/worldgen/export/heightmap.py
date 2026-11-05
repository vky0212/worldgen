"""Terrain of data-first tiles: heightmap, colour and land-cover sections, and the reference mesh.

A tile's terrain ships as data (``HGT ``, ``RGB ``, ``LCC `` sections of its ``.wgt``):
the tile's grid nodes, shared edges included, plus a one-node apron for normals. Viewers
and engines mesh it at runtime; :func:`terrain_mesh_from_sections` is the reference for that
mesh, identical in triangulation and positions to the terrain of a ``baked`` tile.
"""

from __future__ import annotations

import numpy as np

from worldgen.export.container import HeightGrid, decode_grid, decode_heights, encode_grid, encode_heights
from worldgen.export.gltf import MATERIALS
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


def level_node_range(hf: Heightfield, bounds: tuple[float, float, float, float], level: int
                     ) -> tuple[int, int, int, int]:
    """``tile_node_range`` of a tile of ``level``, checked to lie on that level's lattice (every
    ``2**level``-th node). Holds whenever the grid starts at a tile corner and ``2**level``
    divides the nodes per tile, which ``pyramid.top_level`` ensures."""
    k = 1 << level
    rng = tile_node_range(hf, bounds)
    if any(v % k for v in rng):
        raise ValueError(f"tile {bounds} is not on the level-{level} node lattice of the terrain grid")
    return rng


def _halve(a: np.ndarray) -> np.ndarray:
    """One step of the colour pyramid: a [1, 2, 1] / 4 binomial filter along both axes (the
    array's edges repeated), then every other node from the first."""
    for axis in (0, 1):
        n = a.shape[axis]
        if n < 2:
            continue
        p = np.concatenate([a.take([0], axis), a, a.take([n - 1], axis)], axis=axis)
        a = (p.take(range(0, n), axis) + 2.0 * p.take(range(1, n + 1), axis) + p.take(range(2, n + 2), axis)) / 4.0
        a = a.take(range(0, n, 2), axis)
    return a


# Land-cover code of water (landcover.LANDCOVER_CODES): level-0 tiles draw a water surface there.
WATER_CODE = 7


def surface_colors(rgb: np.ndarray, codes: np.ndarray | None) -> np.ndarray:
    """Colours a level-0 tile shows from above: the terrain colour, with the water surface (its
    material's colour at its opacity, as a level-0 tile blends it) over water nodes."""
    if codes is None:
        return rgb
    w = MATERIALS["water"].color
    water = np.asarray(w[:3], dtype=np.float64)
    return np.where((codes == WATER_CODE)[..., None], w[3] * water + (1.0 - w[3]) * rgb, rgb)


def level_colors(rgba: np.ndarray, rng: tuple[int, int, int, int], level: int,
                 codes: np.ndarray | None = None) -> np.ndarray:
    """Colours (0..1, float64) at the level-``level`` nodes of the node range ``rng``: ``level``
    halving steps (``_halve``) of the level-0 ``surface_colors``. Computed in a window with a
    margin of ``2**level`` nodes, which covers the filters' reach (``2**level - 1`` nodes), so a
    node gets the same colour from any tile's window; the grid's own edges are repeated as in
    the window."""
    k = 1 << level
    r0, r1, c0, c1 = rng
    rows, cols = rgba.shape[:2]
    wr0, wr1 = max(r0 - k, 0), min(r1 + k, rows - 1)
    wc0, wc1 = max(c0 - k, 0), min(c1 + k, cols - 1)
    a = surface_colors(rgba[wr0:wr1 + 1, wc0:wc1 + 1, :3].astype(np.float64),
                       codes[wr0:wr1 + 1, wc0:wc1 + 1] if codes is not None else None)
    for _ in range(level):
        a = _halve(a)
    return a[(r0 - wr0) // k:(r1 - wr0) // k + 1, (c0 - wc0) // k:(c1 - wc0) // k + 1]


def terrain_sections(hf: Heightfield, bounds: tuple[float, float, float, float], quantum: float,
                     rgba: np.ndarray, codes: np.ndarray | None, level: int = 0) -> list[tuple[str, bytes, bool]]:
    """``HGT ``, ``RGB `` and ``LCC `` payloads of one tile (``hf`` already rounded to ``quantum``).

    A tile of a coarser ``level`` (``pyramid``) takes every ``2**level``-th node: the level-0 heights
    and land-cover codes there, its apron one coarse node out, and ``level_colors``."""
    k = 1 << level
    r0, r1, c0, c1 = rng = level_node_range(hf, bounds, level)
    wr0 = r0 - k if r0 >= k else r0
    wr1 = r1 + k if r1 + k <= hf.rows - 1 else r1
    wc0 = c0 - k if c0 >= k else c0
    wc1 = c1 + k if c1 + k <= hf.cols - 1 else c1
    n = np.rint(hf.z[wr0:wr1 + 1:k, wc0:wc1 + 1:k] / quantum).astype(np.int64)
    heights = _extrapolate_apron(n, wc0 == c0, wc1 == c1, wr0 == r0, wr1 == r1)
    r = hf.resolution
    x0 = hf.xmin + (c0 - k) * r
    y0 = hf.ymin + (r0 - k) * r
    sections = [("HGT", encode_heights(heights, r * k, quantum, x0, y0), True)]
    if level:
        col = np.clip(np.round(level_colors(rgba, rng, level, codes) * 255.0), 0, 255).astype(np.uint8)
    else:
        col = np.clip(np.round(rgba[r0:r1 + 1, c0:c1 + 1, :3] * 255.0), 0, 255).astype(np.uint8)
    sections.append(("RGB", encode_grid(col), True))
    if codes is not None:
        sections.append(("LCC", encode_grid(codes[r0:r1 + 1:k, c0:c1 + 1:k]), True))
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
