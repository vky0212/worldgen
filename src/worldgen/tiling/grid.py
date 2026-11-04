"""Tile grid, ownership rules and tile-aware mesh pieces.

Ownership rules (each generated piece belongs to exactly one tile):

* terrain: the grid nodes of the tile square (edges shared, identical vertices);
* roads, bridge decks, rivers: centreline pieces split exactly at tile lines;
* intersections, piers, portals, buildings, trees: the tile containing their
  anchor point (node, footprint representative point, trunk position);
* water and land-cover areas: polygon clipped to the tile square.
"""

from __future__ import annotations

import math
from collections.abc import Iterable
from dataclasses import dataclass

import numpy as np

from worldgen.geometry.mesh import Mesh


def tile_id(i: int, j: int) -> str:
    return f"{i}_{j}"


def parse_tile_id(tid: str) -> tuple[int, int]:
    a, b = tid.split("_")
    return int(a), int(b)


@dataclass(frozen=True)
class TileGrid:
    size: float

    def tile_of(self, x: float, y: float) -> str:
        return tile_id(math.floor(x / self.size), math.floor(y / self.size))

    def tiles_of(self, xs: np.ndarray, ys: np.ndarray) -> np.ndarray:
        i = np.floor(np.asarray(xs) / self.size).astype(np.int64)
        j = np.floor(np.asarray(ys) / self.size).astype(np.int64)
        return np.array([f"{a}_{b}" for a, b in zip(i, j)], dtype=object)

    def bounds(self, tid: str) -> tuple[float, float, float, float]:
        i, j = parse_tile_id(tid)
        s = self.size
        return (i * s, j * s, (i + 1) * s, (j + 1) * s)

    def origin(self, tid: str) -> tuple[float, float]:
        b = self.bounds(tid)
        return b[0], b[1]

    def tiles_in_bbox(self, xmin: float, ymin: float, xmax: float, ymax: float) -> list[str]:
        s = self.size
        out = []
        for j in range(math.floor(ymin / s), math.floor(ymax / s) + 1):
            for i in range(math.floor(xmin / s), math.floor(xmax / s) + 1):
                out.append(tile_id(i, j))
        return out

    def insert_tile_crossings(self, xy: np.ndarray, *values: np.ndarray) -> tuple[np.ndarray, ...]:
        """Insert polyline vertices where it crosses tile lines; values are interpolated."""
        s = self.size
        out_xy = [xy[0]]
        out_vals: list[list[float]] = [[float(v[0])] for v in values]
        for k in range(len(xy) - 1):
            a, b = xy[k], xy[k + 1]
            ts = []
            for axis in (0, 1):
                lo, hi = sorted((a[axis], b[axis]))
                first = math.floor(lo / s) + 1
                last = math.ceil(hi / s) - 1
                if hi - lo < 1e-12:
                    continue
                for m in range(first, last + 1):
                    t = (m * s - a[axis]) / (b[axis] - a[axis])
                    if 1e-9 < t < 1 - 1e-9:
                        ts.append(t)
            # A line through a tile corner crosses both grid lines at (almost) the same t.
            uniq: list[float] = []
            for t in sorted(ts):
                if not uniq or (t - uniq[-1]) * float(np.hypot(*(b - a))) > 1e-6:
                    uniq.append(t)
            for t in uniq:
                out_xy.append(a + (b - a) * t)
                for vi, v in enumerate(values):
                    out_vals[vi].append(float(v[k] + (v[k + 1] - v[k]) * t))
            out_xy.append(b)
            for vi, v in enumerate(values):
                out_vals[vi].append(float(v[k + 1]))
        return (np.array(out_xy), *[np.array(v) for v in out_vals])

    def segment_tiles(self, xy: np.ndarray) -> list[str]:
        mid = (xy[:-1] + xy[1:]) / 2.0
        return list(self.tiles_of(mid[:, 0], mid[:, 1]))


@dataclass
class Piece:
    """A mesh fragment owned by one tile, on one layer, for one feature."""

    tile: str
    layer: str
    material: str
    feature_id: str
    mesh: Mesh


def submesh(mesh: Mesh, tri_mask: np.ndarray) -> Mesh:
    idx = mesh.indices[tri_mask]
    used = np.unique(idx)
    remap = -np.ones(mesh.vertex_count, dtype=np.int64)
    remap[used] = np.arange(len(used))
    sel = lambda a: None if a is None else a[used]  # noqa: E731
    return Mesh(mesh.positions[used], remap[idx], sel(mesh.normals), sel(mesh.uvs), sel(mesh.colors),
                sel(mesh.feature), mesh.mode)


def group_pieces(pieces: Iterable[Piece]) -> dict[str, list[Piece]]:
    out: dict[str, list[Piece]] = {}
    for p in pieces:
        out.setdefault(p.tile, []).append(p)
    return out
