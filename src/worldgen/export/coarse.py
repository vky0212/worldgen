"""Buildings and trees of the coarser levels of detail (format 2.5).

A coarse tile (``pyramid``) takes its content from the files of its level-0 tiles, their
``BLD `` and ``TRE `` sections, and writes the same sections, simplified for its level. Coarse
content has no entity ids: no ``IDS ``, no ``TOB ``, and tree records carry no candidate index.

Buildings, at every coarse level. A level keeps a building when its footprint area or its
height above the terrain (top minus the lowest level-0 terrain at its footprint's vertices)
reaches the level's thresholds (``BUILDING_THRESHOLDS``). A kept building becomes flat prisms:

* its footprint polygons simplified (Douglas-Peucker, topology preserved) with a tolerance of
  half the level's node spacing; polygons that vanish are dropped;
* a flat roof halfway between its lowest eave and its top (``(eave + top) / 2``);
* its floor lowered to the lowest coarse terrain node around the footprint, so it never floats
  over the coarse terrain;
* its material.

Trees, at levels 1 to ``TREE_LEVELS``. The tile is cut into ``TREE_CELLS`` x ``TREE_CELLS``
cells; each cell keeps its tallest tree (ties: the southmost, then the westmost), whose crown
radius grows to stand for the cell's canopy, ``sqrt(sum of the cell's crown radii squared)``,
but not past half its height (and never shrinks). A cell of level L + 1 is four cells of
level L, so a tree kept at L + 1 is kept at L: the levels are nested.

Content error (``cerr``, metres): how far the coarse content may be from level 0's; the
largest of

* the height of each building left out;
* half the roof height of each prism, ``(top - lowest eave) / 2``, and the simplification
  tolerance when the tile keeps any building;
* for each tree left out, the smaller of its height and its distance to its cell's kept tree
  (every tree's height above ``TREE_LEVELS``);
* the crown growth of each kept tree.
"""

from __future__ import annotations

import math
import struct
from pathlib import Path
from typing import Any

import numpy as np
import shapely
from shapely.geometry import MultiPolygon, Polygon
from shapely.geometry.polygon import orient

from worldgen.export.container import read_container
from worldgen.export.footprints import COLUMNS, BuildingRecords, decode_buildings, encode_buildings
from worldgen.export.instances import FLAG_OBSERVED, TRE_DTYPE, decode_trees
from worldgen.export.tileset import tile_paths
from worldgen.terrain.heightfield import Heightfield
from worldgen.tiling.grid import TileGrid, tile_id
from worldgen.vegetation.placement import ARCHETYPES
from worldgen.vegetation.regions import ARCHETYPE_ORDER

Key = tuple[int, int]

# Level -> (footprint area m2, height m): a building is kept when it reaches either.
BUILDING_THRESHOLDS: dict[int, tuple[float, float]] = {1: (50.0, 12.0), 2: (300.0, 25.0), 3: (2000.0, 40.0),
                                                       4: (8000.0, 80.0)}
# Coarse levels that keep trees, and cells per tile side (one tree each at most).
TREE_LEVELS = 2
TREE_CELLS = 64
# Crown radius of a kept tree: at most this fraction of its height.
CROWN_HEIGHT_CAP = 0.5
_TREE_ORDER = {a: k for k, a in enumerate(ARCHETYPE_ORDER)}


def _up(v: float) -> float:
    """Rounded up to the millimetre."""
    return math.ceil(v * 1000.0 - 1e-6) / 1000.0


class LeafContent:
    """The buildings and trees of a coarse tile's level-0 tiles, read from their files."""

    def __init__(self) -> None:
        self.buildings: list[tuple[BuildingRecords, float, float]] = []   # (records, tile corner x0, y0)
        parts: dict[str, list[np.ndarray]] = {k: [] for k in ("x", "y", "h", "crown", "rec")}
        self._parts = parts

    def add(self, sections: dict[str, bytes], x0: float, y0: float, tile_size: float) -> None:
        if "BLD" in sections:
            rec = decode_buildings(sections["BLD"])
            if len(rec):
                self.buildings.append((rec, x0, y0))
        if "TRE" in sections:
            rec = decode_trees(sections["TRE"])
            if len(rec):
                p = self._parts
                p["x"].append(x0 + rec["x"].astype(np.float64) / 65535.0 * tile_size)
                p["y"].append(y0 + rec["y"].astype(np.float64) / 65535.0 * tile_size)
                p["h"].append(rec["h"].astype(np.float64) / 100.0)
                p["crown"].append(rec["crown"].astype(np.float64) / 10.0)
                p["rec"].append(rec)

    def trees(self) -> dict[str, np.ndarray]:
        p = self._parts
        if not p["rec"]:
            return {"x": np.zeros(0), "y": np.zeros(0), "h": np.zeros(0), "crown": np.zeros(0),
                    "rec": np.zeros(0, dtype=TRE_DTYPE)}
        return {k: np.concatenate(v) for k, v in p.items()}


def read_leaves(out: Path, leaves: list[Key], tile_size: float) -> LeafContent:
    """Content of the level-0 tiles ``leaves`` (in the given order); tiles without files add none."""
    grid = TileGrid(tile_size)
    content = LeafContent()
    for i, j in leaves:
        tid = tile_id(i, j)
        try:
            data = (out / tile_paths("data", tid)["content"]).read_bytes()
        except FileNotFoundError:
            continue
        x0, y0, _, _ = grid.bounds(tid)
        content.add(read_container(data, only={"BLD", "TRE"}).sections, x0, y0, tile_size)
    return content


def _ring_areas(rec: BuildingRecords) -> np.ndarray:
    """Signed area of every ring (exteriors counter-clockwise: positive; holes negative)."""
    xy = rec["xy"].reshape(-1, 2).astype(np.float64) / 1000.0
    vs = rec.vert_start
    if len(vs) < 2:
        return np.zeros(0)
    nxt = np.arange(len(xy)) + 1
    nxt[vs[1:] - 1] = vs[:-1]
    cross = xy[:, 0] * xy[nxt, 1] - xy[nxt, 0] * xy[:, 1]
    return np.add.reduceat(cross, vs[:-1]) / 2.0


def _per_building(values: np.ndarray, starts: np.ndarray, ufunc: Any) -> np.ndarray:
    """``ufunc.reduceat`` of per-item ``values`` over the buildings' item ranges ``starts`` (B + 1)."""
    out = np.zeros(len(starts) - 1)
    nonempty = starts[1:] > starts[:-1]
    if nonempty.any():
        out[nonempty] = ufunc.reduceat(values, starts[:-1][nonempty])
    return out


def coarse_buildings(content: LeafContent, hf: Heightfield, level: int, bounds: tuple[float, float, float, float]
                     ) -> tuple[list[tuple[str, bytes, bool]], float, list[float] | None, int]:
    """``BLD `` of a coarse tile's prisms; its building error, prism height range and count."""
    min_area, min_height = BUILDING_THRESHOLDS[level]
    k = 1 << level
    r = hf.resolution
    tol = r * k / 2.0
    X0, Y0 = bounds[0], bounds[1]
    cols: dict[str, list[Any]] = {name: [] for name, _, _ in COLUMNS}
    xy: list[np.ndarray] = []
    err = 0.0
    lo, hi = math.inf, -math.inf
    for rec, x0, y0 in content.buildings:
        ps, rs, vs = rec.poly_start, rec.ring_start, rec.vert_start
        area = _per_building(_per_building(_ring_areas(rec), rs, np.add), ps, np.add)
        pts = rec["xy"].reshape(-1, 2).astype(np.float64) / 1000.0 + (x0, y0)
        ground = _per_building(hf.sample(pts[:, 0], pts[:, 1]), vs[rs[ps]], np.minimum)
        height = rec["top"].astype(np.float64) - ground
        keep = (area >= min_area) | (height >= min_height)
        if (~keep).any():
            err = max(err, float(height[~keep].max()))
        for b in np.flatnonzero(keep).tolist():
            polys = []
            for p in range(int(ps[b]), int(ps[b + 1])):
                rings = [pts[vs[q]:vs[q + 1]] for q in range(int(rs[p]), int(rs[p + 1]))]
                s = shapely.simplify(Polygon(rings[0], rings[1:]), tol, preserve_topology=True)
                parts = list(s.geoms) if isinstance(s, MultiPolygon) else [s]
                polys += [orient(q, 1.0) for q in parts if isinstance(q, Polygon) and not q.is_empty and q.area > 0]
            top = float(rec["top"][b])
            if not polys:
                err = max(err, float(height[b]))
                continue
            eave = float(rec["eave"][ps[b]:ps[b + 1]].min())
            roof = (eave + top) / 2.0
            err = max(err, (top - eave) / 2.0, tol)
            v = pts[vs[rs[ps[b]]]:vs[rs[ps[b + 1]]]]
            c0 = max(int(math.floor((v[:, 0].min() - hf.xmin) / (r * k))) * k, 0)
            c1 = min(int(math.ceil((v[:, 0].max() - hf.xmin) / (r * k))) * k, hf.cols - 1)
            r0 = max(int(math.floor((v[:, 1].min() - hf.ymin) / (r * k))) * k, 0)
            r1 = min(int(math.ceil((v[:, 1].max() - hf.ymin) / (r * k))) * k, hf.rows - 1)
            bottom = min(float(rec["bottom"][b]), float(hf.z[r0:r1 + 1:k, c0:c1 + 1:k].min()))
            lo, hi = min(lo, bottom), max(hi, roof)
            cols["bottom"].append(bottom)
            cols["top"].append(roof)
            cols["polygons"].append(len(polys))
            cols["material"].append(int(rec["material"][b]))
            for q in polys:
                rings = [q.exterior, *q.interiors]
                cols["shape"].append(0)
                cols["rings"].append(len(rings))
                cols["eave"].append(roof)
                for name in ("cx", "cy", "ux", "uy", "half"):
                    cols[name].append(0.0)
                for ring in rings:
                    c = np.asarray(ring.coords)[:-1, :2]
                    cols["vertices"].append(len(c))
                    xy.append(np.rint((c - (X0, Y0)) * 1000.0))
    n = len(cols["bottom"])
    sections = [("BLD", encode_buildings(cols, xy), True)] if n else []
    return sections, _up(err), ([lo, hi] if n else None), n


def coarse_trees(content: LeafContent, coarse: Heightfield, level: int, bounds: tuple[float, float, float, float]
                 ) -> tuple[list[tuple[str, bytes, bool]], float, float | None, int]:
    """``TRE `` of a coarse tile's trees (levels up to TREE_LEVELS); its tree error, the highest
    tree top over the coarse terrain and the tree count."""
    t = content.trees()
    h = t["h"]
    if not len(h):
        return [], 0.0, None, 0
    if level > TREE_LEVELS:
        return [], _up(float(h.max())), None, 0
    X0, Y0, X1, _ = bounds
    size = X1 - X0
    g = size / TREE_CELLS
    x, y, crown = t["x"], t["y"], t["crown"]
    cx = np.clip(np.floor((x - X0) / g).astype(np.int64), 0, TREE_CELLS - 1)
    cy = np.clip(np.floor((y - Y0) / g).astype(np.int64), 0, TREE_CELLS - 1)
    cell = cy * TREE_CELLS + cx
    order = np.lexsort((x, y, -t["rec"]["h"].astype(np.int64), cell))
    first = np.r_[True, cell[order][1:] != cell[order][:-1]]
    starts = np.flatnonzero(first)
    kept = order[starts]
    rep = np.empty(len(h), dtype=np.int64)
    rep[order] = kept[np.cumsum(first) - 1]
    omitted = rep != np.arange(len(h))
    err = 0.0
    if omitted.any():
        d = np.hypot(x - x[rep], y - y[rep])
        err = float(np.minimum(h, d)[omitted].max())
    area = np.add.reduceat(crown[order] ** 2, starts)
    grown = np.maximum(crown[kept], np.minimum(np.sqrt(area), CROWN_HEIGHT_CAP * h[kept]))
    err = max(err, float((grown - crown[kept]).max()))
    src = t["rec"][kept]
    rec = np.zeros(len(kept), dtype=TRE_DTYPE)
    rec["x"] = np.clip(np.rint((x[kept] - X0) / size * 65535.0), 0, 65535)
    rec["y"] = np.clip(np.rint((y[kept] - Y0) / size * 65535.0), 0, 65535)
    rec["crown"] = np.clip(np.rint(grown * 10.0), 0, 255)
    for name in ("h", "trunk", "yaw", "arch_cat", "conf"):
        rec[name] = src[name]
    rec["flags"] = src["flags"] & FLAG_OBSERVED
    # Grouped by archetype in ARCHETYPE_ORDER (as level 0), by cell within a group.
    arch = np.array([_TREE_ORDER.get(ARCHETYPES[c] if c < len(ARCHETYPES) else "", len(ARCHETYPE_ORDER))
                     for c in (src["arch_cat"] & 15).tolist()], dtype=np.int64)
    rec = rec[np.lexsort((np.arange(len(kept)), arch))]
    top = float((coarse.sample(x[kept], y[kept]) + h[kept]).max())
    payload = struct.pack("<II", len(rec), TRE_DTYPE.itemsize) + rec.tobytes()
    return [("TRE", payload, True)], _up(err), top, len(rec)
