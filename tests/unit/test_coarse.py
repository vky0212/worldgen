"""Buildings and trees of the coarser levels (export/coarse.py): selection, nesting, prisms, errors."""

from __future__ import annotations

import math
import struct

import numpy as np
import pytest
from shapely.geometry import Polygon, box
from shapely.geometry.polygon import orient

from worldgen.export.coarse import (
    BUILDING_THRESHOLDS,
    CROWN_HEIGHT_CAP,
    TREE_CELLS,
    TREE_LEVELS,
    LeafContent,
    coarse_buildings,
    coarse_trees,
)
from worldgen.export.container import read_container, write_container
from worldgen.export.footprints import COLUMNS, decode_buildings, encode_buildings
from worldgen.export.instances import FLAG_OBSERVED, FLAG_TOB, TRE_DTYPE, decode_trees
from worldgen.export.pyramid import coarse_heightfield, level_bounds
from worldgen.terrain.heightfield import Heightfield

T = 64.0        # level-0 tile size
R = 2.0         # terrain spacing: 32 cells per tile


def _terrain() -> Heightfield:
    """Four level-0 tiles from (0, 0) to (128, 128), sloping up to the east and north."""
    n = int(2 * T / R) + 1
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    return Heightfield(0.0, 0.0, R, 10.0 + 0.5 * xx + 0.25 * yy + np.sin(xx) * 0.3)


def _tre(x: np.ndarray, y: np.ndarray, h: np.ndarray, crown: np.ndarray, x0: float, y0: float,
         flags: int = FLAG_OBSERVED | FLAG_TOB) -> bytes:
    rec = np.zeros(len(x), dtype=TRE_DTYPE)
    rec["x"] = np.rint((x - x0) / T * 65535)
    rec["y"] = np.rint((y - y0) / T * 65535)
    rec["h"] = np.rint(h * 100)
    rec["crown"] = np.rint(crown * 10)
    rec["trunk"] = 20
    rec["yaw"] = np.arange(len(x)) % 256
    rec["arch_cat"] = np.arange(len(x)) % 3
    rec["conf"] = 200
    rec["k"] = 7
    rec["dx"], rec["dy"] = 1, 2
    rec["flags"] = flags
    return struct.pack("<II", len(rec), 16) + rec.tobytes()


def _forest(seed: int = 3) -> tuple[LeafContent, dict[str, np.ndarray]]:
    """Level-0 trees of the four tiles: a dense stand and a few lone trees."""
    rng = np.random.default_rng(seed)
    content = LeafContent()
    for j in range(2):
        for i in range(2):
            x0, y0 = i * T, j * T
            n = 600 if (i, j) == (0, 0) else 40
            x = x0 + rng.uniform(0, T - 1e-3, n)
            y = y0 + rng.uniform(0, T - 1e-3, n)
            h = np.round(rng.uniform(4, 25, n), 2)
            h[:20] = 12.0                      # ties broken by position
            crown = np.round(rng.uniform(1, 4, n), 1)
            content.add({"TRE": _tre(x, y, h, crown, x0, y0)}, x0, y0, T)
    return content, content.trees()


def _decode(sections: list[tuple[str, bytes, bool]], bounds: tuple[float, float, float, float]) -> dict[str, np.ndarray]:
    tre = decode_trees(dict((t, d) for t, d, _ in sections)["TRE"])
    size = bounds[2] - bounds[0]
    return {"x": bounds[0] + tre["x"] / 65535 * size, "y": bounds[1] + tre["y"] / 65535 * size,
            "h": tre["h"] / 100, "crown": tre["crown"] / 10, "rec": tre}


@pytest.mark.parametrize("level", [1, 2])
def test_coarse_trees_keep_the_tallest_tree_of_each_cell(level):
    hf = _terrain()
    content, t = _forest()
    bounds = level_bounds(T, level, 0, 0) if level == 1 else (0.0, 0.0, 4 * T, 4 * T)
    sections, err, top, n = coarse_trees(content, coarse_heightfield(hf, (0, 0, 2 * T, 2 * T), 1), level, bounds)
    out = _decode(sections, bounds)
    g = (bounds[2] - bounds[0]) / TREE_CELLS
    # Brute force: per cell, the tallest tree (then the southmost, then the westmost).
    cells: dict[tuple[int, int], list[int]] = {}
    for k in range(len(t["x"])):
        if bounds[0] <= t["x"][k] < bounds[2] and bounds[1] <= t["y"][k] < bounds[3]:
            cells.setdefault((int((t["x"][k] - bounds[0]) // g), int((t["y"][k] - bounds[1]) // g)), []).append(k)
    assert n == len(cells) == len(out["x"])
    want = {}
    worst = 0.0
    for members in cells.values():
        best = min(members, key=lambda k: (-round(t["h"][k] * 100), t["y"][k], t["x"][k]))
        c = math.sqrt(sum(t["crown"][k] ** 2 for k in members))
        grown = max(t["crown"][best], min(c, CROWN_HEIGHT_CAP * t["h"][best]))
        want[(t["x"][best], t["y"][best])] = (t["h"][best], round(grown, 1))
        worst = max(worst, grown - t["crown"][best])
        for k in members:
            if k != best:
                worst = max(worst, min(t["h"][k], math.hypot(t["x"][k] - t["x"][best], t["y"][k] - t["y"][best])))
    for (x, y), (h, c) in want.items():
        d = np.hypot(out["x"] - x, out["y"] - y)
        k = int(d.argmin())
        assert d[k] < 0.005 and out["h"][k] == pytest.approx(h) and out["crown"][k] == pytest.approx(c, abs=0.051)
    assert worst - 0.06 <= err <= worst + 0.06              # positions and crowns at record precision
    assert err > 0 and top is not None
    # Records without ids: no candidate index, cell or TOB flag; observed kept; grouped by archetype.
    rec = out["rec"]
    assert not (rec["k"].any() or rec["dx"].any() or rec["dy"].any()) and (rec["flags"] == FLAG_OBSERVED).all()
    from worldgen.export.coarse import _TREE_ORDER
    from worldgen.vegetation.placement import ARCHETYPES
    order = [_TREE_ORDER[ARCHETYPES[c & 15]] for c in rec["arch_cat"]]
    assert order == sorted(order)


def test_coarse_trees_nest_and_thin_up_the_levels():
    hf = _terrain()
    content, t = _forest()
    terrain = coarse_heightfield(hf, (0, 0, 2 * T, 2 * T), 1)
    b1, b2 = (0.0, 0.0, 2 * T, 2 * T), (0.0, 0.0, 4 * T, 4 * T)
    one = _decode(coarse_trees(content, terrain, 1, b1)[0], b1)
    two = _decode(coarse_trees(content, terrain, 2, b2)[0], b2)
    assert len(t["x"]) > len(one["x"]) > len(two["x"]) > 0
    for x, y, h in zip(two["x"], two["y"], two["h"]):
        d = np.hypot(one["x"] - x, one["y"] - y)
        k = int(d.argmin())
        assert d[k] < 0.05 and one["h"][k] == h
    # Above TREE_LEVELS: no trees; the error is the tallest tree.
    sections, err, top, n = coarse_trees(content, terrain, TREE_LEVELS + 1, (0.0, 0.0, 8 * T, 8 * T))
    assert sections == [] and n == 0 and top is None and err == pytest.approx(t["h"].max())
    assert coarse_trees(LeafContent(), terrain, 1, b1) == ([], 0.0, None, 0)


def _bld(polys: list[Polygon], bottom: list[float], top: list[float], eave: list[float], material: list[int],
         x0: float, y0: float) -> bytes:
    cols = {name: [] for name, _, _ in COLUMNS}
    xy = []
    for p, b, t, e, m in zip(polys, bottom, top, eave, material):
        p = orient(p, 1.0)                       # BLD: exteriors counter-clockwise, holes clockwise
        cols["bottom"].append(b)
        cols["top"].append(t)
        cols["polygons"].append(1)
        cols["material"].append(m)
        rings = [p.exterior, *p.interiors]
        cols["shape"].append(0 if e == t else 1)
        cols["rings"].append(len(rings))
        cols["eave"].append(e)
        c = p.centroid
        for name, v in (("cx", c.x - x0), ("cy", c.y - y0), ("ux", 1.0), ("uy", 0.0), ("half", 3.0)):
            cols[name].append(v)
        for ring in rings:
            v = np.asarray(ring.coords)[:-1, :2]
            cols["vertices"].append(len(v))
            xy.append(np.rint((v - (x0, y0)) * 1000.0))
    return encode_buildings(cols, xy)


def test_coarse_buildings_keep_large_or_tall_ones_as_flat_prisms():
    hf = _terrain()
    ground = lambda p: float(hf.sample(*np.asarray(p.exterior.coords)[:-1].T).min())  # noqa: E731
    big_low = box(5, 5, 25, 25).difference(box(12, 12, 16, 16))        # 384 m2, a courtyard
    small_tall = box(40, 40, 45, 46)                                   # 30 m2
    small_low = box(50, 10, 54, 14.5)                                  # 18 m2, gabled
    jagged = Polygon([(70, 70), (90, 70), (90, 85), (80.5, 85), (80, 85.6), (79.5, 85), (70, 85)])  # a 0.6 m bump
    polys = [big_low, small_tall, small_low, jagged]
    heights = [8.0, 30.0, 5.0, 26.0]
    bases = [ground(p) for p in polys]
    tops = [b + h for b, h in zip(bases, heights)]
    eaves = [tops[0], tops[1], tops[2] - 2.0, tops[3] - 4.0]
    content = LeafContent()
    # Two of them in tile (0, 0), two in tile (1, 1): their records are relative to their tiles.
    content.add({"BLD": _bld(polys[:3], [b - 1 for b in bases[:3]], tops[:3], eaves[:3], [0, 1, 2], 0, 0)}, 0, 0, T)
    content.add({"BLD": _bld(polys[3:], [b - 1 for b in bases[3:]], tops[3:], eaves[3:], [3], T, T)}, T, T, T)
    assert len(content.buildings) == 2

    def run(level: int) -> tuple[dict, float, list | None, int]:
        bounds = (0.0, 0.0, T * (1 << level), T * (1 << level))
        sections, err, zr, n = coarse_buildings(content, hf, level, bounds)
        return (decode_buildings(sections[0][1]) if sections else None), err, zr, n

    rec, err, zr, n = run(1)
    assert n == 3                                      # small_low (18 m2, 5 m) is left out
    assert 5.0 <= err <= 5.01                          # its height; roofs: (top - eave) / 2 <= 2; tolerance 2
    assert (rec["shape"] == 0).all() and np.array_equal(rec["eave"], np.repeat(rec["top"], rec["polygons"]))
    assert list(rec["material"]) == [0, 1, 3]
    assert rec["top"][2] == np.float32((eaves[3] + tops[3]) / 2)          # halfway up the roof
    assert rec["top"][0] == np.float32(tops[0])                            # flat roof unchanged
    # Floors at or below the coarse terrain around the footprint, and the level-0 floor.
    k = 2
    for b, p in zip(range(3), [big_low, small_tall, jagged]):
        x0, y0, x1, y1 = p.bounds
        z = hf.z[int(y0 // (R * k)) * k:int(math.ceil(y1 / (R * k))) * k + 1:k,
                 int(x0 // (R * k)) * k:int(math.ceil(x1 / (R * k))) * k + 1:k]
        assert rec["bottom"][b] <= z.min() + 1e-4 and rec["bottom"][b] <= [bases[0], bases[1], bases[3]][b] - 1 + 1e-4
    assert zr[0] == pytest.approx(float(rec["bottom"].min()), abs=1e-3) and zr[1] == pytest.approx(float(rec["top"].max()), abs=1e-3)
    # The courtyard survives; the bump below the tolerance (R * 2 / 2 = 2 m) does not.
    polys1 = rec.polygons(0, 0.0, 0.0)
    assert len(polys1[0].interiors) == 1 and polys1[0].area == pytest.approx(big_low.area)
    assert rec.polygons(2, 0.0, 0.0)[0].equals(box(70, 70, 90, 85))

    rec, err, zr, n = run(2)                           # >= 300 m2 or >= 25 m: big_low, small_tall, jagged
    assert n == 3 and err == pytest.approx(5.0, abs=0.01)
    rec, err, zr, n = run(3)                           # >= 2000 m2 or >= 40 m: none
    assert rec is None and n == 0 and zr is None and err == pytest.approx(max(heights), abs=0.01)
    assert BUILDING_THRESHOLDS[3] == (2000.0, 40.0)


def test_read_container_can_skip_sections():
    data = write_container(0, 1, 2, [("HGT", b"heights", True), ("BLD", b"buildings", True), ("GLB", b"glb", False)])
    assert read_container(data).sections == {"HGT": b"heights", "BLD": b"buildings", "GLB": b"glb"}
    assert read_container(data, only={"BLD", "TRE"}).sections == {"BLD": b"buildings"}
