"""Post-processing stage: smooth land-cover borders, shorelines and vegetation edges."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Point, box

from worldgen.config import SmoothingConfig
from worldgen.landcover.smoothing import (
    WATER_CODE,
    _Warp,
    code_lut,
    refine_vegetation,
    shape_shorelines,
    smooth_classes,
    smooth_landcover,
    value_noise,
    warp_offsets,
)
from worldgen.terrain.build import carve_polygons, rasterize_mask
from worldgen.terrain.heightfield import Heightfield
from worldgen.vegetation.classify import CAT, VegetationGrid


def test_value_noise_is_world_space_deterministic_and_smooth():
    x = np.linspace(-500.0, 500.0, 2001)
    y = np.full_like(x, 123.4)
    a = value_noise(x, y, 60.0, 7)
    # Same world points evaluated in a different order / grid give the same values.
    b = value_noise(x[::-1], y, 60.0, 7)[::-1]
    assert np.array_equal(a, b)
    assert a.min() >= -1.0 and a.max() <= 1.0
    assert np.abs(np.diff(a)).max() < 0.05          # 0.5 m steps on a 60 m wavelength
    assert not np.allclose(a, value_noise(x, y, 60.0, 8))
    dx, dy = warp_offsets(x, y, 5.0, 60.0, 7)
    assert np.abs(dx).max() <= 5.0 and np.abs(dy).max() <= 5.0 and np.abs(dx).max() > 1.0


def _border_rows(codes: np.ndarray, cls: int) -> np.ndarray:
    """Per column, the number of rows of class ``cls`` (it fills each column from the bottom)."""
    return (codes == cls).sum(axis=0).astype(float)


def test_smooth_classes_removes_stairs_and_specks():
    res = 2.0
    n = 160
    rr, cc = np.mgrid[0:n, 0:n]
    # A diagonal border drawn with 10 m (5 cell) source pixels, plus a one-pixel speck.
    codes = np.where(rr // 5 < cc // 5, 1, 2).astype(np.uint8)
    codes[100:105, 20:25] = 3
    every = np.ones(codes.shape, dtype=bool)
    out, _ = smooth_classes(codes, every, every, res, sigma_m=6.0)
    assert not (out == 3).any()
    sel = slice(30, 130)
    cols = np.arange(n)[sel]

    def residual(c: np.ndarray) -> float:
        h = _border_rows(c, 2)[sel]
        fit = np.polyval(np.polyfit(cols, h, 1), cols)
        return float(np.abs(h - fit).max())

    before = residual(np.where(rr // 5 < cc // 5, 1, 2))
    after = residual(out)
    assert before >= 2.0
    assert after <= 0.35 * before


def test_smooth_classes_fills_holes_and_keeps_fixed_nodes():
    codes = np.full((40, 40), 3, dtype=np.uint8)
    codes[:, 20:] = 4
    codes[10:30, 5:15] = WATER_CODE          # fixed (water)
    hole = np.zeros(codes.shape, dtype=bool)
    hole[5:8, 30:33] = True                  # to fill from the surroundings
    land = (codes != WATER_CODE) & ~hole
    out, _ = smooth_classes(codes, land, land | hole, 4.0, 6.0)
    assert (out[10:30, 5:15] == WATER_CODE).all()
    assert (out[hole] == 4).all()


def _plane(n: int, res: float, z: float | np.ndarray) -> Heightfield:
    zz = np.broadcast_to(np.asarray(z, dtype=float), (n, n)).copy()
    return Heightfield(-n * res / 2.0, -n * res / 2.0, res, zz)


def test_shorelines_meet_the_water_surface_on_the_outline():
    res = 4.0
    n = 81
    hf = _plane(n, res, 0.0)
    xs, ys = hf.node_coords()
    hf.z = 3.0 + 0.02 * xs                       # gently sloping shore, 1.4 m .. 4.6 m
    lake = Point(3.0, -2.0).buffer(70.0, 256)
    surface, bed = 2.5, 0.5
    plain = hf.copy()
    carve_polygons(plain, [(lake, bed)])
    shape_shorelines(hf, [(lake, surface, bed)], box(hf.xmin, hf.ymin, hf.xmax, hf.ymax), 0.03, 1.0)
    ring = np.asarray(lake.exterior.coords)
    err = np.abs(hf.sample(ring[:, 0], ring[:, 1]) - surface)
    err_plain = np.abs(plain.sample(ring[:, 0], ring[:, 1]) - surface)
    assert err.max() < 0.12
    assert err_plain.max() > 1.0                  # plain node-by-node carving zig-zags
    inner = np.asarray(lake.buffer(-1.0).exterior.coords)
    outer = np.asarray(lake.buffer(1.0).exterior.coords)
    assert (hf.sample(inner[:, 0], inner[:, 1]) < surface).all()
    assert (hf.sample(outer[:, 0], outer[:, 1]) > surface).all()
    # Deep inside: carved to the bed; far outside: untouched.
    assert hf.sample(3.0, -2.0) == pytest.approx(bed)
    far = (xs - 3.0) ** 2 + (ys + 2.0) ** 2 > (70.0 + 6 * res) ** 2
    assert np.array_equal(hf.z[far], (3.0 + 0.02 * xs)[far])


def test_region_edge_is_not_a_shoreline():
    res = 4.0
    hf = _plane(41, res, 3.0)
    extent = box(hf.xmin, hf.ymin, hf.xmax, hf.ymax)
    sea = box(hf.xmin, hf.ymin, 0.0, hf.ymax)      # the region edge cuts the sea on three sides
    shape_shorelines(hf, [(sea, 0.05, -1.5)], extent, 0.03, 1.0)
    xs, _ = hf.node_coords()
    deep = xs < -6 * res
    assert np.allclose(hf.z[deep], -1.5)
    # The waterline runs along x = 0 (mid-cell) everywhere but at the region's corners.
    ys = np.linspace(hf.ymin + 3 * res, hf.ymax - 3 * res, 50)
    assert np.allclose(hf.sample(np.zeros(50), ys), 0.05, atol=0.05)


def test_smooth_landcover_snaps_stray_worldcover_water_and_blends_colours():
    res = 4.0
    n = 80
    hf = _plane(n, res, 0.0)
    codes = np.full((n, n), 3, dtype=np.uint8)       # grass
    codes[:, 50:] = 1                                 # forest
    x0, y0 = hf.xmin + 40.0, hf.ymin + 40.0
    lake = box(x0, y0, x0 + 80.0, y0 + 80.0)
    xs, ys = hf.node_coords()
    in_lake = rasterize_mask(hf, [lake])
    codes[in_lake] = WATER_CODE
    # WorldCover sees a slightly larger lake than the mapped outline.
    stray = ~in_lake & (xs > x0 - 12.5) & (xs < x0 + 92.5) & (ys > y0 - 12.5) & (ys < y0 + 92.5)
    codes[stray] = WATER_CODE
    far = (xs > hf.xmin + 200) & (xs < hf.xmin + 220) & (ys > hf.ymin + 250) & (ys < hf.ymin + 270)
    codes[far] = WATER_CODE                                          # unmapped pond
    cfg = SmoothingConfig(warp_amplitude_m=0.0)
    res_ = smooth_landcover(codes, hf, [lake], [], cfg, 1)
    assert res_.stats["water_mismatch_nodes"] == int(stray.sum())
    assert (res_.codes[stray] == 3).all()
    assert (res_.codes[in_lake] == WATER_CODE).all()
    assert (res_.codes[far] == WATER_CODE).all()
    rgba = res_.colors
    assert rgba.shape == (n, n, 4) and np.isfinite(rgba).all() and (rgba[..., 3] == 1).all()
    lut = code_lut()
    # Grass -> forest: a gradual transition instead of a switch between two nodes.
    row = rgba[60, 40:60, 1]
    assert row[0] == pytest.approx(lut[3, 1], abs=1e-3) and row[-1] == pytest.approx(lut[1, 1], abs=1e-3)
    assert np.all(np.diff(row) <= 1e-6) and np.sum(np.abs(np.diff(row)) > 1e-3) >= 3
    # Submerged nodes next to the bank are wet ground; far out the bed colour.
    r, c = 20, 11                                   # first submerged node row/col of the lake
    assert in_lake[r, c] and not in_lake[r, c - 1]
    assert rgba[r, c, :3].sum() < 0.9 * lut[3].sum()
    assert np.allclose(rgba[20, 20, :3], lut[WATER_CODE], atol=1e-3)


def _veg_grid(res: float, n: int) -> VegetationGrid:
    rr, cc = np.mgrid[0:n, 0:n]
    cat = np.where((rr // 2) < (cc // 2), CAT["dense_forest"], CAT["grassland"]).astype(np.uint8)
    forest = cat == CAT["dense_forest"]
    f32 = np.float32
    return VegetationGrid(
        0.0, 0.0, res, cat, np.full(cat.shape, 2, dtype=np.uint8),
        np.where(forest, 16.0, 0.4).astype(f32), np.zeros(cat.shape, dtype=bool),
        np.where(forest, 0.85, 0.0).astype(f32), np.full(cat.shape, 0.55, dtype=f32),
        np.where(forest, 1.0, 0.0).astype(f32), np.zeros(cat.shape, dtype=np.uint8))


def test_refine_vegetation_smooths_and_thins_forest_edges():
    g = _veg_grid(8.0, 48)
    pond = box(300.0, 20.0, 340.0, 60.0)
    out = refine_vegetation(g, [pond], SmoothingConfig(), 5)
    assert out.shape == (96, 96) and out.resolution == 4.0
    assert out.xmin == g.xmin and out.ymin == g.ymin
    forest = out.category == CAT["dense_forest"]
    # Channels follow the (new) category.
    assert np.allclose(out.canopy_height[forest], 16.0)
    assert np.allclose(out.canopy_height[out.category == CAT["grassland"]], 0.4)
    # Pond cells carry no vegetation.
    x, y = out.cell_centres()
    in_pond = (x > 300) & (x < 340) & (y > 20) & (y < 60)
    assert (out.category[in_pond] == 0).all()
    # Density fades towards the forest edge.
    cov = out.coverage
    assert cov[forest].max() == pytest.approx(0.85, abs=1e-4)
    edge = np.zeros(forest.shape, dtype=bool)
    edge[:-1] = forest[:-1] & ~forest[1:]            # forest fills each column from the bottom
    interior = np.zeros(forest.shape, dtype=bool)
    interior[:-8] = forest[:-8] & forest[8:]
    assert np.median(cov[edge]) < 0.6
    assert np.median(cov[interior & ~np.isin(np.arange(96), [0])[:, None]]) == pytest.approx(0.85, abs=1e-3)
    assert out.meta["smoothing"]["upsample"] == 2
    # Deterministic.
    again = refine_vegetation(g, [pond], SmoothingConfig(), 5)
    assert np.array_equal(again.category, out.category) and np.array_equal(again.coverage, out.coverage)


def test_identity_warp_is_a_noop():
    a = np.arange(12, dtype=np.float32).reshape(3, 4)
    assert _Warp(None, None).apply(a) is a
