"""Terrain built window by window equals the whole-grid build (phase 5a)."""

from __future__ import annotations

import numpy as np
import pytest
from affine import Affine
from scipy import ndimage

from worldgen.build import fixture_region
from worldgen.config import load_config
from worldgen.pipeline import bundle_from_fixture
from worldgen.sources.synthetic import load_fixture
from worldgen.spatial.raster import GeoRaster
from worldgen.spatial.region import Region
from worldgen.spatial.windows import box_filter, grid_windows
from worldgen.terrain.build import build_heightfield


@pytest.mark.parametrize("size", [3, 4, 7, 11])
def test_box_filter_is_a_uniform_filter(size):
    a = np.random.default_rng(size).normal(size=(30, 41))
    assert np.abs(box_filter(a, size) - ndimage.uniform_filter(a, size)).max() < 1e-12
    # Exact stencil: a window holding a node's box gives that node bit-identical values.
    sub = box_filter(a[5:25, 7:33], size)
    h = size // 2 + 1
    assert np.array_equal(sub[h:-h, h:-h], box_filter(a, size)[5 + h:25 - h, 7 + h:33 - h])


def test_grid_windows_cover_every_node_once():
    seen = np.zeros((70, 45), dtype=int)
    for win, core_w, core in grid_windows(seen.shape, 16, 5, align=2):
        assert all(s.start % 2 == 0 for s in win)
        assert (win[0].stop - win[0].start, win[1].stop - win[1].start) >= (core_w[0].stop, core_w[1].stop)
        seen[core] += 1
    assert (seen == 1).all()


def _region_and_cfg(name: str, overrides: list[str] | None = None):
    cfg = load_config("basic", overrides=overrides or [], region=fixture_region(name))
    return Region.from_config(cfg), cfg


@pytest.mark.parametrize("name", ["demo_town", "hillside_curve"])
def test_fixture_terrain_does_not_depend_on_windows(name):
    region, cfg = _region_and_cfg(name)
    b = bundle_from_fixture(load_fixture(name, region.center))
    whole = build_heightfield(region, cfg, b.dem, b.worldcover, b.dem_datum, window=10**9)
    for size in (23, 50):
        part = build_heightfield(region, cfg, b.dem, b.worldcover, b.dem_datum, window=size)
        assert np.array_equal(whole.z, part.z) and np.array_equal(whole.valid, part.valid)
        assert whole.meta == part.meta


def test_coarse_dem_with_nodata_and_dsm_correction():
    """A 30 m DEM with a nodata hole that does not cover the whole region, with built-up and
    tree cells: smoothing and DSM correction run, every node gets a height, coverage marks
    the hole and the uncovered border, and windows do not change anything."""
    region, cfg = _region_and_cfg("demo_town")
    x0, y0, x1, y1 = region.extent
    px = 30.0
    n = int((x1 - x0) / px) - 4                         # the raster stops short of the east edge
    m = int((y1 - y0) / px) + 2
    yy, xx = np.mgrid[0:m, 0:n].astype(float)
    data = (40.0 + 6.0 * np.sin(xx / 7.0) * np.cos(yy / 5.0) + 0.3 * xx).astype(np.float32)
    data[m // 2:m // 2 + 3, n // 3:n // 3 + 4] = np.nan
    dem = GeoRaster(data, Affine(px, 0.0, x0 - px, 0.0, -px, y1 + px), region.frame.local_crs(), name="dem")
    lc = np.where((xx.astype(int) // 5 + yy.astype(int) // 4) % 3 == 0, 50, 30).astype(np.uint8)
    wc = GeoRaster(lc, dem.transform, dem.crs, nodata=0, name="worldcover")
    whole = build_heightfield(region, cfg, dem, wc, "EGM2008", window=10**9)
    assert np.isfinite(whole.z).all()
    assert 0.5 < whole.meta["source_coverage"] < 1.0 and not whole.valid.all()
    assert whole.meta["dsm_correction"]["masked_fraction"] > 0.1
    for size in (17, 64):
        part = build_heightfield(region, cfg, dem, wc, "EGM2008", window=size)
        assert np.array_equal(whole.z, part.z) and np.array_equal(whole.valid, part.valid)
        assert whole.meta == part.meta
