"""Smoothing computed window by window gives exactly the whole-grid result (phase 5a)."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import box

import worldgen.pipeline as pipeline
from worldgen.build import fixture_region
from worldgen.config import SmoothingConfig, load_config
from worldgen.landcover.smoothing import (
    MAX_FILL_M,
    VEGETATION_CHANNELS,
    WATER_CODE,
    _nearest,
    refine_vegetation,
    smooth_landcover,
)
from worldgen.pipeline import bundle_from_fixture, compile_world
from worldgen.sources.synthetic import load_fixture
from worldgen.spatial.region import Region
from worldgen.terrain.heightfield import Heightfield


def test_nearest_breaks_ties_by_offset():
    mask = np.zeros((9, 9), dtype=bool)
    mask[2, 4] = mask[6, 4] = mask[4, 2] = mask[4, 6] = True     # four cells at distance 2
    sel = np.zeros_like(mask)
    sel[4, 4] = True
    ir, ic, sr, sc = _nearest(mask, sel)
    assert (ir.tolist(), ic.tolist(), sr.tolist(), sc.tolist()) == ([2], [4], [4], [4])   # smallest row offset
    mask[2, 4] = False
    ir, ic, *_ = _nearest(mask, sel)
    assert (ir.tolist(), ic.tolist()) == ([4], [2])                    # same row: smallest column offset
    ir, ic, *_ = _nearest(mask, sel, max_cells=1.5)                    # none within 1.5 cells
    assert len(ir) == 0
    # The same cell from a window holding both (indices shifted by the window origin).
    ir, ic, *_ = _nearest(mask[1:8, 1:8], sel[1:8, 1:8])
    assert ((ir + 1).tolist(), (ic + 1).tolist()) == ([4], [2])


def test_far_stray_water_stays_water():
    """A WorldCover water patch touching mapped water is reassigned to the nearest land class
    within MAX_FILL_M of land; farther out it keeps its water class."""
    res = 4.0
    n = 120
    hf = Heightfield(0.0, 0.0, res, np.zeros((n, n)))
    codes = np.full((n, n), 3, dtype=np.uint8)            # land: class 3
    codes[:, 20:] = WATER_CODE                            # WorldCover water east of x = 80 m
    mapped = box(400.0, 0.0, 480.0, 480.0)                # mapped water only in the far east
    out = smooth_landcover(codes, hf, [mapped], [], SmoothingConfig(), seed=1)
    x = np.arange(n) * res
    row = out.codes[60]
    near = (x > 80.0 + 8.0) & (x < 80.0 + MAX_FILL_M - 8.0)
    far = (x > 80.0 + MAX_FILL_M + 8.0) & (x < 400.0 - 8.0)
    assert (row[near] == 3).all()                         # reassigned to the adjacent land class
    assert (row[far] == WATER_CODE).all()                 # too far from land: stays water
    assert (out.codes[:, x >= 404.0] == WATER_CODE).all()


@pytest.mark.parametrize("name", ["demo_town", "forest"])
def test_windows_give_the_whole_grid_result(monkeypatch, name):
    cfg = load_config("basic", region=fixture_region(name))
    region = Region.from_config(cfg)
    calls: list[str] = []

    def landcover(*a, **k):
        whole = smooth_landcover(*a, window=10**9)
        for size in (24, 61):
            part = smooth_landcover(*a, window=size)
            assert np.array_equal(whole.codes, part.codes) and np.array_equal(whole.colors, part.colors)
            assert whole.stats == part.stats
        calls.append("landcover")
        return whole

    def vegetation(*a, **k):
        whole = refine_vegetation(*a, window=10**9)
        for size in (24, 61):
            part = refine_vegetation(*a, window=size)
            for ch in ("category",) + VEGETATION_CHANNELS:
                assert np.array_equal(getattr(whole, ch), getattr(part, ch), equal_nan=True), ch
            assert whole.meta["smoothing"] == part.meta["smoothing"]
        calls.append("vegetation")
        return whole

    monkeypatch.setattr(pipeline, "smooth_landcover", landcover)
    monkeypatch.setattr(pipeline, "refine_vegetation", vegetation)
    compile_world(region, cfg, bundle_from_fixture(load_fixture(name, region.center)), place_trees=False)
    assert calls == ["landcover", "vegetation"]
