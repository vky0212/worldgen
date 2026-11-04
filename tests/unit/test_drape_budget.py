"""Road draping onto terrain triangles, and instance budgets."""

from __future__ import annotations

import numpy as np
import pytest
from shapely.geometry import Polygon

from worldgen.budget import select_within_budget
from worldgen.config import BudgetConfig
from worldgen.geometry.drape import drape_polygons, drape_quads
from worldgen.terrain.heightfield import Heightfield


def _hilly() -> Heightfield:
    rng = np.random.default_rng(3)
    return Heightfield(0.0, 0.0, 4.0, rng.uniform(0, 12, (21, 21)))


def _area(m) -> float:  # noqa: ANN001
    t = m.positions[m.indices][:, :, :2]
    return float(0.5 * np.abs((t[:, 1, 0] - t[:, 0, 0]) * (t[:, 2, 1] - t[:, 0, 1])
                              - (t[:, 1, 1] - t[:, 0, 1]) * (t[:, 2, 0] - t[:, 0, 0])).sum())


def test_draped_quads_follow_terrain_exactly():
    hf = _hilly()
    quads = np.array([[[5, 5], [5, 13], [30, 15], [30, 7]], [[30, 7], [30, 15], [60, 40], [62, 33]]], float)
    m = drape_quads(quads, hf, 0.15)
    centroids = m.positions[m.indices].mean(axis=1)
    # Every triangle lies inside one terrain triangle, parallel to it, 15 cm above.
    assert np.allclose(centroids[:, 2] - hf.sample(centroids[:, 0], centroids[:, 1]), 0.15, atol=1e-9)
    assert np.allclose(m.positions[:, 2] - hf.sample(m.positions[:, 0], m.positions[:, 1]), 0.15, atol=1e-9)
    expected = sum(Polygon(q).area for q in quads)
    assert _area(m) == pytest.approx(expected, rel=1e-6)
    assert (m.normals[:, 2] > 0).all()
    # The general (shapely) path gives the same surface.
    g = drape_polygons([Polygon(q) for q in quads], hf, 0.15)
    assert _area(g) == pytest.approx(expected, rel=1e-6)


def test_concave_and_grouped_draping():
    hf = _hilly()
    bowtie_free_concave = np.array([[[10, 10], [40, 10], [20, 20], [10, 40]]], float)   # non-convex quad
    out = drape_quads(np.concatenate([bowtie_free_concave, bowtie_free_concave + 20]), hf, 0.1,
                      groups=np.array([7, 9]))
    assert set(out) == {7, 9}
    assert _area(out[7]) == pytest.approx(Polygon(bowtie_free_concave[0]).area, rel=1e-6)


def _tiles(n: int, per: int) -> np.ndarray:
    return np.array([f"{k // per}_0" for k in range(n)], dtype=object)


def test_budget_caps_tiles_and_prefers_large_items():
    rng = np.random.default_rng(0)
    n = 4000
    ids = [f"b{k}" for k in range(n)]
    sizes = rng.lognormal(3.0, 1.2, n)
    tiles = _tiles(n, 1000)
    res = select_within_budget(ids, tiles, sizes, np.zeros(n, bool), BudgetConfig(True, 300, 0), 1, "t")
    for t in set(tiles):
        assert res.keep[tiles == t].sum() == 300
    assert res.dropped == 4000 - 4 * 300 and len(res.tiles_over_budget) == 4
    assert np.median(sizes[res.keep]) > 3 * np.median(sizes[~res.keep])     # smaller dropped first
    small = sizes < np.percentile(sizes, 10)
    assert res.keep[small].mean() < 0.05
    mid = (sizes > np.percentile(sizes, 60)) & (sizes < np.percentile(sizes, 75))
    assert 0.05 < res.keep[mid].mean() < 0.95                                # random among similar sizes
    again = select_within_budget(ids, tiles, sizes, np.zeros(n, bool), BudgetConfig(True, 300, 0), 1, "t")
    assert np.array_equal(res.keep, again.keep)                              # deterministic
    alone = select_within_budget(ids[:1000], tiles[:1000], sizes[:1000], np.zeros(1000, bool),
                                 BudgetConfig(True, 300, 0), 1, "t")
    assert np.array_equal(alone.keep, res.keep[:1000])                       # per-tile independent


def test_budget_protection_total_and_disable():
    n = 500
    ids = [f"t{k}" for k in range(n)]
    sizes = np.linspace(1, 100, n)
    protected = np.zeros(n, bool)
    protected[:20] = True                                                    # tiny but observed
    res = select_within_budget(ids, _tiles(n, n), sizes, protected, BudgetConfig(True, 100, 0), 2, "x")
    assert res.keep[:20].all() and res.keep.sum() == 100
    res = select_within_budget(ids, _tiles(n, n), sizes, protected, BudgetConfig(True, 0, 60), 2, "x")
    assert res.keep.sum() == 60
    res = select_within_budget(ids, _tiles(n, n), sizes, protected, BudgetConfig(False, 10, 10), 2, "x")
    assert res.keep.all() and res.dropped == 0
