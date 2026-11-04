"""Vectorised vegetation placement: portable hash, Poisson draws, radii and window independence."""

from __future__ import annotations

import math

import numpy as np
import pytest
from scipy import stats

from worldgen.config import VegetationConfig
from worldgen.hashing import hash_u64, hash_unit
from worldgen.terrain.heightfield import Heightfield
from worldgen.vegetation.classify import CAT, VegetationGrid
from worldgen.vegetation.placement import (
    ARCHETYPES,
    Exclusions,
    _archetype_tables,
    _leaf_adjusted,
    crown_radius,
    crown_radius_codes,
    place_trees,
    poisson_icdf,
    trunk_radius,
    trunk_radius_codes,
)

_M = (1 << 64) - 1


def _reference_hash(*words: int) -> int:
    """The published spec, in plain Python integers."""
    h = 0
    for w in words:
        h = ((h ^ (w & _M)) + 0x9E3779B97F4A7C15) & _M
        h ^= h >> 30
        h = (h * 0xBF58476D1CE4E5B9) & _M
        h ^= h >> 27
        h = (h * 0x94D049BB133111EB) & _M
        h ^= h >> 31
    return h


def test_hash_test_vectors():
    # Fixed vectors for other implementations (the viewer's generators); hash_u64(0) is the
    # first splitmix64 output for seed 0.
    assert int(hash_u64(0)[0]) == 0xE220A8397B1DCDAF
    assert int(hash_u64(1337, 1, 0, 0, 0)[0]) == 0x1CD696F626CE01C3
    assert int(hash_u64(1337, 2, -3, 5, 7, 6)[0]) == 0x90C054CD94FB42C3
    assert int(hash_u64(2**63 + 5, -1)[0]) == 0xAD5616BDEDDAD707


def test_hash_matches_reference_and_broadcasts():
    rng = np.random.default_rng(1)
    i = rng.integers(-5000, 5000, 200)
    j = rng.integers(-5000, 5000, 200)
    got = hash_u64(1337, 2, i, j, 7, 3)
    assert got.shape == (200,) and got.dtype == np.uint64
    assert [int(v) for v in got] == [_reference_hash(1337, 2, int(a), int(b), 7, 3) for a, b in zip(i, j)]
    u = hash_unit(42, 5, i[:, None], j[None, :])
    assert u.shape == (200, 200) and u.min() >= 0.0 and u.max() < 1.0
    assert abs(u.mean() - 0.5) < 0.01


def test_poisson_icdf_matches_scipy():
    rng = np.random.default_rng(3)
    u = rng.random(20000)
    for lam in (0.05, 2.56, 17.6, 80.0):
        n = poisson_icdf(u, lam)
        ref = stats.poisson.ppf(u, lam).astype(np.int64)
        assert np.mean(n == ref) > 0.999                 # only ulp-level boundary cases may differ
        assert abs(n.mean() - lam) < 4 * math.sqrt(lam / len(u)) + 1e-3
    lam = rng.uniform(0.0, 50.0, 1000)
    assert np.array_equal(poisson_icdf(np.zeros(1000), lam), np.zeros(1000, dtype=np.int64))
    assert poisson_icdf(np.full(3, 1.0 - 2.0**-53), 30.0).max() < 200  # terminates near u = 1


@pytest.mark.parametrize("code", range(len(ARCHETYPES)))
def test_radius_codes_match_scalar_functions(code):
    h = np.linspace(0.0, 70.0, 301)
    name = ARCHETYPES[code]
    assert np.array_equal(crown_radius_codes(np.full(len(h), code), h), crown_radius(name, h))
    assert np.array_equal(trunk_radius_codes(np.full(len(h), code), h), trunk_radius(name, h))


def test_leaf_adjusted_matches_searchsorted():
    cum, _ = _archetype_tables(VegetationConfig())
    rng = np.random.default_rng(5)
    cat = rng.integers(0, len(cum), 5000)
    u = rng.random(5000)
    u[:50] = cum[cat[:50], 0]                          # exactly on a cumulative weight
    leaf = rng.integers(0, 4, 5000)
    ref = np.clip([np.searchsorted(cum[c], x, side="right") for c, x in zip(cat, u)], 0, len(ARCHETYPES) - 1)
    ref = np.where(np.isin(ref, [0, 1]) & (leaf == 1), 0, ref)
    ref = np.where(np.isin(ref, [0, 1]) & (leaf == 2), 1, ref)
    assert np.array_equal(_leaf_adjusted(u, cat, leaf, cum), ref)


def _uniform_forest(size: float = 640.0, res: float = 4.0, coverage: float = 0.85, height: float = 16.0
                    ) -> tuple[VegetationGrid, Heightfield]:
    n = int(size / res)
    shape = (n, n)
    grid = VegetationGrid(
        xmin=0.0, ymin=0.0, resolution=res, category=np.full(shape, CAT["dense_forest"], dtype=np.uint8),
        source=np.ones(shape, dtype=np.uint8), canopy_height=np.full(shape, height, dtype=np.float32),
        canopy_observed=np.ones(shape, dtype=bool), coverage=np.full(shape, coverage, dtype=np.float32),
        confidence=np.full(shape, 0.8, dtype=np.float32), tree_fraction_wc=np.ones(shape, dtype=np.float32),
        leaf_type=np.zeros(shape, dtype=np.uint8))
    hf = Heightfield(-8.0, -8.0, res, np.zeros((n + 5, n + 5)))
    return grid, hf


def test_stem_density_matches_target():
    """Matern II with the thinned candidate intensity keeps the target stem density.

    One archetype, so every candidate has the same hard core and the Matern II retention
    (1 - exp(-lam0 a)) / a is exact (mixed archetypes have smaller conifer cores).
    """
    cfg = VegetationConfig()
    cfg.categories["dense_forest"].archetypes = {"broadleaf": 1.0}
    grid, hf = _uniform_forest()
    trees = place_trees(grid, hf, Exclusions(None), [], cfg, bounds=(64.0, 64.0, 576.0, 576.0))
    rc = float(crown_radius("broadleaf", 16.0))
    spacing = cfg.categories["dense_forest"].spacing_factor
    target = min(0.85 / (math.pi * rc ** 2), 0.95 / (math.pi * (spacing * rc) ** 2))
    density = len(trees) / (512.0 * 512.0)
    assert abs(density / target - 1.0) < 0.05, (density, target)


def test_window_independence(compiled):
    world = compiled("demo_town")
    from worldgen.pipeline import vegetation_exclusions

    cfg = world.config.vegetation
    excl = vegetation_exclusions(world, cfg.exclusion_buffer)
    full = place_trees(world.vegetation_grid, world.terrain, excl, world.observed_trees, cfg)
    rng = np.random.default_rng(11)
    g = world.vegetation_grid
    x1, y1 = g.xmin + g.shape[1] * g.resolution, g.ymin + g.shape[0] * g.resolution
    for _ in range(25):
        w, h = rng.uniform(20.0, 300.0, 2)
        bx, by = rng.uniform(g.xmin, x1 - w), rng.uniform(g.ymin, y1 - h)
        part = place_trees(g, world.terrain, excl, world.observed_trees, cfg, bounds=(bx, by, bx + w, by + h))
        inside = (full.x >= bx) & (full.x < bx + w) & (full.y >= by) & (full.y < by + h)
        expect = {i: (x, y) for i, x, y, m in zip(full.ids, full.x, full.y, inside) if m}
        got = dict(zip(part.ids, zip(part.x, part.y)))
        assert got == expect
