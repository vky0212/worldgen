"""Vegetation extension: classification, placement rules, determinism and tiling."""

from __future__ import annotations

import numpy as np

from worldgen.pipeline import vegetation_exclusions
from worldgen.tiling.grid import TileGrid
from worldgen.vegetation.classify import CAT, TREE_CATEGORIES
from worldgen.vegetation.placement import place_trees


def test_forest_classification_and_regions(compiled):
    w = compiled("forest")
    grid = w.vegetation_grid
    rr, cc, _ = grid.cell_of(np.array([0.0]), np.array([0.0]))
    assert grid.category[rr[0], cc[0]] == CAT["dense_forest"]
    assert grid.canopy_observed[rr[0], cc[0]]                       # canopy height from the CHM raster
    assert 10.0 < grid.canopy_height[rr[0], cc[0]] < 19.0
    cats = {r.properties["category"] for r in w.vegetation_regions}
    assert "dense_forest" in cats and "grassland" in cats
    forest = max((r for r in w.vegetation_regions if r.properties["category"] == "dense_forest"),
                 key=lambda r: r.geometry.area)
    assert forest.properties["coverage_fraction"] > 0.5
    assert forest.properties["sources"]                           # source attribution kept


def test_tree_rules(compiled):
    w = compiled("forest")
    tr = w.trees
    gen = np.array([s == "procedural" for s in tr.source]) & np.isin(
        np.array(tr.archetype, dtype=object), ["broadleaf", "conifer", "palm"])
    assert gen.sum() > 200
    grid = w.vegetation_grid
    rr, cc, inside = grid.cell_of(tr.x[gen], tr.y[gen])
    assert inside.all() and np.isin(grid.category[rr, cc], [CAT[c] for c in TREE_CATEGORIES]).all()
    excl = vegetation_exclusions(w, 0.0)
    assert not excl.contains(tr.x[gen], tr.y[gen]).any()          # not on roads, buildings or water
    assert np.allclose(tr.ground_z, w.terrain.sample(tr.x, tr.y))   # on the terrain surface
    assert len(set(tr.ids)) == len(tr.ids)
    assert not tr.observed[gen].any()                               # generated = inferred
    # Mapped trees are kept at their exact positions and flagged as observed.
    for ot in w.observed_trees:
        k = tr.ids.index(ot.id)
        assert tr.x[k] == ot.x and tr.y[k] == ot.y
        assert bool(tr.observed[k]) is (not ot.from_row)
    obs = [k for k, s in enumerate(tr.ids) if s.startswith("tree:osm:node/")]
    assert len(obs) == 3 and all(tr.height[k] in (12.0, 13.0, 14.0) for k in obs)
    # Height follows canopy height where it was measured.
    meas = grid.canopy_observed[rr, cc] & (grid.canopy_height[rr, cc] > 3)
    ratio = np.median(tr.height[gen][meas] / grid.canopy_height[rr, cc][meas])
    assert 0.8 < ratio < 1.25


def test_no_regular_grid_pattern(compiled):
    w = compiled("forest")
    tr = w.trees
    gen = np.array([a == "broadleaf" or a == "conifer" for a in tr.archetype]) & ~tr.observed
    x = tr.x[gen]
    frac = np.mod(x, w.config.vegetation.cell_size) / w.config.vegetation.cell_size
    hist, _ = np.histogram(frac, bins=8)
    assert hist.min() > 0.5 * hist.mean()        # positions uniform within sampling cells


def test_tile_independent_and_deterministic(compiled):
    w = compiled("forest")
    grid = TileGrid(w.config.tiling.tile_size)
    excl = vegetation_exclusions(w, w.config.vegetation.exclusion_buffer)
    full = place_trees(w.vegetation_grid, w.terrain, excl, w.observed_trees, w.config.vegetation)
    again = place_trees(w.vegetation_grid, w.terrain, excl, w.observed_trees, w.config.vegetation)
    assert full.ids == again.ids and np.array_equal(full.x, again.x) and np.array_equal(full.height, again.height)
    tiles = sorted(set(grid.tiles_of(full.x, full.y)))
    assert len(tiles) >= 4
    collected = []
    for t in tiles:
        part = place_trees(w.vegetation_grid, w.terrain, excl, w.observed_trees, w.config.vegetation,
                           bounds=grid.bounds(t))
        collected += part.ids
        idx = [full.ids.index(i) for i in part.ids]
        assert np.array_equal(full.x[idx], part.x) and np.array_equal(full.height[idx], part.height)
    assert sorted(collected) == sorted(full.ids)   # union of tiles == global run, no duplicates


def test_changed_seed_changes_trees(compiled):
    import dataclasses

    w = compiled("forest")
    cfg = dataclasses.replace(w.config.vegetation, seed=99)
    excl = vegetation_exclusions(w, cfg.exclusion_buffer)
    other = place_trees(w.vegetation_grid, w.terrain, excl, w.observed_trees, cfg)
    base = place_trees(w.vegetation_grid, w.terrain, excl, w.observed_trees, w.config.vegetation)
    assert other.ids != base.ids


def test_budgets_drop_small_buildings_and_trees_but_keep_semantics():
    from conftest import fixture_config

    from worldgen.export.tiles import generate_pieces, world_tiles
    from worldgen.pipeline import bundle_from_fixture, compile_world
    from worldgen.sources.synthetic import load_fixture
    from worldgen.spatial.region import Region

    cfg = fixture_config("demo_town", ["buildings.budget.max_per_tile=2", "vegetation.budget.max_per_tile=150"])
    region = Region.from_config(cfg)
    w = compile_world(region, cfg, bundle_from_fixture(load_fixture("demo_town", region.center)))
    grid = TileGrid(cfg.tiling.tile_size)
    dropped = [b for b in w.buildings if b.budget_dropped]
    kept = [b for b in w.buildings if not b.budget_dropped]
    assert dropped and kept and w.stats["buildings_budget_dropped"] == len(dropped)
    pieces = generate_pieces(w, set(world_tiles(w)))
    drawn = {p.feature_id for ps in pieces.values() for p in ps if p.layer == "buildings"}
    assert drawn == {b.id for b in kept}
    tr = w.trees
    rendered = tr.rendered_mask()
    tiles = grid.tiles_of(tr.x, tr.y)
    assert all(rendered[tiles == t].sum() <= 150 for t in set(tiles))
    assert rendered[tr.observed].all()                                    # mapped trees protected
    assert len(tr) > rendered.sum()                                       # semantic table keeps all
    assert any(i["check"] == "instance_budget" for i in w.stage_issues)
