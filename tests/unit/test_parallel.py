"""Worker processes: stages give the same results in workers as in this process (phase 5b)."""

from __future__ import annotations

import operator

import numpy as np
import pytest

from worldgen import parallel
from worldgen.build import fixture_region
from worldgen.config import load_config
from worldgen.landcover import smoothing
from worldgen.pipeline import bundle_from_fixture, compile_world
from worldgen.sources.synthetic import load_fixture
from worldgen.spatial import windows
from worldgen.spatial.region import Region
from worldgen.world.world import CanonicalWorld


def test_spans_and_batches_keep_order_and_spread_over_workers(tmp_path, monkeypatch):
    assert parallel.spans(10, 4) == [(0, 10)] and parallel.spans(0) == []
    assert parallel.batches([1, 2, 3], [5, 1, 1]) == [[1, 2, 3]] and parallel.batches([], []) == []
    monkeypatch.setattr(parallel, "_ACTIVE", parallel.Pool(3, tmp_path))   # no processes started
    spans = parallel.spans(1000, 10)
    assert len(spans) == 12 and spans[0][0] == 0 and spans[-1][1] == 1000
    assert all(a[1] == b[0] for a, b in zip(spans, spans[1:]))
    assert len(parallel.spans(50, 10)) == 5
    weights = [1.0] * 30 + [10.0] * 10
    batches = parallel.batches(list(range(40)), weights)
    assert [x for b in batches for x in b] == list(range(40)) and 1 < len(batches) <= 12
    sums = [sum(weights[x] for x in b) for b in batches]
    assert max(sums) <= sum(weights) / 12 + 10.0                         # within one item of a share


def _compile(name: str) -> CanonicalWorld:
    cfg = load_config("basic", region=fixture_region(name))
    region = Region.from_config(cfg)
    return compile_world(region, cfg, bundle_from_fixture(load_fixture(name, region.center)), place_trees=False)


def _grids(w: CanonicalWorld) -> dict[str, np.ndarray]:
    g = w.vegetation_grid
    out = {"z": w.terrain.z, "valid": w.terrain.valid, "codes": w.terrain_codes, "rgba": w.terrain_rgba}
    out.update({f"veg.{k}": getattr(g, k) for k in ("category",) + smoothing.VEGETATION_CHANNELS})
    return out


@pytest.mark.slow
def test_pool_runs_tasks_and_compile_stages_like_this_process(tmp_path, monkeypatch):
    """Two spawned workers with small windows, row blocks and shore batches, so every
    windowed compile stage hands several tasks to them: the compiled grids are bit-identical
    to the build in this process, and the work folder is removed."""
    monkeypatch.setattr(windows, "POOL_WINDOW_NODES", 48)
    monkeypatch.setattr(windows, "POOL_ROW_BLOCK", 16)
    monkeypatch.setattr(smoothing, "SHORE_BATCH", 32)
    here = _compile("demo_town")
    data = np.arange(300_000, dtype=np.float64)        # 2.4 MB: memory-mapped by the workers
    items = [slice(k, k + 999) for k in range(0, 300_000, 20_000)]
    with parallel.Pool(2, tmp_path / "work") as pool:
        assert parallel.active() is pool
        got = list(parallel.tasks(operator.getitem, data, items))
        tasks_before = pool.tasks_run
        pooled = _compile("demo_town")
        stats = pool.stats()
    assert parallel.active() is None and not (tmp_path / "work").exists()
    assert all(np.array_equal(g, data[s]) for g, s in zip(got, items))
    assert tasks_before == len(items) and stats["tasks"] > tasks_before + 40 and stats["worker_peak_mb"] > 0
    a, b = _grids(here), _grids(pooled)
    for k in a:
        assert a[k].dtype == b[k].dtype and np.array_equal(a[k], b[k], equal_nan=a[k].dtype.kind == "f"), k
    assert pooled.terrain.meta == here.terrain.meta and pooled.vegetation_grid.meta == here.vegetation_grid.meta
    assert {k: v for k, v in pooled.stats.items()} == here.stats
