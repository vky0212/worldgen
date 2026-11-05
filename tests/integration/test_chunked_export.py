"""Chunked export: outputs do not depend on export.chunk_tiles (trees are placed per chunk)."""

from __future__ import annotations

import hashlib
import json
from pathlib import Path

import geopandas as gpd
import pytest

from worldgen.build import BuildOptions, build, fixture_region
from worldgen.config import load_config

pytestmark = pytest.mark.slow

# (export.chunk_tiles, parallel.workers): the whole world in this process, one tile per chunk,
# squares straddling the fixture's tiles, and chunks spread over two worker processes.
CASES = [(0, 1), (1, 1), (3, 1), (2, 2)]


def _build(out: Path, mode: str, chunk: int, workers: int, cap: int):  # noqa: ANN202
    cfg = load_config("basic", overrides=[f"export.mode={mode}", f"export.chunk_tiles={chunk}",
                                          f"parallel.workers={workers}", f"vegetation.budget.max_total={cap}"],
                      region=fixture_region("demo_town"))
    return build(cfg, BuildOptions(out, fixture="demo_town"))


def _digests(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}


@pytest.mark.parametrize("mode,cap", [("baked", 200_000), ("data", 4_000)])
def test_outputs_do_not_depend_on_chunks_or_workers(tmp_path, mode, cap):
    """Every file but trees.parquet (rows grouped by chunk) is byte-identical whatever the
    chunking and the number of worker processes, including a world-wide tree budget that
    drops trees (4,000 of demo_town's ~13,000), which chunks apply through a key floor."""
    results = {c: _build(tmp_path / f"c{c[0]}w{c[1]}", mode, *c, cap) for c in CASES}
    ref_dir = tmp_path / "c0w1"
    ref = _digests(ref_dir)
    assert any(k.startswith("tiles/") for k in ref) and "vectors/trees.parquet" in ref
    stats = results[CASES[0]].manifest["statistics"]
    assert stats["trees"] > cap or cap > stats["trees"]
    if cap < stats["trees"]:
        assert stats["trees_rendered"] == cap
    for c in CASES[1:]:
        got = _digests(tmp_path / f"c{c[0]}w{c[1]}")
        assert set(got) == set(ref)
        differ = sorted(k for k in ref if got[k] != ref[k])
        assert set(differ) <= {"vectors/trees.parquet"}, (c, differ)
        a = gpd.read_parquet(ref_dir / "vectors" / "trees.parquet").sort_values("id").reset_index(drop=True)
        b = gpd.read_parquet(tmp_path / f"c{c[0]}w{c[1]}" / "vectors" / "trees.parquet").sort_values("id")
        assert a.equals(b.reset_index(drop=True))
        for key in ("trees", "trees_observed", "trees_rendered"):
            assert results[c].manifest["statistics"][key] == stats[key] > 0
        assert results[c].manifest["validation"] == results[CASES[0]].manifest["validation"]
    assert not (tmp_path / "c2w2" / ".work").exists()            # the worker snapshot is removed
    diag = json.loads((tmp_path / "c1w1" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diag["summary"]["checks"]["vegetation_placement"]["status"] == "pass"
    assert diag["summary"]["checks"]["vegetation_tile_independence"]["status"] == "pass"


def test_trees_are_released_unless_kept(tmp_path):
    cfg = load_config("basic", overrides=["export.chunk_tiles=1", "vegetation.budget.max_total=0"],
                      region=fixture_region("demo_town"))
    plain = build(cfg, BuildOptions(tmp_path / "a", fixture="demo_town"))
    kept = build(cfg, BuildOptions(tmp_path / "b", fixture="demo_town", keep_trees=True))
    assert len(plain.world.trees) == 0 and not plain.world.trees_deferred
    assert len(kept.world.trees) == kept.manifest["statistics"]["trees"] > 0
    assert len(set(kept.world.trees.ids)) == len(kept.world.trees)
