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

CHUNKS = [0, 1, 3]   # whole world, one tile per chunk, squares straddling the fixture's tiles


def _build(out: Path, mode: str, chunk: int):  # noqa: ANN202
    cfg = load_config("basic", overrides=[f"export.mode={mode}", f"export.chunk_tiles={chunk}"],
                      region=fixture_region("demo_town"))
    return build(cfg, BuildOptions(out, fixture="demo_town"))


def _digests(root: Path) -> dict[str, str]:
    return {p.relative_to(root).as_posix(): hashlib.sha256(p.read_bytes()).hexdigest()
            for p in sorted(root.rglob("*")) if p.is_file() and p.name != "manifest.json"}


@pytest.mark.parametrize("mode", ["baked", "data"])
def test_outputs_do_not_depend_on_chunking(tmp_path, mode):
    results = {c: _build(tmp_path / f"c{c}", mode, c) for c in CHUNKS}
    ref = _digests(tmp_path / "c0")
    assert any(k.startswith("tiles/") for k in ref) and "vectors/trees.parquet" in ref
    for c in CHUNKS[1:]:
        got = _digests(tmp_path / f"c{c}")
        assert set(got) == set(ref)
        differ = sorted(k for k in ref if got[k] != ref[k])
        # Trees reach trees.parquet chunk by chunk; anything else must be byte-identical.
        assert set(differ) <= {"vectors/trees.parquet"}, differ
        if differ:
            a = gpd.read_parquet(tmp_path / "c0" / "vectors" / "trees.parquet").sort_values("id")
            b = gpd.read_parquet(tmp_path / f"c{c}" / "vectors" / "trees.parquet").sort_values("id")
            assert a.reset_index(drop=True).equals(b.reset_index(drop=True))
        for key in ("trees", "trees_observed", "trees_rendered"):
            assert results[c].manifest["statistics"][key] == results[0].manifest["statistics"][key] > 0
        assert results[c].manifest["validation"] == results[0].manifest["validation"]
    diag = json.loads((tmp_path / "c1" / "diagnostics.json").read_text(encoding="utf-8"))
    assert diag["summary"]["checks"]["vegetation_placement"]["status"] == "pass"
    assert diag["summary"]["checks"]["vegetation_tile_independence"]["status"] == "pass"


def test_trees_are_released_unless_kept(tmp_path):
    cfg = load_config("basic", overrides=["export.chunk_tiles=1"], region=fixture_region("demo_town"))
    plain = build(cfg, BuildOptions(tmp_path / "a", fixture="demo_town"))
    kept = build(cfg, BuildOptions(tmp_path / "b", fixture="demo_town", keep_trees=True))
    assert len(plain.world.trees) == 0 and not plain.world.trees_deferred
    assert len(kept.world.trees) == kept.manifest["statistics"]["trees"] > 0
    assert len(set(kept.world.trees.ids)) == len(kept.world.trees)
