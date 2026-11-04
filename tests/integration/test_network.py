"""Live-data smoke test (skipped unless WORLDGEN_NETWORK_TESTS=1)."""

from __future__ import annotations

import pytest

from worldgen.build import BuildOptions, build
from worldgen.config import load_config


@pytest.mark.network
def test_small_real_region_then_offline_rebuild(tmp_path):
    region = {"center_lat": 25.0330, "center_lon": 121.5654, "radius_m": 200.0}
    cfg = load_config("basic", region=region)
    cache = tmp_path / "cache"
    first = build(cfg, BuildOptions(tmp_path / "a", cache_dir=cache))
    statuses = {s["name"]: s.get("status", "ok") for s in first.manifest["sources"]}
    assert statuses["osm"] == "ok" and statuses["dem"] in ("ok", "partial")
    assert first.manifest["statistics"]["roads"] > 10 and first.manifest["statistics"]["buildings"] > 10
    # Rebuild from the cache without network access: identical outputs.
    second = build(cfg, BuildOptions(tmp_path / "b", cache_dir=cache, offline=True))
    from worldgen.validation.assets import compare_worlds

    assert compare_worlds(tmp_path / "a", tmp_path / "b")["identical"]
    assert second.manifest["validation"]["errors"] == 0
