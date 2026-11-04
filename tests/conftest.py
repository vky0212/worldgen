"""Shared test helpers: compile synthetic fixture worlds once per session."""

from __future__ import annotations

import os
from functools import lru_cache

import pytest

from worldgen.build import fixture_region
from worldgen.config import WorldConfig, load_config
from worldgen.pipeline import bundle_from_fixture, compile_world
from worldgen.sources.synthetic import load_fixture
from worldgen.spatial.region import Region
from worldgen.world.world import CanonicalWorld


def fixture_config(name: str, overrides: list[str] | None = None) -> WorldConfig:
    return load_config("basic", overrides=overrides or [], region=fixture_region(name))


@lru_cache(maxsize=None)
def _compiled(name: str) -> CanonicalWorld:
    cfg = fixture_config(name)
    region = Region.from_config(cfg)
    return compile_world(region, cfg, bundle_from_fixture(load_fixture(name, region.center)))


@pytest.fixture
def compiled():
    """compiled(name) -> CanonicalWorld (cached per session; treat as read-only)."""
    return _compiled


def pytest_collection_modifyitems(config, items):  # noqa: ANN001
    if os.environ.get("WORLDGEN_NETWORK_TESTS") == "1":
        return
    skip = pytest.mark.skip(reason="network test; set WORLDGEN_NETWORK_TESTS=1 to run")
    for item in items:
        if "network" in item.keywords:
            item.add_marker(skip)
