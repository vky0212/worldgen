"""Partial nested config sections keep the parent field's defaults for unset keys."""

from __future__ import annotations

import pytest

from worldgen.config import BudgetConfig, WorldConfig, load_config
from worldgen.errors import ConfigError

REGION = {"bbox": [24.51106, 118.400418, 24.542114, 118.435463]}


def test_override_keeps_vegetation_budget_siblings():
    budget = load_config("basic", overrides=["vegetation.budget.max_total=0"], region=REGION).vegetation.budget
    assert budget == BudgetConfig(enabled=True, max_per_tile=2500, max_total=0, keep_observed=True)


def test_override_keeps_buildings_budget_siblings():
    budget = load_config("basic", overrides=["buildings.budget.max_total=5000"], region=REGION).buildings.budget
    assert budget == BudgetConfig(enabled=True, max_per_tile=250, max_total=5000, keep_observed=True)


def test_partial_yaml_section_keeps_factory_defaults(tmp_path):
    user = tmp_path / "user.yaml"
    user.write_text(
        "vegetation:\n"
        "  budget:\n"
        "    keep_observed: false\n"
        "buildings:\n"
        "  budget:\n"
        "    enabled: false\n",
        encoding="utf-8",
    )
    cfg = load_config("basic", config_file=user, region=REGION)
    assert cfg.vegetation.budget == BudgetConfig(True, 2500, 200000, keep_observed=False)
    assert cfg.buildings.budget == BudgetConfig(False, 250, 0)
    # Untouched siblings of the edited sections still come from the defaults.
    assert cfg.vegetation.categories == WorldConfig().vegetation.categories


def test_untouched_config_matches_defaults_and_full_section_wins():
    base = load_config("basic", region=REGION)
    assert base.vegetation.budget == WorldConfig().vegetation.budget
    full = "vegetation.budget={enabled: false, max_per_tile: 1, max_total: 2, keep_observed: false}"
    assert load_config("basic", overrides=[full], region=REGION).vegetation.budget == BudgetConfig(False, 1, 2, False)
    with pytest.raises(ConfigError):
        load_config("basic", overrides=["vegetation.budget.no_such_key=1"], region=REGION)
