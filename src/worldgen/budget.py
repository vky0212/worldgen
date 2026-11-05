"""Instance budgets: drop smaller buildings and trees at random when a tile is too dense.

Selection is weighted random sampling without replacement (Efraimidis-Spirakis):
every item gets the key ``ln(u) / size`` with ``u`` a deterministic hash of its
id, and the items with the largest keys are kept. Larger items are therefore
much more likely to survive, smaller ones are dropped first, and among similar
sizes the choice is random. Budgets are applied per tile (only the tile's own
items matter, so tiles stay independently rebuildable), then optionally to the
whole world. Dropped items stay in the semantic exports, flagged, but are not
rendered.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np

from worldgen.config import BudgetConfig
from worldgen.hashing import hash_unit, unit_hash


@dataclass
class BudgetResult:
    keep: np.ndarray            # bool per item
    dropped: int
    tiles_over_budget: list[str]
    keys: np.ndarray | None = None   # selection key per item (larger is kept first)


def key_floor(keys: np.ndarray, max_total: int) -> float | None:
    """The key of the ``max_total``-th largest of ``keys`` (the alive items of a whole world,
    gathered chunk by chunk): keeping the items with a key at or above it applies the
    world-wide budget without holding the items; None when the budget is not exceeded."""
    if max_total <= 0 or len(keys) <= max_total:
        return None
    return float(np.partition(keys, len(keys) - max_total)[len(keys) - max_total])


# Salt of the tree budget's hash (hashing.hash_unit over the per-tree seed).
TREE_BUDGET_SALT = 7


def tree_budget_uniforms(seed: int, tree_seeds: np.ndarray) -> np.ndarray:
    """The budget's uniform per tree, hashed from the tree's own (already unique) seed column."""
    return hash_unit(seed, TREE_BUDGET_SALT, np.asarray(tree_seeds, dtype=np.uint64))


def select_within_budget(ids: list[str], tiles: np.ndarray, sizes: np.ndarray, protected: np.ndarray,
                         cfg: BudgetConfig, seed: int, salt: str, u: np.ndarray | None = None,
                         floor: float | None = None) -> BudgetResult:
    """``u``: one uniform per item; by default a hash of (seed, salt, id) per item (slow for many).
    ``floor`` (``key_floor`` of the whole world's alive keys) replaces the ``max_total`` cut for
    items selected chunk by chunk: items below it are dropped (ties at it are kept)."""
    n = len(ids)
    keep = np.ones(n, dtype=bool)
    if not cfg.enabled or n == 0:
        return BudgetResult(keep, 0, [])
    if u is None:
        u = np.array([unit_hash(seed, "budget", salt, i) for i in ids])
    u = np.maximum(np.asarray(u, dtype=float), 1e-12)
    key = np.log(u) / np.maximum(np.asarray(sizes, dtype=float), 1e-6)
    key = np.where(protected, np.inf, key)
    over: list[str] = []
    if cfg.max_per_tile > 0:
        order = np.lexsort((np.array(ids, dtype=object), tiles))
        t_sorted = tiles[order]
        starts = np.flatnonzero(np.r_[True, t_sorted[1:] != t_sorted[:-1]])
        ends = np.r_[starts[1:], n]
        for a, b in zip(starts, ends):
            if b - a <= cfg.max_per_tile:
                continue
            members = order[a:b]
            ranked = members[np.argsort(-key[members], kind="stable")]
            keep[ranked[cfg.max_per_tile:]] = False
            over.append(str(t_sorted[a]))
    if floor is not None:
        keep &= key >= floor
    elif cfg.max_total > 0 and keep.sum() > cfg.max_total:
        alive = np.nonzero(keep)[0]
        ranked = alive[np.argsort(-key[alive], kind="stable")]
        keep[ranked[cfg.max_total:]] = False
    # Protected items rank first (infinite key); they are only dropped when they alone exceed the budget.
    return BudgetResult(keep, int((~keep).sum()), sorted(over), key)


def building_sizes(buildings: list) -> np.ndarray:  # noqa: ANN001
    """Volume proxy (footprint area x height, m3)."""
    return np.array([b.geometry.area * float(b.get("height", 3.0) or 3.0) for b in buildings])


def tree_sizes(height: np.ndarray, crown_radius: np.ndarray) -> np.ndarray:
    """Crown volume proxy (m3): grass clusters and shrubs are tiny, mature trees large."""
    return np.asarray(height) * math.pi * np.asarray(crown_radius) ** 2
