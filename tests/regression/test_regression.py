"""Regression snapshot: compiled statistics of the demo town must stay stable.

Run with ``WORLDGEN_UPDATE_GOLDEN=1`` to refresh the snapshot after an intended change.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np

GOLDEN = Path(__file__).parent / "golden_demo_town.json"


def snapshot(world) -> dict:  # noqa: ANN001
    g = world.graph
    tr = world.trees
    return {
        "statistics": {k: v for k, v in world.stats.items() if isinstance(v, (int, float))},
        "graph_node_kinds": world.stats["graph_node_kinds"],
        "bridge_deck_max": {b.id: round(float(max(world.profiles[e].z.max() for e in b.edge_ids)), 1)
                            for b in world.bridges},
        "tunnel_deck_min": {t.id: round(float(min(world.profiles[e].z.min() for e in t.edge_ids)), 1)
                            for t in world.tunnels},
        "building_heights": {b.id: round(float(b.get("height")), 2) for b in world.buildings},
        "terrain_mean": round(float(world.terrain.z.mean()), 2),
        "trees_total_height": round(float(np.sum(tr.height)), 0),
        "first_tree_ids": sorted(tr.ids)[:5],
        "edges": len(g.edges),
    }


def test_demo_town_snapshot(compiled):
    snap = snapshot(compiled("demo_town"))
    if os.environ.get("WORLDGEN_UPDATE_GOLDEN") == "1" or not GOLDEN.exists():
        GOLDEN.write_text(json.dumps(snap, indent=1, sort_keys=True))
    golden = json.loads(GOLDEN.read_text())
    assert json.loads(json.dumps(snap, sort_keys=True)) == golden
