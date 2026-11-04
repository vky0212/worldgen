"""The scale benchmark (benchmarks/scale_bench.py) runs end to end on a fixture."""

from __future__ import annotations

import json
import subprocess
import sys
from pathlib import Path

import pytest

pytestmark = pytest.mark.slow

SCRIPT = Path(__file__).resolve().parents[2] / "benchmarks" / "scale_bench.py"


def test_scale_bench_fixture(tmp_path):
    out = tmp_path / "bench.json"
    proc = subprocess.run([sys.executable, str(SCRIPT), "--fixture", "demo_town", "--fractions", "0.5,1",
                           "--json", str(out)], capture_output=True, text=True, timeout=300)
    assert proc.returncode == 0, proc.stdout + proc.stderr
    doc = json.loads(out.read_text(encoding="utf-8"))
    small, full = doc["runs"]
    assert small["km2"] < full["km2"] and small["tiles"] < full["tiles"]
    for r in doc["runs"]:
        assert r["compile_s"] > 0 and r["export_s"] > 0
        assert "terrain_tile_seams" in r["checks_s"] and "vegetation" in r["stages_s"]
        assert r["output_bytes"]["tiles_glb"] > 0
        assert r["peak_mb"] is None or r["peak_mb"] > 0
        if r["stage_peak_mb"]:  # sampled where the platform reports the current working set
            assert {"vegetation", "validate", "export"} <= set(r["stage_peak_mb"])

    # A run compared against a baseline reports the speed ratio.
    again = subprocess.run([sys.executable, str(SCRIPT), "--fixture", "demo_town", "--fractions", "0.5",
                            "--no-export", "--baseline", str(out)], capture_output=True, text=True, timeout=300)
    assert again.returncode == 0, again.stdout + again.stderr
    assert "vs baseline" in again.stdout
