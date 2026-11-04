"""Progress accounting and JSON-lines output."""

from __future__ import annotations

import io
import json

import pytest

from worldgen.log import stage
from worldgen.progress import STAGES, Progress, current, jsonl_sink, use


def test_weights_sum_to_one_and_overall_is_monotonic():
    assert sum(w for _, _, w in STAGES) == pytest.approx(1.0)
    events: list[dict] = []
    p = Progress(events.append)
    with use(p):
        for name, _, _ in STAGES:
            with stage(name):
                current().step(name, 1, 2)
                current().step(name, 2, 2)
    overall = [e["overall"] for e in events if "overall" in e]
    assert overall == sorted(overall) and overall[-1] == pytest.approx(1.0)
    assert current() is not p                      # reporter uninstalled after the block


def test_failed_stage_does_not_count_and_logs_are_forwarded():
    events: list[dict] = []
    p = Progress(events.append)
    with use(p):
        with pytest.raises(RuntimeError):
            with stage("terrain"):
                raise RuntimeError("boom")
        import logging

        logging.getLogger("worldgen.test").warning("watch out")
    assert not any(e["type"] == "stage" and e.get("state") == "end" for e in events)
    assert p.overall() == 0.0
    assert {"type": "log", "level": "warning", "message": "watch out"}.items() <= next(
        e for e in events if e["type"] == "log").items()


def test_jsonl_sink_and_disabled_default():
    buf = io.StringIO()
    p = Progress(jsonl_sink(buf))
    p.tile("0_0", "built")
    line = json.loads(buf.getvalue())
    assert line["type"] == "tile" and line["id"] == "0_0" and "t" in line
    assert not current().enabled                   # default reporter is a no-op
    current().step("export", 1, 2)                 # and never fails
