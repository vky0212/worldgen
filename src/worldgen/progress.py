"""Build progress events (see docs/studio-api.md).

The pipeline reports through :func:`current`, which is a no-op unless a
reporter has been installed with :func:`use` (``worldgen build --progress
jsonl`` and the studio server do). Events are plain dicts so they can be
serialised as JSON lines or Server-Sent Events.
"""

from __future__ import annotations

import json
import logging
import threading
import time
from collections.abc import Callable, Iterator
from contextlib import contextmanager
from contextvars import ContextVar
from typing import IO, Any

# (name, label, weight) in pipeline order; weights sum to 1.
STAGES: list[tuple[str, str, float]] = [
    ("acquire", "Download sources", 0.25),
    ("normalize", "Normalize source data", 0.02),
    ("terrain", "Terrain", 0.03),
    ("topology", "Road topology", 0.03),
    ("water_levels", "Water levels", 0.01),
    ("road_elevation", "Road & structure elevation", 0.05),
    ("terrain_shaping", "Terrain shaping", 0.03),
    ("buildings", "Buildings", 0.02),
    ("landcover", "Land cover", 0.02),
    ("intersections", "Junctions", 0.03),
    ("vegetation", "Vegetation", 0.10),
    ("validate", "Validation", 0.10),
    ("export", "Export tiles & data", 0.28),
    ("manifest", "Manifest", 0.03),
]
WEIGHTS = {name: w for name, _, w in STAGES}
SOURCES = ["osm", "dem", "worldcover", "canopy_height", "overture", "imagery"]


class Progress:
    """Collects stage weights into an overall fraction and forwards events to a sink."""

    def __init__(self, sink: Callable[[dict[str, Any]], None] | None = None) -> None:
        self.sink = sink
        self.t0 = time.perf_counter()
        self.finished: set[str] = set()
        self.current_stage: str | None = None
        self.stage_fraction = 0.0
        self._lock = threading.Lock()

    @property
    def enabled(self) -> bool:
        return self.sink is not None

    def overall(self) -> float:
        done = sum(WEIGHTS.get(s, 0.0) for s in self.finished)
        if self.current_stage and self.current_stage not in self.finished:
            done += WEIGHTS.get(self.current_stage, 0.0) * self.stage_fraction
        return round(min(done, 1.0), 4)

    def emit(self, type_: str, **fields: Any) -> None:
        if self.sink is None:
            return
        event = {"type": type_, "t": round(time.perf_counter() - self.t0, 3), **fields}
        with self._lock:
            self.sink(event)

    # -- typed helpers -------------------------------------------------------------------
    def plan(self, **fields: Any) -> None:
        stages = [{"name": n, "label": label, "weight": w} for n, label, w in STAGES]
        self.emit("plan", stages=stages, **fields)

    def stage_start(self, name: str) -> None:
        self.current_stage = name
        self.stage_fraction = 0.0
        self.emit("stage", name=name, state="start", overall=self.overall())

    def stage_end(self, name: str, duration_s: float) -> None:
        self.finished.add(name)
        if self.current_stage == name:
            self.current_stage = None
        self.emit("stage", name=name, state="end", duration_s=round(duration_s, 3), overall=self.overall())

    def stage_skipped(self, name: str) -> None:
        self.finished.add(name)
        self.emit("stage", name=name, state="skipped", overall=self.overall())

    def step(self, stage: str, done: int, total: int, detail: str | None = None) -> None:
        if self.sink is None:
            return
        if self.current_stage == stage and total > 0:
            self.stage_fraction = min(max(done / total, 0.0), 1.0)
        fields: dict[str, Any] = {"stage": stage, "done": done, "total": total, "overall": self.overall()}
        if detail:
            fields["detail"] = detail
        self.emit("progress", **fields)

    def source(self, name: str, state: str, detail: str | None = None) -> None:
        self.emit("source", name=name, state=state, **({"detail": detail} if detail else {}))

    def tile(self, tile_id: str, state: str) -> None:
        self.emit("tile", id=tile_id, state=state)

    def log(self, level: str, message: str) -> None:
        self.emit("log", level=level, message=message)


_NULL = Progress(None)
_current: ContextVar[Progress] = ContextVar("worldgen_progress", default=_NULL)


def current() -> Progress:
    return _current.get()


@contextmanager
def use(progress: Progress) -> Iterator[Progress]:
    token = _current.set(progress)
    handler = _LogForwarder(progress)
    logger = logging.getLogger("worldgen")
    logger.addHandler(handler)
    try:
        yield progress
    finally:
        logger.removeHandler(handler)
        _current.reset(token)


class _LogForwarder(logging.Handler):
    """Forwards compiler log records (INFO and above) as ``log`` events."""

    def __init__(self, progress: Progress) -> None:
        super().__init__(level=logging.INFO)
        self.progress = progress

    def emit(self, record: logging.LogRecord) -> None:
        if not self.progress.enabled or record.name == "worldgen.pipeline" and "stage " in record.getMessage():
            return
        level = "error" if record.levelno >= logging.ERROR else "warning" if record.levelno >= logging.WARNING else "info"
        try:
            self.progress.log(level, record.getMessage())
        except Exception:  # never let progress reporting break a build
            pass


def jsonl_sink(stream: IO[str]) -> Callable[[dict[str, Any]], None]:
    def write(event: dict[str, Any]) -> None:
        stream.write(json.dumps(event, ensure_ascii=False, separators=(",", ":")) + "\n")
        stream.flush()

    return write
