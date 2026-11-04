"""Logging setup and a small timing helper for pipeline stages."""

from __future__ import annotations

import logging
import sys
import time
from collections.abc import Iterator
from contextlib import contextmanager

LOGGER_NAME = "worldgen"


def get_logger(name: str | None = None) -> logging.Logger:
    return logging.getLogger(LOGGER_NAME if not name else f"{LOGGER_NAME}.{name}")


class _StderrHandler(logging.StreamHandler):
    """Writes to whatever ``sys.stderr`` is at emit time (robust to stream swapping)."""

    @property  # type: ignore[override]
    def stream(self):  # noqa: ANN201
        return sys.stderr

    @stream.setter
    def stream(self, _value) -> None:  # noqa: ANN001
        pass


def configure_logging(verbosity: int = 0) -> None:
    level = logging.WARNING if verbosity < 0 else logging.INFO if verbosity == 0 else logging.DEBUG
    root = logging.getLogger(LOGGER_NAME)
    root.setLevel(level)
    if not root.handlers:
        h = _StderrHandler()
        h.setFormatter(logging.Formatter("%(asctime)s %(levelname)-7s %(name)s: %(message)s", "%H:%M:%S"))
        root.addHandler(h)
    else:
        for h in root.handlers:
            h.setLevel(level)


@contextmanager
def stage(name: str, timings: dict[str, float] | None = None) -> Iterator[None]:
    from worldgen.progress import current

    log = get_logger("pipeline")
    log.info("stage %s ...", name)
    progress = current()
    progress.stage_start(name)
    t0 = time.perf_counter()
    ok = False
    try:
        yield
        ok = True
    finally:
        dt = time.perf_counter() - t0
        if timings is not None:
            timings[name] = round(dt, 3)
        if ok:
            progress.stage_end(name, dt)
        log.info("stage %s done in %.2fs", name, dt)
