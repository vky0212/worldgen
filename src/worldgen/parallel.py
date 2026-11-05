"""Worker processes shared by a build: the compile stages and the export.

``build`` opens one ``Pool`` of spawned processes for the whole build and makes it the
active pool. A stage hands it independent tasks with ``tasks(fn, ctx, items)``: ``ctx``,
what every task reads, is written once to a snapshot in the pool's work folder and loaded
once by each worker; ``fn(ctx, item)`` runs per item, and the results come back in item
order. Without an active pool, or for a single item, the same function runs here, so a
stage gives the same result whatever the number of processes, as long as ``fn`` does not
modify ``ctx`` (in workers its arrays are read-only).

A snapshot is one pickle in which every large numeric array is a separate ``.npy`` file
that workers map read-only (``np.load(mmap_mode="r")``), so grids are shared through the OS
page cache instead of being copied into every process, and geometries keep the precision
grid that GEOS results depend on.
"""

from __future__ import annotations

import multiprocessing
import os
import pickle
import shutil
from collections import deque
from collections.abc import Callable, Iterable, Iterator
from concurrent.futures import Future, ProcessPoolExecutor
from pathlib import Path
from typing import Any

import numpy as np
import shapely

from worldgen.log import get_logger
from worldgen.memory import peak_rss_mb

log = get_logger("parallel")

# Arrays at least this large go to their own memory-mapped file in a snapshot.
SNAPSHOT_ARRAY_BYTES = 1 << 20
# Thread pools of numeric libraries in workers: one thread each (the processes are the parallelism).
WORKER_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")
# Worlds with fewer tiles build in one process by default (spawning workers costs seconds).
MIN_POOL_TILES = 256
# Default upper bound of worker processes.
MAX_AUTO_WORKERS = 8

_ACTIVE: Pool | None = None


def active() -> Pool | None:
    """The pool of the build in progress (None: everything runs in this process)."""
    return _ACTIVE


def pool_workers(setting: int, n_tiles: int) -> int:
    """Worker processes for a build: ``parallel.workers`` when set (> 0), else one process
    for small worlds and otherwise up to the cores minus one, at most ``MAX_AUTO_WORKERS``."""
    if setting > 0:
        return setting
    if n_tiles < MIN_POOL_TILES:
        return 1
    return max(1, min((os.cpu_count() or 2) - 1, MAX_AUTO_WORKERS))


def tasks(fn: Callable[[Any, Any], Any], ctx: Any, items: Iterable[Any],
          init: Callable[[Any], Any] | None = None, local: bool = False) -> Iterator[Any]:
    """``fn(context, item)`` for every item, in item order: in the active pool's workers, or
    here without one, for a single item or with ``local`` (a context that cannot be pickled).
    The context is ``init(ctx)`` when ``init`` is given (built once per process), else
    ``ctx``. ``fn`` and ``init`` must be module-level functions."""
    items = list(items)
    pool = _ACTIVE
    if local or pool is None or pool.workers <= 1 or len(items) <= 1:
        obj = init(ctx) if init is not None else ctx
        for item in items:
            yield fn(obj, item)
        return
    shared = pool.share(ctx, init)
    try:
        yield from pool.map(fn, shared, items)
    finally:
        pool.release(shared)


def batches(items: list[Any], weights: list[float] | np.ndarray, per_worker: int = 4,
            max_weight: float | None = None) -> list[list[Any]]:
    """``items`` cut into consecutive batches of about equal total weight: ``per_worker``
    batches per worker of the active pool (one batch without one), more where a batch would
    weigh over ``max_weight``."""
    pool = _ACTIVE
    n = per_worker * pool.workers if pool is not None and pool.workers > 1 else 1
    cum = np.cumsum(np.asarray(weights, dtype=float))
    if max_weight and len(cum):
        n = max(n, int(np.ceil(cum[-1] / max_weight)))
    if n <= 1 or len(items) <= 1:
        return [list(items)] if items else []
    cuts = np.searchsorted(cum, cum[-1] * np.arange(1, n) / n, side="right")
    bounds = [0, *sorted(set(int(c) for c in cuts if 0 < c < len(items))), len(items)]
    return [list(items[a:b]) for a, b in zip(bounds[:-1], bounds[1:])]


def spans(n: int, min_size: int = 1, per_worker: int = 4) -> list[tuple[int, int]]:
    """``range(n)`` cut into consecutive ``(start, stop)`` spans of at least ``min_size``:
    ``per_worker`` spans per worker of the active pool, one span without one."""
    pool = _ACTIVE
    k = per_worker * pool.workers if pool is not None and pool.workers > 1 else 1
    k = max(1, min(k, n // max(min_size, 1)))
    edges = [n * i // k for i in range(k + 1)]
    return [(a, b) for a, b in zip(edges[:-1], edges[1:])] if n else []


class Pool:
    """Spawned worker processes for one build; ``with Pool(...)`` makes it the active pool.

    ``workers`` <= 1 starts no process (tasks then run in this process). The work folder
    holds the snapshots and is removed on exit."""

    def __init__(self, workers: int, folder: Path) -> None:
        self.workers = max(1, int(workers))
        self.folder = folder
        self.executor: ProcessPoolExecutor | None = None
        self.saved_env: dict[str, str | None] = {}
        self.shared = 0
        self.tasks_run = 0
        self.worker_peak_mb: float | None = None
        self._previous: Pool | None = None

    def __enter__(self) -> Pool:
        global _ACTIVE
        if self.workers > 1:
            if self.folder.exists():
                shutil.rmtree(self.folder)
            self.folder.mkdir(parents=True)
            self.saved_env = {k: os.environ.get(k) for k in WORKER_THREAD_ENV}
            os.environ.update({k: "1" for k in WORKER_THREAD_ENV})
            self.executor = ProcessPoolExecutor(max_workers=self.workers,
                                                mp_context=multiprocessing.get_context("spawn"))
            # Start every worker now: they import the libraries while this process carries on.
            for _ in range(self.workers):
                self.executor.submit(_warm)
            log.info("building with %d worker processes", self.workers)
        self._previous, _ACTIVE = _ACTIVE, self
        return self

    def __exit__(self, *exc: Any) -> None:
        global _ACTIVE
        _ACTIVE = self._previous
        if self.executor is not None:
            self.executor.shutdown(wait=True, cancel_futures=True)
            self.executor = None
            for k, v in self.saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            shutil.rmtree(self.folder, ignore_errors=True)

    def stats(self) -> dict[str, Any]:
        return {"processes": self.workers, "tasks": self.tasks_run,
                "worker_peak_mb": round(self.worker_peak_mb or 0.0, 1)}

    def share(self, ctx: Any, init: Callable[[Any], Any] | None = None) -> str:
        """Write a task context for the workers; returns its handle."""
        self.shared += 1
        path = write_snapshot((init, ctx), self.folder / f"s{self.shared}")
        return str(path)

    def map(self, fn: Callable[[Any, Any], Any], shared: str, items: list[Any]) -> Iterator[Any]:
        """Results of ``fn`` over ``items`` in the workers, in item order (a bounded number of
        tasks in flight). ``shared`` stays usable until ``release``."""
        assert self.executor is not None
        pending: deque[Future[Any]] = deque()
        it = iter(items)
        ahead = 4 * self.workers
        try:
            for item in it:
                pending.append(self.executor.submit(_call, fn, shared, item))
                if len(pending) >= ahead:
                    break
            while pending:
                result, peak = pending.popleft().result()
                nxt = next(it, _END)
                if nxt is not _END:
                    pending.append(self.executor.submit(_call, fn, shared, nxt))
                self.tasks_run += 1
                if peak is not None:
                    self.worker_peak_mb = max(self.worker_peak_mb or 0.0, peak)
                yield result
        finally:
            for f in pending:
                f.cancel()

    def release(self, shared: str) -> None:
        """Remove a context's files. Workers may still map them (Windows then refuses to
        delete them); whatever is left goes with the work folder."""
        shutil.rmtree(Path(shared).parent, ignore_errors=True)


_END = object()

# --------------------------------------------------------------------------------------
# worker side

_LOADED: tuple[str, Any] | None = None


def _warm() -> None:
    import worldgen.pipeline  # noqa: F401  (the compile stages and their libraries)


def _call(fn: Callable[[Any, Any], Any], shared: str, item: Any) -> tuple[Any, float | None]:
    global _LOADED
    if _LOADED is None or _LOADED[0] != shared:
        _LOADED = None                      # release the previous context first
        init, ctx = read_snapshot(Path(shared))
        _LOADED = (shared, init(ctx) if init is not None else ctx)
    return fn(_LOADED[1], item), peak_rss_mb()


# --------------------------------------------------------------------------------------
# snapshots


def _gridded_geometry(wkb: bytes, grid: float) -> Any:
    # Pointwise: the coordinates are already on the grid, so they stay as they are (ring order
    # included); the geometry gets back the precision grid that GEOS operations depend on.
    return shapely.set_precision(shapely.from_wkb(wkb), grid, mode="pointwise")


class _SnapshotPickler(pickle.Pickler):
    """Pickles with every large numeric array saved to its own ``.npy`` file, and geometries
    with the precision grid they carry (plain pickling drops it, and GEOS results such as
    ``set_precision`` or ``representative_point`` depend on it)."""

    def __init__(self, file: Any, folder: Path) -> None:
        super().__init__(file, protocol=pickle.HIGHEST_PROTOCOL)
        self.folder = folder
        self.names: dict[int, str] = {}
        self.keep: list[np.ndarray] = []   # ids stay unique while pickling

    def reducer_override(self, obj: Any) -> Any:
        if isinstance(obj, shapely.Geometry):
            grid = float(shapely.get_precision(obj))
            if grid > 0:
                return _gridded_geometry, (shapely.to_wkb(obj), grid)
        return NotImplemented

    def persistent_id(self, obj: Any) -> Any:
        if (type(obj) is np.ndarray or isinstance(obj, np.memmap)) and not obj.dtype.hasobject \
                and obj.nbytes >= SNAPSHOT_ARRAY_BYTES:
            name = self.names.get(id(obj))
            if name is None:
                name = f"a{len(self.names)}.npy"
                np.save(self.folder / name, np.asarray(obj), allow_pickle=False)
                self.names[id(obj)] = name
                self.keep.append(obj)
            return ("npy", name)
        return None


class _SnapshotUnpickler(pickle.Unpickler):
    def __init__(self, file: Any, folder: Path) -> None:
        super().__init__(file)
        self.folder = folder

    def persistent_load(self, pid: Any) -> Any:
        kind, name = pid
        if kind != "npy":
            raise pickle.UnpicklingError(f"unknown persistent id {pid!r}")
        return np.load(self.folder / name, mmap_mode="r")


def write_snapshot(obj: Any, folder: Path) -> Path:
    """Write ``obj`` to a new snapshot folder; returns the pickle's path."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    path = folder / "snapshot.pkl"
    with open(path, "wb") as f:
        _SnapshotPickler(f, folder).dump(obj)
    return path


def read_snapshot(path: Path) -> Any:
    with open(path, "rb") as f:
        return _SnapshotUnpickler(f, path.parent).load()
