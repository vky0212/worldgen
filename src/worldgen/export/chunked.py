"""Tile export chunk by chunk, in this process or in worker processes.

The export walks squares of ``export.chunk_tiles`` tiles (``tiles.tile_chunks``). For each
chunk, ``ChunkExporter.run`` places the chunk's deferred trees, hashes its tiles, writes the
dirty ones and returns a ``ChunkResult``: tile states, locator ids and the chunk's tree
outputs (validation issues, the trees.parquet rows, per-tree field contributions, counts).
``ChunkMerger`` applies results strictly in chunk order, so every output is the same
whether chunks ran here or in ``export.workers`` processes, in whatever order they finished.

Workers (spawned processes) get the compiled world through a snapshot written once to
``<output>/.work``: one pickle in which every large numeric array is a separate ``.npy`` file
that workers map read-only (``np.load(mmap_mode="r")``), so grids are shared through the OS
page cache instead of being copied into every process. Tile files are written by the
workers directly to their final paths.
"""

from __future__ import annotations

import json
import multiprocessing
import os
import pickle
import shutil
from collections.abc import Iterator
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import shapely

from worldgen.buildings.buildings import building_plan
from worldgen.export.container import write_container
from worldgen.export.footprints import building_sections
from worldgen.export.heightmap import tile_terrain_sections
from worldgen.export.instances import tree_sections
from worldgen.export.ortho import ortho_baker
from worldgen.export.tiles import (
    NO_TREES,
    PieceSelector,
    archetypes_for,
    build_tile_assets,
    buildings_by_tile,
    entity_strtree,
    generate_pieces,
    terrain_vertex_colors,
    tile_blocks,
    tile_config_hash,
    tile_input_hash,
    trees_by_tile,
)
from worldgen.export.tileset import tile_paths
from worldgen.export.vectors import TreesParquetWriter
from worldgen.log import get_logger
from worldgen.memory import peak_rss_mb
from worldgen.pipeline import apply_tree_budget, place_vegetation, vegetation_exclusions
from worldgen.progress import current
from worldgen.tiling.grid import TileGrid, parse_tile_id
from worldgen.validation.checks import IndependenceSample, Issue, Report, check_vegetation
from worldgen.vegetation.placement import TreeTable, concat_tables
from worldgen.vegetation.regions import add_cell_contributions, tree_cell_contributions
from worldgen.world.world import CanonicalWorld

log = get_logger("export")

# Dirty tiles are generated in blocks of this many tiles per side (memory follows the block).
EXPORT_BLOCK_TILES = 8
# Arrays at least this large go to their own memory-mapped file in a worker snapshot.
SNAPSHOT_ARRAY_BYTES = 1 << 20
# Thread pools of numeric libraries in workers: one thread each (the processes are the parallelism).
WORKER_THREAD_ENV = ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS")


@dataclass
class ExportPlan:
    """What every chunk needs to know besides the world (picklable)."""

    output: Path
    mode: str
    prev_tiles: dict[str, Any]
    force: bool = False
    only_tiles: set[str] | None = None
    keep_trees: bool = False


@dataclass
class ChunkResult:
    index: int
    tiles: list[str]
    hashes: dict[str, str] = field(default_factory=dict)
    built: dict[str, dict[str, Any]] = field(default_factory=dict)       # tile -> build_state entry
    tile_entities: dict[str, list[str]] = field(default_factory=dict)    # data mode: the locator's ids
    trees: dict[str, Any] | None = None                                  # deferred trees' outputs
    peak_mb: float | None = None                                         # worker process peak so far


def _write_json(path: Path, data: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, ensure_ascii=False, allow_nan=False), encoding="utf-8")


def _tob_ids(sections: dict[str, bytes]) -> list[str]:
    """Ids of the mapped trees of a data tile (``TOB `` section; stored or deflated payload)."""
    tob = sections.get("TOB")
    return [r["id"] for r in json.loads(tob)] if tob else []


class ChunkExporter:
    """Exports chunks of one world; the world-level indexes are built once per process."""

    def __init__(self, world: CanonicalWorld, plan: ExportPlan) -> None:
        cfg = world.config
        self.world, self.plan = world, plan
        self.grid = TileGrid(cfg.tiling.tile_size)
        self.data = plan.mode == "data"
        self.etree = entity_strtree(world)
        self.chash = tile_config_hash(cfg)
        self.baker = ortho_baker(world)
        self.selector: PieceSelector | None = None
        self.rgba = terrain_vertex_colors(world)
        self.arche = archetypes_for(world)
        self.eindex = world.entity_index()
        self.bindex = buildings_by_tile(world) if self.data else {}
        self.deferred = world.trees_deferred
        self.check = self.deferred and cfg.validation.enabled and world.vegetation_grid is not None
        self.exclusions0 = vegetation_exclusions(world, 0.0) if self.check else None
        self.sample = IndependenceSample(world) if self.check else None

    def budget_keys(self, bounds: tuple[float, float, float, float] | None) -> np.ndarray:
        """Budget keys of the chunk's trees that the per-tile budget renders (the key pass of a
        world-wide ``vegetation.budget.max_total``)."""
        tr = place_vegetation(self.world, bounds)
        res = apply_tree_budget(tr, self.world.config, chunk=True)
        return res.keys[res.keep] if res is not None and res.keys is not None else np.zeros(0)

    def run(self, index: int, bounds: tuple[float, float, float, float] | None, tiles: list[str],
            floor: float | None = None, whole_world: bool = False) -> ChunkResult:
        """Export a chunk. The tree budget is per tile plus ``floor`` (the world-wide budget's
        key floor), or the full budget when the chunk is the ``whole_world``."""
        w, plan = self.world, self.plan
        cfg = w.config
        out = plan.output
        res = ChunkResult(index, list(tiles))
        budget = None
        if self.deferred:
            w.trees = TreeTable()
            w.trees = place_vegetation(w, bounds)
            budget = apply_tree_budget(w.trees, cfg, chunk=not whole_world, floor=floor)
        tindex = trees_by_tile(w)
        dirty = []
        for t in tiles:
            h = res.hashes[t] = tile_input_hash(w, t, self.etree, tindex, self.chash)
            paths = tile_paths(plan.mode, t)
            files_ok = (out / paths["content"]).is_file() and (out / paths["meta"]).is_file()
            if self.baker is not None:
                files_ok = files_ok and (out / paths["ortho"]).is_file()
            if plan.force or not files_ok or plan.prev_tiles.get(t, {}).get("hash") != h:
                if plan.only_tiles is None or t in plan.only_tiles:
                    dirty.append(t)
        if dirty and self.selector is None:
            self.selector = PieceSelector(w)
        for block in tile_blocks(dirty, EXPORT_BLOCK_TILES):
            pieces = generate_pieces(w, set(block), self.selector, building_meshes=not self.data)
            for t in block:
                self._write_tile(t, pieces.pop(t, []), tindex, res)
        if self.deferred:
            res.trees = self._tree_outputs(w.trees, bounds, tiles, budget)
            w.trees = TreeTable()
        return res

    def _write_tile(self, t: str, pieces: list[Any], tindex: dict[str, np.ndarray], res: ChunkResult) -> None:
        w, cfg, out = self.world, self.world.config, self.plan.output
        paths = tile_paths(self.plan.mode, t)
        bounds = self.grid.bounds(t)
        assets = build_tile_assets(w, t, pieces, None, self.rgba, self.arche,
                                   trees_in_tile=tindex.get(t, NO_TREES), entity_index=self.eindex,
                                   terrain_mesh=not self.data, trees_as_data=self.data,
                                   buildings_as_data=self.bindex.get(t, []) if self.data else None)
        content = out / paths["content"]
        content.parent.mkdir(parents=True, exist_ok=True)
        if self.data:
            i, j = parse_tile_id(t)
            sections = tile_terrain_sections(w, t, self.rgba)
            sections += tree_sections(w.trees, assets["tree_idx"], bounds, cfg.vegetation.cell_size)
            bl = self.bindex.get(t, [])
            if bl:
                sections += building_sections(bl, [building_plan(b, cfg.buildings) for b in bl], bounds)
            sections.append(("GLB", assets["glb"], False))
            content.write_bytes(write_container(0, i, j, sections))
            res.tile_entities[t] = list(assets["meta"]["entities"]) + _tob_ids({s[0]: s[1] for s in sections})
        else:
            content.write_bytes(assets["glb"])
        col = out / paths["collision"]
        if assets["collision"] is not None:
            col.write_bytes(assets["collision"])
        elif col.exists():
            col.unlink()
        _write_json(out / paths["meta"], assets["meta"])
        ortho = out / paths["ortho"]
        if self.baker is not None:
            ortho.write_bytes(self.baker.bake(bounds))
        elif ortho.exists():
            ortho.unlink()
        res.built[t] = {"hash": res.hashes[t], "counts": assets["counts"], "zrange": assets["zrange"],
                        "collision": assets["collision"] is not None, "ortho": self.baker is not None}

    def _tree_outputs(self, tr: Any, bounds: tuple[float, float, float, float] | None, tiles: list[str],
                      budget: Any) -> dict[str, Any]:
        """What the merger needs of the chunk's trees (small, picklable)."""
        w = self.world
        cfg = w.config
        return {
            "issues": check_vegetation(w, tr, bounds, self.exclusions0) if self.check and len(tr) else [],
            "candidates": self.sample.candidates(tr, tiles) if self.check else [],  # type: ignore[union-attr]
            "table": TreesParquetWriter.table_of(w, tr) if cfg.export.geoparquet and len(tr) else None,
            "contrib": tree_cell_contributions(w.vegetation_grid, tr) if cfg.export.fields else None,
            "total": len(tr),
            "observed": int(np.sum(tr.observed)) if len(tr) else 0,
            "rendered": int(tr.rendered_mask().sum()) if len(tr) else 0,
            "dropped": budget.dropped if budget is not None else 0,
            "over_budget": list(budget.tiles_over_budget) if budget is not None else [],
            "kept": tr if self.plan.keep_trees else None,
        }


class ChunkMerger:
    """Applies chunk results in chunk order: tile states, locator ids, progress, and the
    deferred trees' validation, trees.parquet rows, field sums and statistics."""

    def __init__(self, world: CanonicalWorld, report: Report, plan: ExportPlan, n_tiles: int) -> None:
        cfg = world.config
        self.world, self.report, self.plan, self.n_tiles = world, report, plan, n_tiles
        self.hashes: dict[str, str] = {}
        self.built: dict[str, dict[str, Any]] = {}
        self.tile_entities: dict[str, list[str]] = {}
        self.visited = 0
        self.next_index = 0
        self.deferred = world.trees_deferred
        self.check = self.deferred and cfg.validation.enabled and world.vegetation_grid is not None
        self.sample = IndependenceSample(world) if self.check else None
        self.placement_issues: list[Issue] = []
        self.parquet = (TreesParquetWriter(world, plan.output / "vectors" / "trees.parquet")
                        if self.deferred and cfg.export.geoparquet else None)
        self.field_sums: tuple[np.ndarray, np.ndarray] | None = None
        self.kept: list[Any] = []
        self.total = self.observed = self.rendered = self.dropped = 0
        self.over_budget: list[str] = []
        self.worker_peak_mb: float | None = None
        current().step("export", 0, n_tiles, "generating geometry")

    def merge(self, r: ChunkResult) -> None:
        if r.index != self.next_index:
            raise RuntimeError(f"chunk {r.index} merged out of order (expected {self.next_index})")
        self.next_index += 1
        if r.peak_mb is not None:
            self.worker_peak_mb = max(self.worker_peak_mb or 0.0, r.peak_mb)
        progress = current()
        self.hashes.update(r.hashes)
        for t in r.tiles:
            if t not in r.built:
                progress.tile(t, "unchanged")
                self.visited += 1
        for t, state in r.built.items():
            self.built[t] = state
            progress.tile(t, "built")
            self.visited += 1
            progress.step("export", self.visited, self.n_tiles, f"tile {t}")
        self.tile_entities.update(r.tile_entities)
        tr = r.trees
        if tr is None:
            return
        if self.check:
            self.placement_issues += tr["issues"]
            self.sample.merge(tr["candidates"])  # type: ignore[union-attr]
        if self.parquet is not None and tr["table"] is not None:
            self.parquet.add_table(tr["table"])
        if tr["contrib"] is not None:
            self.field_sums = add_cell_contributions(self.world.vegetation_grid, tr["contrib"], self.field_sums)
        self.total += tr["total"]
        self.observed += tr["observed"]
        self.rendered += tr["rendered"]
        self.dropped += tr["dropped"]
        self.over_budget += tr["over_budget"]
        if tr["kept"] is not None:
            self.kept.append(tr["kept"])

    def finish(self) -> Any:
        """Close the tree outputs, record checks and statistics; the trees to leave in the world."""
        from worldgen.budget import BudgetResult
        from worldgen.pipeline import budget_issue

        w = self.world
        if self.parquet is not None:
            self.parquet.close()
        if self.check:
            self.report.run("vegetation_placement", lambda: self.placement_issues)
            self.report.run("vegetation_tile_independence", self.sample.check)  # type: ignore[union-attr]
        if self.total:
            res = BudgetResult(np.zeros(0, dtype=bool), self.dropped, sorted(set(self.over_budget)))
            before = len(w.stage_issues)
            budget_issue(w, w.config, res)
            # Where validate_world would have listed it: after the earlier stage issues.
            for k, raw in enumerate(w.stage_issues[before:]):
                self.report.issues.insert(min(before + k, len(self.report.issues)), Issue(
                    raw["check"], raw["severity"], raw["message"], raw.get("entity_ids", [])))
        w.stats["trees"] = self.total
        w.stats["trees_observed"] = self.observed
        w.stats["trees_rendered"] = self.rendered
        w.trees_deferred = False
        return concat_tables(self.kept) if self.kept else TreeTable()


# --------------------------------------------------------------------------------------
# running chunks


def export_workers(setting: int, n_tiles: int, n_chunks: int) -> int:
    """Worker processes for an export: ``export.workers`` when set (> 0), else one process for
    small worlds (spawning workers costs seconds) and otherwise up to the cores minus one."""
    if setting > 0:
        return max(1, min(setting, n_chunks))
    if n_chunks < 4 or n_tiles < 256:
        return 1
    return max(1, min((os.cpu_count() or 2) - 1, n_chunks, 8))


class ChunkRunner:
    """Runs chunk tasks here (``workers`` <= 1) or in a pool of worker processes sharing one
    world snapshot; results come back in chunk order."""

    def __init__(self, world: CanonicalWorld, plan: ExportPlan, workers: int, n_chunks: int) -> None:
        self.world, self.plan = world, plan
        self.workers = workers if n_chunks > 1 else 1
        self.exporter: ChunkExporter | None = None
        self.pool: ProcessPoolExecutor | None = None
        self.saved_env: dict[str, str | None] = {}

    def __enter__(self) -> ChunkRunner:
        if self.workers <= 1:
            self.exporter = ChunkExporter(self.world, self.plan)
            return self
        snapshot = write_snapshot(self.world, self.plan.output / ".work")
        self.saved_env = {k: os.environ.get(k) for k in WORKER_THREAD_ENV}
        os.environ.update({k: "1" for k in WORKER_THREAD_ENV})
        self.pool = ProcessPoolExecutor(max_workers=self.workers, mp_context=multiprocessing.get_context("spawn"),
                                        initializer=_init_worker, initargs=(str(snapshot), self.plan))
        log.info("exporting with %d worker processes", self.workers)
        return self

    def __exit__(self, *exc: Any) -> None:
        if self.pool is not None:
            self.pool.shutdown(wait=True, cancel_futures=True)
            for k, v in self.saved_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v
            shutil.rmtree(self.plan.output / ".work", ignore_errors=True)

    def keys(self, chunks: list[tuple[Any, list[str]]]) -> list[np.ndarray]:
        bounds = [b for b, _ in chunks]
        if self.pool is None:
            return [self.exporter.budget_keys(b) for b in bounds]  # type: ignore[union-attr]
        return list(self.pool.map(_chunk_keys, bounds))

    def export(self, chunks: list[tuple[Any, list[str]]], floor: float | None, whole_world: bool
               ) -> Iterator[ChunkResult]:
        if self.pool is None:
            for k, (b, t) in enumerate(chunks):
                yield self.exporter.run(k, b, t, floor, whole_world)  # type: ignore[union-attr]
            return
        n = len(chunks)
        yield from self.pool.map(_run_chunk, range(n), [b for b, _ in chunks], [t for _, t in chunks],
                                 [floor] * n, [whole_world] * n)


_EXPORTER: ChunkExporter | None = None


def _init_worker(snapshot: str, plan: ExportPlan) -> None:
    global _EXPORTER
    _EXPORTER = ChunkExporter(read_snapshot(Path(snapshot)), plan)


def _chunk_keys(bounds: tuple[float, float, float, float] | None) -> np.ndarray:
    assert _EXPORTER is not None
    return _EXPORTER.budget_keys(bounds)


def _run_chunk(index: int, bounds: tuple[float, float, float, float] | None, tiles: list[str],
               floor: float | None, whole_world: bool) -> ChunkResult:
    assert _EXPORTER is not None
    res = _EXPORTER.run(index, bounds, tiles, floor, whole_world)
    res.peak_mb = peak_rss_mb()
    return res


# --------------------------------------------------------------------------------------
# world snapshot


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


def write_snapshot(world: CanonicalWorld, folder: Path) -> Path:
    """Write ``world`` for worker processes; returns the pickle's path."""
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    path = folder / "world.pkl"
    with open(path, "wb") as f:
        _SnapshotPickler(f, folder).dump(world)
    return path


def read_snapshot(path: Path) -> CanonicalWorld:
    with open(path, "rb") as f:
        return _SnapshotUnpickler(f, path.parent).load()
