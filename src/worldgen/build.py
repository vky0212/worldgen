"""Top-level build: region -> sources -> canonical world -> assets + manifest."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import shutil
import time
from collections.abc import Callable
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.transform import Affine

from worldgen import SCHEMA_VERSION, __version__, parallel
from worldgen.config import WorldConfig
from worldgen.errors import ValidationFailed
from worldgen.budget import key_floor
from worldgen.export.chunked import ChunkMerger, ChunkRunner, ExportPlan, _tob_ids
from worldgen.export.container import read_container
from worldgen.export.footprints import tileset_entry as buildings_tileset_entry
from worldgen.export.graphcore import graph_files
from worldgen.export.instances import archetypes_glb
from worldgen.export.locate import locate_files
from worldgen.export.merge import merge_glbs
from worldgen.export.ortho import imagery_hash, ortho_baker, tile_pixels
from worldgen.export.pyramid import finish_levels, level_work, plan_levels, remove_stale_levels, state_key, top_level
from worldgen.export.tiles import (
    LAYER_ORDER,
    tile_chunks,
    world_tiles,
)
from worldgen.export.tileset import DATA_SCHEMA_VERSION, data_index, level_paths, tile_paths
from worldgen.export.vectors import write_geoparquet
from worldgen.hashing import sha256_file
from worldgen.landcover.landcover import LANDCOVER_CODES
from worldgen.log import get_logger, stage
from worldgen.pipeline import (
    acquire,
    bundle_from_fixture,
    compile_world,
)
from worldgen.progress import current
from worldgen.sources.base import GEOTIFF_BLOCK_ROWS, SourceRecord, write_geotiff_rows
from worldgen.sources.synthetic import load_fixture
from worldgen.spatial.region import Region
from worldgen.tiling.grid import TileGrid, parse_tile_id
from worldgen.validation.checks import Report, validate_world
from worldgen.vegetation.classify import CATEGORIES
from worldgen.vegetation.placement import ARCHETYPES
from worldgen.vegetation.regions import FIELD_NAMES, environmental_fields, tree_cell_sums
from worldgen.world.world import CanonicalWorld

log = get_logger("build")


@dataclass
class BuildOptions:
    output: Path
    cache_dir: Path = Path(".worldgen-cache")
    offline: bool = False
    fixture: str | None = None
    only_tiles: set[str] | None = None
    force: bool = False
    # Keep every tree in BuildResult.world.trees (the export places and releases them chunk by
    # chunk; tests and tools that inspect them ask for them).
    keep_trees: bool = False


@dataclass
class BuildResult:
    world: CanonicalWorld
    manifest: dict[str, Any]
    report: Report
    tiles_written: list[str]
    tiles_skipped: list[str]


def fixture_region(name: str) -> dict[str, Any]:
    from worldgen.sources.synthetic import ORIGIN

    fx = load_fixture(name)
    return {"center_lat": ORIGIN.lat, "center_lon": ORIGIN.lon, "radius_m": fx.radius_m, "name": f"fixture:{name}"}


def build_pool(cfg: WorldConfig, region: Region, output: Path) -> parallel.Pool:
    """The worker pool of a build of ``region`` (``parallel.workers``; work folder in the output)."""
    return parallel.Pool(parallel.pool_workers(cfg.parallel.workers, len(region.tiles())), output / ".work")


def build(cfg: WorldConfig, opts: BuildOptions) -> BuildResult:
    t0 = time.perf_counter()
    region = Region.from_config(cfg)
    timings: dict[str, float] = {}
    progress = current()
    if progress.enabled:
        progress.plan(**plan_payload(region, cfg))
    # One pool of worker processes for the whole build (parallel.py): the windowed compile
    # stages and the export chunks run in it. It starts now, so the workers import their
    # libraries while the sources load.
    with build_pool(cfg, region, opts.output) as pool:
        with stage("acquire", timings):
            if opts.fixture:
                fx = load_fixture(opts.fixture, region.center, cfg.projection.method)
                bundle = bundle_from_fixture(fx)
                if not any(r.name == "imagery" for r in bundle.records):
                    im = cfg.sources.imagery
                    bundle.records.append(SourceRecord(
                        "imagery", "synthetic", "-", "-", status="missing" if im.enabled else "disabled",
                        notes=["synthetic fixtures carry no imagery"] if im.enabled else []))
                for rec in bundle.records:
                    progress.source(rec.name, rec.status, "synthetic fixture")
            else:
                bundle = acquire(region, cfg, opts.cache_dir, offline=opts.offline)
        world = compile_world(region, cfg, bundle, place_trees=False)
        world.timings = {**timings, **world.timings}
        with stage("validate", world.timings):
            report = validate_world(world) if cfg.validation.enabled else Report()
        with stage("export", world.timings):
            written, skipped = export_world(world, report, opts)
        if pool.workers > 1:
            world.stats["workers"] = pool.stats()
    with stage("manifest", world.timings):
        manifest = write_manifest(world, report, opts, time.perf_counter() - t0)
    errors = manifest["validation"]["errors"]
    if cfg.validation.fail_on_error and errors:
        raise ValidationFailed(f"validation found {errors} error(s); see {opts.output / 'diagnostics.json'}")
    return BuildResult(world, manifest, report, written, skipped)


# --------------------------------------------------------------------------------------
# export


def _write_json(path: Path, data: Any, indent: int | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=indent, sort_keys=False, ensure_ascii=False, allow_nan=False),
                    encoding="utf-8")


def export_world(world: CanonicalWorld, report: Report, opts: BuildOptions) -> tuple[list[str], list[str]]:
    cfg = world.config
    out = opts.output
    mode = cfg.export.mode
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    _remove_other_layout(out, mode)
    grid = TileGrid(cfg.tiling.tile_size)
    tiles = world_tiles(world)
    paths = {t: tile_paths(mode, t) for t in tiles}

    state_path = out / "build_state.json"
    prev: dict[str, Any] = {}
    if state_path.is_file():
        try:
            prev = json.loads(state_path.read_text(encoding="utf-8"))
        except json.JSONDecodeError:
            prev = {}
    if prev.get("world_key") != _world_key(world):
        prev = {}
    prev_tiles = prev.get("tiles", {})
    if opts.only_tiles is not None:
        unknown = opts.only_tiles - set(tiles)
        if unknown:
            raise ValueError(f"unknown tiles requested: {sorted(unknown)}")
    # Chunk by chunk (export/chunked.py): deferred trees are placed for a chunk of tiles, the
    # chunk's tile hashes decide what to rebuild, its dirty tiles are generated block by block,
    # and its trees go to validation, vectors/trees.parquet, the fields and the statistics
    # before the next chunk replaces them. Chunks may run in worker processes; results are
    # merged in chunk order, and every tile gets the same trees, in the same order, whatever
    # the chunking, so the files depend on neither.
    g = world.vegetation_grid if world.trees_deferred else None
    extent = ((g.xmin, g.ymin, g.xmin + g.shape[1] * g.resolution, g.ymin + g.shape[0] * g.resolution)
              if g is not None else None)
    chunks = tile_chunks(tiles, cfg.export.chunk_tiles, cfg.tiling.tile_size, extent)
    plan = ExportPlan(out, mode, prev_tiles, opts.force, opts.only_tiles, opts.keep_trees)
    merger = ChunkMerger(world, report, plan, len(tiles))
    budget = cfg.vegetation.budget
    with ChunkRunner(world, plan, len(chunks)) as runner:
        # A world-wide tree budget over several chunks: a first pass gathers the keys of the
        # trees each chunk's per-tile budget keeps, and the export keeps those at or above the
        # max_total-th largest key (budget.key_floor) - the trees a whole-world pass keeps.
        whole = len(chunks) == 1
        floor = None
        if world.trees_deferred and budget.enabled and budget.max_total > 0 and not whole:
            keys = runner.keys(chunks)
            floor = key_floor(np.concatenate(keys) if keys else np.zeros(0), budget.max_total)
        for result in runner.export(chunks, floor, whole):
            merger.merge(result)
        # Coarser levels of detail (data mode, export/pyramid.py), from the finished level 0.
        levels: dict[int, dict[tuple[int, int], list[tuple[int, int]]]] = {}
        level_state: dict[int, dict[tuple[int, int], dict[str, Any]]] = {}
        if mode == "data":
            levels = plan_levels(tiles, top_level(cfg.export.lod_levels, cfg.tiling.tile_size,
                                                  world.terrain.resolution))
            # Coarse content comes from the level-0 files: a tile left stale (only_tiles) keys
            # its coarse tiles by what its files hold too.
            leaves = {}
            for t in tiles:
                on_disk = merger.built[t]["hash"] if t in merger.built else prev_tiles.get(t, {}).get("hash")
                leaves[parse_tile_id(t)] = merger.hashes[t] + ("" if on_disk == merger.hashes[t] else f"|{on_disk}")
            items, level_state = level_work(levels, leaves, prev.get("levels", {}), out, opts.force,
                                            {"quantum": cfg.height_quantum(), "imagery": imagery_hash(world)})
            for level, key, state in runner.coarse(items):
                level_state[level][key] = state
            remove_stale_levels(out, levels)
            if levels:
                log.info("coarser levels: %s tiles, %d (re)built",
                         "+".join(str(len(levels[k])) for k in sorted(levels)), len(items))
    if world.trees_deferred:
        world.trees = merger.finish()
    hashes, built, tile_entities = merger.hashes, merger.built, merger.tile_entities
    baker = ortho_baker(world)
    data = mode == "data"
    dirty = [t for t in tiles if t in built]
    skipped = [t for t in tiles if t not in built]
    log.info("tiles: %d total, %d (re)built, %d unchanged", len(tiles), len(dirty), len(skipped))
    tile_state = dict(prev_tiles)
    for t in dirty:  # in tile order, as build_state.json has always been written
        tile_state[t] = built[t]
    for t in list(tile_state):
        if t not in tiles:
            tile_state.pop(t)

    ortho_info = baker.describe() if baker is not None else None
    if ortho_info is not None:
        world.stats["ortho"] = ortho_info
    if data:
        finished = finish_levels(levels, tile_state, level_state, world.terrain.resolution)
        level_ortho = {}
        if baker is not None:
            for level in finished:
                size = cfg.tiling.tile_size * (1 << level)
                px = tile_pixels(size, baker.effective_m_per_px, cfg.sources.imagery.tile_px_max)
                level_ortho[level] = {"pixels": px, "m_per_px": round(size / px, 4)}
        tileset, subtrees = data_index(tiles, tile_state, cfg.tiling.tile_size, world.terrain.resolution,
                                       cfg.height_quantum(), LAYER_ORDER, ortho_info,
                                       {str(k): v for k, v in LANDCOVER_CODES.items()}, finished, level_ortho)
        tileset["archetypes"] = "tiles/archetypes.glb"
        tileset["trees"] = {"sections": ["TRE", "TOB"], "seed": cfg.vegetation.seed,
                            "cell_size": cfg.vegetation.cell_size, "archetypes": ARCHETYPES,
                            "categories": CATEGORIES, "seed_salts": {"tree": 3, "ground": 6}}
        tileset["buildings"] = buildings_tileset_entry()
        for t in tiles:
            if t not in tile_entities:  # unchanged tile: its ids from the files on disk
                meta = json.loads((out / paths[t]["meta"]).read_text(encoding="utf-8"))
                tile_entities[t] = list(meta["entities"]) + _tob_ids(
                    read_container((out / paths[t]["content"]).read_bytes()).sections)
        tileset["locate"], locate_docs = locate_files(tile_entities, DATA_SCHEMA_VERSION)
        if (out / "tiles" / "locate").exists():
            shutil.rmtree(out / "tiles" / "locate")
        (out / "tiles" / "locate").mkdir(parents=True)
        for rel, doc in locate_docs.items():
            _write_json(out / rel, doc, indent=None)
        (out / "tiles" / "archetypes.glb").write_bytes(archetypes_glb(cfg.vegetation.quality))
        for rel, doc in subtrees.items():
            _write_json(out / rel, doc, indent=None)
        _write_json(out / "tiles" / "tileset.json", tileset, indent=1)
    else:
        index = {"schema": "worldgen.tile_index", "schema_version": SCHEMA_VERSION,
                 "tile_size": cfg.tiling.tile_size, "layers": LAYER_ORDER, "ortho": ortho_info, "tiles": []}
        for t in tiles:
            st = tile_state.get(t, {})
            index["tiles"].append({
                "id": t, "i": int(t.split("_")[0]), "j": int(t.split("_")[1]), "bounds": list(grid.bounds(t)),
                "zrange": st.get("zrange"), "glb": f"tiles/{t}.glb",
                "collision": f"tiles/{t}.collision.glb" if st.get("collision") else None,
                "meta": f"tiles/{t}.meta.json",
                "ortho": f"tiles/{t}.ortho.jpg" if st.get("ortho") else None,
                "hash": hashes[t], "counts": st.get("counts", {}),
            })
        _write_json(out / "tiles" / "index.json", index, indent=1)
    state: dict[str, Any] = {"world_key": _world_key(world), "tiles": tile_state}
    if data:
        state["levels"] = {state_key(level, *key): level_state[level][key]
                           for level in sorted(level_state) for key in levels[level]}
    _write_json(state_path, state, indent=1)

    combined = out / "world.glb"
    if not data and _combined_glb_wanted(cfg, len(tiles)):
        files = [(out / "tiles" / f"{t}.glb").read_bytes() for t in tiles if (out / "tiles" / f"{t}.glb").is_file()]
        if files:
            combined.write_bytes(merge_glbs(files, "world", {"origin": _origin(world)}))
    else:
        if cfg.export.combined_glb and data:
            log.info("world.glb not written: data-first tiles carry no terrain mesh to combine")
        elif cfg.export.combined_glb:
            log.info("world.glb not written: %d tiles exceed export.combined_glb_max_tiles=%d",
                     len(tiles), cfg.export.combined_glb_max_tiles)
        if combined.exists():
            combined.unlink()  # a world.glb from an earlier build no longer matches the tiles

    _write_json(out / "graph" / "road_graph.json", world.graph.to_dict(world.region.frame, _origin(world)))
    if data:
        core, strings = graph_files(world.graph, _origin(world), world.region.frame.proj_string, DATA_SCHEMA_VERSION)
        (out / "graph" / "core.wgt").write_bytes(core)
        _write_json(out / "graph" / "strings.json", strings, indent=None)
    if cfg.export.geoparquet:
        write_geoparquet(world, out / "vectors", trees=merger.parquet is None)
    if cfg.export.heightfield:
        export_terrain(world, out / "terrain")
    if cfg.export.fields and world.vegetation_grid is not None:
        export_fields(world, out / "fields", merger.field_sums)
    _write_json(out / "diagnostics.json", report.to_dict(world), indent=1)
    (out / "ATTRIBUTION.txt").write_text(attribution_text(world), encoding="utf-8")
    return dirty, skipped


def _remove_other_layout(out: Path, mode: str) -> None:
    """Delete files of the other export mode left by an earlier build into ``out``."""
    tiles = out / "tiles"
    if mode == "data" and (tiles / "index.json").exists():
        for p in tiles.iterdir():
            if p.is_file():
                p.unlink()
    elif mode == "baked":
        if (tiles / "tileset.json").exists():
            for p in tiles.iterdir():
                if p.is_dir() and (p.name in ("sub", "locate") or p.name.isdigit()):
                    shutil.rmtree(p)
            for name in ("tileset.json", "archetypes.glb"):
                if (tiles / name).exists():
                    (tiles / name).unlink()
        for name in ("core.wgt", "strings.json"):
            if (out / "graph" / name).exists():
                (out / "graph" / name).unlink()


def _combined_glb_wanted(cfg: WorldConfig, n_tiles: int) -> bool:
    limit = cfg.export.combined_glb_max_tiles
    return cfg.export.combined_glb and (limit <= 0 or n_tiles <= limit)


def plan_payload(region: Region, cfg: WorldConfig) -> dict[str, Any]:
    """Region outline and tile rings in WGS84 (lat, lon) for progress displays and previews."""
    grid = TileGrid(cfg.tiling.tile_size)
    frame = region.frame

    def ring(xs: list[float], ys: list[float]) -> list[list[float]]:
        lon, lat = frame.to_wgs84(xs, ys)
        return [[round(float(a), 7), round(float(b), 7)] for a, b in zip(lat, lon)]

    tiles = []
    for i, j in region.tiles():
        tid = f"{i}_{j}"
        x0, y0, x1, y1 = grid.bounds(tid)
        tiles.append({"id": tid, "ring": ring([x0, x1, x1, x0], [y0, y0, y1, y1])})
    fp = np.asarray(region.footprint.exterior.coords)
    sources = [n for n in ("osm", "dem", "worldcover", "canopy_height", "overture", "imagery")
               if getattr(cfg.sources, n).enabled and (n != "canopy_height" or cfg.vegetation.enabled)]
    return {"region": region.describe(), "tiles": tiles, "footprint": ring(list(fp[:, 0]), list(fp[:, 1])),
            "tile_size": cfg.tiling.tile_size, "sources": sources}


def _world_key(world: CanonicalWorld) -> dict[str, Any]:
    return {"origin": _origin(world), "tile_size": world.config.tiling.tile_size,
            "projection": world.region.frame.proj_string}


def _origin(world: CanonicalWorld) -> dict[str, float]:
    return {"lat": world.region.center.lat, "lon": world.region.center.lon, "height": 0.0}


def export_terrain(world: CanonicalWorld, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    hf = world.terrain
    crs = world.region.frame.local_crs()
    rows, cols = hf.z.shape

    def north_up(grid: np.ndarray) -> Callable[[int, int], np.ndarray]:
        # File rows r0:r1 (row 0 = north) are grid rows rows-r1 .. rows-r0, flipped.
        return lambda r0, r1: grid[rows - r1:rows - r0][::-1][None]

    # Written in row blocks: no whole-grid copies of the heights or codes.
    write_geotiff_rows(out / "heightfield.tif", rows, cols, 1, np.float32, crs, hf.geotransform(), north_up(hf.z),
                       ["elevation_m"])
    with open(out / "heightfield.f32", "wb") as f:
        for r0 in range(0, rows, GEOTIFF_BLOCK_ROWS):
            f.write(hf.z[r0:r0 + GEOTIFF_BLOCK_ROWS].astype("<f4").tobytes())
    _write_json(out / "heightfield.json", {
        "schema": "worldgen.heightfield", "schema_version": SCHEMA_VERSION, "xmin": hf.xmin, "ymin": hf.ymin,
        "resolution": hf.resolution, "cols": hf.cols, "rows": hf.rows, "dtype": "float32", "byte_order": "little",
        "row_order": "south_to_north", "vertical_datum": hf.vertical_datum, "units": "m", "file": "heightfield.f32",
        "triangulation": "quad (i,j)-(i+1,j+1) diagonal",
        "processing": {k: v for k, v in hf.meta.items()},
        "source_coverage": float(hf.valid.mean()) if hf.valid is not None else None,
    }, indent=1)
    if world.terrain_codes is not None:
        write_geotiff_rows(out / "landcover.tif", rows, cols, 1, np.uint8, crs, hf.geotransform(),
                           north_up(world.terrain_codes), ["landcover_code"])
        _write_json(out / "landcover.json", {"codes": {str(k): v for k, v in LANDCOVER_CODES.items()}}, indent=1)


def export_fields(world: CanonicalWorld, out: Path, sums: tuple[np.ndarray, np.ndarray] | None = None) -> None:
    """``sums``: the trees' ``tree_cell_sums`` when they are not all in ``world.trees``."""
    out.mkdir(parents=True, exist_ok=True)
    g = world.vegetation_grid
    if sums is None:
        sums = tree_cell_sums(g, world.trees)
    rows, cols = g.shape
    transform = Affine(g.resolution, 0.0, g.xmin, 0.0, -g.resolution, g.ymin + rows * g.resolution)

    def block(r0: int, r1: int) -> np.ndarray:
        # File rows r0:r1 (row 0 = north): the fields of grid rows rows-r1 .. rows-r0, flipped.
        f = environmental_fields(g, sums=sums, rows=slice(rows - r1, rows - r0))
        return np.stack([f[n][::-1] for n in FIELD_NAMES]).astype(np.float32)

    write_geotiff_rows(out / "vegetation_fields.tif", rows, cols, len(FIELD_NAMES), np.float32,
                       world.region.frame.local_crs(), transform, block, FIELD_NAMES, nodata=float("nan"))
    _write_json(out / "fields.json", {
        "schema": "worldgen.fields", "schema_version": SCHEMA_VERSION, "file": "vegetation_fields.tif",
        "resolution": g.resolution, "xmin": g.xmin, "ymin": g.ymin, "cols": g.shape[1], "rows": g.shape[0],
        "bands": [
            {"band": 1, "name": "canopy_height_m", "units": "m above ground (not elevation)"},
            {"band": 2, "name": "canopy_coverage", "units": "fraction 0-1"},
            {"band": 3, "name": "vegetation_density", "units": "stems per hectare (from placed trees)"},
            {"band": 4, "name": "ground_obstruction", "units": "fraction 0-1 (shrubs + trunk footprints)"},
            {"band": 5, "name": "visual_occlusion", "units": "approx. fraction of a 10 m horizontal sight line blocked"},
            {"band": 6, "name": "confidence", "units": "0-1"},
            {"band": 7, "name": "vegetation_category", "units": "code", "legend": dict(enumerate(CATEGORIES))},
        ],
        "canopy_height_observed_fraction": g.meta.get("canopy_observed_fraction"),
        "note": "Environmental descriptors only; movement, visibility or concealment rules are application-specific.",
    }, indent=1)


def attribution_text(world: CanonicalWorld) -> str:
    lines = ["Data sources used to compile this world", "=" * 40, ""]
    for s in world.sources:
        if s.status == "disabled":
            continue
        lines.append(f"* {s.name} ({s.provider}) - {s.status}")
        lines.append(f"  {s.attribution}")
        lines.append(f"  License: {s.license}")
        if s.dataset_version:
            lines.append(f"  Version: {s.dataset_version}")
        lines.append("")
    return "\n".join(lines)


def recorded_hashes(out: Path, mode: str) -> dict[str, list[Any]]:
    """Output path -> ``[sha256, size, mtime_ns]`` of the tile files, as their writers hashed
    them (``hashing.write_hashed``; build_state.json ``files`` of every tile and coarse tile)."""
    try:
        state = json.loads((out / "build_state.json").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return {}
    known: dict[str, list[Any]] = {}
    for tid, st in state.get("tiles", {}).items():
        paths = tile_paths(mode, tid)
        for kind, rec in st.get("files", {}).items():
            known[paths[kind]] = rec
    for key, st in state.get("levels", {}).items():
        level, ij = key.split("/")
        paths = level_paths(int(level), *parse_tile_id(ij))
        for kind, rec in st.get("files", {}).items():
            known[paths[kind]] = rec
    return known


def write_manifest(world: CanonicalWorld, report: Report, opts: BuildOptions, duration: float) -> dict[str, Any]:
    out = opts.output
    cfg = world.config
    region = world.region
    hf = world.terrain
    # Tile files keep the hash their writer took while their size and modification time are
    # unchanged; the others are read and hashed here.
    known = recorded_hashes(out, cfg.export.mode)
    outputs: dict[str, Any] = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json", "build_state.json"):
            rel = p.relative_to(out).as_posix()
            st = p.stat()
            rec = known.get(rel)
            sha = rec[0] if rec is not None and rec[1] == st.st_size and rec[2] == st.st_mtime_ns else sha256_file(p)
            outputs[rel] = {"sha256": sha, "bytes": st.st_size}
    zr = [float(hf.z.min()), float(hf.z.max())] if hf is not None else [0.0, 0.0]
    xmin, ymin, xmax, ymax = region.extent
    summary = report.to_dict(world)["summary"]
    libs = {}
    for lib in ("numpy", "shapely", "pyproj", "rasterio", "geopandas", "pyarrow", "trimesh", "networkx", "scipy"):
        try:
            libs[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    data = cfg.export.mode == "data"
    tiling: dict[str, Any] = {"tile_size": cfg.tiling.tile_size, "tile_count": len(world_tiles(world))}
    if data:
        tiling.update(mode="data", tileset="tiles/tileset.json")
    else:
        tiling["index"] = "tiles/index.json"
    manifest = {
        "schema": "worldgen.manifest", "schema_version": DATA_SCHEMA_VERSION if data else SCHEMA_VERSION,
        "compiler": {"name": "worldgen", "version": __version__, "python": platform.python_version(),
                     "libraries": libs},
        "build": {"created_utc": _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                  "duration_s": round(duration, 2), "stage_timings_s": world.timings,
                  "fixture": opts.fixture, "offline": opts.offline},
        "region": region.describe(),
        "crs": {"geographic": "EPSG:4326", "local": region.frame.describe(),
                "vertical_datum": hf.vertical_datum if hf is not None else None, "units": "m",
                "axes": "x=east,y=north,z=up"},
        "origin": _origin(world),
        "gltf": {"up": "+Y", "mapping": "gltf = (x, z, -y)", "tile_root_translation": "tile minimum corner"},
        "local_bounds": {"xmin": xmin, "ymin": ymin, "xmax": xmax, "ymax": ymax,
                         "zmin": round(zr[0], 3), "zmax": round(zr[1], 3)},
        "tiling": tiling,
        "layers": LAYER_ORDER,
        "config": cfg.to_dict(), "config_hash": cfg.hash(),
        "sources": [s.to_dict() for s in world.sources],
        "outputs": outputs,
        "statistics": world.stats,
        "validation": {"errors": summary["errors"], "warnings": summary["warnings"], "info": summary["info"],
                       "report": "diagnostics.json"},
        "files": {"graph": "graph/road_graph.json", "diagnostics": "diagnostics.json",
                  "combined_glb": "world.glb" if (out / "world.glb").is_file() else None,
                  "heightfield": "terrain/heightfield.json", "fields": "fields/fields.json",
                  "vectors": "vectors/", "attribution": "ATTRIBUTION.txt"},
    }
    if data:
        manifest["files"]["graph_core"] = "graph/core.wgt"
        manifest["files"]["graph_strings"] = "graph/strings.json"
    _write_json(out / "manifest.json", manifest, indent=1)
    return manifest
