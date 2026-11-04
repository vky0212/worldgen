"""Top-level build: region -> sources -> canonical world -> assets + manifest."""

from __future__ import annotations

import datetime as _dt
import json
import platform
import time
from dataclasses import dataclass
from importlib import metadata
from pathlib import Path
from typing import Any

import numpy as np
from rasterio.transform import Affine

from worldgen import SCHEMA_VERSION, __version__
from worldgen.config import WorldConfig
from worldgen.errors import ValidationFailed
from worldgen.export.merge import merge_glbs
from worldgen.export.ortho import ortho_baker
from worldgen.export.tiles import (
    LAYER_ORDER,
    NO_TREES,
    archetypes_for,
    build_tile_assets,
    entity_strtree,
    generate_pieces,
    terrain_render_inputs,
    tile_input_hash,
    trees_by_tile,
    world_tiles,
)
from worldgen.export.vectors import write_geoparquet
from worldgen.hashing import sha256_file
from worldgen.landcover.landcover import LANDCOVER_CODES
from worldgen.log import get_logger, stage
from worldgen.pipeline import acquire, bundle_from_fixture, compile_world
from worldgen.progress import current
from worldgen.sources.base import SourceRecord, write_geotiff
from worldgen.sources.synthetic import load_fixture
from worldgen.spatial.raster import GeoRaster
from worldgen.spatial.region import Region
from worldgen.tiling.grid import TileGrid
from worldgen.validation.checks import Report, validate_world
from worldgen.vegetation.classify import CATEGORIES
from worldgen.vegetation.regions import environmental_fields
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


def build(cfg: WorldConfig, opts: BuildOptions) -> BuildResult:
    t0 = time.perf_counter()
    region = Region.from_config(cfg)
    timings: dict[str, float] = {}
    progress = current()
    if progress.enabled:
        progress.plan(**plan_payload(region, cfg))
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
    world = compile_world(region, cfg, bundle)
    world.timings = {**timings, **world.timings}
    with stage("validate", world.timings):
        report = validate_world(world) if cfg.validation.enabled else Report()
    with stage("export", world.timings):
        written, skipped = export_world(world, report, opts)
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
    (out / "tiles").mkdir(parents=True, exist_ok=True)
    grid = TileGrid(cfg.tiling.tile_size)
    tiles = world_tiles(world)

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
    # World-level indexes, built once: per-tile lookups must not scan every tree or entity.
    etree = entity_strtree(world)
    tindex = trees_by_tile(world)
    hashes = {t: tile_input_hash(world, t, etree, tindex) for t in tiles}
    baker = ortho_baker(world)
    dirty = []
    for t in tiles:
        files_ok = (out / "tiles" / f"{t}.glb").is_file() and (out / "tiles" / f"{t}.meta.json").is_file()
        if baker is not None:
            files_ok = files_ok and (out / "tiles" / f"{t}.ortho.jpg").is_file()
        if opts.force or not files_ok or prev_tiles.get(t, {}).get("hash") != hashes[t]:
            dirty.append(t)
    if opts.only_tiles is not None:
        unknown = opts.only_tiles - set(tiles)
        if unknown:
            raise ValueError(f"unknown tiles requested: {sorted(unknown)}")
        dirty = [t for t in dirty if t in opts.only_tiles]
    skipped = [t for t in tiles if t not in dirty]
    log.info("tiles: %d total, %d to (re)build, %d unchanged", len(tiles), len(dirty), len(skipped))
    progress = current()
    for t in skipped:
        progress.tile(t, "unchanged")
    total_steps = len(dirty)
    progress.step("export", 0, total_steps, "generating geometry")

    pieces = generate_pieces(world, set(dirty)) if dirty else {}
    normals, rgba = terrain_render_inputs(world)
    arche = archetypes_for(world)
    eindex = world.entity_index()
    tile_state = dict(prev_tiles)
    for k, t in enumerate(dirty):
        progress.tile(t, "building")
        assets = build_tile_assets(world, t, pieces.get(t, []), normals, rgba, arche,
                                   trees_in_tile=tindex.get(t, NO_TREES), entity_index=eindex)
        (out / "tiles" / f"{t}.glb").write_bytes(assets["glb"])
        col = out / "tiles" / f"{t}.collision.glb"
        if assets["collision"] is not None:
            col.write_bytes(assets["collision"])
        elif col.exists():
            col.unlink()
        _write_json(out / "tiles" / f"{t}.meta.json", assets["meta"])
        ortho = out / "tiles" / f"{t}.ortho.jpg"
        if baker is not None:
            ortho.write_bytes(baker.bake(grid.bounds(t)))
        elif ortho.exists():
            ortho.unlink()
        tile_state[t] = {"hash": hashes[t], "counts": assets["counts"], "zrange": assets["zrange"],
                         "collision": assets["collision"] is not None, "ortho": baker is not None}
        progress.tile(t, "built")
        progress.step("export", k + 1, total_steps, f"tile {t}")
    for t in list(tile_state):
        if t not in tiles:
            tile_state.pop(t)

    ortho_info = baker.describe() if baker is not None else None
    if ortho_info is not None:
        world.stats["ortho"] = ortho_info
    index = {"schema": "worldgen.tile_index", "schema_version": SCHEMA_VERSION, "tile_size": cfg.tiling.tile_size,
             "layers": LAYER_ORDER, "ortho": ortho_info, "tiles": []}
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
    _write_json(state_path, {"world_key": _world_key(world), "tiles": tile_state}, indent=1)

    if cfg.export.combined_glb:
        files = [(out / "tiles" / f"{t}.glb").read_bytes() for t in tiles if (out / "tiles" / f"{t}.glb").is_file()]
        if files:
            (out / "world.glb").write_bytes(merge_glbs(files, "world", {"origin": _origin(world)}))

    _write_json(out / "graph" / "road_graph.json", world.graph.to_dict(world.region.frame, _origin(world)))
    if cfg.export.geoparquet:
        write_geoparquet(world, out / "vectors")
    if cfg.export.heightfield:
        export_terrain(world, out / "terrain")
    if cfg.export.fields and world.vegetation_grid is not None:
        export_fields(world, out / "fields")
    _write_json(out / "diagnostics.json", report.to_dict(world), indent=1)
    (out / "ATTRIBUTION.txt").write_text(attribution_text(world), encoding="utf-8")
    return dirty, skipped


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
    north_up = np.flipud(hf.z).astype(np.float32)
    write_geotiff(out / "heightfield.tif", GeoRaster(north_up, hf.geotransform(), crs, None, "heightfield"),
                  ["elevation_m"])
    hf.z.astype("<f4").tofile(out / "heightfield.f32")
    _write_json(out / "heightfield.json", {
        "schema": "worldgen.heightfield", "schema_version": SCHEMA_VERSION, "xmin": hf.xmin, "ymin": hf.ymin,
        "resolution": hf.resolution, "cols": hf.cols, "rows": hf.rows, "dtype": "float32", "byte_order": "little",
        "row_order": "south_to_north", "vertical_datum": hf.vertical_datum, "units": "m", "file": "heightfield.f32",
        "triangulation": "quad (i,j)-(i+1,j+1) diagonal",
        "processing": {k: v for k, v in hf.meta.items()},
        "source_coverage": float(hf.valid.mean()) if hf.valid is not None else None,
    }, indent=1)
    if world.terrain_codes is not None:
        codes = np.flipud(world.terrain_codes).astype(np.uint8)
        write_geotiff(out / "landcover.tif", GeoRaster(codes, hf.geotransform(), crs, None, "landcover"),
                      ["landcover_code"])
        _write_json(out / "landcover.json", {"codes": {str(k): v for k, v in LANDCOVER_CODES.items()}}, indent=1)


def export_fields(world: CanonicalWorld, out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    g = world.vegetation_grid
    fields = environmental_fields(g, world.trees)
    names = list(fields)
    stack = np.stack([np.flipud(fields[n]) for n in names]).astype(np.float32)
    rows = g.shape[0]
    transform = Affine(g.resolution, 0.0, g.xmin, 0.0, -g.resolution, g.ymin + rows * g.resolution)
    write_geotiff(out / "vegetation_fields.tif",
                  GeoRaster(stack, transform, world.region.frame.local_crs(), float("nan"), "fields"), names)
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


def write_manifest(world: CanonicalWorld, report: Report, opts: BuildOptions, duration: float) -> dict[str, Any]:
    out = opts.output
    cfg = world.config
    region = world.region
    hf = world.terrain
    outputs: dict[str, Any] = {}
    for p in sorted(out.rglob("*")):
        if p.is_file() and p.name not in ("manifest.json", "build_state.json"):
            outputs[p.relative_to(out).as_posix()] = {"sha256": sha256_file(p), "bytes": p.stat().st_size}
    zr = [float(hf.z.min()), float(hf.z.max())] if hf is not None else [0.0, 0.0]
    xmin, ymin, xmax, ymax = region.extent
    summary = report.to_dict(world)["summary"]
    libs = {}
    for lib in ("numpy", "shapely", "pyproj", "rasterio", "geopandas", "pyarrow", "trimesh", "networkx", "scipy"):
        try:
            libs[lib] = metadata.version(lib)
        except metadata.PackageNotFoundError:
            pass
    manifest = {
        "schema": "worldgen.manifest", "schema_version": SCHEMA_VERSION,
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
        "tiling": {"tile_size": cfg.tiling.tile_size, "tile_count": len(world_tiles(world)),
                   "index": "tiles/index.json"},
        "layers": LAYER_ORDER,
        "config": cfg.to_dict(), "config_hash": cfg.hash(),
        "sources": [s.to_dict() for s in world.sources],
        "outputs": outputs,
        "statistics": world.stats,
        "validation": {"errors": summary["errors"], "warnings": summary["warnings"], "info": summary["info"],
                       "report": "diagnostics.json"},
        "files": {"graph": "graph/road_graph.json", "diagnostics": "diagnostics.json",
                  "combined_glb": "world.glb" if cfg.export.combined_glb else None,
                  "heightfield": "terrain/heightfield.json", "fields": "fields/fields.json",
                  "vectors": "vectors/", "attribution": "ATTRIBUTION.txt"},
    }
    _write_json(out / "manifest.json", manifest, indent=1)
    return manifest
