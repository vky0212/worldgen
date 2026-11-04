"""Compiler pipeline: acquisition -> normalization -> canonical world -> compilation."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import shapely
from shapely.ops import unary_union

from worldgen.budget import building_sizes, select_within_budget, tree_budget_uniforms, tree_sizes
from worldgen.buildings.buildings import assign_base_elevations, prepare_buildings
from worldgen.config import WorldConfig
from worldgen.errors import SourceError
from worldgen.landcover.landcover import classify_terrain
from worldgen.landcover.smoothing import refine_vegetation, shape_shorelines, smooth_landcover
from worldgen.landcover.water import (
    carve_inputs,
    compute_water_levels,
    sea_from_coastlines,
    shore_inputs,
    waterway_outlines,
    waterways_covered_by_areas,
)
from worldgen.log import get_logger, stage
from worldgen.progress import SOURCES as SOURCE_ORDER
from worldgen.progress import current
from worldgen.sources.base import SourceCache, SourceRecord
from worldgen.sources.imagery import fetch_imagery
from worldgen.sources.osm import OverpassClient
from worldgen.sources.osm_normalize import normalize_osm
from worldgen.sources.overture import fetch_overture_buildings
from worldgen.sources.rasters import fetch_canopy_height, fetch_dem, fetch_worldcover
from worldgen.sources.synthetic import Fixture
from worldgen.spatial.region import Region
from worldgen.terrain.build import Corridor, apply_corridors, build_heightfield, carve_polygons
from worldgen.topology.crossings import find_road_crossings, find_water_crossings
from worldgen.topology.graph import RoadGraph
from worldgen.transportation.elevation import compute_elevations
from worldgen.transportation.junctions import compute_junctions
from worldgen.transportation.meshing import build_intersections
from worldgen.vegetation.classify import classify_vegetation
from worldgen.vegetation.placement import Exclusions, concat_tables, place_ground_cover, place_trees
from worldgen.vegetation.regions import extract_regions
from worldgen.tiling.grid import TileGrid
from worldgen.world.world import CanonicalWorld, SourceBundle

log = get_logger("pipeline")


# --------------------------------------------------------------------------------------
# acquisition


def acquire(region: Region, cfg: WorldConfig, cache_dir: str | Path, offline: bool = False) -> SourceBundle:
    """Fetch (or read from cache) every enabled source for the region."""
    cache = SourceCache(cache_dir, offline=offline)
    bundle = SourceBundle()
    margin = cfg.sources.osm.margin_m
    bbox_osm = region.wgs84_bbox(margin)
    bbox_raster = region.wgs84_bbox(max(margin, 4 * cfg.terrain.resolution) + 60.0)

    progress = current()

    def guarded(name: str, fn):  # noqa: ANN001, ANN202
        progress.source(name, "start")
        n_before = len(bundle.records)
        try:
            res = fn()
        except SourceError as exc:
            log.error("%s acquisition failed: %s", name, exc)
            bundle.records.append(SourceRecord(name, "error", "-", "-", status="failed", notes=[str(exc)]))
            progress.source(name, "failed", str(exc))
            return None
        progress.step("acquire", SOURCE_ORDER.index(name) + 1, len(SOURCE_ORDER), name)
        return res

    def report(rec: SourceRecord) -> None:
        detail = "from cache" if cache.last_hit else "downloaded"
        if rec.status == "disabled":
            detail = "disabled in config"
        elif rec.dataset_version:
            detail = f"{detail} · {rec.dataset_version}"
        progress.source(rec.name, rec.status, detail)

    if cfg.sources.osm.enabled:
        res = guarded("osm", lambda: OverpassClient(cfg.sources.osm, cache).fetch(bbox_osm))
        if res is not None:
            bundle.osm, rec = res
            bundle.records.append(rec)
            report(rec)
    else:
        bundle.records.append(SourceRecord("osm", "overpass", "-", "-", status="disabled"))
        report(bundle.records[-1])
    dem = guarded("dem", lambda: fetch_dem(cfg.sources.dem, bbox_raster, cache))
    if dem is not None:
        bundle.dem = dem.raster
        bundle.records.append(dem.record)
        report(dem.record)
        bundle.dem_datum = dem.record.vertical_datum or cfg.sources.dem.vertical_datum
    wc = guarded("worldcover", lambda: fetch_worldcover(cfg.sources.worldcover, bbox_raster, cache))
    if wc is not None:
        bundle.worldcover, rec = wc
        bundle.records.append(rec)
        report(rec)
    if not cfg.vegetation.enabled:
        progress.source("canopy_height", "disabled", "vegetation disabled")
    else:
        ch = guarded("canopy_height", lambda: fetch_canopy_height(cfg.sources.canopy_height, bbox_raster, cache))
        if ch is not None:
            bundle.canopy, rec = ch
            bundle.records.append(rec)
            report(rec)
    ov = guarded("overture", lambda: fetch_overture_buildings(cfg.sources.overture, bbox_osm, region.frame, cache))
    if ov is not None:
        bundle.overture_buildings, rec = ov
        bundle.records.append(rec)
        report(rec)
    im = guarded("imagery", lambda: fetch_imagery(cfg.sources.imagery, bbox_raster, cache))
    if im is not None:
        bundle.imagery, rec = im
        bundle.records.append(rec)
        report(rec)
    return bundle


def bundle_from_fixture(fx: Fixture) -> SourceBundle:
    return SourceBundle(osm=fx.osm, dem=fx.dem, worldcover=fx.worldcover, canopy=fx.canopy, records=fx.records())


# --------------------------------------------------------------------------------------
# compilation


def compile_world(region: Region, cfg: WorldConfig, bundle: SourceBundle) -> CanonicalWorld:
    world = CanonicalWorld(region=region, config=cfg, sources=list(bundle.records), imagery=bundle.imagery)
    t = world.timings
    extent = region.extent_box

    with stage("normalize", t):
        if bundle.osm is not None:
            norm = normalize_osm(bundle.osm, region.frame, extent, cfg)
            world.roads = {r.id: r for r in norm.roads}
            osm_buildings = norm.buildings
            world.water = norm.water
            world.landcover = norm.landcover
            world.observed_trees = norm.trees
            world.restrictions = norm.restrictions
            world.coastlines = norm.coastlines
            world.stats["osm_skipped"] = norm.skipped
        else:
            osm_buildings = []
        world.water += sea_from_coastlines(world.coastlines, extent)
        world.buildings, bstats = prepare_buildings(osm_buildings, bundle.overture_buildings, extent, cfg)
        world.stats["buildings_prep"] = bstats

    with stage("terrain", t):
        hf = build_heightfield(region, cfg, bundle.dem, bundle.worldcover, bundle.dem_datum)
        world.terrain_base = hf.copy()
        world.terrain = hf

    with stage("topology", t):
        graph = RoadGraph.build(list(world.roads.values()), world.restrictions)
        world.graph = graph
        world.crossings = find_road_crossings(graph)
        world.water_crossings = find_water_crossings(graph, [w for w in world.water if not w.properties.get("tunnel")])

    with stage("water_levels", t):
        levels, wprofiles = compute_water_levels(world.water, world.terrain_base, cfg)
        world.water_levels = levels
        world.waterway_profiles = wprofiles
        world.covered_waterways = waterways_covered_by_areas(world.water)

    with stage("road_elevation", t):
        er = compute_elevations(graph, world.roads, world.terrain_base, world.crossings, world.water_crossings,
                                {w.id: w for w in world.water}, levels, cfg)
        world.profiles = er.profiles
        world.bridges = er.bridges
        world.tunnels = er.tunnels
        world.stage_issues += er.issues
        for c in world.crossings:
            if c.kind != "grade_separated":
                continue
            for eid in (c.upper, c.lower):
                sid = graph.edges[eid].structure_id
                if sid is None:
                    continue
                st = next((s for s in world.bridges + world.tunnels if s.id == sid), None)
                other = graph.edges[c.lower if eid == c.upper else c.upper].road_id
                if st is not None and other not in st.crossings and other not in st.road_ids:
                    st.crossings.append(other)

    with stage("terrain_shaping", t):
        _shape_terrain(world, cfg)

    with stage("buildings", t):
        assign_base_elevations(world.buildings, world.terrain)
        _building_budget(world, cfg)

    with stage("landcover", t):
        world.terrain_codes = classify_terrain(world.terrain, region, bundle.worldcover, world.landcover, world.water)
        if cfg.vegetation.enabled:
            world.vegetation_grid = classify_vegetation(region, cfg.vegetation, bundle.worldcover, bundle.canopy,
                                                        world.landcover, world.water)

    if cfg.smoothing.enabled:
        with stage("smoothing", t):
            _smooth_borders(world, cfg)
    else:
        current().stage_skipped("smoothing")

    with stage("intersections", t):
        layout = compute_junctions(graph, world.profiles, cfg)
        if not cfg.roads.intersections:
            layout.junctions.clear()
            layout.trims.clear()
        world.junctions = layout
        world.intersections, _ = build_intersections(graph, layout, TileGrid(cfg.tiling.tile_size))

    if cfg.vegetation.enabled:
        with stage("vegetation", t):
            _vegetation(world, cfg)
    else:
        current().stage_skipped("vegetation")

    _assign_edge_tiles(world)
    world.stats.update(_stats(world))
    return world


def _shape_terrain(world: CanonicalWorld, cfg: WorldConfig) -> None:
    hf = world.terrain
    assert hf is not None
    graph = world.graph
    polys, wcorr = carve_inputs(world.water, world.water_levels, world.waterway_profiles,
                                world.covered_waterways, cfg)
    sm = cfg.smoothing
    if sm.enabled and sm.shorelines:
        # The terrain meets lake/river/sea surfaces on their mapped outline (no grid zig-zag).
        n_water = shape_shorelines(hf, shore_inputs(world.water, world.water_levels, cfg),
                                   world.region.extent_box, sm.shore_slope_min, sm.shore_slope_max)
    else:
        n_water = carve_polygons(hf, polys)
    n_water += apply_corridors(hf, wcorr)
    n_road = 0
    if cfg.terrain.road_conformance:
        road_corr = []
        for eid in sorted(graph.edges):
            e = graph.edges[eid]
            p = world.profiles[eid]
            if p.kind != "ground":
                continue
            # Conform one full terrain cell beyond the road edge so no terrain triangle touching
            # the road surface can rise above it.
            road_corr.append(Corridor(p.xy, p.z, e.width / 2.0 + 1.5 * hf.resolution, cfg.terrain.road_shoulder_m))
        n_road = apply_corridors(hf, road_corr)
    # Last, so nothing raises it again: terrain never rises above a bridge deck (a deck next to
    # a higher ground road would otherwise be buried by that road's conformance).
    bridge_corr = []
    for b in world.bridges:
        for eid in b.edge_ids:
            p = world.profiles[eid]
            bridge_corr.append(Corridor(p.xy, p.z - 0.4, graph.edges[eid].width / 2.0 + hf.resolution, 1.5,
                                        mode="max"))
    n_bridge = apply_corridors(hf, bridge_corr)
    world.stats["terrain_shaping"] = {"water_nodes": n_water, "bridge_nodes": n_bridge, "road_nodes": n_road}


def _smooth_borders(world: CanonicalWorld, cfg: WorldConfig) -> None:
    """Post-processing: smooth, natural land-cover and vegetation borders (landcover/smoothing.py)."""
    sm = cfg.smoothing
    seed = cfg.vegetation.seed
    progress = current()
    areas = [w.geometry for w in world.water if w.is_area]
    res = smooth_landcover(world.terrain_codes, world.terrain, areas,
                           waterway_outlines(world.water, world.waterway_profiles), sm, seed)
    world.terrain_codes = res.codes
    world.terrain_rgba = res.colors
    stats = dict(res.stats)
    progress.step("smoothing", 1, 2, "land cover")
    if world.vegetation_grid is not None:
        world.vegetation_grid = refine_vegetation(world.vegetation_grid, areas, sm, seed)
        stats["vegetation_cells_changed"] = world.vegetation_grid.meta["smoothing"]["cells_changed"]
    progress.step("smoothing", 2, 2, "vegetation")
    world.stats["smoothing"] = stats


def _vegetation(world: CanonicalWorld, cfg: WorldConfig) -> None:
    vc = cfg.vegetation
    progress = current()
    grid = world.vegetation_grid
    # Region outlines keep the configured minimum size when the grid was refined.
    world.vegetation_regions = extract_regions(grid, world.landcover, vc, min_area=2.0 * vc.resolution ** 2)
    progress.step("vegetation", 1, 3, f"{len(world.vegetation_regions)} regions")
    excl = vegetation_exclusions(world, vc.exclusion_buffer)
    trees = place_trees(grid, world.terrain, excl, world.observed_trees, vc)  # type: ignore[arg-type]
    progress.step("vegetation", 2, 3, f"{len(trees)} trees placed")
    ground = place_ground_cover(grid, world.terrain, excl, vc)  # type: ignore[arg-type]
    world.trees = concat_tables([trees, ground])
    _tree_budget(world, cfg)


def _building_budget(world: CanonicalWorld, cfg: WorldConfig) -> None:
    bc = cfg.buildings.budget
    for b in world.buildings:
        b.budget_dropped = False
    # Building parts and outlines rendered by parts form landmarks; they are never dropped.
    cand = [b for b in world.buildings if not b.is_part and not b.rendered_by_parts]
    if not cand:
        return
    grid = TileGrid(cfg.tiling.tile_size)
    pts = [b.geometry.representative_point() for b in cand]
    tiles = grid.tiles_of(np.array([p.x for p in pts]), np.array([p.y for p in pts]))
    res = select_within_budget([b.id for b in cand], tiles, building_sizes(cand),
                               np.zeros(len(cand), dtype=bool), bc, cfg.vegetation.seed, "buildings")
    for b, k in zip(cand, res.keep):
        b.budget_dropped = not bool(k)
    world.stats["buildings_budget_dropped"] = res.dropped
    if res.dropped:
        world.stage_issues.append({
            "check": "instance_budget", "severity": "info", "entity_ids": [],
            "message": f"{res.dropped} smaller buildings not rendered (budget {bc.max_per_tile}/tile"
                       f"{f', {bc.max_total} total' if bc.max_total else ''}) in {len(res.tiles_over_budget)} "
                       f"tile(s): {', '.join(res.tiles_over_budget[:8])}; still present in vectors/buildings.parquet",
        })


def _tree_budget(world: CanonicalWorld, cfg: WorldConfig) -> None:
    vb = cfg.vegetation.budget
    tr = world.trees
    if not len(tr):
        return
    grid = TileGrid(cfg.tiling.tile_size)
    protected = tr.observed.copy() if vb.keep_observed else np.zeros(len(tr), dtype=bool)
    res = select_within_budget(tr.ids, grid.tiles_of(tr.x, tr.y), tree_sizes(tr.height, tr.crown_radius),
                               protected, vb, cfg.vegetation.seed, "trees",
                               u=tree_budget_uniforms(cfg.vegetation.seed, tr.seed))
    tr.rendered = res.keep
    world.stats["trees_budget_dropped"] = res.dropped
    if res.dropped:
        world.stage_issues.append({
            "check": "instance_budget", "severity": "info", "entity_ids": [],
            "message": f"{res.dropped} smaller trees/plants not rendered (budget {vb.max_per_tile}/tile"
                       f"{f', {vb.max_total} total' if vb.max_total else ''}) in {len(res.tiles_over_budget)} "
                       f"tile(s); still present in vectors/trees.parquet",
        })


def vegetation_exclusions(world: CanonicalWorld, buffer: float) -> Exclusions:
    parts = [b.geometry.buffer(buffer) for b in world.buildings]
    graph = world.graph
    for e in graph.edges.values():
        if e.tunnel:
            continue
        parts.append(e.line.buffer(e.width / 2.0 + buffer, cap_style="flat"))
    for w in world.water:
        if w.is_area:
            parts.append(w.geometry)
        elif w.id in world.waterway_profiles and w.id not in world.covered_waterways:
            parts.append(w.geometry.buffer(float(w.get("width", 3.0)) / 2.0))
    for i in world.intersections:
        parts.append(i.geometry)
    if not parts:
        return Exclusions(None)
    u = unary_union(parts)
    shapely.prepare(u)
    return Exclusions(u)


def _assign_edge_tiles(world: CanonicalWorld) -> None:
    grid = TileGrid(world.config.tiling.tile_size)
    for e in world.graph.edges.values():
        xy, = grid.insert_tile_crossings(e.coords)
        e.tiles = sorted(set(grid.segment_tiles(xy)))


def _stats(world: CanonicalWorld) -> dict:
    g = world.graph
    kinds: dict[str, int] = {}
    for n in g.nodes.values():
        kinds[n.kind] = kinds.get(n.kind, 0) + 1
    return {
        "roads": len(world.roads),
        "graph_nodes": len(g.nodes),
        "graph_edges": len(g.edges),
        "graph_node_kinds": kinds,
        "turn_restrictions": len(g.restrictions),
        "turn_restrictions_enforced": sum(1 for r in g.restrictions if r.enforced),
        "bridges": len(world.bridges),
        "tunnels": len(world.tunnels),
        "intersections": len(world.intersections),
        "grade_separated_crossings": sum(1 for c in world.crossings if c.kind == "grade_separated"),
        "unconnected_same_level_crossings": sum(1 for c in world.crossings if c.kind == "same_level_unconnected"),
        "buildings": len(world.buildings),
        "buildings_with_observed_height": sum(1 for b in world.buildings if b.is_observed("height")),
        "water_bodies": len(world.water),
        "landcover_areas": len(world.landcover),
        "vegetation_regions": len(world.vegetation_regions),
        "trees": len(world.trees),
        "trees_observed": int(np.sum(world.trees.observed)) if len(world.trees) else 0,
        "trees_rendered": int(world.trees.rendered_mask().sum()) if len(world.trees) else 0,
        "buildings_rendered": sum(1 for b in world.buildings if not b.budget_dropped),
    }
