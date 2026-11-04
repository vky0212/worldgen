"""GeoParquet export of the semantic layers (EPSG:4326 geometry)."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
from shapely.geometry import LineString, Point

from worldgen.world.model import Entity
from worldgen.world.world import CanonicalWorld

COMMON = ["id", "kind", "source", "source_id", "confidence", "uncertain"]


def _frame(rows: list[dict[str, Any]], geoms: list[Any]) -> gpd.GeoDataFrame:
    df = gpd.GeoDataFrame(rows, geometry=geoms, crs="EPSG:4326")
    return df


def entity_rows(world: CanonicalWorld, ents: list[Entity], extra: list[str]) -> gpd.GeoDataFrame:
    rows, geoms = [], []
    frame = world.region.frame
    for e in ents:
        rec = e.record()
        row = {
            "id": e.id, "kind": e.kind, "source": e.source, "source_id": e.source_id,
            "confidence": float(e.confidence), "uncertain": bool(e.uncertain),
            "properties": json.dumps(rec["properties"], sort_keys=True, ensure_ascii=False),
            "inferred_properties": json.dumps(rec["inferred"], sort_keys=True, ensure_ascii=False),
            "missing": json.dumps(rec["missing"]),
        }
        for k in extra:
            v = e.get(k)
            if isinstance(v, (list, dict)):
                v = json.dumps(v, sort_keys=True)
            row[k] = v
            row[f"{k}_inferred"] = (k not in e.properties or e.properties.get(k) is None) and k in e.inferred_properties
        rows.append(row)
        geoms.append(frame.geom_to_wgs84(e.geometry))
    return _frame(rows, geoms)


def write_geoparquet(world: CanonicalWorld, out_dir: Path) -> dict[str, Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    def write(name: str, df: gpd.GeoDataFrame) -> None:
        p = out_dir / f"{name}.parquet"
        for col in df.columns:
            if df[col].dtype == object and col != "geometry":
                df[col] = df[col].map(lambda v: None if v is None else (v if isinstance(v, str) else json.dumps(v)))
        df.to_parquet(p, index=False, compression="zstd", write_covering_bbox=False)
        files[name] = p

    roads = [world.roads[k] for k in sorted(world.roads)]
    write("roads", entity_rows(world, roads, ["class", "name", "width", "lanes", "layer", "bridge", "tunnel",
                                              "direction", "surface", "maxspeed_kmh"]))
    write("bridges", entity_rows(world, world.bridges, ["name", "width", "length", "clearance",
                                                        "deck_elevation_range", "elevation_confidence"]))
    write("tunnels", entity_rows(world, world.tunnels, ["name", "width", "length", "deck_elevation_range",
                                                        "elevation_confidence"]))
    write("buildings", entity_rows(world, world.buildings, ["class", "name", "height", "levels", "min_height",
                                                            "roof_shape", "ground_elevation"]))
    write("water", entity_rows(world, world.water, ["class", "name", "width", "surface_elevation"]))
    write("landcover", entity_rows(world, world.landcover, ["class", "tag", "name"]))
    write("vegetation_regions", entity_rows(world, world.vegetation_regions,
                                            ["category", "coverage_fraction", "canopy_height_mean",
                                             "density_stems_per_ha", "area_m2"]))
    write("intersections", entity_rows(world, world.intersections, ["degree"]))
    write("trees", trees_frame(world))
    nodes, edges = graph_frames(world)
    write("graph_nodes", nodes)
    write("graph_edges", edges)
    return files


def trees_frame(world: CanonicalWorld) -> gpd.GeoDataFrame:
    t = world.trees
    frame = world.region.frame
    if len(t) == 0:
        return _frame([], [])
    lon, lat = frame.to_wgs84(t.x, t.y)
    rows = {
        "id": t.ids, "x": t.x, "y": t.y, "ground_elevation": t.ground_z, "height": t.height,
        "crown_radius": t.crown_radius, "trunk_radius": t.trunk_radius, "yaw": t.yaw,
        "archetype": t.archetype, "species": t.species, "category": t.category,
        "placement_confidence": t.confidence, "observed": t.observed, "seed": t.seed.astype(np.int64),
        "source": t.source, "source_id": t.source_id,
        "inferred_properties": [json.dumps(sorted(v)) for v in t.inferred],
    }
    return gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")


def graph_frames(world: CanonicalWorld) -> tuple[gpd.GeoDataFrame, gpd.GeoDataFrame]:
    g = world.graph
    frame = world.region.frame
    nrows, ngeo = [], []
    for nid in sorted(g.nodes):
        n = g.nodes[nid]
        nrows.append({"id": nid, "kind": n.kind, "degree": n.degree, "x": n.x, "y": n.y, "z": n.z})
        ngeo.append(frame.geom_to_wgs84(Point(n.x, n.y)))
    erows, egeo = [], []
    for eid in sorted(g.edges):
        e = g.edges[eid]
        erows.append({"id": eid, "road_id": e.road_id, "from": e.u, "to": e.v, "direction": e.direction,
                      "length_m": e.length, "class": e.road_class, "bridge": e.bridge, "tunnel": e.tunnel,
                      "layer": e.layer, "structure_id": e.structure_id,
                      "motor_vehicle": e.modes.get("motor_vehicle"), "bicycle": e.modes.get("bicycle"),
                      "foot": e.modes.get("foot")})
        egeo.append(frame.geom_to_wgs84(LineString(e.coords)))
    return _frame(nrows, ngeo), _frame(erows, egeo)
