"""GeoParquet export of the semantic layers (EPSG:4326 geometry)."""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import geopandas as gpd
import numpy as np
import pyarrow as pa
import pyarrow.parquet as pq
from geopandas.io.arrow import _geopandas_to_arrow
from shapely.geometry import LineString, Point

from worldgen.world.model import Entity, TreeTable
from worldgen.world.world import CanonicalWorld

COMMON = ["id", "kind", "source", "source_id", "confidence", "uncertain"]
# trees.parquet: row groups of 128k rows (a bounded merge buffer for the chunked writer), and
# dictionary pages only for the low-cardinality columns (on unique floats, ids and points they
# only cost space before falling back) - smaller than one dictionary-encoded row group.
TREES_PARQUET = {"compression": "zstd", "row_group_size": 1 << 17,
                 "use_dictionary": ["archetype", "species", "category", "observed", "source", "source_id",
                                    "inferred_properties", "rendered"]}


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
            "budget_dropped": bool(e.budget_dropped),
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


def _json_columns(df: gpd.GeoDataFrame) -> gpd.GeoDataFrame:
    """Object columns as strings (non-string values JSON-encoded), as every table is written."""
    for col in df.columns:
        if df[col].dtype == object and col != "geometry":
            df[col] = df[col].map(lambda v: None if v is None else (v if isinstance(v, str) else json.dumps(v)))
    return df


def write_geoparquet(world: CanonicalWorld, out_dir: Path, trees: bool = True) -> dict[str, Path]:
    """Every vector table; without ``trees`` the caller writes ``trees.parquet``
    (``TreesParquetWriter``, chunk by chunk)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    files: dict[str, Path] = {}

    def write(name: str, df: gpd.GeoDataFrame) -> None:
        p = out_dir / f"{name}.parquet"
        opts = TREES_PARQUET if name == "trees" else {"compression": "zstd"}
        _json_columns(df).to_parquet(p, index=False, write_covering_bbox=False, **opts)
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
    if trees:
        write("trees", trees_frame(world))
    nodes, edges = graph_frames(world)
    write("graph_nodes", nodes)
    write("graph_edges", edges)
    return files


def trees_frame(world: CanonicalWorld, trees: Any = None) -> gpd.GeoDataFrame:
    t = world.trees if trees is None else trees
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
        "rendered": t.rendered_mask(),
    }
    return gpd.GeoDataFrame(rows, geometry=gpd.points_from_xy(lon, lat), crs="EPSG:4326")


class TreesParquetWriter:
    """``trees.parquet`` written table by table: the same columns, types, encoding and GeoParquet
    metadata as ``write_geoparquet``, rows in the order the tables arrive. Tables stream into a
    part file; ``close`` copies its row groups, one at a time, into the final file, whose schema
    carries the GeoParquet metadata (bbox of all rows) and the column types a single table
    would have had (a column empty in every table stays null-typed)."""

    ROW_GROUP = TREES_PARQUET["row_group_size"]

    def __init__(self, world: CanonicalWorld, path: Path) -> None:
        self.world = world
        self.path = path
        self.part = path.with_name(path.name + ".part")
        self.writer: pq.ParquetWriter | None = None
        self.schema: pa.Schema | None = None
        self.geo: dict[str, Any] | None = None
        self.bbox = [math.inf, math.inf, -math.inf, -math.inf]
        self.typed: set[str] = set()   # columns not all-null in some table

    def add(self, trees: Any) -> None:
        if not len(trees):
            return
        table = _geopandas_to_arrow(_json_columns(trees_frame(self.world, trees)), index=False,
                                    write_covering_bbox=False)
        meta = dict(table.schema.metadata or {})
        geo = json.loads(meta.pop(b"geo"))
        self.typed |= {f.name for f in table.schema if not pa.types.is_null(f.type)}
        b = geo["columns"]["geometry"].get("bbox")
        if b:
            self.bbox = [min(self.bbox[0], b[0]), min(self.bbox[1], b[1]), max(self.bbox[2], b[2]),
                         max(self.bbox[3], b[3])]
        if self.writer is None:
            fields = [pa.field(f.name, pa.string()) if pa.types.is_null(f.type) else f for f in table.schema]
            self.schema = pa.schema(fields, metadata=meta)
            self.geo = geo
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.writer = pq.ParquetWriter(self.part, self.schema, compression="zstd")
        self.writer.write_table(table.replace_schema_metadata(meta).cast(self.schema))

    def close(self) -> None:
        self.path.parent.mkdir(parents=True, exist_ok=True)
        if self.writer is None:
            _json_columns(trees_frame(self.world, TreeTable())).to_parquet(
                self.path, index=False, write_covering_bbox=False, **TREES_PARQUET)
            return
        self.writer.close()
        geo = dict(self.geo or {})
        geo["columns"]["geometry"]["bbox"] = self.bbox
        # Columns empty in every table keep the null type a single table would have given them.
        nulls = [k for k, f in enumerate(self.schema) if f.name not in self.typed]
        fields = [pa.field(f.name, pa.null()) if k in nulls else f for k, f in enumerate(self.schema)]
        schema = pa.schema(fields, metadata={**(self.schema.metadata or {}), b"geo": json.dumps(geo).encode("utf-8")})
        src = pq.ParquetFile(self.part)
        with pq.ParquetWriter(self.path, schema, compression=TREES_PARQUET["compression"],
                              use_dictionary=TREES_PARQUET["use_dictionary"]) as w:
            # One chunk at a time: with dictionaries only on low-cardinality columns, small row
            # groups compress as well as large ones, so nothing needs merging in memory.
            for k in range(src.num_row_groups):
                t = src.read_row_group(k)
                for c in nulls:
                    t = t.set_column(c, schema.field(c), pa.nulls(len(t)))
                w.write_table(t.replace_schema_metadata(schema.metadata), row_group_size=self.ROW_GROUP)
                del t
        src.close()
        self.part.unlink()
        pa.default_memory_pool().release_unused()  # the copy buffers; Arrow's pool keeps them otherwise


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
