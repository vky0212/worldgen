"""Overture Maps adapter (buildings theme).

Reads Overture GeoParquet either from the public S3 release
(``s3://overturemaps-us-west-2/release/<release>/theme=buildings/type=building``)
or from a local file/directory with the same schema. Rows are filtered by the
``bbox`` struct column so only intersecting row groups are read.

Overture building records that originate from OpenStreetMap are skipped (OSM is
read directly); the remaining footprints (e.g. ML-derived) are conflated with
OSM buildings by the buildings stage.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.dataset as pads
import requests
import shapely
from shapely.geometry.base import BaseGeometry

from worldgen.config import OvertureSourceConfig
from worldgen.errors import SourceError
from worldgen.hashing import sha256_bytes
from worldgen.log import get_logger
from worldgen.sources.base import SourceCache, SourceRecord
from worldgen.spatial.frame import LocalFrame
from worldgen.world.model import Building

log = get_logger("sources.overture")

OVERTURE_LICENSE = "ODbL-1.0 (buildings theme)"
OVERTURE_ATTRIBUTION = "© Overture Maps Foundation and contributors (includes © OpenStreetMap contributors)"
LISTING_URL = "https://overturemaps-us-west-2.s3.us-west-2.amazonaws.com/?list-type=2&prefix=release/&delimiter=/"

COLUMNS = ["id", "geometry", "bbox", "sources", "class", "subtype", "height", "num_floors",
           "min_height", "min_floor", "roof_shape", "roof_height", "names", "is_underground"]


def latest_release() -> str:
    try:
        xml = requests.get(LISTING_URL, timeout=30).text
    except requests.RequestException as exc:
        raise SourceError(f"cannot list Overture releases: {exc}") from exc
    rel = sorted(re.findall(r"<Prefix>release/([^<]+)/</Prefix>", xml))
    if not rel:
        raise SourceError("no Overture releases found")
    return rel[-1]


def _bbox_filter(bbox: tuple[float, float, float, float]) -> Any:
    w, s, e, n = bbox
    return (
        (pc.field("bbox", "xmin") <= e) & (pc.field("bbox", "xmax") >= w)
        & (pc.field("bbox", "ymin") <= n) & (pc.field("bbox", "ymax") >= s)
    )


def read_overture_table(cfg: OvertureSourceConfig, bbox: tuple[float, float, float, float],
                        release: str) -> pa.Table:
    if cfg.local_path:
        dataset = pads.dataset(cfg.local_path, format="parquet")
    else:
        from pyarrow import fs

        s3 = fs.S3FileSystem(anonymous=True, region="us-west-2")
        root = cfg.base_uri.removeprefix("s3://").rstrip("/")
        dataset = pads.dataset(f"{root}/{release}/theme=buildings/type=building", filesystem=s3, format="parquet")
    names = set(dataset.schema.names)
    cols = [c for c in COLUMNS if c in names]
    return dataset.to_table(filter=_bbox_filter(bbox), columns=cols)


def fetch_overture_buildings(
    cfg: OvertureSourceConfig, bbox: tuple[float, float, float, float], frame: LocalFrame, cache: SourceCache
) -> tuple[list[Building], SourceRecord]:
    if not cfg.enabled:
        return [], SourceRecord("overture", "overture_maps", "-", "-", status="disabled")
    release = cfg.release
    if release == "latest" and not cfg.local_path:
        meta = cache.read_meta("overture", "latest_release")
        if cache.offline and meta:
            release = meta["release"]
        else:
            release = latest_release()

    def fetch() -> tuple[bytes, dict[str, Any]]:
        table = read_overture_table(cfg, bbox, release)
        sink = pa.BufferOutputStream()
        import pyarrow.parquet as pq

        pq.write_table(table, sink)
        return sink.getvalue().to_pybytes(), {"release": release, "source": cfg.local_path or cfg.base_uri}

    raw, meta = cache.get_bytes("overture", ["buildings", release, cfg.local_path, [round(v, 7) for v in bbox]],
                                fetch, suffix=".parquet")
    import pyarrow.parquet as pq

    table = pq.read_table(pa.BufferReader(raw))
    buildings = overture_rows_to_buildings(table, frame)
    rec = SourceRecord(
        name="overture", provider="overture_maps", license=OVERTURE_LICENSE, attribution=OVERTURE_ATTRIBUTION,
        dataset_version=meta.get("release", release), url=meta.get("source"),
        retrieved_utc=meta.get("retrieved_utc"), snapshot_hash=meta.get("sha256") or sha256_bytes(raw),
        coverage={"rows": table.num_rows, "non_osm_buildings": len(buildings)},
    )
    return buildings, rec


def overture_rows_to_buildings(table: pa.Table, frame: LocalFrame) -> list[Building]:
    out: list[Building] = []
    rows = table.to_pylist()
    for row in sorted(rows, key=lambda r: str(r.get("id"))):
        srcs = row.get("sources") or []
        datasets = {str(s.get("dataset", "")) for s in srcs if isinstance(s, dict)}
        if any("openstreetmap" in d.lower() for d in datasets):
            continue
        if row.get("is_underground"):
            continue
        geom_wkb = row.get("geometry")
        if geom_wkb is None:
            continue
        g: BaseGeometry = shapely.from_wkb(geom_wkb)
        if g.geom_type not in ("Polygon", "MultiPolygon"):
            continue
        local = frame.geom_to_local(g)
        if not local.is_valid:
            local = shapely.make_valid(local)
        if local.is_empty or local.area < 1.0:
            continue
        props: dict[str, Any] = {"class": row.get("class") or row.get("subtype") or "yes"}
        if row.get("height") is not None:
            props["height"] = float(row["height"])
        if row.get("num_floors") is not None:
            props["levels"] = float(row["num_floors"])
        if row.get("min_height") is not None:
            props["min_height"] = float(row["min_height"])
        if row.get("roof_shape"):
            props["roof_shape"] = row["roof_shape"]
        if row.get("roof_height") is not None:
            props["roof_height"] = float(row["roof_height"])
        names = row.get("names") or {}
        if isinstance(names, dict) and names.get("primary"):
            props["name"] = names["primary"]
        confs = [float(s["confidence"]) for s in srcs if isinstance(s, dict) and s.get("confidence") is not None]
        conf = max(confs) if confs else 0.6
        props["datasets"] = sorted(datasets)
        oid = str(row["id"])
        out.append(Building(id=f"building:overture:{oid}", geometry=local, properties=props,
                            source="overture", source_id=oid, confidence=round(min(conf, 0.85), 3)))
    return out


def write_overture_fixture(path: Path, rows: list[dict[str, Any]]) -> None:
    """Write a minimal Overture-schema GeoParquet file (used by tests and offline demos)."""
    import pyarrow.parquet as pq

    table = pa.Table.from_pylist(rows)
    pq.write_table(table, path)
