"""OpenStreetMap acquisition through the Overpass API.

This module only downloads and parses raw OSM elements into :class:`OsmData`.
Turning tags into canonical entities happens in ``osm_normalize``.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from typing import Any

import requests
from urllib3.exceptions import ConnectTimeoutError, NewConnectionError

from worldgen import __version__
from worldgen.config import OsmSourceConfig
from worldgen.errors import SourceError
from worldgen.log import get_logger
from worldgen.sources.base import SourceCache, SourceRecord

log = get_logger("sources.osm")

OSM_LICENSE = "ODbL-1.0"
OSM_ATTRIBUTION = "© OpenStreetMap contributors, available under the Open Database License (ODbL)"


@dataclass
class OsmNode:
    id: int
    lon: float
    lat: float
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class OsmWay:
    id: int
    refs: list[int]
    tags: dict[str, str] = field(default_factory=dict)

    @property
    def closed(self) -> bool:
        return len(self.refs) >= 4 and self.refs[0] == self.refs[-1]


@dataclass
class OsmMember:
    type: str  # node | way | relation
    ref: int
    role: str


@dataclass
class OsmRelation:
    id: int
    members: list[OsmMember]
    tags: dict[str, str] = field(default_factory=dict)


@dataclass
class OsmData:
    nodes: dict[int, OsmNode] = field(default_factory=dict)
    ways: dict[int, OsmWay] = field(default_factory=dict)
    relations: dict[int, OsmRelation] = field(default_factory=dict)
    timestamp: str | None = None

    @staticmethod
    def from_overpass_json(doc: dict[str, Any]) -> OsmData:
        out = OsmData(timestamp=(doc.get("osm3s") or {}).get("timestamp_osm_base"))
        for el in doc.get("elements", []):
            t = el.get("type")
            tags = {str(k): str(v) for k, v in (el.get("tags") or {}).items()}
            if t == "node":
                nid = int(el["id"])
                prev = out.nodes.get(nid)
                # "out skel" repeats nodes without tags; keep the tagged copy.
                if prev is None or (not prev.tags and tags):
                    out.nodes[nid] = OsmNode(nid, float(el["lon"]), float(el["lat"]), tags)
            elif t == "way":
                wid = int(el["id"])
                prev_w = out.ways.get(wid)
                if prev_w is None or (not prev_w.tags and tags):
                    out.ways[wid] = OsmWay(wid, [int(r) for r in el.get("nodes", [])], tags)
            elif t == "relation":
                rid = int(el["id"])
                members = [
                    OsmMember(str(m["type"]), int(m["ref"]), str(m.get("role", "")))
                    for m in el.get("members", [])
                ]
                prev_r = out.relations.get(rid)
                if prev_r is None or (not prev_r.tags and tags):
                    out.relations[rid] = OsmRelation(rid, members, tags)
        return out

    def to_overpass_json(self) -> dict[str, Any]:
        """Inverse of ``from_overpass_json`` (used for synthetic fixtures and caching)."""
        elements: list[dict[str, Any]] = []
        for n in sorted(self.nodes.values(), key=lambda n: n.id):
            e: dict[str, Any] = {"type": "node", "id": n.id, "lat": n.lat, "lon": n.lon}
            if n.tags:
                e["tags"] = n.tags
            elements.append(e)
        for w in sorted(self.ways.values(), key=lambda w: w.id):
            elements.append({"type": "way", "id": w.id, "nodes": w.refs, "tags": w.tags})
        for r in sorted(self.relations.values(), key=lambda r: r.id):
            elements.append({
                "type": "relation", "id": r.id, "tags": r.tags,
                "members": [{"type": m.type, "ref": m.ref, "role": m.role} for m in r.members],
            })
        return {"version": 0.6, "osm3s": {"timestamp_osm_base": self.timestamp}, "elements": elements}


def build_query(bbox: tuple[float, float, float, float], timeout_s: int) -> str:
    """Overpass QL for everything the compiler consumes. bbox = (west, south, east, north)."""
    w, s, e, n = bbox
    b = f"{s:.7f},{w:.7f},{n:.7f},{e:.7f}"
    return f"""[out:json][timeout:{timeout_s}][bbox:{b}];
(
  way["highway"];
  way["building"];
  relation["building"]["type"="multipolygon"];
  way["building:part"];
  relation["building:part"]["type"="multipolygon"];
  way["natural"~"^(water|wood|scrub|grassland|heath|wetland|bare_rock|sand|beach|scree|coastline|tree_row)$"];
  relation["natural"~"^(water|wood|scrub|grassland|heath|wetland|bare_rock|sand|beach)$"]["type"="multipolygon"];
  way["waterway"];
  relation["waterway"="riverbank"];
  way["water"];
  way["landuse"];
  relation["landuse"]["type"="multipolygon"];
  way["leisure"~"^(park|garden|nature_reserve|golf_course|pitch|playground|recreation_ground)$"];
  relation["leisure"~"^(park|garden|nature_reserve|golf_course)$"]["type"="multipolygon"];
  node["natural"="tree"];
  relation["type"="restriction"];
);
out body;
>;
out skel qt;
"""


def _unreachable(exc: requests.ConnectionError) -> bool:
    """True if no TCP connection could be made (refused, no route, DNS, connect timeout)."""
    if isinstance(exc, requests.ConnectTimeout):
        return True
    reason = getattr(exc.args[0], "reason", None) if exc.args else None
    return isinstance(reason, (NewConnectionError, ConnectTimeoutError))


class OverpassClient:
    def __init__(self, cfg: OsmSourceConfig, cache: SourceCache) -> None:
        self.cfg = cfg
        self.cache = cache

    def fetch(self, bbox: tuple[float, float, float, float]) -> tuple[OsmData, SourceRecord]:
        query = build_query(bbox, self.cfg.timeout_s)
        used_endpoint: dict[str, str] = {}

        def do_fetch() -> tuple[bytes, dict[str, Any]]:
            last_err: Exception | None = None
            candidates = [ep for ep in self.cfg.endpoints if ep.covers(bbox)]
            for ep in self.cfg.endpoints:
                if ep not in candidates:
                    log.info("Overpass endpoint %s skipped: region outside its coverage", ep.name or ep.url)
            if not candidates:
                raise SourceError("no Overpass endpoint covers the region (check sources.osm.endpoints)")
            for ep in candidates:
                endpoint = ep.url
                for attempt in range(3):
                    try:
                        log.info("Overpass query to %s (attempt %d)", endpoint, attempt + 1)
                        r = requests.post(
                            endpoint,
                            data={"data": query},
                            timeout=(self.cfg.connect_timeout_s, self.cfg.timeout_s + 30),
                            headers={"User-Agent": f"worldgen/{__version__} (geospatial world compiler)"},
                        )
                        if r.status_code in (429, 504):
                            raise SourceError(f"Overpass busy ({r.status_code})")
                        r.raise_for_status()
                        doc = r.json()
                        if "remark" in doc and "runtime error" in str(doc["remark"]):
                            raise SourceError(f"Overpass runtime error: {doc['remark']}")
                        used_endpoint["url"] = endpoint
                        return r.content, {"endpoint": endpoint, "query": query}
                    except requests.ConnectionError as exc:
                        last_err = exc
                        if not _unreachable(exc):
                            log.warning("Overpass request failed: %s", exc)
                            if attempt < 2:
                                time.sleep(2.0 * (attempt + 1))
                            continue
                        # Unreachable (e.g. the LAN server from off-site): next endpoint at once.
                        log.warning("Overpass endpoint %s unreachable: %s", endpoint, exc)
                        break
                    except (requests.RequestException, ValueError, SourceError) as exc:
                        last_err = exc
                        log.warning("Overpass request failed: %s", exc)
                        if attempt < 2:
                            time.sleep(2.0 * (attempt + 1))
            raise SourceError(f"all Overpass endpoints failed: {last_err}")

        raw, meta = self.cache.get_bytes("osm", ["overpass", query], do_fetch, suffix=".json")
        doc = json.loads(raw.decode("utf-8"))
        data = OsmData.from_overpass_json(doc)
        record = SourceRecord(
            name="osm",
            provider="overpass",
            license=OSM_LICENSE,
            attribution=OSM_ATTRIBUTION,
            dataset_version=data.timestamp,
            url=meta.get("endpoint"),
            retrieved_utc=meta.get("retrieved_utc"),
            snapshot_hash=meta.get("sha256"),
            coverage={
                "query_bbox_wgs84": list(bbox),
                "nodes": len(data.nodes),
                "ways": len(data.ways),
                "relations": len(data.relations),
            },
        )
        if not data.ways:
            record.status = "partial"
            record.notes.append("Overpass returned no ways for the region")
        return data, record
