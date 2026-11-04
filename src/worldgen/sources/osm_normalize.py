"""OSM tags and elements -> canonical entities.

All OSM-specific interpretation lives here: tag parsing, unit conversion,
multipolygon assembly and clipping of ways to the region extent. Output is the
source-independent model of :mod:`worldgen.world.model`.

Road connectivity comes exclusively from shared OSM node ids; two ways that
cross in 2D without sharing a node are never connected.
"""

from __future__ import annotations

import math
import re
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Point, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.ops import linemerge, polygonize, unary_union

from worldgen.config import WorldConfig
from worldgen.log import get_logger
from worldgen.sources.osm import OsmData, OsmRelation, OsmWay
from worldgen.spatial.frame import LocalFrame
from worldgen.world.model import (
    Building,
    LandcoverArea,
    ObservedTree,
    Road,
    TurnRestriction,
    Waterbody,
)

log = get_logger("sources.osm_normalize")

SOURCE = "osm"

# --------------------------------------------------------------------------------------
# tag parsing


_NUM = re.compile(r"^\s*(-?\d+(?:[.,]\d+)?)\s*(m|meters?|metres?|ft|feet|'|km|mi|mph|km/h|kmh|kph)?\s*$", re.I)
_FEET_INCHES = re.compile(r"^\s*(\d+)\s*'\s*(\d+(?:\.\d+)?)\s*\"?\s*$")


def parse_length(value: str | None) -> float | None:
    """Parse an OSM length ('12', '12 m', "40'", "12'6\"") to metres; None if unparsable."""
    if value is None:
        return None
    v = value.strip().split(";")[0]
    m = _FEET_INCHES.match(v)
    if m:
        return int(m.group(1)) * 0.3048 + float(m.group(2)) * 0.0254
    m = _NUM.match(v)
    if not m:
        return None
    num = float(m.group(1).replace(",", "."))
    unit = (m.group(2) or "m").lower()
    if unit in ("ft", "feet", "'"):
        num *= 0.3048
    elif unit == "km":
        num *= 1000.0
    elif unit == "mi":
        num *= 1609.344
    return num if math.isfinite(num) else None


def parse_speed(value: str | None) -> float | None:
    """Parse maxspeed to km/h."""
    if value is None:
        return None
    v = value.strip().lower()
    if v in ("walk",):
        return 6.0
    m = re.match(r"^(\d+(?:\.\d+)?)\s*(mph|km/h|kmh|kph)?$", v)
    if not m:
        return None
    s = float(m.group(1))
    return s * 1.609344 if m.group(2) == "mph" else s


def parse_int(value: str | None) -> int | None:
    if value is None:
        return None
    try:
        return int(round(float(value.strip().split(";")[0].replace(",", "."))))
    except ValueError:
        return None


def parse_float(value: str | None) -> float | None:
    if value is None:
        return None
    try:
        f = float(value.strip().split(";")[0].replace(",", "."))
        return f if math.isfinite(f) else None
    except ValueError:
        return None


def truthy(value: str | None) -> bool:
    return value is not None and value.lower() not in ("no", "false", "0", "")


# --------------------------------------------------------------------------------------
# road semantics

CAR_CLASSES = {
    "motorway", "trunk", "primary", "secondary", "tertiary", "unclassified", "residential",
    "service", "living_street", "road", "motorway_link", "trunk_link", "primary_link",
    "secondary_link", "tertiary_link", "track",
}
NO_FOOT_CLASSES = {"motorway", "motorway_link"}
NO_BIKE_CLASSES = {"motorway", "motorway_link", "steps"}


def road_modes(tags: dict[str, str]) -> dict[str, bool]:
    """Legal access per mode, from class defaults refined by access tags."""
    hw = tags.get("highway", "")
    car = hw in CAR_CLASSES
    foot = hw not in NO_FOOT_CLASSES
    bike = hw not in NO_BIKE_CLASSES and hw not in ("footway", "pedestrian", "corridor")

    def apply(current: bool, *keys: str) -> bool:
        for k in keys:
            v = tags.get(k)
            if v is None:
                continue
            if v in ("no", "private", "agricultural", "forestry", "delivery", "use_sidepath"):
                return False
            if v in ("yes", "designated", "permissive", "destination", "customers"):
                return True
        return current

    general = tags.get("access")
    if general in ("no", "private"):
        car = bike = foot = False
    elif general in ("yes", "permissive", "destination"):
        pass
    car = apply(car, "motor_vehicle", "motorcar", "vehicle")
    bike = apply(bike, "bicycle", "vehicle")
    foot = apply(foot, "foot")
    return {"motor_vehicle": car, "bicycle": bike, "foot": foot}


def oneway_direction(tags: dict[str, str]) -> tuple[str, bool]:
    """Return (direction, observed). direction is 'both', 'forward' or 'backward'."""
    v = tags.get("oneway")
    if v is not None:
        v = v.lower()
        if v in ("yes", "true", "1"):
            return "forward", True
        if v in ("-1", "reverse"):
            return "backward", True
        return "both", True
    hw = tags.get("highway", "")
    if hw in ("motorway",) or tags.get("junction") in ("roundabout", "circular"):
        return "forward", False
    return "both", False


def road_properties(tags: dict[str, str], cfg: WorldConfig) -> tuple[dict[str, Any], dict[str, tuple[Any, str, float]]]:
    """Observed properties and inferences (name -> (value, method, confidence))."""
    p: dict[str, Any] = {"class": tags.get("highway")}
    inf: dict[str, tuple[Any, str, float]] = {}
    for k in ("name", "ref", "surface", "smoothness", "lit", "sidewalk", "junction"):
        if k in tags:
            p[k] = tags[k]
    if "name:en" in tags:
        p["name_en"] = tags["name:en"]
    width = parse_length(tags.get("width") or tags.get("est_width"))
    if width is not None and 0.5 < width < 80:
        p["width"] = width
    lanes = parse_int(tags.get("lanes"))
    if lanes is not None and 0 < lanes < 20:
        p["lanes"] = lanes
    layer = parse_int(tags.get("layer"))
    if layer is not None:
        p["layer"] = max(-5, min(5, layer))
    bridge = tags.get("bridge")
    p["bridge"] = truthy(bridge)
    if p["bridge"] and bridge not in ("yes",):
        p["bridge_type"] = bridge
    if "bridge:structure" in tags:
        p["bridge_structure"] = tags["bridge:structure"]
    if "bridge:name" in tags:
        p["bridge_name"] = tags["bridge:name"]
    tunnel = tags.get("tunnel")
    p["tunnel"] = truthy(tunnel) and tunnel != "building_passage"
    if truthy(tunnel):
        p["tunnel_type"] = tunnel
    if tags.get("covered") == "yes":
        p["covered"] = True
    maxspeed = parse_speed(tags.get("maxspeed"))
    if maxspeed is not None:
        p["maxspeed_kmh"] = maxspeed
    maxheight = parse_length(tags.get("maxheight"))
    if maxheight is not None:
        p["maxheight"] = maxheight
    direction, observed = oneway_direction(tags)
    if observed:
        p["direction"] = direction
    else:
        inf["direction"] = (direction, "osm_implied_oneway" if direction != "both" else "osm_default", 0.9)
    p["modes"] = road_modes(tags)
    for k in ("access", "motor_vehicle", "bicycle", "foot"):
        if k in tags:
            p[f"access_{k}"] = tags[k]

    # Width estimate.
    hw = tags.get("highway", "road")
    if "width" not in p:
        if lanes:
            inf["width"] = (lanes * cfg.roads.lane_width + 1.0, "lanes*lane_width+shoulders", 0.6)
        else:
            inf["width"] = (cfg.roads.default_widths.get(hw, 6.0), f"class_default:{hw}", 0.3)
    # Layer: the OSM default is 0, but mapped bridges without layer conventionally sit one level up.
    if "layer" not in p:
        if p["bridge"]:
            inf["layer"] = (1, "osm_convention:bridge_implies_layer_1", 0.6)
        elif p["tunnel"]:
            inf["layer"] = (-1, "osm_convention:tunnel_implies_layer_-1", 0.6)
        else:
            inf["layer"] = (0, "osm_default", 0.9)
    # A non-zero layer without a bridge/tunnel tag still means the way passes over or under
    # ground-level features without connecting (raised walkways, underground corridors,
    # cuttings). Record an implied structure as an estimate, never as an observation.
    layer_v = p.get("layer")
    if layer_v and not p["bridge"] and not p["tunnel"] and tags.get("highway") not in ("elevator",):
        if layer_v > 0:
            del p["bridge"]
            inf["bridge"] = (True, "osm_layer_above_ground_without_bridge_tag", 0.4)
        else:
            del p["tunnel"]
            inf["tunnel"] = (True, "osm_layer_below_ground_without_tunnel_tag", 0.4)
    return p, inf


# --------------------------------------------------------------------------------------
# geometry assembly


@dataclass
class NormalizedOsm:
    roads: list[Road] = field(default_factory=list)
    buildings: list[Building] = field(default_factory=list)
    water: list[Waterbody] = field(default_factory=list)
    landcover: list[LandcoverArea] = field(default_factory=list)
    trees: list[ObservedTree] = field(default_factory=list)
    restrictions: list[TurnRestriction] = field(default_factory=list)
    coastlines: list[LineString] = field(default_factory=list)
    skipped: dict[str, int] = field(default_factory=dict)
    node_positions: dict[str, tuple[float, float]] = field(default_factory=dict)


class _Projector:
    def __init__(self, osm: OsmData, frame: LocalFrame) -> None:
        ids = np.array(sorted(osm.nodes), dtype=np.int64)
        lons = np.array([osm.nodes[i].lon for i in ids])
        lats = np.array([osm.nodes[i].lat for i in ids])
        xs, ys = frame.to_local(lons, lats) if len(ids) else (np.zeros(0), np.zeros(0))
        self.xy = {int(i): (float(x), float(y)) for i, x, y in zip(ids, xs, ys)}

    def coords(self, refs: list[int]) -> list[tuple[float, float]] | None:
        try:
            return [self.xy[r] for r in refs]
        except KeyError:
            return None


def _ring_polygon(coords: list[tuple[float, float]]) -> Polygon | None:
    if len(coords) < 4:
        return None
    poly = Polygon(coords)
    if not poly.is_valid:
        poly = _polygonal(shapely.make_valid(poly))
    return poly if poly is not None and not poly.is_empty and poly.area > 0.5 else None


def _polygonal(g: BaseGeometry) -> Any:
    """Keep only polygonal parts of a geometry (after make_valid)."""
    if g.is_empty:
        return None
    if isinstance(g, (Polygon, MultiPolygon)):
        return g
    parts = [p for p in getattr(g, "geoms", []) if isinstance(p, (Polygon, MultiPolygon))]
    if not parts:
        return None
    u = unary_union(parts)
    return u if not u.is_empty else None


def assemble_multipolygon(rel: OsmRelation, osm: OsmData, proj: _Projector) -> BaseGeometry | None:
    outers: list[LineString] = []
    inners: list[LineString] = []
    for m in rel.members:
        if m.type != "way" or m.ref not in osm.ways:
            continue
        cs = proj.coords(osm.ways[m.ref].refs)
        if cs is None or len(cs) < 2:
            continue
        (inners if m.role == "inner" else outers).append(LineString(cs))
    if not outers:
        return None
    outer_polys = list(polygonize(linemerge(outers) if len(outers) > 1 else outers))
    if not outer_polys:
        return None
    shape = unary_union(outer_polys)
    if inners:
        inner_polys = list(polygonize(linemerge(inners) if len(inners) > 1 else inners))
        if inner_polys:
            shape = shape.difference(unary_union(inner_polys))
    shape = _polygonal(shapely.make_valid(shape)) if not shape.is_valid else shape
    if shape is None or shape.is_empty or shape.area < 0.5:
        return None
    return shape


# --------------------------------------------------------------------------------------
# classification tables

WATER_AREA_LANDUSE = {"reservoir", "basin"}
WATERWAY_LINES = {"river", "stream", "canal", "drain", "ditch", "brook", "tidal_channel"}

LANDCOVER_TAGS: dict[tuple[str, str], str] = {
    ("landuse", "forest"): "forest", ("natural", "wood"): "forest",
    ("natural", "scrub"): "scrub", ("natural", "heath"): "heath",
    ("natural", "grassland"): "grass", ("landuse", "grass"): "grass",
    ("landuse", "meadow"): "grass", ("landuse", "village_green"): "grass",
    ("landuse", "recreation_ground"): "grass", ("leisure", "pitch"): "grass",
    ("leisure", "golf_course"): "grass", ("leisure", "playground"): "grass",
    ("leisure", "park"): "park", ("leisure", "garden"): "park",
    ("leisure", "nature_reserve"): "nature_reserve", ("leisure", "recreation_ground"): "grass",
    ("landuse", "farmland"): "farmland", ("landuse", "orchard"): "orchard",
    ("landuse", "vineyard"): "orchard", ("landuse", "allotments"): "farmland",
    ("landuse", "plant_nursery"): "farmland", ("landuse", "flowerbed"): "grass",
    ("landuse", "cemetery"): "grass",
    ("natural", "wetland"): "wetland", ("natural", "bare_rock"): "bare",
    ("natural", "scree"): "bare", ("natural", "sand"): "sand", ("natural", "beach"): "sand",
    ("landuse", "residential"): "residential", ("landuse", "commercial"): "commercial",
    ("landuse", "retail"): "commercial", ("landuse", "industrial"): "industrial",
    ("landuse", "construction"): "construction", ("landuse", "brownfield"): "construction",
    ("landuse", "railway"): "industrial", ("landuse", "military"): "industrial",
    ("landuse", "religious"): "residential", ("landuse", "education"): "residential",
}

LANDCOVER_PRIORITY = ("leisure", "landuse", "natural")


def landcover_class(tags: dict[str, str]) -> tuple[str, str] | None:
    for key in ("natural", "landuse", "leisure"):
        v = tags.get(key)
        if v is not None and (key, v) in LANDCOVER_TAGS:
            return LANDCOVER_TAGS[(key, v)], f"{key}={v}"
    return None


def is_water_area(tags: dict[str, str]) -> bool:
    return (
        tags.get("natural") == "water"
        or tags.get("waterway") == "riverbank"
        or tags.get("landuse") in WATER_AREA_LANDUSE
        or ("water" in tags and tags.get("natural") in (None, "water") and "building" not in tags
            and "highway" not in tags and tags.get("water") not in ("no",))
    )


# --------------------------------------------------------------------------------------
# main entry


def _clip_way_runs(
    refs: list[int], proj: _Projector, extent: Polygon
) -> list[tuple[list[tuple[float, float]], list[str]]]:
    """Split a way into runs inside ``extent``. Boundary crossings get synthetic node ids."""
    cs = proj.coords(refs)
    if cs is None:
        return []
    xmin, ymin, xmax, ymax = extent.bounds
    inside = [xmin <= x <= xmax and ymin <= y <= ymax for x, y in cs]
    if all(inside):
        return [(cs, [f"n:osm:{r}" for r in refs])]
    runs: list[tuple[list[tuple[float, float]], list[str]]] = []
    cur_c: list[tuple[float, float]] = []
    cur_n: list[str] = []
    boundary = extent.exterior
    for k in range(len(cs)):
        if inside[k]:
            if not cur_c and k > 0:
                p = LineString([cs[k - 1], cs[k]]).intersection(boundary)
                if not p.is_empty:
                    q = _first_point(p, cs[k - 1])
                    cur_c.append(q)
                    cur_n.append(f"n:bnd:{refs[k - 1]}-{refs[k]}")
            cur_c.append(cs[k])
            cur_n.append(f"n:osm:{refs[k]}")
        else:
            if cur_c:
                p = LineString([cs[k - 1], cs[k]]).intersection(boundary)
                if not p.is_empty:
                    q = _first_point(p, cs[k - 1])
                    cur_c.append(q)
                    cur_n.append(f"n:bnd:{refs[k - 1]}-{refs[k]}")
                if len(cur_c) >= 2:
                    runs.append((cur_c, cur_n))
                cur_c, cur_n = [], []
            elif k > 0:
                # Segment between two outside vertices may still cross the extent.
                seg = LineString([cs[k - 1], cs[k]]).intersection(extent)
                if isinstance(seg, LineString) and seg.length > 0.5:
                    a, b = list(seg.coords)[0], list(seg.coords)[-1]
                    da = math.dist(a, cs[k - 1])
                    db = math.dist(b, cs[k - 1])
                    if db < da:
                        a, b = b, a
                    runs.append(([a, b], [f"n:bnd:{refs[k - 1]}-{refs[k]}:a", f"n:bnd:{refs[k - 1]}-{refs[k]}:b"]))
    if len(cur_c) >= 2:
        runs.append((cur_c, cur_n))
    return runs


def _first_point(g: BaseGeometry, near: tuple[float, float]) -> tuple[float, float]:
    pts = [g] if g.geom_type == "Point" else [p for p in getattr(g, "geoms", []) if p.geom_type == "Point"]
    if not pts:
        c = g.representative_point()
        return (c.x, c.y)
    pts.sort(key=lambda p: (p.x - near[0]) ** 2 + (p.y - near[1]) ** 2)
    return (pts[0].x, pts[0].y)


def normalize_osm(osm: OsmData, frame: LocalFrame, extent: Polygon, cfg: WorldConfig) -> NormalizedOsm:
    out = NormalizedOsm()
    proj = _Projector(osm, frame)
    skipped: dict[str, int] = {}

    def skip(reason: str) -> None:
        skipped[reason] = skipped.get(reason, 0) + 1

    exclude = set(cfg.roads.exclude_classes)
    clip_box = extent
    area_clip = extent.buffer(1.0, join_style="mitre")
    shapely.prepare(area_clip)

    # -- multipolygon relations first; member ways used as outlines are not re-used as areas
    consumed_ways: set[int] = set()
    for rid in sorted(osm.relations):
        rel = osm.relations[rid]
        tags = rel.tags
        if tags.get("type") != "multipolygon":
            continue
        kind = _area_kind(tags, cfg)
        if kind is None:
            continue
        geom = assemble_multipolygon(rel, osm, proj)
        if geom is None:
            skip(f"{kind}:invalid_multipolygon")
            continue
        if not geom.intersects(area_clip):
            continue
        for m in rel.members:
            if m.type == "way" and m.role == "outer" and m.ref in osm.ways and not _area_kind(osm.ways[m.ref].tags, cfg):
                consumed_ways.add(m.ref)
        _add_area(out, kind, f"relation/{rid}", geom, tags, cfg, area_clip)

    for wid in sorted(osm.ways):
        way = osm.ways[wid]
        tags = way.tags
        if not tags:
            continue
        hw = tags.get("highway")
        if hw is not None and tags.get("area") != "yes" and hw not in exclude:
            _add_road(out, way, proj, clip_box, cfg, skip)
            continue
        if tags.get("natural") == "coastline":
            cs = proj.coords(way.refs)
            if cs and len(cs) >= 2:
                out.coastlines.append(LineString(cs))
            continue
        if tags.get("natural") == "tree_row":
            cs = proj.coords(way.refs)
            if cs and len(cs) >= 2:
                _add_tree_row(out, wid, LineString(cs), tags, cfg, area_clip)
            continue
        ww = tags.get("waterway")
        if ww in WATERWAY_LINES and not way.closed:
            cs = proj.coords(way.refs)
            if cs and len(cs) >= 2:
                line = LineString(cs)
                if line.intersects(area_clip):
                    _add_waterway(out, wid, line.intersection(area_clip), tags, cfg)
            continue
        kind = _area_kind(tags, cfg)
        if kind is None or wid in consumed_ways:
            continue
        if not way.closed:
            skip(f"{kind}:unclosed_way")
            continue
        cs = proj.coords(way.refs)
        if cs is None:
            skip(f"{kind}:missing_nodes")
            continue
        poly = _ring_polygon(cs)
        if poly is None:
            skip(f"{kind}:invalid_polygon")
            continue
        if not poly.intersects(area_clip):
            continue
        _add_area(out, kind, f"way/{wid}", poly, tags, cfg, area_clip, repaired=not Polygon(cs).is_valid)

    # -- trees
    for nid in sorted(osm.nodes):
        n = osm.nodes[nid]
        if n.tags.get("natural") != "tree":
            continue
        x, y = proj.xy[nid]
        if not area_clip.contains(Point(x, y)):
            continue
        out.trees.append(ObservedTree(f"tree:osm:node/{nid}", x, y, _tree_props(n.tags), SOURCE, f"node/{nid}"))

    # -- turn restrictions
    for rid in sorted(osm.relations):
        rel = osm.relations[rid]
        r = rel.tags.get("restriction") or rel.tags.get("restriction:motorcar") or rel.tags.get("restriction:motor_vehicle")
        if rel.tags.get("type") != "restriction" or not r:
            continue
        frm = [m.ref for m in rel.members if m.role == "from" and m.type == "way"]
        to = [m.ref for m in rel.members if m.role == "to" and m.type == "way"]
        via_n = [m.ref for m in rel.members if m.role == "via" and m.type == "node"]
        via_w = [m.ref for m in rel.members if m.role == "via" and m.type == "way"]
        if len(frm) != 1 or len(to) != 1 or (not via_n and not via_w):
            skip("restriction:unsupported_members")
            continue
        out.restrictions.append(
            TurnRestriction(
                id=f"r:osm:relation/{rid}",
                restriction=r,
                from_way=f"way/{frm[0]}",
                to_way=f"way/{to[0]}",
                via_node=f"n:osm:{via_n[0]}" if via_n else None,
                via_ways=[f"way/{w}" for w in via_w],
                source_id=f"relation/{rid}",
            )
        )
    out.skipped = skipped
    return out


def _area_kind(tags: dict[str, str], cfg: WorldConfig) -> str | None:
    b = tags.get("building")
    if b is not None and b != "no":
        return "building"
    if cfg.buildings.building_parts and tags.get("building:part") not in (None, "no"):
        return "building_part"
    if is_water_area(tags):
        return "water"
    if landcover_class(tags) is not None:
        return "landcover"
    return None


def _add_area(
    out: NormalizedOsm, kind: str, sid: str, geom: BaseGeometry, tags: dict[str, str],
    cfg: WorldConfig, clip: Polygon, repaired: bool = False,
) -> None:
    if kind in ("building", "building_part"):
        b = _building(sid, geom, tags, kind == "building_part")
        if repaired:
            b.infer("footprint_repaired", True, "shapely.make_valid", 0.8)
        out.buildings.append(b)
        return
    clipped = _polygonal(geom.intersection(clip)) if not clip.contains(geom) else geom
    if clipped is None:
        return
    if kind == "water":
        props: dict[str, Any] = {"class": tags.get("water") or tags.get("landuse") or
                                 ("river" if tags.get("waterway") == "riverbank" else "lake")}
        for k in ("name", "intermittent", "salt"):
            if k in tags:
                props[k] = tags[k]
        w = Waterbody(id=f"water:osm:{sid}", geometry=clipped, properties=props, source=SOURCE,
                      source_id=sid, confidence=0.9)
        w.mark_missing("surface_elevation")
        out.water.append(w)
    elif kind == "landcover":
        cls = landcover_class(tags)
        assert cls is not None
        props = {"class": cls[0], "tag": cls[1]}
        for k in ("name", "leaf_type", "leaf_cycle", "wetland", "species", "genus"):
            if k in tags:
                props[k] = tags[k]
        out.landcover.append(LandcoverArea(id=f"landcover:osm:{sid}", geometry=clipped, properties=props,
                                           source=SOURCE, source_id=sid, confidence=0.85))


def _building(sid: str, geom: BaseGeometry, tags: dict[str, str], is_part: bool) -> Building:
    props: dict[str, Any] = {"class": tags.get("building") if not is_part else tags.get("building:part")}
    h = parse_length(tags.get("height"))
    if h is not None and 0 < h < 1000:
        props["height"] = h
    mh = parse_length(tags.get("min_height"))
    if mh is not None and 0 <= mh < 1000:
        props["min_height"] = mh
    lv = parse_float(tags.get("building:levels"))
    if lv is not None and 0 <= lv < 300:
        props["levels"] = lv
    ml = parse_float(tags.get("building:min_level"))
    if ml is not None and 0 <= ml < 300:
        props["min_level"] = ml
    rs = tags.get("roof:shape")
    if rs:
        props["roof_shape"] = rs
    rh = parse_length(tags.get("roof:height"))
    if rh is not None and 0 <= rh < 200:
        props["roof_height"] = rh
    rl = parse_float(tags.get("roof:levels"))
    if rl is not None and 0 <= rl < 50:
        props["roof_levels"] = rl
    for k, dst in (("name", "name"), ("name:en", "name_en"), ("building:colour", "colour"),
                   ("roof:colour", "roof_colour"), ("building:material", "material"),
                   ("amenity", "amenity"), ("addr:housenumber", "housenumber"), ("addr:street", "street")):
        if k in tags:
            props[dst] = tags[k]
    b = Building(id=f"building:osm:{sid}", geometry=geom, properties=props, source=SOURCE,
                 source_id=sid, confidence=0.9, is_part=is_part)
    return b


def _add_road(out: NormalizedOsm, way: OsmWay, proj: _Projector, clip_box: Polygon,
              cfg: WorldConfig, skip: Any) -> None:
    runs = _clip_way_runs(way.refs, proj, clip_box)
    if not runs:
        if proj.coords(way.refs) is None:
            skip("road:missing_nodes")
        return
    props, inf = road_properties(way.tags, cfg)
    multi = len(runs) > 1
    for k, (coords, node_ids) in enumerate(runs):
        # Drop consecutive duplicate vertices (zero-length segments).
        cc: list[tuple[float, float]] = []
        nn: list[str] = []
        for c, n in zip(coords, node_ids):
            if cc and math.dist(cc[-1], c) < 1e-6:
                continue
            cc.append(c)
            nn.append(n)
        if len(cc) < 2:
            continue
        rid = f"road:osm:way/{way.id}" + (f":p{k}" if multi else "")
        road = Road(id=rid, geometry=LineString(cc), properties=dict(props), source=SOURCE,
                    source_id=f"way/{way.id}", confidence=0.9, node_ids=nn)
        for name, (value, method, conf) in inf.items():
            road.infer(name, value, method, conf)
        if road.is_bridge or road.is_tunnel:
            road.mark_missing("deck_elevation")
        out.roads.append(road)
        for c, n in zip(cc, nn):
            out.node_positions[n] = c


def _add_waterway(out: NormalizedOsm, wid: int, geom: BaseGeometry, tags: dict[str, str], cfg: WorldConfig) -> None:
    lines = [g for g in ([geom] if geom.geom_type == "LineString" else getattr(geom, "geoms", []))
             if g.geom_type == "LineString" and g.length > 0.5]
    if not lines:
        return
    ww = tags["waterway"]
    props: dict[str, Any] = {"class": ww}
    width = parse_length(tags.get("width"))
    if width is not None and 0.2 < width < 2000:
        props["width"] = width
    for k in ("name", "intermittent"):
        if k in tags:
            props[k] = tags[k]
    if truthy(tags.get("tunnel")):
        props["tunnel"] = tags["tunnel"]
    line = lines[0] if len(lines) == 1 else linemerge(lines)
    w = Waterbody(id=f"water:osm:way/{wid}", geometry=line, properties=props, source=SOURCE,
                  source_id=f"way/{wid}", confidence=0.85)
    if "width" not in props:
        w.infer("width", cfg.water.default_widths.get(ww, 3.0), f"class_default:{ww}", 0.3)
    w.mark_missing("surface_elevation")
    out.water.append(w)


def _tree_props(tags: dict[str, str]) -> dict[str, Any]:
    p: dict[str, Any] = {}
    h = parse_length(tags.get("height"))
    if h is not None and 0.5 < h < 120:
        p["height"] = h
    dc = parse_length(tags.get("diameter_crown"))
    if dc is not None and 0.3 < dc < 60:
        p["crown_radius"] = dc / 2.0
    circ = parse_length(tags.get("circumference"))
    if circ is not None and 0.05 < circ < 30:
        p["trunk_radius"] = circ / (2 * math.pi)
    for k in ("species", "genus", "leaf_type", "leaf_cycle", "denotation", "species:en"):
        if k in tags:
            p[k.replace(":", "_")] = tags[k]
    return p


def _add_tree_row(out: NormalizedOsm, wid: int, line: LineString, tags: dict[str, str],
                  cfg: WorldConfig, clip: Polygon) -> None:
    spacing = cfg.vegetation.tree_row_spacing
    n = max(1, int(line.length // spacing) + 1)
    props = _tree_props(tags)
    for k in range(n):
        p = line.interpolate(min(k * spacing, line.length))
        if not clip.contains(p):
            continue
        out.trees.append(
            ObservedTree(f"tree:osm:way/{wid}:{k}", p.x, p.y, dict(props), SOURCE, f"way/{wid}", from_row=True)
        )
