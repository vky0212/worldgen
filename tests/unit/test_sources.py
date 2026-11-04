"""OSM tag parsing and normalization, Overture conflation and the source cache."""

from __future__ import annotations

import pytest
import shapely
from shapely.geometry import box

from worldgen.config import load_config
from worldgen.errors import OfflineCacheMiss
from worldgen.sources.base import SourceCache
from worldgen.sources.osm import OsmData, build_query
from worldgen.sources.osm_normalize import normalize_osm, oneway_direction, parse_length, parse_speed, road_modes
from worldgen.sources.overture import overture_rows_to_buildings
from worldgen.sources.synthetic import FixtureBuilder, ORIGIN
from worldgen.spatial.frame import LocalFrame
from worldgen.buildings.buildings import prepare_buildings

REGION = {"center_lat": ORIGIN.lat, "center_lon": ORIGIN.lon, "radius_m": 300.0}


@pytest.mark.parametrize("raw,expected", [("12", 12.0), ("12 m", 12.0), ("12.5m", 12.5), ("40'", 12.192),
                                          ("12'6\"", 3.81), ("3,5", 3.5), ("1 km", 1000.0), ("tall", None),
                                          (None, None), ("7;8", 7.0)])
def test_parse_length(raw, expected):
    v = parse_length(raw)
    assert (v is None and expected is None) or v == pytest.approx(expected, abs=1e-3)


def test_parse_speed_and_oneway():
    assert parse_speed("30 mph") == pytest.approx(48.28, abs=0.01)
    assert parse_speed("50") == 50.0
    assert oneway_direction({"highway": "residential", "oneway": "yes"}) == ("forward", True)
    assert oneway_direction({"highway": "residential", "oneway": "-1"}) == ("backward", True)
    assert oneway_direction({"highway": "motorway"}) == ("forward", False)        # implied
    assert oneway_direction({"highway": "primary", "junction": "roundabout"}) == ("forward", False)
    assert oneway_direction({"highway": "primary"}) == ("both", False)


def test_modes():
    assert road_modes({"highway": "footway"}) == {"motor_vehicle": False, "bicycle": False, "foot": True}
    assert road_modes({"highway": "motorway"}) == {"motor_vehicle": True, "bicycle": False, "foot": False}
    assert road_modes({"highway": "residential", "access": "private"})["motor_vehicle"] is False
    assert road_modes({"highway": "footway", "bicycle": "yes"})["bicycle"] is True


def _norm(b: FixtureBuilder, extent_half: float = 300.0):
    cfg = load_config("basic", region=REGION)
    return normalize_osm(b.osm, b.frame, box(-extent_half, -extent_half, extent_half, extent_half), cfg)


def test_road_normalization_keeps_observed_and_inferred_apart():
    b = FixtureBuilder(LocalFrame(ORIGIN))
    b.way([(-100, 0), (100, 0)], {"highway": "primary", "lanes": "3", "name": "A"})
    b.way([(0, -100), (0, 100)], {"highway": "residential", "width": "7.5", "bridge": "yes"}, share=False)
    n = _norm(b)
    a, br = sorted(n.roads, key=lambda r: r.id)
    assert "width" not in a.properties and a.inferred_properties["width"].method == "lanes*lane_width+shoulders"
    assert a.get("width") == pytest.approx(3 * 3.25 + 1.0)
    assert br.properties["width"] == 7.5 and "width" not in br.inferred_properties
    assert br.inferred_properties["layer"].value == 1          # bridge implies layer 1 (estimate only)
    assert "layer" not in br.properties
    assert "deck_elevation" in br.missing


def test_ways_are_clipped_with_boundary_nodes():
    b = FixtureBuilder(LocalFrame(ORIGIN))
    b.way([(-500, 10), (0, 10), (500, 10)], {"highway": "secondary"})
    b.way([(-500, -500), (500, 500)], {"highway": "tertiary"})  # both vertices outside, passes through
    n = _norm(b, 200.0)
    ids = sorted(r.id for r in n.roads)
    assert len(ids) == 2
    for r in n.roads:
        x0, y0, x1, y1 = r.geometry.bounds
        assert x0 >= -200.001 and x1 <= 200.001
        assert r.node_ids[0].startswith("n:bnd:") and r.node_ids[-1].startswith("n:bnd:")


def test_multipolygon_with_hole_and_invalid_footprint():
    b = FixtureBuilder(LocalFrame(ORIGIN))
    outer = b.way([(-40, -40), (40, -40), (40, 40), (-40, 40)], {}, share=False, close=True)
    inner = b.way([(-15, -15), (15, -15), (15, 15), (-15, 15)], {}, share=False, close=True)
    b.relation([("way", outer, "outer"), ("way", inner, "inner")], {"type": "multipolygon", "building": "yes"})
    b.way([(60, 60), (90, 90), (90, 60), (60, 90)], {"building": "shed"}, share=False, close=True)  # bow-tie
    n = _norm(b)
    court = next(x for x in n.buildings if x.source_id.startswith("relation/"))
    assert court.geometry.area == pytest.approx(80 * 80 - 30 * 30)
    assert len(court.geometry.interiors) == 1
    cfg = load_config("basic", region=REGION)
    prepared, stats = prepare_buildings(n.buildings, [], box(-300, -300, 300, 300), cfg)
    assert all(p.geometry.is_valid for p in prepared)
    shed = next(p for p in prepared if p.properties["class"] == "shed")
    assert shed.inferred_properties["footprint_repaired"].value is True


def test_overture_rows_skip_osm_and_conflate():
    frame = LocalFrame(ORIGIN)
    poly_a = frame.geom_to_wgs84(box(0, 0, 20, 20))
    poly_b = frame.geom_to_wgs84(box(100, 100, 120, 115))
    poly_c = frame.geom_to_wgs84(box(1, 1, 19, 19))
    rows = [
        {"id": "osm-derived", "geometry": shapely.to_wkb(poly_a), "height": None,
         "sources": [{"dataset": "OpenStreetMap"}]},
        {"id": "ml-1", "geometry": shapely.to_wkb(poly_b), "height": 12.0,
         "sources": [{"dataset": "Microsoft ML Buildings", "confidence": 0.9}]},
        {"id": "ml-2", "geometry": shapely.to_wkb(poly_c), "height": None,
         "sources": [{"dataset": "Google Open Buildings"}]},
    ]
    import pyarrow as pa

    ov = overture_rows_to_buildings(pa.Table.from_pylist(rows), frame)
    assert sorted(b.source_id for b in ov) == ["ml-1", "ml-2"]
    b = FixtureBuilder(frame)
    b.way([(0, 0), (20, 0), (20, 20), (0, 20)], {"building": "yes"}, share=False, close=True)
    osm = _norm(b).buildings
    cfg = load_config("basic", region=REGION)
    out, stats = prepare_buildings(osm, ov, box(-300, -300, 300, 300), cfg)
    assert stats["overture_duplicates"] == 1 and stats["overture_added"] == 1
    added = next(x for x in out if x.source == "overture")
    assert added.properties["height"] == 12.0 and added.confidence <= 0.85


def test_overpass_json_roundtrip_and_query():
    b = FixtureBuilder(LocalFrame(ORIGIN))
    b.way([(0, 0), (10, 0)], {"highway": "service"})
    doc = b.osm.to_overpass_json()
    again = OsmData.from_overpass_json(doc)
    assert again.ways.keys() == b.osm.ways.keys() and again.nodes.keys() == b.osm.nodes.keys()
    q = build_query((121.5, 25.0, 121.6, 25.1), 60)
    assert "[bbox:25.0000000,121.5000000,25.1000000,121.6000000]" in q and 'relation["type"="restriction"]' in q


def test_cache_offline(tmp_path):
    cache = SourceCache(tmp_path)
    data, meta = cache.get_bytes("osm", ["q"], lambda: (b"payload", {"endpoint": "x"}))
    assert data == b"payload" and meta["sha256"]
    offline = SourceCache(tmp_path, offline=True)
    data2, meta2 = offline.get_bytes("osm", ["q"], lambda: (_ for _ in ()).throw(AssertionError("network used")))
    assert data2 == b"payload" and meta2["sha256"] == meta["sha256"]
    with pytest.raises(OfflineCacheMiss):
        offline.get_bytes("osm", ["other"], lambda: (b"", {}))
