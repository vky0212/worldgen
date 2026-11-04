"""Synthetic geographic scenarios for automated tests and offline demos.

Each fixture produces OSM-shaped data (nodes/ways/relations with real tags) and
rasters exactly like the live adapters do, so the complete pipeline runs on
them unchanged. Coordinates are authored in local metres around a fixed origin
and converted to WGS84 with the same LocalFrame the compiler uses.

Fixtures (plan §6):
 1 simple_intersection      5 tunnel_hill
 2 crossing_no_connection   6 hillside_curve
 3 bridge_river             7 multi_tile_road
 4 overpass                 8 building_holes
 + forest (vegetation extension) and demo_town (everything combined)
"""

from __future__ import annotations

import math
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
from pyproj import CRS

from worldgen.sources.base import SourceRecord
from worldgen.sources.osm import OsmData, OsmMember, OsmNode, OsmRelation, OsmWay
from worldgen.sources.rasters import synthetic_geo_raster
from worldgen.spatial.frame import GeoPoint, LocalFrame
from worldgen.spatial.raster import GeoRaster

ORIGIN = GeoPoint(24.0, 121.0)
SYNTH_LICENSE = "CC0-1.0 (synthetic test data)"


@dataclass
class FixtureBuilder:
    frame: LocalFrame
    osm: OsmData = field(default_factory=OsmData)
    _next_node: int = 1000
    _next_way: int = 100
    _node_at: dict[tuple[float, float], int] = field(default_factory=dict)

    def node(self, x: float, y: float, tags: dict[str, str] | None = None, share: bool = True) -> int:
        key = (round(x, 3), round(y, 3))
        if share and key in self._node_at and not tags:
            return self._node_at[key]
        nid = self._next_node
        self._next_node += 1
        lon, lat = self.frame.to_wgs84(x, y)
        self.osm.nodes[nid] = OsmNode(nid, float(lon), float(lat), dict(tags or {}))
        if share:
            self._node_at[key] = nid
        return nid

    def way(self, pts: list[tuple[float, float]], tags: dict[str, str], share: bool = True,
            close: bool = False) -> int:
        refs = [self.node(x, y, share=share) for x, y in pts]
        if close:
            refs.append(refs[0])
        wid = self._next_way
        self._next_way += 1
        self.osm.ways[wid] = OsmWay(wid, refs, dict(tags))
        return wid

    def relation(self, members: list[tuple[str, int, str]], tags: dict[str, str]) -> int:
        rid = 9000 + len(self.osm.relations)
        self.osm.relations[rid] = OsmRelation(rid, [OsmMember(t, r, role) for t, r, role in members], dict(tags))
        return rid


@dataclass
class Fixture:
    name: str
    description: str
    radius_m: float
    osm: OsmData
    dem: GeoRaster
    worldcover: GeoRaster | None = None
    canopy: GeoRaster | None = None
    expectations: dict[str, Any] = field(default_factory=dict)

    def records(self) -> list[SourceRecord]:
        recs = [
            SourceRecord("osm", "synthetic", SYNTH_LICENSE, f"synthetic fixture '{self.name}'",
                         dataset_version="fixture-1"),
            SourceRecord("dem", "synthetic", SYNTH_LICENSE, f"synthetic terrain '{self.name}'",
                         dataset_version="fixture-1", vertical_datum="EGM2008",
                         coverage={"valid_fraction": 1.0}),
        ]
        recs.append(SourceRecord("worldcover", "synthetic", SYNTH_LICENSE, "synthetic land cover",
                                 status="ok" if self.worldcover is not None else "missing"))
        recs.append(SourceRecord("canopy_height", "synthetic", SYNTH_LICENSE, "synthetic canopy height",
                                 status="ok" if self.canopy is not None else "missing"))
        return recs


def _raster(frame: LocalFrame, half: float, res: float, fn: Callable[[np.ndarray, np.ndarray], np.ndarray],
            dtype: str = "float32", nodata: Any = None) -> GeoRaster:
    """Raster in the local CRS covering [-half, half]^2 (row 0 = north)."""
    n = int(round(2 * half / res))
    xs = -half + (np.arange(n) + 0.5) * res
    ys = half - (np.arange(n) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    data = fn(gx, gy).astype(dtype)
    return synthetic_geo_raster(data, -half, half, res, CRS.from_user_input(frame.local_crs()), nodata, "synthetic")


def _flat(z: float = 10.0) -> Callable[[np.ndarray, np.ndarray], np.ndarray]:
    return lambda x, y: np.full_like(x, z)


def fixture_simple_intersection(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)
    b.way([(-200, 0), (0, 0), (200, 0)], {"highway": "primary", "name": "East Road", "lanes": "2"})
    b.way([(0, -200), (0, 0)], {"highway": "residential", "name": "North Street"})
    b.way([(0, 0), (0, 150), (40, 190)], {"highway": "residential", "name": "North Street"})
    b.way([(40, 190), (120, 190)], {"highway": "service"})  # dead end
    return Fixture("simple_intersection", "Four-way junction with a dead-end service road", 250, b.osm,
                   _raster(frame, 320, 30, _flat()),
                   expectations={"intersections": 1, "dead_ends": 1})


def fixture_crossing_no_connection(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)
    b.way([(-200, 0), (200, 0)], {"highway": "secondary"})
    # Crosses the first road in 2D at (0, 0) but shares no node (data has no junction).
    b.way([(0, -200), (0, 200)], {"highway": "secondary"}, share=False)
    return Fixture("crossing_no_connection", "Two roads crossing without a shared node", 250, b.osm,
                   _raster(frame, 320, 30, _flat()),
                   expectations={"components": 2, "same_level_crossings": 1})


def fixture_bridge_river(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)

    def terrain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        # Valley along y = 0, river at the bottom.
        return 20.0 - 8.0 * np.exp(-((y / 45.0) ** 2))

    b.way([(0, -250), (0, -40)], {"highway": "tertiary", "name": "River Road"})
    b.way([(0, -40), (0, 40)], {"highway": "tertiary", "name": "River Road", "bridge": "yes", "layer": "1"})
    b.way([(0, 40), (0, 250)], {"highway": "tertiary", "name": "River Road"})
    river = b.way([(-250, -12), (-60, -12), (60, -12), (250, -12), (250, 12), (60, 12), (-60, 12), (-250, 12)],
                  {"natural": "water", "water": "river", "name": "Test River"}, share=False, close=True)
    b.way([(-250, 0), (250, 0)], {"waterway": "river", "name": "Test River"}, share=False)
    del river
    return Fixture("bridge_river", "Road bridge over a river in a shallow valley", 260, b.osm,
                   _raster(frame, 330, 10, terrain),
                   expectations={"bridges": 1})


def fixture_overpass(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)
    b.way([(-250, 0), (250, 0)], {"highway": "primary", "name": "Ground Avenue"})
    b.way([(0, -250), (0, -120)], {"highway": "trunk", "name": "Elevated Way"})
    b.way([(0, -120), (0, 120)], {"highway": "trunk", "name": "Elevated Way", "bridge": "viaduct", "layer": "1"},
          share=False)
    b.way([(0, 120), (0, 250)], {"highway": "trunk", "name": "Elevated Way"})
    # Re-link the bridge ends to the approaches (shared nodes at the ramps only).
    osm = b.osm
    ways = sorted(osm.ways)
    w_app1, w_bridge, w_app2 = ways[1], ways[2], ways[3]
    osm.ways[w_bridge].refs[0] = osm.ways[w_app1].refs[-1]
    osm.ways[w_bridge].refs[-1] = osm.ways[w_app2].refs[0]
    return Fixture("overpass", "Elevated road crossing above a ground road", 260, b.osm,
                   _raster(frame, 330, 30, _flat(15.0)),
                   expectations={"bridges": 1, "grade_separated": 1})


def fixture_tunnel_hill(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)

    def terrain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 10.0 + 45.0 * np.exp(-((x / 90.0) ** 2 + (y / 160.0) ** 2))

    b.way([(-250, 0), (-110, 0)], {"highway": "secondary"})
    b.way([(-110, 0), (110, 0)], {"highway": "secondary", "tunnel": "yes", "layer": "-1", "name": "Hill Tunnel"})
    b.way([(110, 0), (250, 0)], {"highway": "secondary"})
    b.way([(-250, 60), (-150, 60), (-60, 120), (0, 200)], {"highway": "track"})
    return Fixture("tunnel_hill", "Tunnel through an elevated hill", 260, b.osm,
                   _raster(frame, 330, 10, terrain), expectations={"tunnels": 1})


def fixture_hillside_curve(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)

    def terrain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 30.0 + 0.18 * y + 6.0 * np.sin(x / 60.0)

    pts = [(160 * math.cos(a), 160 * math.sin(a)) for a in np.linspace(-math.pi / 2, math.pi / 2, 40)]
    b.way([(-220, -160)] + pts + [(-220, 160)], {"highway": "unclassified", "name": "Hill Loop"})
    return Fixture("hillside_curve", "Curved road climbing a hillside", 260, b.osm,
                   _raster(frame, 330, 10, terrain))


def fixture_multi_tile_road(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)
    b.way([(-600, -300), (-200, -100), (100, 50), (550, 320)], {"highway": "primary", "name": "Long Road"})
    b.way([(-600, 300), (600, -300)], {"highway": "secondary", "name": "Cross Road"})
    # Junction with Long Road through an explicitly shared node.
    osm = b.osm
    ways = sorted(osm.ways)
    long_w, cross = osm.ways[ways[0]], osm.ways[ways[1]]
    mid = b.node(0.0, 0.0)
    cross.refs.insert(1, mid)
    long_w.refs.insert(2, mid)
    return Fixture("multi_tile_road", "Roads spanning several 256 m tiles", 650, b.osm,
                   _raster(frame, 720, 30, lambda x, y: 20.0 + 0.01 * x), expectations={"components": 1})


def fixture_building_holes(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)
    outer = b.way([(-40, -40), (40, -40), (40, 40), (-40, 40)], {}, share=False, close=True)
    inner = b.way([(-15, -15), (15, -15), (15, 15), (-15, 15)], {}, share=False, close=True)
    b.relation([("way", outer, "outer"), ("way", inner, "inner")],
               {"type": "multipolygon", "building": "apartments", "building:levels": "6"})
    b.way([(60, 60), (90, 60), (90, 80), (60, 80)], {"building": "house", "roof:shape": "gabled"},
          share=False, close=True)
    b.way([(-90, 60), (-60, 60), (-60, 90), (-90, 90)], {"building": "yes", "height": "21 m", "roof:shape": "pyramidal"},
          share=False, close=True)
    # Self-intersecting (bow-tie) footprint that must be repaired.
    b.way([(-90, -90), (-60, -60), (-60, -90), (-90, -60)], {"building": "shed"}, share=False, close=True)
    b.way([(-200, -120), (200, -120)], {"highway": "residential"})
    return Fixture("building_holes", "Courtyard building (multipolygon with hole) and roof shapes", 220, b.osm,
                   _raster(frame, 300, 30, _flat(5.0)), expectations={"buildings": 4})


def fixture_forest(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)

    def terrain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return 50.0 + 0.12 * x + 10.0 * np.exp(-(((x - 100) / 80.0) ** 2 + ((y + 50) / 80.0) ** 2))

    b.way([(-120, -150), (150, -150), (150, 120), (-120, 120)],
          {"landuse": "forest", "leaf_type": "mixed"}, share=False, close=True)
    b.way([(-250, 0), (250, 0)], {"highway": "track"})
    b.way([(-60, 30), (-30, 30), (-30, 55), (-60, 55)], {"building": "hut"}, share=False, close=True)
    b.way([(-230, -230), (-150, -230), (-150, -170), (-230, -170)], {"landuse": "grass"}, share=False, close=True)
    for k, (x, y) in enumerate([(-200, 150), (-190, 160), (-180, 170)]):
        b.node(x, y, {"natural": "tree", "height": f"{12 + k}", "leaf_type": "broadleaved"}, share=False)
    b.way([(-240, 210), (-100, 210)], {"natural": "tree_row"})

    def worldcover(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.full(x.shape, 30, dtype=np.uint8)
        out[(x > -100) & (x < 230) & (y > -170) & (y < 140)] = 10
        out[(x < -150) & (y > 120)] = 50
        return out

    def canopy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        h = np.where((x > -120) & (x < 150) & (y > -150) & (y < 120), 14.0 + 4.0 * np.sin(x / 40.0), 0.0)
        return h.astype(np.float32)

    return Fixture("forest", "Forest polygon, WorldCover tree cover, canopy heights and mapped trees", 260, b.osm,
                   _raster(frame, 330, 10, terrain), _raster(frame, 330, 10, worldcover, "uint8", 0),
                   _raster(frame, 330, 10, canopy), expectations={"observed_trees": 3})


def fixture_turn_restriction(frame: LocalFrame) -> Fixture:
    b = FixtureBuilder(frame)
    w_ab = b.way([(-200, 0), (0, 0)], {"highway": "secondary", "name": "West"})
    w_bc = b.way([(0, 0), (200, 0)], {"highway": "secondary", "name": "East"})
    w_bd = b.way([(0, 0), (0, 200)], {"highway": "secondary", "name": "North"})
    b.way([(0, 200), (200, 200), (200, 0)], {"highway": "residential", "name": "Loop"})
    via = b.osm.ways[w_ab].refs[-1]
    b.relation([("way", w_ab, "from"), ("node", via, "via"), ("way", w_bd, "to")],
               {"type": "restriction", "restriction": "no_left_turn"})
    b.way([(-200, -100), (200, -100)], {"highway": "residential", "oneway": "yes", "name": "One Way"})
    b.way([(-200, -100), (-200, 0)], {"highway": "residential"})
    del w_bc
    return Fixture("turn_restriction", "No-left-turn restriction and a one-way street", 260, b.osm,
                   _raster(frame, 330, 30, _flat()))


def fixture_demo_town(frame: LocalFrame) -> Fixture:
    """A small town combining all features (used for offline demos and regression)."""
    b = FixtureBuilder(frame)

    def terrain(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        valley = -9.0 * np.exp(-(((y + 20) / 55.0) ** 2))
        hill = 70.0 * np.exp(-(((x - 330) / 150.0) ** 2 + ((y - 300) / 170.0) ** 2))
        return 25.0 + valley + hill + 0.02 * x

    # Grid of streets north of the river.
    for gy in (80, 160, 240):
        b.way([(-450, gy), (-250, gy), (-100, gy), (50, gy)], {"highway": "residential", "name": f"Street {gy}"})
    for gx in (-250, -100):
        b.way([(gx, 80), (gx, 160), (gx, 240), (gx, 330)], {"highway": "residential", "name": f"Avenue {gx}"})
    b.way([(-450, 80), (-450, 160), (-450, 240)], {"highway": "tertiary", "name": "West Lane"})
    # Main road crossing the river on a bridge and continuing south.
    b.way([(50, 450), (50, 240), (50, 160), (50, 80), (50, 30)], {"highway": "primary", "name": "Main Road", "lanes": "2"})
    b.way([(50, 30), (50, -70)], {"highway": "primary", "name": "Main Road", "bridge": "yes", "layer": "1", "lanes": "2"})
    b.way([(50, -70), (50, -200), (60, -450)], {"highway": "primary", "name": "Main Road", "lanes": "2"})
    # Ring road over the main road (overpass) south of the river.
    b.way([(-450, -250), (-120, -250)], {"highway": "trunk", "name": "Ring"})
    b.way([(-120, -250), (220, -250)], {"highway": "trunk", "name": "Ring", "bridge": "yes", "layer": "1"}, share=False)
    b.way([(220, -250), (450, -250)], {"highway": "trunk", "name": "Ring"})
    osm = b.osm
    ws = sorted(osm.ways)
    ring_a, ring_b, ring_c = osm.ways[ws[-3]], osm.ways[ws[-2]], osm.ways[ws[-1]]
    ring_b.refs[0] = ring_a.refs[-1]
    ring_b.refs[-1] = ring_c.refs[0]
    # Tunnel through the eastern hill.
    b.way([(50, 240), (170, 240)], {"highway": "secondary", "name": "Hill Road"})
    b.way([(170, 240), (470, 260)], {"highway": "secondary", "name": "Hill Road", "tunnel": "yes", "layer": "-1"})
    b.way([(470, 260), (560, 260)], {"highway": "secondary", "name": "Hill Road"})
    # River and stream.
    b.way([(-560, -40), (560, -40), (560, 0), (-560, 0)], {"natural": "water", "water": "river", "name": "Demo River"},
          share=False, close=True)
    b.way([(-200, 450), (-180, 300), (-170, 120), (-160, 0)], {"waterway": "stream"}, share=False)
    # Buildings along streets.
    for gx in (-400, -330, -190, -160, -40, 0):
        for gy in (100, 180, 260):
            w, h = 22.0, 16.0
            tags = {"building": "residential", "building:levels": str(2 + (int(gx + gy) // 10) % 4)}
            if gx == -40 and gy == 180:
                tags = {"building": "commercial", "height": "28"}
            if gx == 0 and gy == 100:
                tags = {"building": "house", "roof:shape": "gabled"}
            b.way([(gx, gy), (gx + w, gy), (gx + w, gy + h), (gx, gy + h)], tags, share=False, close=True)
    # Forest on the hill and a park.
    b.way([(200, 120), (560, 120), (560, 560), (200, 560)], {"landuse": "forest"}, share=False, close=True)
    b.way([(-450, -200), (-200, -200), (-200, -100), (-450, -100)], {"leisure": "park", "name": "Town Park"},
          share=False, close=True)
    for k in range(5):
        b.node(-430 + 40 * k, -150, {"natural": "tree", "species": "Ficus microcarpa"}, share=False)

    def worldcover(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        out = np.full(x.shape, 30, dtype=np.uint8)
        out[(x < 100) & (y > 60) & (y < 350)] = 50
        out[(x > 180) & (y > 100)] = 10
        out[(x > -450) & (x < -200) & (y > -200) & (y < -100) & (np.sin(x / 15.0) > 0)] = 10
        out[(y > -40) & (y < 0)] = 80
        return out

    def canopy(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        return np.where((x > 180) & (y > 100), 18.0 + 3.0 * np.sin(y / 50.0), 0.0).astype(np.float32)

    return Fixture("demo_town", "Small town: grid streets, river bridge, overpass, tunnel, forest, park", 600, b.osm,
                   _raster(frame, 720, 10, terrain), _raster(frame, 720, 10, worldcover, "uint8", 0),
                   _raster(frame, 720, 10, canopy))


FIXTURES: dict[str, Callable[[LocalFrame], Fixture]] = {
    "simple_intersection": fixture_simple_intersection,
    "crossing_no_connection": fixture_crossing_no_connection,
    "bridge_river": fixture_bridge_river,
    "overpass": fixture_overpass,
    "tunnel_hill": fixture_tunnel_hill,
    "hillside_curve": fixture_hillside_curve,
    "multi_tile_road": fixture_multi_tile_road,
    "building_holes": fixture_building_holes,
    "forest": fixture_forest,
    "turn_restriction": fixture_turn_restriction,
    "demo_town": fixture_demo_town,
}


def load_fixture(name: str, origin: GeoPoint = ORIGIN, method: str = "tmerc") -> Fixture:
    if name not in FIXTURES:
        raise KeyError(f"unknown fixture {name!r}; available: {', '.join(sorted(FIXTURES))}")
    return FIXTURES[name](LocalFrame(origin, method))
