"""The eight synthetic scenarios of plan §6, plus turn restrictions.

Each test compiles a fixture through the real pipeline and checks the
geographic/topological property the scenario exists for.
"""

from __future__ import annotations

import numpy as np
import pytest
import trimesh

from worldgen.buildings.buildings import building_mesh
from worldgen.export.tiles import generate_pieces
from worldgen.validation.checks import validate_world


def _node_at(g, x: float, y: float, tol: float = 0.5):  # noqa: ANN001, ANN202
    for n in g.nodes.values():
        if abs(n.x - x) < tol and abs(n.y - y) < tol:
            return n
    return None


def _errors(world) -> list:  # noqa: ANN001
    return [i for i in validate_world(world).issues if i.severity == "error"]


def test_1_simple_intersection(compiled):
    w = compiled("simple_intersection")
    g = w.graph
    centre = _node_at(g, 0, 0)
    assert centre is not None and centre.kind == "intersection" and centre.degree == 4
    dead = [n for n in g.nodes.values() if n.kind == "dead_end"]
    assert any(abs(n.x - 120) < 0.5 and abs(n.y - 190) < 0.5 for n in dead)   # service road stays a dead end
    assert len(w.intersections) == 1
    route = g.shortest_path(_node_at(g, -200, 0).id, _node_at(g, 120, 190).id)
    assert route is not None and route[2] == pytest.approx(200 + 150 + np.hypot(40, 40) + 80, abs=1.0)
    assert not _errors(w)


def test_2_crossing_without_connection(compiled):
    w = compiled("crossing_no_connection")
    g = w.graph
    assert _node_at(g, 0, 0) is None                      # no node invented at the 2D crossing
    assert len(g.components()) == 2
    assert [c.kind for c in w.crossings] == ["same_level_unconnected"]
    a, b = _node_at(g, -200, 0), _node_at(g, 0, -200)
    assert g.shortest_path(a.id, b.id) is None
    issues = [i for i in validate_world(w).issues if i.check == "false_intersection"]
    assert len(issues) == 1 and issues[0].x == pytest.approx(0, abs=0.01)


def test_3_bridge_over_river(compiled):
    w = compiled("bridge_river")
    assert len(w.bridges) == 1
    br = w.bridges[0]
    assert len(br.anchor_node_ids) == 2                   # connects to roads at both ends
    river = next(x for x in w.water if x.is_area)
    assert river.id in br.crossings and br.id in river.crossings
    p = w.profiles[br.edge_ids[0]]
    level = w.water_levels[river.id]
    over_water = np.abs(p.xy[:, 1]) < 12
    assert (p.z[over_water] - w.config.bridges.deck_thickness >= level + w.config.bridges.min_clearance_water - 1e-6).all()
    # Deck continuous with the approach roads, no jumps.
    assert np.max(np.abs(np.diff(p.z))) < 1.0
    for a in br.anchor_node_ids:
        n = w.graph.nodes[a]
        assert n.z == pytest.approx(float(w.terrain_base.sample(n.x, n.y)))
    assert "clearance" in br.missing or "clearance" in br.inferred_properties
    assert not _errors(w)


def test_4_road_below_elevated_road(compiled):
    w = compiled("overpass")
    g = w.graph
    sep = [c for c in w.crossings if c.kind == "grade_separated"]
    assert len(sep) == 1
    up, lo = w.profiles[sep[0].upper], w.profiles[sep[0].lower]
    gap = up.z_at(sep[0].x, sep[0].y) - lo.z_at(sep[0].x, sep[0].y)
    assert gap >= w.config.bridges.min_clearance_road + w.config.bridges.deck_thickness - 1e-6
    # The two roads are not connected anywhere.
    ground = _node_at(g, -250, 0) or _node_at(g, -256, 0, 10)
    elevated = _node_at(g, 0, -250) or _node_at(g, 0, -256, 10)
    assert g.shortest_path(ground.id, elevated.id, mode=None) is None
    assert w.bridges[0].level == 1 and "viaduct" == w.bridges[0].properties.get("bridge_type")
    assert not _errors(w)


def test_5_tunnel_through_hill(compiled):
    w = compiled("tunnel_hill")
    assert len(w.tunnels) == 1 and not w.bridges
    t = w.tunnels[0]
    assert t.kind == "tunnel" and len(t.anchor_node_ids) == 2
    p = w.profiles[t.edge_ids[0]]
    mid = len(p.z) // 2
    terrain_mid = float(w.terrain.sample(*p.xy[mid]))
    assert terrain_mid - p.z[mid] > w.config.tunnels.clearance_height + 5   # well under the hill
    assert np.max(np.abs(np.diff(p.z) / np.maximum(np.diff(p.s), 1e-6))) <= w.config.tunnels.max_grade + 1e-6
    # Route goes through the tunnel.
    g = w.graph
    a, b = _node_at(g, -250, 0, 7), _node_at(g, 250, 0, 7)
    nodes, edges, _ = g.shortest_path(a.id, b.id)
    assert any(g.edges[e].tunnel for e in edges)
    assert not _errors(w)


def test_6_curved_hillside_road(compiled):
    w = compiled("hillside_curve")
    eid = next(iter(w.graph.edges))
    p = w.profiles[eid]
    base = w.terrain_base.sample(p.xy[:, 0], p.xy[:, 1])
    assert np.max(np.abs(p.z - base)) < 1.5                # follows the terrain (small cut/fill)
    grade = np.abs(np.diff(p.z)) / np.maximum(np.diff(p.s), 1e-9)
    assert grade.max() <= w.config.roads.max_grade + 1e-6  # within the drivable grade limit
    shaped = w.terrain.sample(p.xy[:, 0], p.xy[:, 1])
    assert np.max(np.abs(shaped - p.z)) < 0.05              # terrain conformed under the road
    pieces = [x for ps in generate_pieces(w, set(w.graph.edges[eid].tiles)).values() for x in ps
              if x.layer == "roads"]
    line = w.graph.edges[eid].line
    from shapely.geometry import Point

    width = w.graph.edges[eid].width
    for piece in pieces:
        d = [line.distance(Point(x, y)) for x, y, _ in piece.mesh.positions]
        assert max(d) <= width / 2.0 * 3.0 + 1e-6        # miter-limited offsets around the centreline
        z_t = w.terrain.sample(piece.mesh.positions[:, 0], piece.mesh.positions[:, 1])
        assert np.all(piece.mesh.positions[:, 2] - z_t > -0.1)   # road surface above the terrain
    assert not _errors(w)


def test_7_road_across_tiles(compiled):
    w = compiled("multi_tile_road")
    g = w.graph
    assert len(g.components()) == 1
    multi = [e for e in g.edges.values() if len(e.tiles) > 2]
    assert multi
    pieces = generate_pieces(w, {t for e in multi for t in e.tiles})
    for road_id in {e.road_id for e in multi}:
        rp = [p for ps in pieces.values() for p in ps if p.feature_id == road_id and p.layer == "roads"]
        assert len({p.tile for p in rp}) >= 2
        # Pieces of one edge share their tile-boundary vertices exactly (no cracks): the pieces
        # form at most one connected chain per graph edge of the road.
        verts = [set(map(tuple, np.round(p.mesh.positions, 6))) for p in rp]
        parent = list(range(len(rp)))

        def find(i: int) -> int:
            while parent[i] != i:
                i = parent[i]
            return i

        for i in range(len(rp)):
            for j in range(i + 1, len(rp)):
                if verts[i] & verts[j]:
                    parent[find(i)] = find(j)
        n_edges = sum(1 for e in g.edges.values() if e.road_id == road_id)
        assert len({find(i) for i in range(len(rp))}) <= n_edges
    a, b = _node_at(g, -600, -300, 30) or min(g.nodes.values(), key=lambda n: n.x + n.y), None
    far = max(g.nodes.values(), key=lambda n: n.x + n.y)
    assert g.shortest_path(a.id, far.id) is not None
    assert not _errors(w)


def test_8_building_with_holes(compiled):
    w = compiled("building_holes")
    court = next(b for b in w.buildings if b.source_id.startswith("relation/"))
    assert len(court.geometry.interiors) == 1
    assert court.get("height") == pytest.approx(6 * 3.0) and not court.is_observed("height")
    m = building_mesh(court, w.config.buildings)
    tm = trimesh.Trimesh(m.positions, m.indices, process=True)
    tm.merge_vertices()
    assert tm.is_watertight and tm.volume > 0
    expected = court.geometry.area * (court.get("height") + w.config.buildings.foundation_depth)
    assert tm.volume == pytest.approx(expected, rel=1e-6)
    tall = next(b for b in w.buildings if b.properties.get("height") == 21.0)
    assert tall.is_observed("height") and tall.get("roof_shape") == "pyramidal"
    gabled = next(b for b in w.buildings if b.properties.get("roof_shape") == "gabled")
    gm = building_mesh(gabled, w.config.buildings)
    gt = trimesh.Trimesh(gm.positions, gm.indices, process=True)
    gt.merge_vertices()
    assert gt.is_watertight and gt.volume > gabled.geometry.area * 3.0
    assert gm.positions[:, 2].max() == pytest.approx(gabled.base_elevation + gabled.get("height"))
    shed = next(b for b in w.buildings if b.properties.get("class") == "shed")
    assert shed.geometry.is_valid and "footprint_repaired" in shed.inferred_properties
    assert not _errors(w)


def test_turn_restriction_and_oneway(compiled):
    w = compiled("turn_restriction")
    g = w.graph
    r = g.restrictions[0]
    assert r.enforced and r.restriction == "no_left_turn"
    west, north = _node_at(g, -200, 0), _node_at(g, 0, 200)
    nodes, edges, length = g.shortest_path(west.id, north.id)
    assert length > 200 + 200 + 1                     # detour, not the forbidden left turn
    assert g.shortest_path(west.id, north.id, restrictions=False)[2] == pytest.approx(400, abs=0.5)
    a, b = _node_at(g, -200, -100), _node_at(g, 200, -100)
    assert g.shortest_path(a.id, b.id)[2] == pytest.approx(400, abs=0.5)
    back = g.shortest_path(b.id, a.id)
    assert back is None or back[2] > 400.5            # cars cannot drive against the one-way
    assert g.shortest_path(b.id, a.id, mode="foot")[2] == pytest.approx(400, abs=0.5)


def test_untagged_layers_and_conflicting_anchors():
    """Lessons from real data (Taipei): layer without bridge/tunnel tags, and anchors at
    heights the grade limit cannot connect, must still give separated, continuous decks."""
    from worldgen.build import fixture_region
    from worldgen.config import load_config
    from worldgen.pipeline import compile_world
    from worldgen.sources.synthetic import ORIGIN, FixtureBuilder, _raster
    from worldgen.spatial.frame import LocalFrame
    from worldgen.spatial.region import Region
    from worldgen.world.world import SourceBundle

    frame = LocalFrame(ORIGIN)
    b = FixtureBuilder(frame)
    b.way([(-200, 0), (200, 0)], {"highway": "secondary"})
    # Underground corridor crossing the street: layer=-1 but no tunnel tag.
    b.way([(-50, -150), (-50, -60)], {"highway": "footway"})
    b.way([(-50, -60), (-50, 60)], {"highway": "corridor", "layer": "-1"}, share=False)
    b.way([(-50, 60), (-50, 150)], {"highway": "footway"})
    ways = sorted(b.osm.ways)
    corridor = b.osm.ways[ways[2]]
    corridor.refs[0] = b.osm.ways[ways[1]].refs[-1]
    corridor.refs[-1] = b.osm.ways[ways[3]].refs[0]
    # Tunnel whose middle node is joined by an untagged road up on the hill.
    b.way([(100, -200), (100, -120)], {"highway": "primary"})
    b.way([(100, -120), (100, -60), (100, 120)], {"highway": "primary", "tunnel": "yes"})
    b.way([(100, -60), (160, -60)], {"highway": "service"})

    def terrain(x, y):  # noqa: ANN001, ANN202
        return 10.0 + 30.0 * np.exp(-(((x - 120) / 60.0) ** 2 + ((y + 50) / 60.0) ** 2))

    cfg = load_config("basic", region={**fixture_region("simple_intersection"), "radius_m": 220.0})
    region = Region.from_config(cfg)
    w = compile_world(region, cfg, SourceBundle(osm=b.osm, dem=_raster(frame, 300, 10, terrain)))
    corr = next(r for r in w.roads.values() if r.road_class == "corridor")
    assert corr.is_tunnel and corr.structure_implied and "tunnel" not in corr.properties
    assert corr.inferred_properties["tunnel"].method == "osm_layer_below_ground_without_tunnel_tag"
    sep = [c for c in w.crossings if c.kind == "grade_separated"]
    assert sep
    for c in sep:
        gap = w.profiles[c.upper].z_at(c.x, c.y) - w.profiles[c.lower].z_at(c.x, c.y)
        assert gap > 2.5
    tunnel = next(t for t in w.tunnels if not t.structure_implied if False) if False else \
        next(t for t in w.tunnels if any(w.roads[r].is_observed("tunnel") for r in t.road_ids))
    assert len(tunnel.anchor_node_ids) >= 2
    errors = [i for i in validate_world(w).issues if i.severity == "error"]
    assert not errors, [e.message for e in errors]
    assert tunnel.get("design_grade") >= w.config.tunnels.max_grade


def _ribbon_ends(w, eid):  # noqa: ANN001, ANN202
    from worldgen.geometry.shapes import ribbon
    from worldgen.transportation.junctions import trim_polyline

    e, p, lay = w.graph.edges[eid], w.profiles[eid], w.junctions
    t0, t1 = lay.trim(eid)
    xy, _, z = trim_polyline(p.xy, p.s, p.z, t0, t1)
    m = ribbon(xy, z, e.width, start_offset=lay.mitres.get((eid, 0)) if t0 == 0 else None,
               end_offset=lay.mitres.get((eid, 1)) if t1 == 0 else None)
    return m.positions[:2, :2], m.positions[-2:, :2], m


def test_junction_polygons_meet_ribbons_without_overlap(compiled):
    from shapely.geometry import Polygon

    w = compiled("simple_intersection")
    centre = _node_at(w.graph, 0, 0)
    j = w.junctions.junctions[centre.id]
    assert j.degree == 4 and j.polygon.is_valid
    outline = np.asarray(j.polygon.exterior.coords)[:, :2]
    for eid in centre.edges:
        start, end, m = _ribbon_ends(w, eid)
        cut = start if w.graph.edges[eid].u == centre.id else end
        for v in cut:   # both ribbon end vertices are junction vertices: no gap
            assert np.min(np.hypot(*(outline - v).T)) < 1e-6
        pos = m.positions[:, :2]
        footprint = Polygon(np.vstack([pos[0::2], pos[1::2][::-1]]))
        assert footprint.intersection(j.polygon).area < 1e-3     # and no overlap (no z-fighting)
    # Corners lie where the road edges meet: a 7.5 m road (2 lanes) crossing a 6 m road.
    xs, ys = outline[:, 0], outline[:, 1]
    assert xs.max() == pytest.approx(3.0, abs=0.01) and ys.max() == pytest.approx(3.75, abs=0.01)


def test_mitred_joint_shares_vertices():
    from worldgen.build import fixture_region
    from worldgen.config import load_config
    from worldgen.pipeline import compile_world
    from worldgen.sources.synthetic import ORIGIN, FixtureBuilder, _raster
    from worldgen.spatial.frame import LocalFrame
    from worldgen.spatial.region import Region
    from worldgen.world.world import SourceBundle

    frame = LocalFrame(ORIGIN)
    b = FixtureBuilder(frame)
    b.way([(-150, 0), (0, 0)], {"highway": "residential", "name": "A"})
    b.way([(0, 0), (100, 60)], {"highway": "residential", "name": "B"})     # 31 degree bend at a way joint
    cfg = load_config("basic", region={**fixture_region("simple_intersection"), "radius_m": 200.0})
    w = compile_world(Region.from_config(cfg), cfg,
                      SourceBundle(osm=b.osm, dem=_raster(frame, 300, 30, lambda x, y: np.full_like(x, 5.0))))
    joint = _node_at(w.graph, 0, 0)
    assert joint.degree == 2 and joint.id not in w.junctions.junctions
    e1, e2 = sorted(joint.edges)
    _, end1, _ = _ribbon_ends(w, e1)
    start2, _, _ = _ribbon_ends(w, e2)
    a = end1 if w.graph.edges[e1].v == joint.id else _ribbon_ends(w, e1)[0]
    c = start2 if w.graph.edges[e2].u == joint.id else _ribbon_ends(w, e2)[1]
    # Same two points (order may differ): the joint is closed on both sides.
    assert sorted(map(tuple, np.round(a, 6))) == sorted(map(tuple, np.round(c, 6)))
