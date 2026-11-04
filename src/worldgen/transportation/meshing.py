"""Road, intersection, bridge and tunnel geometry generation.

Produces tile-owned :class:`Piece` objects. Road surfaces are flat ribbons that
follow the source centreline and the computed elevation profile; junctions
(see ``junctions.py``) are corner polygons whose cut vertices are the ribbon ends; bridges add a deck box, railings
and piers; tunnels add an inward-facing tube and portal frames.
"""

from __future__ import annotations

import math
from collections.abc import Iterator

import numpy as np
from shapely.geometry import Point
from shapely.strtree import STRtree

from worldgen.config import WorldConfig
from worldgen.geometry.shapes import box, polyline_frames, ribbon, swept_box
from worldgen.tiling.grid import Piece, TileGrid
from worldgen.topology.graph import GraphEdge, RoadGraph
from worldgen.transportation.elevation import EdgeProfile
from worldgen.transportation.junctions import MAJOR, PATHS, JunctionLayout, surface_offset, trim_polyline
from worldgen.world.model import Bridge, Intersection, Tunnel

def road_material(cls: str) -> str:
    if cls in MAJOR:
        return "road_major"
    if cls in PATHS:
        return "road_path"
    return "road_minor"


def _runs(tiles: list[str]) -> Iterator[tuple[int, int, str]]:
    """(start_station, end_station, tile) for consecutive segments in the same tile."""
    start = 0
    for k in range(1, len(tiles) + 1):
        if k == len(tiles) or tiles[k] != tiles[start]:
            yield start, k, tiles[start]
            start = k


def build_intersections(graph: RoadGraph, layout: JunctionLayout, grid: TileGrid
                        ) -> tuple[list[Intersection], list[Piece]]:
    """Intersection entities (3+ road ends) and the surfaces of every junction polygon."""
    ents: list[Intersection] = []
    pieces: list[Piece] = []
    for nid in sorted(layout.junctions):
        j = layout.junctions[nid]
        n = graph.nodes[nid]
        if j.degree >= 3:
            names = sorted({graph.edges[e].name for e in n.edges if graph.edges[e].name})
            classes = sorted({graph.edges[e].road_class for e in n.edges})
            ents.append(Intersection(
                id=j.feature_id, geometry=j.polygon,
                properties={"node_id": nid, "degree": n.degree, "road_classes": classes,
                            **({"streets": names} if names else {})},
                source="derived", source_id=nid, confidence=0.8, node_id=nid, elevation=n.z))
        if not j.mesh.is_empty():
            pieces.append(Piece(grid.tile_of(n.x, n.y), "roads", "road_major" if j.major else "road_minor",
                                j.feature_id, j.mesh))
    return ents, pieces


def build_road_pieces(graph: RoadGraph, profiles: dict[str, EdgeProfile], layout: JunctionLayout,
                      cfg: WorldConfig, grid: TileGrid, wanted: set[str] | None = None) -> list[Piece]:
    pieces: list[Piece] = []
    exclude = set(cfg.roads.mesh_exclude_classes)
    for eid in sorted(graph.edges):
        e = graph.edges[eid]
        if e.road_class in exclude:
            continue
        p = profiles[eid]
        t0, t1 = layout.trim(eid)
        cut = trim_polyline(p.xy, p.s, p.z, t0, t1)
        if cut is None:
            continue
        xy, s, z = cut
        xy, s, z = grid.insert_tile_crossings(xy, s, z)
        tiles = grid.segment_tiles(xy)
        if wanted is not None and not (set(tiles) & wanted):
            continue
        off = surface_offset(cfg, p.kind, e.road_class)
        mat = road_material(e.road_class)
        start_mitre = layout.mitres.get((eid, 0)) if t0 == 0 else None
        end_mitre = layout.mitres.get((eid, 1)) if t1 == 0 else None
        last = len(tiles)
        for a, b, tile in _runs(tiles):
            if wanted is not None and tile not in wanted:
                continue
            m = ribbon(xy[a:b + 1], z[a:b + 1] + off, e.width, s0=float(s[a]),
                       start_offset=start_mitre if a == 0 else None,
                       end_offset=end_mitre if b == last else None)
            pieces.append(Piece(tile, _layer_for(p.kind), mat, e.road_id, m))
    return pieces


def _layer_for(kind: str) -> str:
    return {"ground": "roads", "bridge": "bridges", "tunnel": "tunnels"}[kind]


def build_bridge_pieces(bridges: list[Bridge], graph: RoadGraph, profiles: dict[str, EdgeProfile],
                        terrain_sample, cfg: WorldConfig, grid: TileGrid,  # noqa: ANN001
                        wanted: set[str] | None = None) -> list[Piece]:
    bc = cfg.bridges
    pieces: list[Piece] = []
    ground_lines = [e.line for e in graph.edges.values() if not e.bridge]
    ground_w = [e.width for e in graph.edges.values() if not e.bridge]
    tree = STRtree(ground_lines) if ground_lines else None
    for br in bridges:
        for eid in br.edge_ids:
            e = graph.edges[eid]
            p = profiles[eid]
            xy, s, z = grid.insert_tile_crossings(p.xy, p.s, p.z)
            tiles = grid.segment_tiles(xy)
            for a, b, tile in _runs(tiles):
                if wanted is not None and tile not in wanted:
                    continue
                seg_xy, seg_z = xy[a:b + 1], z[a:b + 1]
                pieces.append(Piece(tile, "bridges", "bridge_deck", br.id,
                                    swept_box(seg_xy, seg_z, e.width + 0.6, bc.deck_thickness)))
                if bc.railings and len(seg_xy) >= 2:
                    left, scale = polyline_frames(seg_xy)
                    for side in (1.0, -1.0):
                        off = side * (e.width / 2.0 + 0.15) * scale[:, None] * left
                        pieces.append(Piece(tile, "bridges", "bridge_railing", br.id,
                                            swept_box(seg_xy + off, seg_z + bc.railing_height, 0.25,
                                                      bc.railing_height)))
            if bc.supports == "simple":
                pieces += _piers(br, e, p, terrain_sample, bc, grid, tree, ground_w, wanted)
    return pieces


def _piers(br: Bridge, e: GraphEdge, p: EdgeProfile, terrain_sample, bc, grid: TileGrid,  # noqa: ANN001
           tree: STRtree | None, ground_w: list[float], wanted: set[str] | None) -> list[Piece]:
    out = []
    total = float(p.s[-1])
    n = int(total // bc.pier_spacing)
    if n < 1:
        return out
    step = total / (n + 1)
    for k in range(1, n + 1):
        sk = k * step
        x = float(np.interp(sk, p.s, p.xy[:, 0]))
        y = float(np.interp(sk, p.s, p.xy[:, 1]))
        z = float(np.interp(sk, p.s, p.z))
        tz = float(terrain_sample(x, y))
        bottom = z - bc.deck_thickness
        if bottom - tz < 1.5:
            continue
        if tree is not None:
            hit = False
            for j in tree.query(Point(x, y).buffer(bc.pier_width + 8.0)).tolist():
                if tree.geometries[j].distance(Point(x, y)) < ground_w[j] / 2.0 + bc.pier_width:
                    hit = True
                    break
            if hit:
                continue
        tile = grid.tile_of(x, y)
        if wanted is not None and tile not in wanted:
            continue
        i = int(np.searchsorted(p.s, sk).clip(1, len(p.s) - 1))
        d = p.xy[i] - p.xy[i - 1]
        yaw = math.atan2(d[1], d[0])
        out.append(Piece(tile, "bridges", "bridge_pier", br.id,
                         box(x, y, tz - 0.5, bottom, bc.pier_width, max(e.width * 0.6, bc.pier_width), yaw)))
    return out


def build_tunnel_pieces(tunnels: list[Tunnel], graph: RoadGraph, profiles: dict[str, EdgeProfile],
                        cfg: WorldConfig, grid: TileGrid, wanted: set[str] | None = None) -> list[Piece]:
    tc = cfg.tunnels
    pieces: list[Piece] = []
    for tn in tunnels:
        for eid in tn.edge_ids:
            e = graph.edges[eid]
            p = profiles[eid]
            xy, s, z = grid.insert_tile_crossings(p.xy, p.s, p.z)
            tiles = grid.segment_tiles(xy)
            for a, b, tile in _runs(tiles):
                if wanted is not None and tile not in wanted:
                    continue
                pieces.append(Piece(tile, "tunnels", "tunnel_wall", tn.id,
                                    swept_box(xy[a:b + 1], z[a:b + 1] + tc.clearance_height, e.width + 1.5,
                                              tc.clearance_height + 0.3, inward=True)))
        if tc.portals:
            for nid in tn.anchor_node_ids:
                n = graph.nodes[nid]
                tile = grid.tile_of(n.x, n.y)
                if wanted is not None and tile not in wanted:
                    continue
                eids = [x for x in n.edges if x in tn.edge_ids]
                if not eids:
                    continue
                e = graph.edges[eids[0]]
                c = e.coords
                d = (c[1] - c[0]) if e.u == nid else (c[-2] - c[-1])
                yaw = math.atan2(d[1], d[0])
                w = e.width + 1.5
                h = tc.clearance_height
                ux, uy = -math.sin(yaw), math.cos(yaw)   # across the road
                parts = []
                for side in (1.0, -1.0):
                    cx = n.x + side * ux * (w / 2.0 + 0.6)
                    cy = n.y + side * uy * (w / 2.0 + 0.6)
                    parts.append(box(cx, cy, n.z - 0.3, n.z + h + 1.2, 1.2, 1.2, yaw))
                parts.append(box(n.x, n.y, n.z + h, n.z + h + 1.2, 1.2, w + 2.4, yaw))
                for m in parts:
                    pieces.append(Piece(tile, "tunnels", "tunnel_portal", tn.id, m))
    return pieces


def build_collision_pieces(graph: RoadGraph, profiles: dict[str, EdgeProfile], cfg: WorldConfig,
                           grid: TileGrid, wanted: set[str] | None = None) -> list[Piece]:
    """Coarse road/deck surfaces for physics (sparser stations, simplified centrelines)."""
    pieces: list[Piece] = []
    sp = cfg.roads.collision_spacing
    for eid in sorted(graph.edges):
        e = graph.edges[eid]
        if e.road_class in PATHS:
            continue
        p = profiles[eid]
        n = max(2, int(math.ceil(p.s[-1] / sp)) + 1)
        sk = np.linspace(0.0, float(p.s[-1]), n)
        xy = np.column_stack([np.interp(sk, p.s, p.xy[:, 0]), np.interp(sk, p.s, p.xy[:, 1])])
        z = np.interp(sk, p.s, p.z) + (cfg.roads.surface_offset if p.kind == "ground" else 0.0)
        xy, sk, z = grid.insert_tile_crossings(xy, sk, z)
        tiles = grid.segment_tiles(xy)
        for a, b, tile in _runs(tiles):
            if wanted is not None and tile not in wanted:
                continue
            pieces.append(Piece(tile, "collision", "collision", e.road_id,
                                ribbon(xy[a:b + 1], z[a:b + 1], e.width, s0=float(sk[a]))))
    return pieces
