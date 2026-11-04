"""Detection and classification of 2D crossings without a shared node.

A crossing between two road edges that share no graph node at the crossing point
is either a legitimate grade separation (different layer, or bridge/tunnel
involved) or a suspicious same-level crossing that the source data does not
connect. Neither case ever creates a connection.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import Point
from shapely.strtree import STRtree

from worldgen.topology.graph import GraphEdge, RoadGraph
from worldgen.world.model import Waterbody


@dataclass
class Crossing:
    upper: str            # edge id (or water id) on top
    lower: str
    x: float
    y: float
    kind: str             # "grade_separated" | "same_level_unconnected" | "water"
    layer_gap: int = 0


def _level_key(e: GraphEdge) -> tuple[int, int]:
    """Sort key for vertical order: layer first, then bridge above ground above tunnel."""
    return (e.layer, 1 if e.bridge else (-1 if e.tunnel else 0))


def find_road_crossings(graph: RoadGraph) -> list[Crossing]:
    ids = sorted(graph.edges)
    if not ids:
        return []
    lines = [graph.edges[i].line for i in ids]
    tree = STRtree(lines)
    left, right = tree.query(lines, predicate="intersects")
    out: list[Crossing] = []
    for a, b in zip(left.tolist(), right.tolist()):
        if a >= b:
            continue
        ea, eb = graph.edges[ids[a]], graph.edges[ids[b]]
        shared = {ea.u, ea.v} & {eb.u, eb.v}
        inter = lines[a].intersection(lines[b])
        pts = [inter] if inter.geom_type == "Point" else [g for g in getattr(inter, "geoms", []) if g.geom_type == "Point"]
        for p in pts:
            # Intersections at a shared graph node are real connections.
            if any(np.hypot(graph.nodes[n].x - p.x, graph.nodes[n].y - p.y) < 0.05 for n in shared):
                continue
            # Touching at a shared interior vertex id would have been split into a node already.
            ka, kb = _level_key(ea), _level_key(eb)
            if ka == kb:
                kind = "same_level_unconnected"
                upper, lower = ea, eb
            else:
                kind = "grade_separated"
                upper, lower = (ea, eb) if ka > kb else (eb, ea)
            out.append(Crossing(upper.id, lower.id, float(p.x), float(p.y), kind, abs(ka[0] - kb[0])))
    out.sort(key=lambda c: (c.upper, c.lower, round(c.x, 3), round(c.y, 3)))
    return out


def find_water_crossings(graph: RoadGraph, water: list[Waterbody]) -> list[Crossing]:
    """Road edges crossing water areas or waterway centrelines."""
    if not water or not graph.edges:
        return []
    ids = sorted(graph.edges)
    lines = [graph.edges[i].line for i in ids]
    geoms = [w.geometry for w in water]
    tree = STRtree(geoms)
    out: list[Crossing] = []
    for k, line in enumerate(lines):
        for j in tree.query(line, predicate="intersects").tolist():
            w = water[j]
            inter = line.intersection(w.geometry)
            if inter.is_empty:
                continue
            p = inter.representative_point() if not isinstance(inter, Point) else inter
            if w.is_area:
                # Use the middle of the crossed stretch.
                p = shapely.line_interpolate_point(inter, 0.5, normalized=True) if inter.geom_type in (
                    "LineString", "MultiLineString") else p
            out.append(Crossing(ids[k], w.id, float(p.x), float(p.y), "water"))
    out.sort(key=lambda c: (c.upper, c.lower))
    return out
