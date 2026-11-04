"""Routable road graph built from canonical roads.

Physical road geometry (``Road``) and topology are separate: graph nodes are
road vertices that are shared by several roads (by source node id), road
endpoints, region-boundary cuts and turn-restriction via nodes. Edges are the
road pieces between consecutive graph nodes. Nothing is ever connected because
two lines merely cross in 2D.
"""

from __future__ import annotations

import heapq
import math
from collections import defaultdict
from dataclasses import dataclass, field
from typing import Any

import networkx as nx
import numpy as np
from scipy.spatial import cKDTree
from shapely.geometry import LineString

from worldgen.world.model import Road, TurnRestriction

MODES = ("motor_vehicle", "bicycle", "foot")


@dataclass
class GraphNode:
    id: str
    x: float
    y: float
    z: float = 0.0
    kind: str = "connector"
    edges: list[str] = field(default_factory=list)

    @property
    def osm_id(self) -> int | None:
        if self.id.startswith("n:osm:"):
            try:
                return int(self.id[6:])
            except ValueError:
                return None
        return None

    @property
    def degree(self) -> int:
        return len(self.edges)


@dataclass
class GraphEdge:
    id: str
    road_id: str
    u: str
    v: str
    coords: np.ndarray                 # (n, 2) local xy, u -> v
    vertex_node_ids: list[str]
    direction: str                     # both | forward | backward
    road_class: str
    name: str | None
    bridge: bool
    tunnel: bool
    layer: int
    modes: dict[str, bool]
    maxspeed_kmh: float | None
    width: float
    structure_id: str | None = None
    structure_implied: bool = False    # bridge/tunnel status inferred from OSM layer only
    z: np.ndarray | None = None        # per vertex of ``coords``
    profile: np.ndarray | None = None  # (m, 3) densified deck profile for bridges/tunnels
    tiles: list[str] = field(default_factory=list)

    @property
    def length(self) -> float:
        return float(np.sum(np.linalg.norm(np.diff(self.coords, axis=0), axis=1)))

    @property
    def line(self) -> LineString:
        return LineString(self.coords)

    def allows(self, mode: str | None) -> bool:
        return True if mode is None else bool(self.modes.get(mode, False))


@dataclass
class RoadGraph:
    nodes: dict[str, GraphNode] = field(default_factory=dict)
    edges: dict[str, GraphEdge] = field(default_factory=dict)
    restrictions: list[TurnRestriction] = field(default_factory=list)

    # -- construction --------------------------------------------------------------------
    @staticmethod
    def build(roads: list[Road], restrictions: list[TurnRestriction] | None = None) -> RoadGraph:
        g = RoadGraph()
        usage: dict[str, int] = defaultdict(int)
        positions: dict[str, tuple[float, float]] = {}
        for road in roads:
            coords = list(road.geometry.coords)
            for nid, c in zip(road.node_ids, coords):
                usage[nid] += 1
                positions.setdefault(nid, (c[0], c[1]))
        via_nodes = {r.via_node for r in restrictions or [] if r.via_node}
        for road in sorted(roads, key=lambda r: r.id):
            ids = road.node_ids
            coords = np.asarray(road.geometry.coords)[:, :2]
            split = [0]
            for k in range(1, len(ids) - 1):
                if usage[ids[k]] > 1 or ids[k] in via_nodes or ids[k].startswith("n:bnd:"):
                    split.append(k)
            split.append(len(ids) - 1)
            for e_idx, (a, b) in enumerate(zip(split[:-1], split[1:])):
                if b <= a:
                    continue
                eid = f"e:{road.id.removeprefix('road:')}:{e_idx}"
                edge = GraphEdge(
                    id=eid, road_id=road.id, u=ids[a], v=ids[b], coords=coords[a:b + 1].copy(),
                    vertex_node_ids=ids[a:b + 1], direction=str(road.get("direction", "both")),
                    road_class=road.road_class, name=road.properties.get("name"),
                    bridge=road.is_bridge, tunnel=road.is_tunnel, layer=road.level,
                    modes=dict(road.properties.get("modes") or {m: True for m in MODES}),
                    maxspeed_kmh=road.properties.get("maxspeed_kmh"), width=road.width,
                    structure_implied=road.structure_implied,
                )
                if edge.length < 1e-6:
                    continue
                g.edges[eid] = edge
                for nid in (edge.u, edge.v):
                    if nid not in g.nodes:
                        x, y = positions[nid]
                        g.nodes[nid] = GraphNode(nid, x, y)
                    g.nodes[nid].edges.append(eid)
        for n in g.nodes.values():
            n.kind = _node_kind(n)
        g.restrictions = list(restrictions or [])
        g._resolve_restrictions(roads)
        return g

    def _resolve_restrictions(self, roads: list[Road]) -> None:
        by_way: dict[str, list[str]] = defaultdict(list)
        for e in self.edges.values():
            src = e.road_id.removeprefix("road:osm:").split(":p")[0]
            by_way[src].append(e.id)
        for r in self.restrictions:
            r.enforced = False
            if r.via_node is None:
                # Via-way restrictions are preserved in the data but not enforced by the router.
                r.via_edges = [eid for w in r.via_ways for eid in by_way.get(w, [])]
                frm = [e for e in by_way.get(r.from_way, [])]
                to = [e for e in by_way.get(r.to_way, [])]
                r.from_edge = frm[0] if len(frm) == 1 else None
                r.to_edge = to[0] if len(to) == 1 else None
                continue
            if r.via_node not in self.nodes:
                continue
            frm = [e for e in by_way.get(r.from_way, []) if r.via_node in (self.edges[e].u, self.edges[e].v)]
            to = [e for e in by_way.get(r.to_way, []) if r.via_node in (self.edges[e].u, self.edges[e].v)]
            if len(frm) == 1 and len(to) >= 1:
                r.from_edge = frm[0]
                r.to_edge = to[0] if len(to) == 1 else (to[0] if r.from_way != r.to_way else frm[0])
                r.enforced = True

    # -- queries -------------------------------------------------------------------------
    def arcs(self, mode: str | None = None) -> dict[str, list[tuple[str, str, float]]]:
        """Directed adjacency: node -> [(edge id, next node, cost)]."""
        adj: dict[str, list[tuple[str, str, float]]] = defaultdict(list)
        for e in self.edges.values():
            if not e.allows(mode):
                continue
            # Pedestrians may walk against one-way traffic.
            direction = "both" if mode == "foot" else e.direction
            length = e.length
            if direction in ("both", "forward"):
                adj[e.u].append((e.id, e.v, length))
            if direction in ("both", "backward"):
                adj[e.v].append((e.id, e.u, length))
        for k in adj:
            adj[k].sort()
        return adj

    def _restriction_tables(self) -> tuple[dict[tuple[str, str], set[str]], dict[tuple[str, str], str]]:
        banned: dict[tuple[str, str], set[str]] = defaultdict(set)
        only: dict[tuple[str, str], str] = {}
        for r in self.restrictions:
            if not r.enforced or r.from_edge is None or r.to_edge is None or r.via_node is None:
                continue
            key = (r.from_edge, r.via_node)
            if r.mandatory:
                only[key] = r.to_edge
            else:
                banned[key].add(r.to_edge)
        return banned, only

    def shortest_path(self, source: str, target: str, mode: str | None = "motor_vehicle",
                      restrictions: bool = True) -> tuple[list[str], list[str], float] | None:
        """Edge-based Dijkstra honouring one-way rules and via-node turn restrictions.

        Returns (node ids, edge ids, length in metres) or None when unreachable.
        """
        if source not in self.nodes or target not in self.nodes:
            return None
        if source == target:
            return [source], [], 0.0
        adj = self.arcs(mode)
        banned, only = self._restriction_tables() if restrictions else ({}, {})
        start = (source, "")
        dist: dict[tuple[str, str], float] = {start: 0.0}
        prev: dict[tuple[str, str], tuple[str, str]] = {}
        heap: list[tuple[float, str, str]] = [(0.0, source, "")]
        best_end: tuple[str, str] | None = None
        while heap:
            d, node, inc = heapq.heappop(heap)
            if d > dist.get((node, inc), math.inf):
                continue
            if node == target:
                best_end = (node, inc)
                break
            for eid, nxt, cost in adj.get(node, []):
                if inc:
                    # No immediate U-turn back along the same edge, except at dead ends.
                    if eid == inc and self.nodes[node].degree > 1:
                        continue
                    key = (inc, node)
                    if eid in banned.get(key, ()):
                        continue
                    req = only.get(key)
                    if req is not None and eid != req:
                        continue
                state = (nxt, eid)
                nd = d + cost
                if nd < dist.get(state, math.inf):
                    dist[state] = nd
                    prev[state] = (node, inc)
                    heapq.heappush(heap, (nd, nxt, eid))
        if best_end is None:
            return None
        nodes, edges = [best_end[0]], []
        cur = best_end
        while cur != start:
            edges.append(cur[1])
            cur = prev[cur]
            nodes.append(cur[0])
        nodes.reverse()
        edges.reverse()
        return nodes, edges, dist[best_end]

    def nearest_node(self, x: float, y: float, mode: str | None = None) -> str | None:
        ids = [n.id for n in self.nodes.values() if any(self.edges[e].allows(mode) for e in n.edges)]
        if not ids:
            return None
        pts = np.array([[self.nodes[i].x, self.nodes[i].y] for i in ids])
        _, k = cKDTree(pts).query([x, y])
        return ids[int(k)]

    def to_networkx(self, mode: str | None = None) -> nx.MultiDiGraph:
        g = nx.MultiDiGraph()
        for n in self.nodes.values():
            g.add_node(n.id, x=n.x, y=n.y, z=n.z, kind=n.kind)
        for u, lst in self.arcs(mode).items():
            for eid, v, cost in lst:
                g.add_edge(u, v, key=eid, length=cost)
        return g

    def components(self, mode: str | None = None) -> list[set[str]]:
        """Weakly connected components (by node id), largest first."""
        g = nx.Graph()
        g.add_nodes_from(n for n in self.nodes)
        for e in self.edges.values():
            if e.allows(mode):
                g.add_edge(e.u, e.v)
        comps = [c for c in nx.connected_components(g) if len(c) > 1 or mode is None]
        return sorted(comps, key=lambda c: (-len(c), min(c)))

    def strongly_connected_fraction(self, mode: str = "motor_vehicle") -> float:
        g = nx.DiGraph()
        for u, lst in self.arcs(mode).items():
            for _, v, _ in lst:
                g.add_edge(u, v)
        if g.number_of_nodes() == 0:
            return 1.0
        largest = max(nx.strongly_connected_components(g), key=len)
        return len(largest) / g.number_of_nodes()

    # -- serialization -------------------------------------------------------------------
    def to_dict(self, frame: Any = None, origin: dict[str, Any] | None = None) -> dict[str, Any]:
        from worldgen import SCHEMA_VERSION

        node_ids = sorted(self.nodes)
        lon = lat = None
        if frame is not None and node_ids:
            lon, lat = frame.to_wgs84([self.nodes[i].x for i in node_ids], [self.nodes[i].y for i in node_ids])
        nodes = []
        for k, nid in enumerate(node_ids):
            n = self.nodes[nid]
            rec: dict[str, Any] = {"id": nid, "osm_id": n.osm_id, "kind": n.kind, "x": round(n.x, 3),
                                   "y": round(n.y, 3), "z": round(n.z, 3), "degree": n.degree}
            if lon is not None:
                rec["lon"] = round(float(lon[k]), 8)
                rec["lat"] = round(float(lat[k]), 8)
            nodes.append(rec)
        edges = []
        for eid in sorted(self.edges):
            e = self.edges[eid]
            z = e.z if e.z is not None else np.zeros(len(e.coords))
            pts = e.profile if e.profile is not None else np.column_stack([e.coords, z])
            edges.append({
                "id": e.id, "road_id": e.road_id, "from": e.u, "to": e.v, "direction": e.direction,
                "length_m": round(e.length, 3), "class": e.road_class, "name": e.name,
                "bridge": e.bridge, "tunnel": e.tunnel, "structure_implied": e.structure_implied,
                "layer": e.layer, "structure_id": e.structure_id,
                "modes": e.modes, "maxspeed_kmh": e.maxspeed_kmh, "width_m": round(e.width, 2),
                "geometry": [[round(float(x), 3), round(float(y), 3), round(float(zz), 3)] for x, y, zz in pts],
                "tiles": e.tiles,
            })
        restr = [{
            "id": r.id, "restriction": r.restriction, "kind": "mandatory" if r.mandatory else "prohibitory",
            "from_edge": r.from_edge, "via_node": r.via_node, "to_edge": r.to_edge, "via_edges": r.via_edges,
            "from_way": r.from_way, "to_way": r.to_way, "source_id": r.source_id, "enforced": r.enforced,
        } for r in sorted(self.restrictions, key=lambda r: r.id)]
        return {
            "schema": "worldgen.road_graph", "schema_version": SCHEMA_VERSION,
            "origin": origin, "crs": {"local": frame.proj_string if frame is not None else None},
            "nodes": nodes, "edges": edges, "restrictions": restr,
        }

    @staticmethod
    def from_dict(doc: dict[str, Any]) -> RoadGraph:
        if doc.get("schema") != "worldgen.road_graph":
            raise ValueError("not a worldgen road graph document")
        major = str(doc.get("schema_version", "0")).split(".")[0]
        if major != "1":
            raise ValueError(f"unsupported road graph schema version {doc.get('schema_version')}")
        g = RoadGraph()
        for n in doc["nodes"]:
            g.nodes[n["id"]] = GraphNode(n["id"], n["x"], n["y"], n.get("z", 0.0), n.get("kind", "connector"))
        for e in doc["edges"]:
            geom = np.asarray(e["geometry"], dtype=float)
            edge = GraphEdge(
                id=e["id"], road_id=e["road_id"], u=e["from"], v=e["to"], coords=geom[:, :2],
                vertex_node_ids=[], direction=e["direction"], road_class=e["class"], name=e.get("name"),
                bridge=e["bridge"], tunnel=e["tunnel"], layer=e["layer"], modes=e["modes"],
                maxspeed_kmh=e.get("maxspeed_kmh"), width=e.get("width_m", 6.0),
                structure_id=e.get("structure_id"), structure_implied=e.get("structure_implied", False),
                z=geom[:, 2], tiles=e.get("tiles", []),
            )
            g.edges[edge.id] = edge
            g.nodes[edge.u].edges.append(edge.id)
            g.nodes[edge.v].edges.append(edge.id)
        for r in doc.get("restrictions", []):
            g.restrictions.append(TurnRestriction(
                id=r["id"], restriction=r["restriction"], from_way=r.get("from_way", ""), to_way=r.get("to_way", ""),
                via_node=r.get("via_node"), via_ways=[], source_id=r.get("source_id", ""),
                from_edge=r.get("from_edge"), to_edge=r.get("to_edge"), via_edges=r.get("via_edges", []),
                enforced=r.get("enforced", False),
            ))
        return g


def _node_kind(n: GraphNode) -> str:
    if n.id.startswith("n:bnd:"):
        return "boundary"
    if n.degree == 1:
        return "dead_end"
    if n.degree >= 3:
        return "intersection"
    return "connector"
