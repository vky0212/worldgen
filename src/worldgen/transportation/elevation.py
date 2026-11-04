"""Road elevation profiles and bridge/tunnel vertical alignment.

Ground roads follow the terrain. Bridges and tunnels are grouped into
structures (connected components of bridge or tunnel edges); each structure is
solved on a densified *station graph*:

* anchors: structure nodes that connect to ground roads keep the ground height;
* bounds: bridges must clear what they cross (roads below, water) and the
  terrain; tunnels must stay below what crosses over them and below ground;
* grade: the deck may not rise or fall faster than ``max_grade``.

With ``d(i, j)`` the along-structure distance, the solution for a bridge is::

    U_i = min_a (z_a + g d(a, i))            # highest reachable from anchors
    L_i = max_j (lower_j - g d(j, i))        # lowest height that still meets hard bounds
    P_i = max_j (prefer_j - g d(j, i))       # soft default height where nothing is known
    A_i = inverse-distance anchor interpolation (linear between two anchors)
    z_i = min(U_i, max(A_i, P_i, L_i))

and symmetrically for tunnels. Every component is Lipschitz in ``d`` with
constant ``g`` (or the anchor-to-anchor slope), so decks are continuous and
grade-limited. OSM ``layer`` only orders crossing items relative to each other;
it is never mapped to an absolute height. Where a bound cannot be met within
the grade limit, or the structure has no ground anchor, the structure is
marked uncertain instead of pretending to be authoritative.
"""

from __future__ import annotations

import math
from collections import defaultdict
from dataclasses import dataclass, field

import numpy as np
from scipy.sparse import coo_matrix
from scipy.sparse.csgraph import dijkstra

from worldgen.config import WorldConfig
from worldgen.geometry.shapes import resample_polyline
from worldgen.log import get_logger
from worldgen.terrain.heightfield import Heightfield
from worldgen.topology.crossings import Crossing
from worldgen.topology.graph import GraphEdge, RoadGraph
from worldgen.world.model import Bridge, Road, Structure, Tunnel, Waterbody

log = get_logger("transportation.elevation")


@dataclass
class EdgeProfile:
    edge_id: str
    xy: np.ndarray        # stations (n, 2), u -> v
    s: np.ndarray         # arc length
    z: np.ndarray         # road surface centreline elevation (deck top for bridges)
    terrain: np.ndarray   # original terrain at stations
    kind: str             # ground | bridge | tunnel
    abutment: np.ndarray | None = None   # bool per station: inside an expected abutment zone

    def z_at(self, x: float, y: float) -> float:
        d = np.hypot(self.xy[:, 0] - x, self.xy[:, 1] - y)
        k = int(np.argmin(d))
        return float(self.z[k])


@dataclass
class ElevationResult:
    profiles: dict[str, EdgeProfile] = field(default_factory=dict)
    bridges: list[Bridge] = field(default_factory=list)
    tunnels: list[Tunnel] = field(default_factory=list)
    issues: list[dict] = field(default_factory=list)


def _smooth(z: np.ndarray, passes: int = 2) -> np.ndarray:
    z = z.copy()
    for _ in range(passes):
        if len(z) > 2:
            z[1:-1] = 0.25 * z[:-2] + 0.5 * z[1:-1] + 0.25 * z[2:]
    return z


def limit_grade(z: np.ndarray, s: np.ndarray, g: float) -> np.ndarray:
    """Make a profile with fixed ends satisfy ``|dz/ds| <= g`` with minimal cut and fill.

    The cut envelope (largest g-Lipschitz profile below ``z``) and the fill envelope
    (smallest above) are averaged, then clamped into the band reachable from both ends.
    Min/max/averages of g-Lipschitz functions stay g-Lipschitz, so the result honours the
    limit everywhere. Where the terrain is steeper than a road can be, the road runs in a
    cutting or on an embankment (the terrain is later conformed to it). If the ends
    themselves are further apart than ``g`` allows, the end-to-end slope is used.
    """
    if len(z) < 3:
        return z
    total = float(s[-1])
    z0, z1 = float(z[0]), float(z[-1])
    g = max(g, abs(z1 - z0) / max(total, 1e-9) * (1 + 1e-9))
    ds = np.diff(s)
    cut = z.astype(float).copy()
    fill = z.astype(float).copy()
    for i in range(1, len(z)):
        cut[i] = min(cut[i], cut[i - 1] + g * ds[i - 1])
        fill[i] = max(fill[i], fill[i - 1] - g * ds[i - 1])
    for i in range(len(z) - 2, -1, -1):
        cut[i] = min(cut[i], cut[i + 1] + g * ds[i])
        fill[i] = max(fill[i], fill[i + 1] - g * ds[i])
    upper = np.minimum(z0 + g * s, z1 + g * (total - s))
    lower = np.maximum(z0 - g * s, z1 - g * (total - s))
    return np.clip(0.5 * (cut + fill), lower, upper)


def group_structures(graph: RoadGraph, roads: dict[str, Road]) -> tuple[list[Bridge], list[Tunnel]]:
    parent: dict[str, str] = {}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    def union(a: str, b: str) -> None:
        ra, rb = find(a), find(b)
        if ra != rb:
            parent[max(ra, rb)] = min(ra, rb)

    def kind(e: GraphEdge) -> str | None:
        return "bridge" if e.bridge else ("tunnel" if e.tunnel else None)

    for e in graph.edges.values():
        if kind(e):
            parent[e.id] = e.id
    for n in graph.nodes.values():
        by_kind: dict[str, list[str]] = defaultdict(list)
        for eid in n.edges:
            k = kind(graph.edges[eid])
            if k:
                by_kind[k].append(eid)
        for lst in by_kind.values():
            for other in lst[1:]:
                union(lst[0], other)
    comps: dict[str, list[str]] = defaultdict(list)
    for eid in parent:
        comps[find(eid)].append(eid)
    bridges: list[Bridge] = []
    tunnels: list[Tunnel] = []
    for _, eids in sorted(comps.items()):
        eids.sort()
        k = kind(graph.edges[eids[0]])
        road_ids = sorted({graph.edges[e].road_id for e in eids})
        ways = sorted({roads[r].source_id for r in road_ids}, key=_way_sort_key)
        sid = f"{k}:{roads[road_ids[0]].source}:{ways[0]}"
        nodes = {n for e in eids for n in (graph.edges[e].u, graph.edges[e].v)}
        anchors = sorted(
            n for n in nodes
            if any(kind(graph.edges[x]) != k for x in graph.nodes[n].edges)
        )
        from shapely.geometry import MultiLineString

        geom = MultiLineString([graph.edges[e].coords for e in eids])
        props: dict = {}
        names = sorted({str(roads[r].properties.get("bridge_name") or roads[r].properties.get("name"))
                        for r in road_ids if roads[r].properties.get("bridge_name") or roads[r].properties.get("name")})
        if names:
            props["name"] = names[0]
        btypes = sorted({str(roads[r].properties.get(f"{k}_type")) for r in road_ids if roads[r].properties.get(f"{k}_type")})
        if btypes:
            props[f"{k}_type"] = btypes[0]
        if k == "bridge":
            structs = sorted({str(roads[r].properties["bridge_structure"]) for r in road_ids
                              if roads[r].properties.get("bridge_structure")})
            if structs:
                props["structure"] = structs[0]
        props["width"] = max(graph.edges[e].width for e in eids)
        props["length"] = round(sum(graph.edges[e].length for e in eids), 2)
        props["road_classes"] = sorted({graph.edges[e].road_class for e in eids})
        cls = Bridge if k == "bridge" else Tunnel
        st: Structure = cls(id=sid, geometry=geom, properties=props, source=roads[road_ids[0]].source,
                            source_id=",".join(ways), confidence=0.9, road_ids=road_ids, edge_ids=eids,
                            anchor_node_ids=anchors, level=max(graph.edges[e].layer for e in eids))
        if k == "bridge" and "structure" not in props:
            st.mark_missing("structure")
        if all(roads[r].structure_implied for r in road_ids):
            st.infer("structure_type", f"implied {k}", f"osm_layer_{'above' if k == 'bridge' else 'below'}_ground", 0.4)
            st.confidence = 0.5
        for e in eids:
            graph.edges[e].structure_id = sid
        (bridges if k == "bridge" else tunnels).append(st)  # type: ignore[arg-type]
    return bridges, tunnels


def _way_sort_key(s: str) -> tuple[str, int]:
    t, _, n = s.partition("/")
    try:
        return (t, int(n))
    except ValueError:
        return (t, 0)


def compute_elevations(
    graph: RoadGraph,
    roads: dict[str, Road],
    terrain: Heightfield,
    crossings: list[Crossing],
    water_crossings: list[Crossing],
    water: dict[str, Waterbody],
    water_levels: dict[str, float],
    cfg: WorldConfig,
) -> ElevationResult:
    res = ElevationResult()
    spacing = cfg.roads.sample_spacing
    bridges, tunnels = group_structures(graph, roads)
    res.bridges, res.tunnels = bridges, tunnels

    # Ground node heights.
    for n in graph.nodes.values():
        n.z = float(terrain.sample(n.x, n.y))

    # Stations for every edge.
    stations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]] = {}
    for eid in sorted(graph.edges):
        e = graph.edges[eid]
        xy, s = resample_polyline(e.line, spacing)
        stations[eid] = (xy, s, terrain.sample(xy[:, 0], xy[:, 1]))

    # Ground profiles.
    for eid in sorted(graph.edges):
        e = graph.edges[eid]
        if e.bridge or e.tunnel:
            continue
        xy, s, t = stations[eid]
        z = _smooth(t)
        z[0] = graph.nodes[e.u].z
        z[-1] = graph.nodes[e.v].z
        if e.road_class not in cfg.roads.grade_exempt_classes:
            z = limit_grade(z, s, cfg.roads.max_grade)
        res.profiles[eid] = EdgeProfile(eid, xy, s, z, t, "ground")

    by_upper: dict[str, list[Crossing]] = defaultdict(list)
    by_lower: dict[str, list[Crossing]] = defaultdict(list)
    for c in crossings:
        if c.kind == "grade_separated":
            by_upper[c.upper].append(c)
            by_lower[c.lower].append(c)
    water_by_edge: dict[str, list[Crossing]] = defaultdict(list)
    for c in water_crossings:
        water_by_edge[c.upper].append(c)

    for st in sorted(bridges, key=lambda b: (b.level, b.id)):
        _solve_structure(st, "bridge", graph, roads, stations, res, terrain, by_upper, water_by_edge,
                         water, water_levels, cfg)
    for st in sorted(tunnels, key=lambda t: (-t.level, t.id)):
        _solve_structure(st, "tunnel", graph, roads, stations, res, terrain, by_lower, water_by_edge,
                         water, water_levels, cfg)

    # Edge vertex elevations for the serialized graph and road entities. Structures also keep
    # their densified deck profile so the exported topology shows the vertical alignment.
    by_road: dict[str, list[GraphEdge]] = defaultdict(list)
    for e in graph.edges.values():
        p = res.profiles[e.id]
        e.z = np.array([p.z_at(x, y) for x, y in e.coords])
        e.profile = np.column_stack([p.xy, p.z]) if p.kind != "ground" else None
        by_road[e.road_id].append(e)
    for road in roads.values():
        coords = np.asarray(road.geometry.coords)[:, :2]
        z = np.zeros(len(coords))
        edges = by_road.get(road.id, [])
        for k, (x, y) in enumerate(coords):
            best = math.inf
            for e in edges:
                d = np.hypot(e.coords[:, 0] - x, e.coords[:, 1] - y)
                j = int(np.argmin(d))
                if d[j] < best:
                    best = float(d[j])
                    z[k] = e.z[j]  # type: ignore[index]
        road.elevation = z
        if road.is_bridge or road.is_tunnel:
            road.infer("deck_elevation", [round(float(z.min()), 2), round(float(z.max()), 2)],
                       "structure_alignment_solver", 0.4)
    return res


def _solve_structure(
    st: Structure, kind: str, graph: RoadGraph, roads: dict[str, Road],
    stations: dict[str, tuple[np.ndarray, np.ndarray, np.ndarray]], res: ElevationResult,
    terrain: Heightfield, crossings_by_edge: dict[str, list[Crossing]],
    water_by_edge: dict[str, list[Crossing]], water: dict[str, Waterbody],
    water_levels: dict[str, float], cfg: WorldConfig,
) -> None:
    bc, tc = cfg.bridges, cfg.tunnels
    g = bc.max_grade if kind == "bridge" else tc.max_grade
    sign = 1.0 if kind == "bridge" else -1.0   # tunnels are solved as mirrored bridges

    # Station graph: shared graph nodes + interior stations.
    index: dict[object, int] = {}
    pos: list[tuple[float, float]] = []
    terr: list[float] = []
    rows: list[int] = []
    cols: list[int] = []
    wts: list[float] = []
    edge_station_ids: dict[str, list[int]] = {}

    def sid(key: object, x: float, y: float, t: float) -> int:
        if key not in index:
            index[key] = len(pos)
            pos.append((x, y))
            terr.append(t)
        return index[key]

    for eid in st.edge_ids:
        e = graph.edges[eid]
        xy, s, t = stations[eid]
        ids = []
        for k in range(len(xy)):
            key: object = e.u if k == 0 else (e.v if k == len(xy) - 1 else (eid, k))
            ids.append(sid(key, xy[k, 0], xy[k, 1], t[k]))
        edge_station_ids[eid] = ids
        for k in range(len(ids) - 1):
            d = float(s[k + 1] - s[k])
            rows += [ids[k], ids[k + 1]]
            cols += [ids[k + 1], ids[k]]
            wts += [max(d, 1e-6)] * 2
    n = len(pos)
    P = np.array(pos)
    T = np.array(terr)
    # Parallel edges between the same stations must not add up (coo->csr sums duplicates).
    links: dict[tuple[int, int], float] = {}
    for a_, b_, w_ in zip(rows, cols, wts):
        links[(a_, b_)] = min(w_, links.get((a_, b_), math.inf))
    rows = [k[0] for k in links]
    cols = [k[1] for k in links]
    wts = list(links.values())
    G = coo_matrix((wts, (rows, cols)), shape=(n, n)).tocsr()

    # Anchors.
    anchors: dict[int, float] = {}
    uncertain_reasons: list[str] = []
    for nid in st.anchor_node_ids:
        if (nid) in index:
            anchors[index[nid]] = graph.nodes[nid].z
    if not anchors:
        uncertain_reasons.append("no ground anchor (structure not connected to ground roads in the data)")
        # Use structure ends (degree-1 within the structure) at terrain height.
        deg = np.diff(G.indptr)
        ends = [i for i in range(n) if deg[i] <= 1] or [0]
        for i in ends:
            anchors[i] = float(T[i])
    elif len(anchors) == 1:
        uncertain_reasons.append("only one ground anchor")

    # Bounds, expressed for the mirrored problem (bridges as is, tunnels negated). ``lower`` are
    # hard requirements (terrain and crossing clearance); ``prefer`` is the soft default height
    # used where nothing is known, which never makes a structure infeasible.
    total_len = float(st.properties.get("length", 0.0))
    if kind == "bridge":
        base_h = bc.default_deck_height if total_len >= bc.short_bridge_length else bc.short_bridge_height
        lower_terrain = T + bc.deck_thickness + 0.3
        prefer = T + max(base_h, bc.deck_thickness + 0.3)
    else:
        lower_terrain = -(T - (tc.clearance_height + tc.wall_thickness))
        prefer = -(T - (tc.clearance_height + tc.wall_thickness + tc.min_cover))
    # Clearance demanded by crossed roads and water (-inf where nothing is crossed).
    lower = np.full(n, -np.inf)
    constrained = False
    clearances: list[float] = []
    observed_clearance: list[float] = []
    for eid in st.edge_ids:
        ids = edge_station_ids[eid]
        sp = P[ids]
        for c in crossings_by_edge.get(eid, []):
            other_id = c.lower if kind == "bridge" else c.upper
            other = graph.edges[other_id]
            other_prof = res.profiles.get(other_id)
            oz = other_prof.z_at(c.x, c.y) if other_prof is not None else float(terrain.sample(c.x, c.y))
            gap = max(c.layer_gap, 1)
            if kind == "bridge":
                mh = roads[other.road_id].properties.get("maxheight")
                clear = float(mh) if mh else bc.min_clearance_road
                if mh:
                    observed_clearance.append(float(mh))
                need = oz + max(clear + bc.deck_thickness, bc.layer_spacing * gap)
            else:
                need = -(oz - max(tc.clearance_height + tc.wall_thickness + 0.5, bc.layer_spacing * gap))
            reach = other.width / 2.0 + 2.0
            near = np.hypot(sp[:, 0] - c.x, sp[:, 1] - c.y) <= reach
            if not near.any():
                near = np.zeros(len(ids), dtype=bool)
                near[int(np.argmin(np.hypot(sp[:, 0] - c.x, sp[:, 1] - c.y)))] = True
            for i in np.asarray(ids)[near]:
                lower[i] = max(lower[i], need)
            constrained = True
            clearances.append(need)
        if kind == "bridge":
            for c in water_by_edge.get(eid, []):
                w = water.get(c.lower)
                lvl = water_levels.get(c.lower)
                if w is None or lvl is None:
                    continue
                need = lvl + bc.min_clearance_water + bc.deck_thickness
                if w.is_area:
                    inside = np.array([w.geometry.covers(_pt(x, y)) for x, y in sp])
                else:
                    width = float(w.get("width", 3.0))
                    inside = np.array([w.geometry.distance(_pt(x, y)) <= width / 2.0 + 1.0 for x, y in sp])
                for i in np.asarray(ids)[inside]:
                    lower[i] = max(lower[i], need)
                if inside.any():
                    constrained = True
                if w.id not in st.crossings:
                    st.crossings.append(w.id)
                if st.id not in w.crossings:
                    w.crossings.append(st.id)

    anchor_ids = sorted(anchors)
    za = np.array([sign * anchors[i] for i in anchor_ids])
    D = dijkstra(G, directed=False, indices=anchor_ids)          # (A, n)
    D = np.atleast_2d(D)
    # Anchors taken from the terrain can be inconsistent with the grade limit (e.g. a junction
    # inside a tunnel with an untagged road on the hill above). Anchors must stay exact, so the
    # structure then uses the steepest grade its anchors imply, and the conflict is reported.
    g_eff = g
    if len(anchor_ids) >= 2:
        Daa = D[:, anchor_ids]
        with np.errstate(divide="ignore", invalid="ignore"):
            implied = np.abs(za[:, None] - za[None, :]) / Daa
        implied = implied[np.isfinite(implied)]
        if implied.size and implied.max() > g:
            g_eff = float(implied.max()) * (1.0 + 1e-6)
            uncertain_reasons.append(f"ground anchors imply a {g_eff:.0%} grade (limit {g:.0%})")
            k = anchor_ids[int(np.argmax(np.abs(za - np.median(za))))]
            res.issues.append({
                "check": "bridge_elevation_continuity" if kind == "bridge" else "tunnel_cover",
                "severity": "warning",
                "message": f"{st.id}: connecting roads force a {g_eff:.0%} grade (limit {g:.0%}); "
                           "likely an untagged connection at a different level",
                "entity_ids": [st.id], "x": float(P[k, 0]), "y": float(P[k, 1]), "z": float(sign * za[0]),
            })
    g = g_eff
    st.infer("design_grade", round(g, 4), "max_grade_or_anchor_implied", 0.5)
    with np.errstate(invalid="ignore"):
        U = np.min(za[:, None] + g * D, axis=0)

    def envelope(bound: np.ndarray) -> np.ndarray:
        """max_j (bound_j - g d(j, i)) via one Dijkstra from a virtual super-source."""
        finite = np.isfinite(bound)
        if not finite.any():
            return np.full(n, -np.inf)
        shift = float(np.max(bound[finite])) / g
        Gs = coo_matrix(
            (list(wts) + list(shift - bound[finite] / g + 1e-9),
             (rows + [n] * int(finite.sum()), cols + list(np.nonzero(finite)[0]))),
            shape=(n + 1, n + 1),
        ).tocsr()
        ds = dijkstra(Gs, directed=True, indices=n)[:n]
        return -g * (ds - shift)

    L_cross = envelope(lower)
    L_terr = envelope(lower_terrain)
    L = np.maximum(L_cross, L_terr)
    Pf = envelope(np.maximum(prefer, np.maximum(lower, lower_terrain)))
    # Anchor interpolation (inverse graph distance; linear for two anchors on a chain).
    with np.errstate(divide="ignore", invalid="ignore"):
        Wt = 1.0 / np.maximum(D, 1e-9)
        A = (Wt * za[:, None]).sum(axis=0) / Wt.sum(axis=0)
    for k, i in enumerate(anchor_ids):
        A[i] = za[k]
    unreachable = ~np.isfinite(U)
    if unreachable.any():
        U[unreachable] = np.max(za) + g * 1e6
        A[unreachable] = np.max(za)
    z = np.minimum(U, np.maximum(np.maximum(A, Pf), L))
    # Crossing clearance that cannot be met within the grade limit is a real data problem.
    infeasible = L_cross > U + 0.05
    # Abutment / portal zone: the deck is still ramping from an anchor and cannot clear the
    # terrain yet (expected at every bridge end and tunnel portal; not a geometry error).
    near_anchor = L_terr > U + 0.05
    z = sign * z

    for eid in st.edge_ids:
        ids = edge_station_ids[eid]
        xy, s, t = stations[eid]
        res.profiles[eid] = EdgeProfile(eid, xy, s, z[ids], t, kind, abutment=near_anchor[ids])
    for nid in {graph.edges[e].u for e in st.edge_ids} | {graph.edges[e].v for e in st.edge_ids}:
        if nid not in st.anchor_node_ids:
            graph.nodes[nid].z = float(z[index[nid]])

    # Provenance and uncertainty.
    hi = z
    above = (z - T) if kind == "bridge" else (T - z)
    st.infer("deck_elevation_range", [round(float(hi.min()), 2), round(float(hi.max()), 2)],
             "structure_alignment_solver", 0.4)
    st.infer("max_height_above_terrain" if kind == "bridge" else "max_depth_below_terrain",
             round(float(above.max()), 2), "structure_alignment_solver", 0.35)
    if observed_clearance:
        st.properties["clearance"] = min(observed_clearance)
    elif kind == "bridge":
        st.mark_missing("clearance")
        if constrained:
            st.infer("clearance", bc.min_clearance_road, "config:min_clearance_road", 0.3)
    if infeasible.any():
        uncertain_reasons.append("clearance bound not reachable within max grade")
        k = int(np.argmax(np.where(infeasible, L_cross - U, -np.inf)))
        res.issues.append({
            "check": "bridge_clearance" if kind == "bridge" else "tunnel_cover",
            "severity": "warning",
            "message": f"{st.id}: required {'clearance' if kind == 'bridge' else 'depth'} not reachable "
                       f"within max grade {g:.0%} (short of {float(L_cross[k] - U[k]):.1f} m)",
            "entity_ids": [st.id], "x": float(P[k, 0]), "y": float(P[k, 1]), "z": float(z[k]),
        })
    if not constrained:
        uncertain_reasons.append(f"no crossing constraints; default {'deck height' if kind == 'bridge' else 'depth'} used")
    confidence = 0.65 if constrained and len(anchors) >= 2 else 0.35
    if any("anchor" in r or "reachable" in r for r in uncertain_reasons):
        confidence = min(confidence, 0.2)
    st.infer("elevation_confidence", confidence, "structure_alignment_solver", confidence)
    st.uncertain = confidence < 0.5
    if uncertain_reasons:
        st.infer("uncertainty_reasons", uncertain_reasons, "structure_alignment_solver", 1.0)


def _pt(x: float, y: float):  # noqa: ANN202
    from shapely.geometry import Point

    return Point(x, y)
