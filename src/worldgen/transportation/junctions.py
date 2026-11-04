"""Junction geometry: where road ribbons meet at graph nodes.

Two cases, both producing watertight, non-overlapping road surfaces:

* **Mitred joints** – two road ends of similar width that continue each other
  (deflection up to ``MAX_MITRE_DEFLECTION``). Both ribbons end on the same
  mitre line through the node, so they share their end vertices.
* **Junction polygons** – three or more ends (or two ends that cannot be
  mitred). The ends are sorted by angle; the edge line of each road is
  intersected with the facing edge line of its neighbour, giving one corner
  point per gap (a bevel when the lines diverge, a chamfer when a very acute
  angle would push the corner too far). Every road is trimmed back to its
  furthest corner, and the polygon is ``cut(right) → cut(left) → corner → …``
  around the node, so its cut vertices are exactly the ribbon end vertices.

Elevations: cut vertices take the ribbon's elevation, corners the node's.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry import Polygon

from worldgen.config import WorldConfig
from worldgen.geometry.mesh import Mesh, empty_mesh
from worldgen.geometry.shapes import polygons_of, triangulate_polygon
from worldgen.topology.graph import RoadGraph
from worldgen.transportation.elevation import EdgeProfile

MAJOR = {"motorway", "trunk", "primary", "secondary", "motorway_link", "trunk_link", "primary_link",
         "secondary_link"}
PATHS = {"footway", "path", "cycleway", "steps", "pedestrian", "bridleway", "corridor", "track"}
MAX_MITRE_DEFLECTION = math.radians(75.0)
MITRE_WIDTH_TOLERANCE = 0.15
PATH_LIFT = 0.03


def surface_offset(cfg: WorldConfig, kind: str, road_class: str) -> float:
    """Height of a road surface above its profile (paths sit just above roads they overlap)."""
    base = cfg.roads.surface_offset if kind == "ground" else 0.08
    return base + (PATH_LIFT if road_class in PATHS else 0.0)


def _rot90(v: np.ndarray) -> np.ndarray:
    return np.array([-v[1], v[0]])


def _cross(a: np.ndarray, b: np.ndarray) -> float:
    return float(a[0] * b[1] - a[1] * b[0])


@dataclass
class _End:
    edge_id: str
    end: int                 # 0: the edge starts at the node, 1: it ends there
    d: np.ndarray            # outward unit direction from the node
    hw: float
    length: float
    kind: str
    road_class: str


@dataclass
class Junction:
    node_id: str
    polygon: Polygon         # 2D outline (local metres)
    mesh: Mesh               # surface with per-vertex elevation
    degree: int
    feature_id: str          # entity the surface belongs to (intersection or the widest road)
    major: bool


@dataclass
class JunctionLayout:
    trims: dict[tuple[str, int], float] = field(default_factory=dict)
    mitres: dict[tuple[str, int], np.ndarray] = field(default_factory=dict)   # left offset at that end
    junctions: dict[str, Junction] = field(default_factory=dict)

    def trim(self, edge_id: str) -> tuple[float, float]:
        return self.trims.get((edge_id, 0), 0.0), self.trims.get((edge_id, 1), 0.0)


def trim_polyline(xy: np.ndarray, s: np.ndarray, z: np.ndarray, t0: float, t1: float
                  ) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Cut a station polyline to the arc-length interval [t0, total - t1]."""
    total = float(s[-1])
    a, b = t0, total - t1
    if b - a < 0.25:
        return None
    keep = (s > a + 1e-9) & (s < b - 1e-9)
    pa = np.array([np.interp(a, s, xy[:, 0]), np.interp(a, s, xy[:, 1])])
    pb = np.array([np.interp(b, s, xy[:, 0]), np.interp(b, s, xy[:, 1])])
    return (np.vstack([pa, xy[keep], pb]), np.concatenate([[a], s[keep], [b]]),
            np.concatenate([[np.interp(a, s, z)], z[keep], [np.interp(b, s, z)]]))


def _outward_direction(p: EdgeProfile, end: int, probe: float) -> np.ndarray:
    xy = p.xy if end == 0 else p.xy[::-1]
    s = p.s if end == 0 else float(p.s[-1]) - p.s[::-1]
    probe = min(probe, float(s[-1]))
    q = np.array([np.interp(probe, s, xy[:, 0]), np.interp(probe, s, xy[:, 1])])
    d = q - xy[0]
    n = float(np.hypot(*d))
    if n < 1e-9:
        d = xy[-1] - xy[0]
        n = float(np.hypot(*d)) or 1.0
    return d / n


def compute_junctions(graph: RoadGraph, profiles: dict[str, EdgeProfile], cfg: WorldConfig) -> JunctionLayout:
    layout = JunctionLayout()
    excluded = set(cfg.roads.mesh_exclude_classes)
    pending: list[tuple[str, list[_End], dict[int, list[np.ndarray]]]] = []
    for nid in sorted(graph.nodes):
        n = graph.nodes[nid]
        eids = sorted({e for e in n.edges if graph.edges[e].road_class not in excluded})
        if not eids:
            continue
        w_max = max(graph.edges[e].width for e in eids)
        ends: list[_End] = []
        for eid in eids:
            e = graph.edges[eid]
            p = profiles[eid]
            probe = min(0.45 * float(p.s[-1]), max(2.0, 1.5 * w_max))
            for which in (0, 1):
                if (e.u if which == 0 else e.v) == nid:
                    ends.append(_End(eid, which, _outward_direction(p, which, probe), e.width / 2.0,
                                     float(p.s[-1]), p.kind, e.road_class))
        if len(ends) < 2:
            continue
        if len(ends) == 2 and _mitrable(ends):
            _mitre(layout, ends)
            continue
        if any(x.kind != "ground" for x in ends):
            continue
        corners = _corners(layout, ends, w_max)
        pending.append((nid, ends, corners))
    # Polygons need the final trims of both ends of every edge (short edges), so build them last.
    for nid, ends, corners in pending:
        j = _junction_polygon(graph, profiles, cfg, layout, nid, ends, corners)
        if j is not None:
            layout.junctions[nid] = j
    return layout


def _mitrable(ends: list[_End]) -> bool:
    a, b = ends
    if a.kind != b.kind or (a.edge_id == b.edge_id and a.length < 1.0):
        return False
    if abs(a.hw - b.hw) > MITRE_WIDTH_TOLERANCE * max(a.hw, b.hw):
        return False
    deflection = math.pi - math.acos(float(np.clip(np.dot(a.d, b.d), -1.0, 1.0)))
    return deflection <= MAX_MITRE_DEFLECTION


def _mitre(layout: JunctionLayout, ends: list[_End]) -> None:
    a, b = ends
    hw = (a.hw + b.hw) / 2.0
    s = a.d + b.d
    if float(np.hypot(*s)) < 1e-6:                    # straight continuation
        v = _rot90(b.d) * hw
    else:
        bis = s / float(np.hypot(*s))
        half = math.acos(float(np.clip(np.dot(a.d, b.d), -1.0, 1.0))) / 2.0
        v = bis * hw / max(math.sin(half), 1e-3)
    for x in ends:
        travel_left = _rot90(x.d) if x.end == 0 else -_rot90(x.d)
        layout.mitres[(x.edge_id, x.end)] = v if float(np.dot(v, travel_left)) > 0 else -v


def _corners(layout: JunctionLayout, ends: list[_End], w_max: float) -> dict[int, list[np.ndarray]]:
    """Corner points (relative to the node) in each gap; records each end's trim."""
    ends.sort(key=lambda x: math.atan2(x.d[1], x.d[0]))
    k = len(ends)
    maxt = [min(0.45 * x.length, 3.0 * w_max + 2.0) for x in ends]
    t_left = [0.0] * k
    t_right = [0.0] * k
    corners: dict[int, list[np.ndarray]] = {}
    for i in range(k):
        j = (i + 1) % k
        ei, ej = ends[i], ends[j]
        A = _rot90(ei.d) * ei.hw          # left edge line of i (faces the gap towards j)
        B = -_rot90(ej.d) * ej.hw         # right edge line of j
        cr = _cross(ei.d, ej.d)
        t = s = 0.0
        if abs(cr) > 0.05:
            diff = B - A
            t = _cross(diff, ej.d) / cr
            s = _cross(diff, ei.d) / cr
        if abs(cr) <= 0.05 or t < 0 or s < 0:
            pts = [A, B]                  # lines diverge (gap >= 180 degrees): bevel at the node
            t = s = 0.0
        elif t > maxt[i] or s > maxt[j]:
            t, s = min(t, maxt[i]), min(s, maxt[j])
            pts = [A + ei.d * t, B + ej.d * s]   # very acute: chamfer
        else:
            pts = [A + ei.d * t]
        t_left[i] = t
        t_right[j] = s
        corners[i] = pts
    for i, x in enumerate(ends):
        trim = max(t_left[i], t_right[i])
        if trim > 1e-6:
            layout.trims[(x.edge_id, x.end)] = trim
    return corners


def _junction_polygon(graph: RoadGraph, profiles: dict[str, EdgeProfile], cfg: WorldConfig,
                      layout: JunctionLayout, nid: str, ends: list[_End],
                      corners: dict[int, list[np.ndarray]]) -> Junction | None:
    n = graph.nodes[nid]
    centre = np.array([n.x, n.y])
    ring: list[tuple[float, float, float]] = []
    corner_z = n.z + min(surface_offset(cfg, "ground", x.road_class) for x in ends)
    for i, x in enumerate(ends):
        p = profiles[x.edge_id]
        cut = trim_polyline(p.xy, p.s, p.z, *layout.trim(x.edge_id))
        off = surface_offset(cfg, p.kind, x.road_class)
        if cut is not None:
            xy, _, z = cut
            if x.end == 0:
                p0, seg, z0 = xy[0], xy[1] - xy[0], z[0]
            else:
                p0, seg, z0 = xy[-1], xy[-1] - xy[-2], z[-1]
            travel_left = _rot90(seg / max(float(np.hypot(*seg)), 1e-12))
            out_left = travel_left if x.end == 0 else -travel_left
            left_pt, right_pt = p0 + out_left * x.hw, p0 - out_left * x.hw
            zc = float(z0) + off
        else:
            t = layout.trims.get((x.edge_id, x.end), 0.0)
            base = centre + x.d * t
            left_pt, right_pt = base + _rot90(x.d) * x.hw, base - _rot90(x.d) * x.hw
            zc = corner_z
        ring.append((right_pt[0], right_pt[1], zc))
        ring.append((left_pt[0], left_pt[1], zc))
        for c in corners[i]:
            q = centre + c
            ring.append((q[0], q[1], corner_z))
    pts = np.array(ring)
    keep = [0]
    for k in range(1, len(pts)):
        if np.hypot(*(pts[k, :2] - pts[keep[-1], :2])) > 1e-3:
            keep.append(k)
    if len(keep) > 1 and np.hypot(*(pts[keep[-1], :2] - pts[keep[0], :2])) <= 1e-3:
        keep.pop()
    pts = pts[keep]
    if len(pts) < 3:
        return None
    poly = Polygon(pts[:, :2])
    if not poly.is_valid:
        # Corners further out than short edges can be trimmed (very wide roads split into short
        # segments) make the ring self-intersect. Fall back to the star-shaped polygon of the same
        # points ordered by angle around the node: always simple and gap-free; it may overlap the
        # ribbons slightly, so it is lifted 1 cm to avoid z-fighting.
        ang = np.arctan2(pts[:, 1] - centre[1], pts[:, 0] - centre[0])
        pts = pts[np.argsort(ang, kind="stable")]
        pts[:, 2] += 0.01
        poly = Polygon(pts[:, :2])
        if not poly.is_valid:
            poly = shapely.convex_hull(shapely.MultiPoint(pts[:, :2]))
            if poly.geom_type != "Polygon":
                return None
    if poly.area < 0.05:
        return None
    v2, tri = triangulate_polygon(poly)
    if len(tri) == 0:
        mesh = empty_mesh()
    else:
        _, idx = cKDTree(pts[:, :2]).query(v2)
        mesh = Mesh(np.column_stack([v2, pts[idx, 2]]), tri, uvs=v2 / 8.0)
        mesh.ensure_normals()
    degree = len(ends)
    widest = max(ends, key=lambda x: (x.hw, x.edge_id))
    feature = (f"intersection:{nid.removeprefix('n:')}" if degree >= 3
               else graph.edges[widest.edge_id].road_id)
    return Junction(nid, poly, mesh, degree, feature, any(x.road_class in MAJOR for x in ends))
