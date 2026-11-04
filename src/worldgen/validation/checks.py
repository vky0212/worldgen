"""Automated validation of a compiled world.

Checks are grouped as in plan §6 (geographic, topological, geometric,
reproducibility) plus the vegetation checks of the extension. Every issue has a
severity, a message, the involved entity ids and, where meaningful, a location
so the viewer can show it.
"""

from __future__ import annotations

import dataclasses
from collections.abc import Callable
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
import trimesh
from scipy.spatial import cKDTree

from worldgen import SCHEMA_VERSION
from worldgen.buildings.buildings import building_mesh
from worldgen.geometry.shapes import polygons_of
from worldgen.tiling.grid import TileGrid, parse_tile_id, tile_id
from worldgen.vegetation.classify import CAT, TREE_CATEGORIES
from worldgen.world.world import CanonicalWorld


@dataclass
class Issue:
    check: str
    severity: str
    message: str
    entity_ids: list[str] = field(default_factory=list)
    x: float | None = None
    y: float | None = None
    z: float | None = None


@dataclass
class Report:
    issues: list[Issue] = field(default_factory=list)
    checks: dict[str, dict[str, Any]] = field(default_factory=dict)
    max_per_check: int = 200

    def add(self, issue: Issue) -> None:
        self.issues.append(issue)

    def run(self, name: str, fn: Callable[[], list[Issue] | None], details: dict[str, Any] | None = None) -> None:
        try:
            found = fn() or []
        except Exception as exc:  # a crashing check is itself a validation error
            found = [Issue(name, "error", f"check crashed: {type(exc).__name__}: {exc}")]
        shown = found[: self.max_per_check]
        self.issues.extend(shown)
        sev = {i.severity for i in found}
        status = "fail" if "error" in sev else ("warn" if "warning" in sev else "pass")
        entry: dict[str, Any] = {"status": status, "issues": len(found)}
        if len(found) > len(shown):
            entry["truncated"] = len(found) - len(shown)
        if details:
            entry.update(details)
        self.checks[name] = entry

    def to_dict(self, world: CanonicalWorld) -> dict[str, Any]:
        frame = world.region.frame
        grid = TileGrid(world.config.tiling.tile_size)
        out = []
        counters: dict[str, int] = {}
        for i in self.issues:
            k = counters.get(i.check, 0) + 1
            counters[i.check] = k
            loc = None
            tile = None
            if i.x is not None and i.y is not None:
                lon, lat = frame.to_wgs84(i.x, i.y)
                z = i.z if i.z is not None else (float(world.terrain.sample(i.x, i.y)) if world.terrain else 0.0)
                loc = {"x": round(i.x, 2), "y": round(i.y, 2), "z": round(z, 2),
                       "lon": round(float(lon), 8), "lat": round(float(lat), 8)}
                tile = grid.tile_of(i.x, i.y)
            out.append({"id": f"{i.check}:{k:04d}", "check": i.check, "severity": i.severity,
                        "message": i.message, "entity_ids": i.entity_ids, "location": loc, "tile": tile})
        summary = {s: sum(1 for i in self.issues if i.severity == s) for s in ("error", "warning", "info")}
        return {"schema": "worldgen.diagnostics", "schema_version": SCHEMA_VERSION,
                "summary": {"errors": summary["error"], "warnings": summary["warning"], "info": summary["info"],
                            "checks": self.checks},
                "issues": out}


def validate_world(world: CanonicalWorld) -> Report:
    cfg = world.config
    rep = Report(max_per_check=cfg.validation.max_issues_per_check)
    for raw in world.stage_issues:
        rep.add(Issue(raw["check"], raw["severity"], raw["message"], raw.get("entity_ids", []),
                      raw.get("x"), raw.get("y"), raw.get("z")))
    # Geographic.
    rep.run("coordinate_roundtrip", lambda: check_coordinates(world))
    rep.run("polygon_validity", lambda: check_polygons(world))
    rep.run("source_coverage", lambda: check_sources(world))
    # Topological.
    rep.run("road_graph_connectivity", lambda: check_connectivity(world), _connectivity_details(world))
    rep.run("false_intersection", lambda: check_false_intersections(world))
    rep.run("grade_separation", lambda: check_grade_separation(world))
    rep.run("bridge_connectivity", lambda: check_structure_connectivity(world, "bridge"))
    rep.run("tunnel_connectivity", lambda: check_structure_connectivity(world, "tunnel"))
    rep.run("dead_ends", lambda: check_dead_ends(world))
    rep.run("turn_restrictions", lambda: check_restrictions(world))
    rep.run("cross_tile_connectivity", lambda: check_cross_tile(world))
    # Geometric.
    rep.run("road_continuity", lambda: check_road_continuity(world))
    rep.run("bridge_elevation_continuity", lambda: check_structure_grades(world))
    rep.run("bridge_terrain_clearance", lambda: check_bridge_terrain(world))
    rep.run("terrain_tile_seams", lambda: check_terrain_seams(world))
    rep.run("building_meshes", lambda: check_building_meshes(world))
    # Vegetation.
    if cfg.vegetation.enabled and world.vegetation_grid is not None:
        rep.run("vegetation_placement", lambda: check_vegetation(world))
        rep.run("vegetation_tile_independence", lambda: check_tree_tile_independence(world))
    return rep


# --------------------------------------------------------------------------------------
# geographic


def check_coordinates(world: CanonicalWorld) -> list[Issue]:
    frame = world.region.frame
    xmin, ymin, xmax, ymax = world.region.extent
    xs = np.linspace(xmin, xmax, 7)
    ys = np.linspace(ymin, ymax, 7)
    gx, gy = np.meshgrid(xs, ys)
    lon, lat = frame.to_wgs84(gx.ravel(), gy.ravel())
    bx, by = frame.to_local(lon, lat)
    err = float(np.max(np.hypot(bx - gx.ravel(), by - gy.ravel())))
    out = []
    if err > 1e-4:
        out.append(Issue("coordinate_roundtrip", "error", f"WGS84 round trip error {err:.2e} m exceeds 1e-4 m"))
    k = max(abs(frame.scale_factor(x, y) - 1.0) for x, y in [(xmin, ymin), (xmax, ymax), (xmin, ymax)])
    if k > 1e-4:
        out.append(Issue("coordinate_roundtrip", "warning",
                         f"projection scale error {k:.1e} at the region edge; consider smaller worlds"))
    return out


def check_polygons(world: CanonicalWorld) -> list[Issue]:
    out = []
    for e in world.buildings + world.water + world.landcover:
        g = e.geometry
        if g.geom_type in ("Polygon", "MultiPolygon") and not g.is_valid:
            c = g.representative_point()
            out.append(Issue("polygon_validity", "error", f"{e.id}: invalid polygon ({shapely.is_valid_reason(g)})",
                             [e.id], c.x, c.y))
    return out


def check_sources(world: CanonicalWorld) -> list[Issue]:
    out = []
    for s in world.sources:
        if s.status in ("missing", "failed"):
            out.append(Issue("source_coverage", "warning", f"source {s.name}: {s.status} "
                             f"({'; '.join(s.notes) or 'no data for the region'})"))
        elif s.status == "partial":
            out.append(Issue("source_coverage", "info", f"source {s.name}: partial coverage {s.coverage}"))
    hf = world.terrain
    if hf is not None and hf.valid is not None and not hf.valid.all():
        frac = 1.0 - float(hf.valid.mean())
        rr, cc = np.nonzero(~hf.valid)
        k = len(rr) // 2
        out.append(Issue("source_coverage", "warning",
                         f"{frac:.1%} of terrain nodes have no DEM data (filled from neighbours)",
                         ["terrain"], hf.xmin + cc[k] * hf.resolution, hf.ymin + rr[k] * hf.resolution))
    # OSM building completeness heuristic: built-up land cover far from any mapped building.
    vg = world.vegetation_grid
    if vg is not None and world.terrain_codes is not None and world.buildings:
        codes = world.terrain_codes
        built = codes == 5
        if built.mean() > 0.05:
            pts = np.array([[b.geometry.centroid.x, b.geometry.centroid.y] for b in world.buildings])
            tree = cKDTree(pts)
            r, c = np.nonzero(built)
            sel = slice(None, None, max(1, len(r) // 4000))
            px = hf.xmin + c[sel] * hf.resolution
            py = hf.ymin + r[sel] * hf.resolution
            d, _ = tree.query(np.column_stack([px, py]))
            far = float(np.mean(d > 60.0))
            if far > 0.25:
                out.append(Issue("source_coverage", "warning",
                                 f"{far:.0%} of built-up land is > 60 m from any mapped building: "
                                 "building data may be incomplete"))
    elif not world.buildings:
        out.append(Issue("source_coverage", "info", "no buildings in the region"))
    if not world.roads:
        out.append(Issue("source_coverage", "warning", "no roads in the region"))
    return out


# --------------------------------------------------------------------------------------
# topological


def _connectivity_details(world: CanonicalWorld) -> dict[str, Any]:
    g = world.graph
    comps = g.components("motor_vehicle")
    n = sum(len(c) for c in comps)
    return {"components_motor_vehicle": len(comps),
            "largest_component_fraction": round(len(comps[0]) / n, 4) if comps else 1.0,
            "strongly_connected_fraction": round(g.strongly_connected_fraction(), 4) if g.edges else 1.0}


def check_connectivity(world: CanonicalWorld) -> list[Issue]:
    g = world.graph
    out = []
    comps = g.components("motor_vehicle")
    for c in comps[1:]:
        if any(g.nodes[n].kind == "boundary" for n in c):
            continue  # pieces cut by the region boundary are expected
        n = g.nodes[min(c)]
        out.append(Issue("road_graph_connectivity", "info",
                         f"isolated drivable component with {len(c)} nodes (not connected to the main network "
                         "inside the region)", [g.edges[n.edges[0]].road_id] if n.edges else [], n.x, n.y, n.z))
    return out


def check_false_intersections(world: CanonicalWorld) -> list[Issue]:
    g = world.graph
    out = []
    for c in world.crossings:
        if c.kind != "same_level_unconnected":
            continue
        a, b = g.edges[c.upper], g.edges[c.lower]
        # Footpaths crossing roads without a junction node are common and harmless.
        sev = "info" if "footway" in (a.road_class, b.road_class) or "path" in (a.road_class, b.road_class) else "warning"
        out.append(Issue("false_intersection", sev,
                         f"{a.road_id} and {b.road_id} cross at the same level without a shared node; "
                         "kept unconnected (no false intersection created)", [a.road_id, b.road_id],
                         c.x, c.y, float(world.terrain.sample(c.x, c.y))))
    return out


def check_grade_separation(world: CanonicalWorld) -> list[Issue]:
    g = world.graph
    out = []
    bc = world.config.bridges
    for c in world.crossings:
        if c.kind != "grade_separated":
            continue
        up = world.profiles[c.upper]
        lo = world.profiles[c.lower]
        zu = up.z_at(c.x, c.y)
        zl = lo.z_at(c.x, c.y)
        sep = zu - zl
        need = 2.5
        if sep < need:
            ids = [g.edges[c.upper].road_id, g.edges[c.lower].road_id]
            sid = g.edges[c.upper].structure_id or g.edges[c.lower].structure_id
            st = next((s for s in world.bridges + world.tunnels if s.id == sid), None)
            sev = "warning" if st is not None and st.uncertain else "error"
            out.append(Issue("grade_separation", sev,
                             f"vertical separation {sep:.1f} m at grade-separated crossing is below {need} m "
                             f"(min clearance target {bc.min_clearance_road} m)", ids, c.x, c.y, zu))
    return out


def check_structure_connectivity(world: CanonicalWorld, kind: str) -> list[Issue]:
    g = world.graph
    structs = world.bridges if kind == "bridge" else world.tunnels
    out = []
    for s in structs:
        anchors = s.anchor_node_ids
        if len(anchors) < 2:
            ends = [n for e in s.edge_ids for n in (g.edges[e].u, g.edges[e].v)
                    if g.nodes[n].kind in ("dead_end", "boundary")]
            boundary = any(g.nodes[n].kind == "boundary" for n in ends)
            n = g.nodes[anchors[0]] if anchors else g.nodes[g.edges[s.edge_ids[0]].u]
            out.append(Issue(f"{kind}_connectivity", "info" if boundary else "warning",
                             f"{s.id}: connects to ground roads at {len(anchors)} end(s)"
                             + (" (cut by the region boundary)" if boundary else ""), [s.id], n.x, n.y, n.z))
        for a in anchors:
            n = g.nodes[a]
            for eid in n.edges:
                p = world.profiles[eid]
                zend = p.z[0] if g.edges[eid].u == a else p.z[-1]
                if abs(zend - n.z) > 0.05:
                    out.append(Issue(f"{kind}_connectivity", "error",
                                     f"{s.id}: elevation discontinuity of {abs(zend - n.z):.2f} m at anchor {a}",
                                     [s.id, g.edges[eid].road_id], n.x, n.y, n.z))
    return out


def check_dead_ends(world: CanonicalWorld) -> list[Issue]:
    g = world.graph
    out = []
    dead = [n for n in g.nodes.values() if n.kind == "dead_end"]
    if not dead:
        return out
    others = [n for n in g.nodes.values()]
    pts = np.array([[n.x, n.y] for n in others])
    tree = cKDTree(pts)
    for n in dead:
        for k in tree.query_ball_point([n.x, n.y], 1.0):
            m = others[k]
            if m.id == n.id or set(m.edges) & set(n.edges):
                continue
            out.append(Issue("dead_ends", "info",
                             f"dead end {n.id} lies {np.hypot(m.x - n.x, m.y - n.y):.2f} m from {m.id} without "
                             "a connection (kept as a dead end)", [g.edges[n.edges[0]].road_id], n.x, n.y, n.z))
            break
    return out


def check_restrictions(world: CanonicalWorld) -> list[Issue]:
    g = world.graph
    out = []
    for r in g.restrictions:
        if r.via_node is not None and not r.enforced:
            out.append(Issue("turn_restrictions", "warning",
                             f"{r.id} ({r.restriction}) could not be mapped onto graph edges "
                             "(member ways outside the region or inconsistent data)", [r.id]))
        elif r.via_node is None:
            out.append(Issue("turn_restrictions", "info",
                             f"{r.id} ({r.restriction}) uses via ways; preserved in the graph but not enforced "
                             "by the router", [r.id]))
    return out


def check_cross_tile(world: CanonicalWorld) -> list[Issue]:
    g = world.graph
    out = []
    for e in g.edges.values():
        tiles = [parse_tile_id(t) for t in e.tiles]
        if len(tiles) <= 1:
            continue
        # The tile sequence along an edge must be 8-connected (no skipped tile).
        s = set(tiles)
        start = tiles[0]
        seen = {start}
        stack = [start]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    q = (i + di, j + dj)
                    if q in s and q not in seen:
                        seen.add(q)
                        stack.append(q)
        if seen != s:
            out.append(Issue("cross_tile_connectivity", "error", f"{e.id}: tile pieces are not contiguous",
                             [e.road_id], float(e.coords[0, 0]), float(e.coords[0, 1])))
    return out


# --------------------------------------------------------------------------------------
# geometric


def check_road_continuity(world: CanonicalWorld) -> list[Issue]:
    g = world.graph
    out = []
    for n in g.nodes.values():
        zs = []
        for eid in n.edges:
            p = world.profiles[eid]
            zs.append(p.z[0] if g.edges[eid].u == n.id else p.z[-1])
        if zs and max(zs) - min(zs) > 0.05:
            out.append(Issue("road_continuity", "error",
                             f"road ends meeting at {n.id} differ by {max(zs) - min(zs):.2f} m",
                             [g.edges[e].road_id for e in n.edges], n.x, n.y, n.z))
    exempt = set(world.config.roads.grade_exempt_classes)
    for eid, p in world.profiles.items():
        if p.kind != "ground" or len(p.z) < 2 or g.edges[eid].road_class in exempt:
            continue
        ds = np.maximum(np.diff(p.s), 1e-6)
        grade = np.abs(np.diff(p.z)) / ds
        k = int(np.argmax(grade))
        if grade[k] > max(0.35, world.config.roads.max_grade * 1.5) and ds[k] > 1.0:
            out.append(Issue("road_continuity", "warning",
                             f"{g.edges[eid].road_id}: road grade {grade[k]:.0%} (terrain data artefact or steep road)",
                             [g.edges[eid].road_id], float(p.xy[k, 0]), float(p.xy[k, 1]), float(p.z[k])))
    return out


def check_structure_grades(world: CanonicalWorld) -> list[Issue]:
    g = world.graph
    out = []
    for s in world.bridges + world.tunnels:
        lim = (world.config.bridges.max_grade if s.kind == "bridge" else world.config.tunnels.max_grade)
        for eid in s.edge_ids:
            p = world.profiles[eid]
            if len(p.z) < 2:
                continue
            ds = np.maximum(np.diff(p.s), 1e-6)
            dz = np.diff(p.z)
            grade = np.abs(dz) / ds
            allowed = max(lim, float(s.get("design_grade", lim))) * 1.05 + 1e-3
            bad = np.nonzero((grade > allowed) & (np.abs(dz) > 0.05))[0]
            if len(bad):
                k = int(bad[np.argmax(grade[bad])])
                out.append(Issue("bridge_elevation_continuity", "warning",
                                 f"{s.id}: deck grade {grade[k]:.1%} exceeds {allowed:.1%}", [s.id],
                                 float(p.xy[k, 0]), float(p.xy[k, 1]), float(p.z[k])))
            jumps = np.nonzero(np.abs(dz) > 2.0)[0]
            for k in jumps[:3]:
                out.append(Issue("bridge_elevation_continuity", "error",
                                 f"{s.id}: deck discontinuity of {abs(dz[k]):.2f} m", [s.id],
                                 float(p.xy[k, 0]), float(p.xy[k, 1]), float(p.z[k])))
    return out


def check_bridge_terrain(world: CanonicalWorld) -> list[Issue]:
    """Deck underside must clear the (shaped) terrain except in abutment zones."""
    out = []
    hf = world.terrain
    bc = world.config.bridges
    for b in world.bridges:
        worst = None
        for eid in b.edge_ids:
            p = world.profiles[eid]
            t = hf.sample(p.xy[:, 0], p.xy[:, 1])
            under = p.z - bc.deck_thickness - t
            ab = p.abutment if p.abutment is not None else np.zeros(len(p.z), dtype=bool)
            bad = (under < -0.05) & ~ab
            if bad.any():
                k = int(np.argmin(np.where(bad, under, np.inf)))
                if worst is None or under[k] < worst[0]:
                    worst = (float(under[k]), float(p.xy[k, 0]), float(p.xy[k, 1]), float(p.z[k]))
            top = p.z - t
            if (top < 0.2).any() and not ab[top < 0.2].all():
                k = int(np.argmin(top))
                out.append(Issue("bridge_terrain_clearance", "error",
                                 f"{b.id}: terrain rises above the deck surface", [b.id],
                                 float(p.xy[k, 0]), float(p.xy[k, 1]), float(p.z[k])))
        if worst is not None:
            out.append(Issue("bridge_terrain_clearance", "warning" if b.uncertain else "error",
                             f"{b.id}: deck underside {abs(worst[0]):.2f} m below terrain outside abutments",
                             [b.id], worst[1], worst[2], worst[3]))
    return out


def check_terrain_seams(world: CanonicalWorld) -> list[Issue]:
    hf = world.terrain
    grid = TileGrid(world.config.tiling.tile_size)
    tiles = {parse_tile_id(t) for t in [tile_id(i, j) for i, j in world.region.tiles()]}
    out = []
    for i, j in sorted(tiles):
        for di, dj in ((1, 0), (0, 1)):
            q = (i + di, j + dj)
            if q not in tiles:
                continue
            a = hf.tile_mesh(*grid.bounds(tile_id(i, j)))
            b = hf.tile_mesh(*grid.bounds(tile_id(*q)))
            if di:
                edge = grid.bounds(tile_id(*q))[0]
                ea = a.positions[np.isclose(a.positions[:, 0], edge)]
                eb = b.positions[np.isclose(b.positions[:, 0], edge)]
            else:
                edge = grid.bounds(tile_id(*q))[1]
                ea = a.positions[np.isclose(a.positions[:, 1], edge)]
                eb = b.positions[np.isclose(b.positions[:, 1], edge)]
            ea = ea[np.lexsort(ea[:, :2].T)]
            eb = eb[np.lexsort(eb[:, :2].T)]
            if ea.shape != eb.shape or not np.allclose(ea, eb, atol=1e-9):
                out.append(Issue("terrain_tile_seams", "error",
                                 f"terrain tiles {tile_id(i, j)} and {tile_id(*q)} do not share edge vertices",
                                 [f"terrain:{tile_id(i, j)}"], edge if di else (i + 1) * grid.size, edge))
    return out


def check_building_meshes(world: CanonicalWorld) -> list[Issue]:
    out = []
    bc = world.config.buildings
    for b in world.buildings:
        if b.rendered_by_parts:
            continue
        c = b.geometry.representative_point()
        # Each footprint part is its own closed solid (parts may touch at a vertex or edge).
        for part in polygons_of(b.geometry):
            m = building_mesh(dataclasses.replace(b, geometry=part), bc)
            if m.is_empty():
                out.append(Issue("building_meshes", "error", f"{b.id}: empty mesh", [b.id], c.x, c.y))
                continue
            if not np.isfinite(m.normals).all() or (np.linalg.norm(m.normals, axis=1) < 0.5).any():  # type: ignore[arg-type]
                out.append(Issue("building_meshes", "error", f"{b.id}: invalid normals", [b.id], c.x, c.y))
            tm = trimesh.Trimesh(m.positions, m.indices, process=True)
            tm.merge_vertices()
            areas = tm.area_faces
            if (areas < 1e-8).sum() > 0:
                out.append(Issue("building_meshes", "warning", f"{b.id}: {(areas < 1e-8).sum()} degenerate triangles",
                                 [b.id], c.x, c.y))
            if not tm.is_watertight:
                out.append(Issue("building_meshes", "warning", f"{b.id}: mesh is not watertight", [b.id], c.x, c.y))
            elif tm.volume <= 0:
                out.append(Issue("building_meshes", "error", f"{b.id}: inverted surfaces (negative volume)",
                                 [b.id], c.x, c.y))
    return out


# --------------------------------------------------------------------------------------
# vegetation


def check_vegetation(world: CanonicalWorld) -> list[Issue]:
    from worldgen.pipeline import vegetation_exclusions

    tr = world.trees
    out: list[Issue] = []
    if len(tr) == 0:
        return out
    grid = world.vegetation_grid
    gen = ~tr.observed & np.isin(np.array(tr.archetype, dtype=object), ["broadleaf", "conifer", "palm"])
    gen &= np.array([s == "procedural" for s in tr.source])
    rr, cc, inside = grid.cell_of(tr.x, tr.y)
    tree_codes = [CAT[c] for c in TREE_CATEGORIES]
    bad = gen & ~(inside & np.isin(grid.category[rr, cc], tree_codes))
    for k in np.nonzero(bad)[0][:20]:
        out.append(Issue("vegetation_placement", "error", f"{tr.ids[k]} placed outside a forest/woodland region",
                         [tr.ids[k]], float(tr.x[k]), float(tr.y[k])))
    excl = vegetation_exclusions(world, 0.0)
    procedural = np.array([s == "procedural" for s in tr.source])
    inside_excl = excl.contains(tr.x, tr.y) & procedural
    for k in np.nonzero(inside_excl)[0][:20]:
        out.append(Issue("vegetation_placement", "error",
                         f"{tr.ids[k]} placed on a road, building or water surface", [tr.ids[k]],
                         float(tr.x[k]), float(tr.y[k])))
    dz = np.abs(tr.ground_z - world.terrain.sample(tr.x, tr.y))
    if (dz > 1e-6).any():
        k = int(np.argmax(dz))
        out.append(Issue("vegetation_placement", "error", f"tree ground elevation differs from terrain by {dz[k]:.3f} m",
                         [tr.ids[k]], float(tr.x[k]), float(tr.y[k])))
    if len(set(tr.ids)) != len(tr.ids):
        out.append(Issue("vegetation_placement", "error", "duplicate tree ids"))
    pts = np.column_stack([tr.x, tr.y])
    pairs = cKDTree(pts).query_pairs(0.01, output_type="ndarray")
    if len(pairs):
        out.append(Issue("vegetation_placement", "error", f"{len(pairs)} duplicate tree positions",
                         [tr.ids[int(pairs[0, 0])]], float(tr.x[pairs[0, 0]]), float(tr.y[pairs[0, 0]])))
    # Observed trees preserved at their source positions.
    pos = {t: k for k, t in enumerate(tr.ids)}
    hf = world.terrain
    for ot in world.observed_trees:
        if not hf.contains(ot.x, ot.y):
            continue
        k = pos.get(ot.id)
        if k is None:
            out.append(Issue("vegetation_placement", "error", f"observed tree {ot.id} missing from output", [ot.id],
                             ot.x, ot.y))
        elif abs(tr.x[k] - ot.x) > 1e-6 or abs(tr.y[k] - ot.y) > 1e-6:
            out.append(Issue("vegetation_placement", "error", f"observed tree {ot.id} moved", [ot.id], ot.x, ot.y))
        elif not ot.from_row and not tr.observed[k]:
            out.append(Issue("vegetation_placement", "error", f"observed tree {ot.id} lost its observed flag", [ot.id]))
    if (gen & tr.observed).any():
        out.append(Issue("vegetation_placement", "error", "generated trees flagged as observed"))
    # Canopy height consistency for generated trees where canopy height was measured.
    meas = gen & inside & grid.canopy_observed[rr, cc] & (grid.canopy_height[rr, cc] >= 3.0)
    if meas.sum() >= 20:
        ratio = float(np.median(tr.height[meas] / grid.canopy_height[rr, cc][meas]))
        if not 0.7 <= ratio <= 1.4:
            out.append(Issue("vegetation_placement", "warning",
                             f"generated tree heights deviate from measured canopy height (median ratio {ratio:.2f})"))
    return out


def check_tree_tile_independence(world: CanonicalWorld) -> list[Issue]:
    """Re-generate two tiles in isolation and compare with the global placement."""
    from worldgen.pipeline import vegetation_exclusions
    from worldgen.vegetation.placement import place_trees

    tr = world.trees
    if len(tr) == 0:
        return []
    grid = TileGrid(world.config.tiling.tile_size)
    tiles = [tile_id(i, j) for i, j in world.region.tiles()]
    gen_mask = np.array([s == "procedural" and a in ("broadleaf", "conifer", "palm")
                         for s, a in zip(tr.source, tr.archetype)])
    tile_of = grid.tiles_of(tr.x, tr.y)
    counts = {t: int(np.sum(gen_mask & (tile_of == t))) for t in tiles}
    sample = sorted(tiles, key=lambda t: (-counts[t], t))[:2]
    excl = vegetation_exclusions(world, world.config.vegetation.exclusion_buffer)
    out = []
    for t in sample:
        if counts[t] == 0:
            continue
        local = place_trees(world.vegetation_grid, world.terrain, excl, world.observed_trees,
                            world.config.vegetation, bounds=grid.bounds(t))
        a = sorted(i for i, o in zip(local.ids, local.observed) if i.startswith("tree:gen:"))
        b = sorted(tr.ids[k] for k in np.nonzero(gen_mask & (tile_of == t))[0])
        if a != b:
            out.append(Issue("vegetation_tile_independence", "error",
                             f"tile {t}: isolated tree generation differs from global run "
                             f"({len(a)} vs {len(b)} trees)", [], *grid.origin(t)))
    return out
