"""Building preparation (repair, conflation, heights) and mesh generation."""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient
from shapely.ops import split
from shapely.strtree import STRtree

from worldgen.config import BuildingsConfig, WorldConfig
from worldgen.geometry.mesh import Mesh, merge_meshes
from worldgen.geometry.shapes import extrude_polygon, polygons_of, triangulate_polygon, walls_to_roof
from worldgen.hashing import unit_hash
from worldgen.log import get_logger
from worldgen.terrain.heightfield import Heightfield
from worldgen.tiling.grid import Piece, TileGrid
from worldgen.world.model import Building

log = get_logger("buildings")

ROOF_SHAPES_SUPPORTED = {"flat", "gabled", "pyramidal", "hipped", "skillion"}


def repair_footprint(geom: BaseGeometry) -> BaseGeometry | None:
    """Make valid, keep polygonal parts, drop slivers and duplicate vertices."""
    g = geom if geom.is_valid else shapely.make_valid(geom)
    polys = [p for p in polygons_of(g) if p.area > 1.0]
    if not polys:
        return None
    polys = [orient(Polygon(p.exterior.coords, [i.coords for i in p.interiors if Polygon(i).area > 0.5]), 1.0)
             for p in polys]
    out = shapely.union_all(polys) if len(polys) > 1 else polys[0]
    out = shapely.set_precision(out, 0.001)
    if out.is_empty or not out.is_valid:
        out = shapely.make_valid(out)
    polys = [p for p in polygons_of(out) if p.area > 1.0]
    if not polys:
        return None
    return polys[0] if len(polys) == 1 else shapely.MultiPolygon(polys)


def prepare_buildings(osm: list[Building], overture: list[Building], extent: Polygon,
                      cfg: WorldConfig) -> tuple[list[Building], dict[str, int]]:
    stats = {"repaired": 0, "dropped_invalid": 0, "overture_added": 0, "overture_duplicates": 0,
             "outside_extent": 0, "parts_linked": 0}
    out: list[Building] = []
    for b in osm + overture:
        fixed = repair_footprint(b.geometry)
        if fixed is None:
            stats["dropped_invalid"] += 1
            continue
        if not b.geometry.is_valid:
            b.infer("footprint_repaired", True, "shapely.make_valid", 0.8)
        if "footprint_repaired" in b.inferred_properties:
            stats["repaired"] += 1
        b.geometry = fixed
        rp = fixed.representative_point()
        if not extent.contains(rp):
            stats["outside_extent"] += 1
            continue
        out.append(b)
    osm_b = [b for b in out if b.source != "overture"]
    ov_b = [b for b in out if b.source == "overture"]
    kept_ov: list[Building] = []
    if ov_b and osm_b:
        tree = STRtree([b.geometry for b in osm_b])
        for b in ov_b:
            dup = False
            for j in tree.query(b.geometry, predicate="intersects").tolist():
                o = osm_b[j].geometry
                inter = b.geometry.intersection(o).area
                if inter / max(min(b.geometry.area, o.area), 1e-9) >= cfg.sources.overture.min_overlap_to_skip:
                    dup = True
                    break
            if dup:
                stats["overture_duplicates"] += 1
            else:
                kept_ov.append(b)
    else:
        kept_ov = ov_b
    stats["overture_added"] = len(kept_ov)
    result = sorted(osm_b + kept_ov, key=lambda b: b.id)
    _link_parts(result, stats)
    for b in result:
        estimate_height(b, cfg.buildings)
    return result, stats


def _link_parts(buildings: list[Building], stats: dict[str, int]) -> None:
    outlines = [b for b in buildings if not b.is_part]
    parts = [b for b in buildings if b.is_part]
    if not parts or not outlines:
        return
    tree = STRtree([b.geometry for b in outlines])
    for p in parts:
        rp = p.geometry.representative_point()
        hits = [outlines[j] for j in tree.query(rp, predicate="within").tolist()]
        if not hits:
            continue
        parent = min(hits, key=lambda b: b.geometry.area)
        p.parent_id = parent.id
        parent.rendered_by_parts = True
        stats["parts_linked"] += 1


def estimate_height(b: Building, bc: BuildingsConfig) -> None:
    """Fill height / levels / roof estimates; observed values are never overwritten."""
    cls = str(b.properties.get("class") or "yes")
    levels = b.properties.get("levels")
    height = b.properties.get("height")
    roof_levels = b.properties.get("roof_levels") or 0
    if height is None:
        if levels is not None:
            h = (float(levels) + float(roof_levels)) * bc.floor_height
            b.infer("height", max(h, bc.min_height), "levels*floor_height", 0.6)
        else:
            lv = bc.class_levels.get(cls)
            if lv is None:
                h, method, conf = bc.default_height, "default_height", 0.2
            else:
                h, method, conf = lv * bc.floor_height, f"class_default_levels:{cls}", 0.3
            if bc.procedural_variation > 0:
                f = 1.0 + bc.procedural_variation * (2.0 * unit_hash("height", b.id) - 1.0)
                h *= f
                method += "+procedural_variation"
            b.infer("height", round(max(h, bc.min_height), 2), method, conf)
    if levels is None:
        h = float(b.get("height"))
        b.infer("levels", max(1, int(round(h / bc.floor_height))), "height/floor_height", 0.4)
    if b.properties.get("min_height") is None and b.properties.get("min_level") is not None:
        b.infer("min_height", float(b.properties["min_level"]) * bc.floor_height, "min_level*floor_height", 0.6)
    if b.properties.get("roof_shape") is None:
        if bc.procedural_roofs and cls in ("house", "detached", "semidetached_house", "bungalow", "farm",
                                           "hut", "shed") and _rectangularity(b.geometry) > 0.85:
            b.infer("roof_shape", "gabled", "procedural:residential_rectangular", 0.25)
        else:
            b.mark_missing("roof_shape")


def _rectangularity(g: BaseGeometry) -> float:
    if g.geom_type != "Polygon" or len(g.interiors) > 0:
        return 0.0
    mrr = g.minimum_rotated_rectangle
    return float(g.area / mrr.area) if mrr.area > 0 else 0.0


def assign_base_elevations(buildings: list[Building], terrain: Heightfield) -> None:
    by_id = {b.id: b for b in buildings}
    for b in buildings:
        pts = []
        for p in polygons_of(b.geometry):
            pts.append(np.asarray(p.exterior.segmentize(4.0).coords)[:, :2])
        rp = b.geometry.representative_point()
        pts.append(np.array([[rp.x, rp.y]]))
        P = np.concatenate(pts)
        inside = terrain.contains(P[:, 0], P[:, 1])
        P = P[inside] if inside.any() else P
        z = terrain.sample(P[:, 0], P[:, 1])
        b.base_elevation = float(np.min(z))
        b.properties.pop("ground_elevation", None)
        b.inferred_properties.pop("ground_elevation", None)
        b.infer("ground_elevation", round(b.base_elevation, 2), "terrain_min_under_footprint", 0.7)
    # Parts share the base of their outline so towers and podiums line up.
    for b in buildings:
        if b.is_part and b.parent_id in by_id:
            b.base_elevation = by_id[b.parent_id].base_elevation  # type: ignore[union-attr]


def _roof_height(b: Building, poly: Polygon, bc: BuildingsConfig, total: float) -> float:
    rh = b.properties.get("roof_height")
    if rh is not None:
        return min(float(rh), total * 0.8)
    rl = b.properties.get("roof_levels")
    if rl:
        return min(float(rl) * bc.floor_height, total * 0.8)
    mrr = poly.minimum_rotated_rectangle
    c = np.asarray(mrr.exterior.coords)[:4]
    short = min(np.hypot(*(c[1] - c[0])), np.hypot(*(c[2] - c[1])))
    return min(math.tan(math.radians(bc.max_roof_pitch_deg)) * short / 2.0, total * 0.5)


def building_mesh(b: Building, bc: BuildingsConfig) -> Mesh:
    base = (b.base_elevation or 0.0)
    total = float(b.get("height", bc.default_height))
    min_h = float(b.get("min_height", 0.0) or 0.0)
    if min_h >= total - 0.5:
        min_h = 0.0
    bottom = base + min_h - (bc.foundation_depth if min_h == 0 else 0.0)
    top = base + total
    shape = str(b.get("roof_shape") or "flat")
    parts: list[Mesh] = []
    for poly in polygons_of(b.geometry):
        if bc.roofs and shape in ("gabled", "pyramidal", "hipped", "skillion") and not poly.interiors:
            m = _pitched(poly, bottom, top, shape, _roof_height(b, poly, bc, total - min_h))
            if m is not None:
                parts.append(m)
                continue
        parts.append(extrude_polygon(poly, bottom, top))
    return merge_meshes(parts)


def _pitched(poly: Polygon, bottom: float, top: float, shape: str, roof_h: float) -> Mesh | None:
    eave = top - roof_h
    if roof_h < 0.3 or eave <= bottom + 1.0:
        return None
    poly = orient(poly, 1.0)
    if shape in ("pyramidal", "hipped"):
        if not poly.equals(poly.convex_hull.buffer(0)) and poly.area / poly.convex_hull.area < 0.98:
            return None
        c = poly.centroid
        ring = np.asarray(poly.exterior.coords)[:-1, :2]
        walls = walls_to_roof(poly, bottom, lambda x, y: np.full(len(x), eave))
        pos, idx = [], []
        for i in range(len(ring)):
            a, bb = ring[i], ring[(i + 1) % len(ring)]
            k = len(pos)
            pos += [(a[0], a[1], eave), (bb[0], bb[1], eave), (c.x, c.y, top)]
            idx.append((k, k + 1, k + 2))
        roof = Mesh(np.array(pos), np.array(idx, dtype=np.int64))
        roof.ensure_normals()
        floor = _cap(poly, bottom, down=True)
        return merge_meshes([walls, roof, floor])
    # Gabled / skillion: ridge along the long axis of the minimum rotated rectangle.
    mrr = poly.minimum_rotated_rectangle
    c = np.asarray(mrr.exterior.coords)[:4]
    e1, e2 = c[1] - c[0], c[2] - c[1]
    axis = e1 if np.hypot(*e1) >= np.hypot(*e2) else e2
    across = e2 if axis is e1 else e1
    half = np.hypot(*across) / 2.0
    if half < 1.0:
        return None
    u = axis / np.hypot(*axis)
    v = np.array([-u[1], u[0]])
    centre = c.mean(axis=0)

    def roof_z(x: np.ndarray, y: np.ndarray) -> np.ndarray:
        d = (np.column_stack([x, y]) - centre) @ v
        if shape == "skillion":
            return eave + roof_h * np.clip((d + half) / (2 * half), 0, 1)
        return eave + roof_h * np.clip(1.0 - np.abs(d) / half, 0, 1)

    if shape == "gabled":
        far = 4.0 * max(np.hypot(*axis), 1.0)
        ridge = LineString([centre - u * far, centre + u * far])
        # Put the ridge/outline intersections into the ring so wall tops meet the roof exactly.
        outline = _with_line_vertices(poly, centre, u)
        try:
            halves = [g for g in split(outline, ridge).geoms if isinstance(g, Polygon)]
        except (ValueError, shapely.errors.GEOSException):
            return None
        if len(halves) < 2:
            return None
    else:
        halves = [poly]
        outline = poly
    walls = walls_to_roof(outline, bottom, roof_z)
    roofs = []
    for h in halves:
        h = orient(h, 1.0)
        v2, tri = triangulate_polygon(h)
        if len(tri) == 0:
            continue
        m = Mesh(np.column_stack([v2, roof_z(v2[:, 0], v2[:, 1])]), tri, uvs=v2 / 4.0)
        m.ensure_normals()
        roofs.append(m)
    floor = _cap(outline, bottom, down=True)
    return merge_meshes([walls, *roofs, floor])


def _with_line_vertices(poly: Polygon, p0: np.ndarray, u: np.ndarray) -> Polygon:
    """Insert the intersections of the infinite line (p0, u) with the exterior ring."""
    ring = np.asarray(poly.exterior.coords)[:-1, :2]
    n = np.array([-u[1], u[0]])
    side = (ring - p0) @ n
    out = []
    for i in range(len(ring)):
        j = (i + 1) % len(ring)
        out.append(ring[i])
        if side[i] * side[j] < 0:
            t = side[i] / (side[i] - side[j])
            out.append(ring[i] + (ring[j] - ring[i]) * t)
    return orient(Polygon(out), 1.0)


def _cap(poly: Polygon, z: float, down: bool) -> Mesh:
    v2, tri = triangulate_polygon(poly)
    m = Mesh(np.column_stack([v2, np.full(len(v2), z)]), tri if not down else tri[:, [0, 2, 1]],
             normals=np.tile([0.0, 0.0, -1.0 if down else 1.0], (len(v2), 1)), uvs=v2 / 4.0)
    return m


def building_material(b: Building) -> str:
    cls = str(b.get("class") or "yes")
    if cls in ("house", "detached", "semidetached_house", "terrace", "residential", "apartments",
               "bungalow", "dormitory"):
        return "building_residential"
    if cls in ("commercial", "retail", "office", "hotel", "supermarket", "kiosk"):
        return "building_commercial"
    if cls in ("industrial", "warehouse", "factory", "manufacture", "garage", "garages", "shed", "service"):
        return "building_industrial"
    if cls in ("school", "university", "hospital", "church", "temple", "public", "civic", "government",
               "train_station", "college", "kindergarten", "mosque", "shrine"):
        return "building_civic"
    return "building_default"


def build_building_pieces(buildings: list[Building], bc: BuildingsConfig, grid: TileGrid,
                          wanted: set[str] | None = None) -> list[Piece]:
    pieces = []
    for b in buildings:
        if b.rendered_by_parts:
            continue
        rp = b.geometry.representative_point()
        tile = grid.tile_of(rp.x, rp.y)
        if wanted is not None and tile not in wanted:
            continue
        m = building_mesh(b, bc)
        if not m.is_empty():
            pieces.append(Piece(tile, "buildings", building_material(b), b.id, m))
    return pieces


def build_building_collision(buildings: list[Building], bc: BuildingsConfig, grid: TileGrid,
                             wanted: set[str] | None = None) -> list[Piece]:
    pieces = []
    for b in buildings:
        if b.rendered_by_parts:
            continue
        rp = b.geometry.representative_point()
        tile = grid.tile_of(rp.x, rp.y)
        if wanted is not None and tile not in wanted:
            continue
        top = (b.base_elevation or 0.0) + float(b.get("height", bc.default_height))
        bottom = (b.base_elevation or 0.0) - bc.foundation_depth
        for p in polygons_of(b.geometry.simplify(0.5)):
            if p.area > 1.0:
                pieces.append(Piece(tile, "collision", "collision", b.id,
                                    extrude_polygon(Polygon(p.exterior), bottom, top)))
    return pieces
