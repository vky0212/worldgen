"""Water: surface levels, coastline seas, terrain carving inputs and water meshes."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import shapely
from shapely.geometry import LineString, Polygon
from shapely.ops import polygonize, unary_union

from worldgen.config import WorldConfig
from worldgen.geometry.shapes import flat_polygon_mesh, polygons_of, resample_polyline, ribbon
from worldgen.terrain.build import Corridor, polygon_boundary_samples
from worldgen.terrain.heightfield import Heightfield
from worldgen.tiling.grid import Piece, TileGrid
from worldgen.world.model import Waterbody


@dataclass
class WaterwayProfile:
    xy: np.ndarray
    s: np.ndarray
    z: np.ndarray
    width: float


def sea_from_coastlines(coastlines: list[LineString], extent: Polygon) -> list[Waterbody]:
    """Sea polygons: OSM coastlines have land on the left, water on the right."""
    if not coastlines:
        return []
    merged = shapely.line_merge(shapely.union_all(coastlines))
    lines = [g for g in getattr(merged, "geoms", [merged]) if g.geom_type == "LineString"]
    clipped = [ln.intersection(extent) for ln in lines]
    pieces = list(polygonize(unary_union([*clipped, extent.exterior])))
    seas = []
    for poly in pieces:
        if _is_sea_side(poly, lines):
            seas.append(poly)
    out = []
    for k, poly in enumerate(sorted(seas, key=lambda p: (round(p.centroid.x, 2), round(p.centroid.y, 2)))):
        w = Waterbody(id=f"water:osm:coastline:{k}", geometry=poly, properties={"class": "sea"}, source="osm",
                      source_id="natural=coastline", confidence=0.8)
        w.infer("surface_elevation", 0.0, "mean_sea_level", 0.7)
        out.append(w)
    return out


def _is_sea_side(poly: Polygon, coast: list[LineString]) -> bool:
    rp = poly.representative_point()
    best = None
    for ln in coast:
        d = ln.distance(rp)
        if best is None or d < best[0]:
            best = (d, ln)
    if best is None:
        return False
    ln = best[1]
    s = ln.project(rp)
    a = ln.interpolate(max(s - 0.5, 0.0))
    b = ln.interpolate(min(s + 0.5, ln.length))
    p = ln.interpolate(s)
    cross = (b.x - a.x) * (rp.y - p.y) - (b.y - a.y) * (rp.x - p.x)
    return cross < 0  # right of the line direction


def compute_water_levels(water: list[Waterbody], terrain: Heightfield, cfg: WorldConfig
                         ) -> tuple[dict[str, float], dict[str, WaterwayProfile]]:
    levels: dict[str, float] = {}
    profiles: dict[str, WaterwayProfile] = {}
    for w in water:
        if w.is_area:
            if w.properties.get("class") == "sea":
                levels[w.id] = 0.0
                continue
            samples = polygon_boundary_samples(terrain, w.geometry, terrain.resolution)
            if len(samples) == 0:
                continue
            lvl = float(np.percentile(samples, cfg.water.lake_level_percentile))
            levels[w.id] = lvl
            w.infer("surface_elevation", round(lvl, 2),
                    f"terrain_shoreline_p{cfg.water.lake_level_percentile:g}", 0.5)
        else:
            if w.properties.get("tunnel"):
                continue
            lines = [w.geometry] if w.geometry.geom_type == "LineString" else list(getattr(w.geometry, "geoms", []))
            line = max(lines, key=lambda g: g.length)
            xy, s = resample_polyline(line, terrain.resolution)
            z = terrain.sample(xy[:, 0], xy[:, 1])
            # OSM waterways point downstream: the surface never rises along the line.
            z = np.minimum.accumulate(z)
            width = float(w.get("width", 3.0))
            profiles[w.id] = WaterwayProfile(xy, s, z, width)
            levels[w.id] = float(np.median(z))
            w.infer("surface_elevation", [round(float(z.min()), 2), round(float(z.max()), 2)],
                    "terrain_running_min_downstream", 0.4)
    return levels, profiles


def waterways_covered_by_areas(water: list[Waterbody]) -> set[str]:
    areas = [w.geometry for w in water if w.is_area]
    if not areas:
        return set()
    u = shapely.union_all(areas)
    shapely.prepare(u)
    out = set()
    for w in water:
        if not w.is_area and w.geometry.length > 0:
            if w.geometry.intersection(u).length / w.geometry.length > 0.5:
                out.add(w.id)
    return out


def carve_inputs(water: list[Waterbody], levels: dict[str, float], profiles: dict[str, WaterwayProfile],
                 covered: set[str], cfg: WorldConfig) -> tuple[list[tuple[Polygon, float]], list[Corridor]]:
    depth = cfg.terrain.water_carve_depth
    polys = [(w.geometry, levels[w.id] - depth) for w in water if w.is_area and w.id in levels]
    corridors = [
        Corridor(p.xy, p.z - depth, half_width=p.width / 2.0, shoulder=max(p.width / 2.0, 2.0), mode="max")
        for wid, p in sorted(profiles.items()) if wid not in covered
    ]
    return polys, corridors


def shore_inputs(water: list[Waterbody], levels: dict[str, float], cfg: WorldConfig
                 ) -> list[tuple[Polygon, float, float]]:
    """(area, surface elevation, bed elevation) of every water area with a level (shoreline shaping)."""
    depth = cfg.terrain.water_carve_depth
    off = cfg.water.surface_offset
    return [(w.geometry, levels[w.id] + off, levels[w.id] - depth) for w in water if w.is_area and w.id in levels]


def waterway_outlines(water: list[Waterbody], profiles: dict[str, WaterwayProfile]) -> list[Polygon]:
    """Outlines of the rendered water lines (centreline buffered to the profile width)."""
    return [w.geometry.buffer(profiles[w.id].width / 2.0, cap_style="flat")
            for w in water if not w.is_area and w.id in profiles]


def build_water_pieces(water: list[Waterbody], levels: dict[str, float], profiles: dict[str, WaterwayProfile],
                       covered: set[str], cfg: WorldConfig, grid: TileGrid, wanted: set[str] | None = None
                       ) -> list[Piece]:
    pieces: list[Piece] = []
    off = cfg.water.surface_offset
    for w in water:
        if w.is_area:
            if w.id not in levels:
                continue
            xmin, ymin, xmax, ymax = w.geometry.bounds
            for t in grid.tiles_in_bbox(xmin, ymin, xmax, ymax):
                if wanted is not None and t not in wanted:
                    continue
                clip = w.geometry.intersection(shapely.box(*grid.bounds(t)))
                for p in polygons_of(clip):
                    if p.area < 0.5:
                        continue
                    m = flat_polygon_mesh(p, levels[w.id] + off, uv_scale=20.0)
                    if not m.is_empty():
                        pieces.append(Piece(t, "water", "water", w.id, m))
        elif w.id in profiles and w.id not in covered:
            p = profiles[w.id]
            xy, s, z = grid.insert_tile_crossings(p.xy, p.s, p.z)
            tiles = grid.segment_tiles(xy)
            start = 0
            for k in range(1, len(tiles) + 1):
                if k == len(tiles) or tiles[k] != tiles[start]:
                    t = tiles[start]
                    if wanted is None or t in wanted:
                        pieces.append(Piece(t, "water", "water", w.id,
                                            ribbon(xy[start:k + 1], z[start:k + 1] + off, p.width, s0=float(s[start]),
                                                   uv_scale=20.0)))
                    start = k
    return pieces
