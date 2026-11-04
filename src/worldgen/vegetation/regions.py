"""Vegetation regions, environmental fields and vegetation geometry."""

from __future__ import annotations

import math

import numpy as np
import shapely
from rasterio import features as rfeatures
from shapely.geometry import shape as shapely_shape

from worldgen.config import VegetationConfig
from worldgen.geometry.mesh import Mesh, merge_meshes
from worldgen.geometry.shapes import cone, cylinder, icosphere, line_strip_mesh, polygons_of
from worldgen.terrain.heightfield import Heightfield
from worldgen.tiling.grid import Piece, TileGrid
from worldgen.vegetation.classify import CAT, CATEGORIES, SOURCE_NAMES, VegetationGrid
from worldgen.vegetation.placement import ARCHETYPES
from worldgen.world.model import LandcoverArea, TreeTable, VegetationRegion


def extract_regions(grid: VegetationGrid, landcover: list[LandcoverArea], cfg: VegetationConfig,
                    min_area: float | None = None) -> list[VegetationRegion]:
    min_area = min_area if min_area is not None else 2.0 * grid.resolution ** 2
    polys: list[tuple[str, shapely.Geometry]] = []
    for geom, value in rfeatures.shapes(grid.category, mask=grid.category > 0, transform=grid.transform()):
        name = CATEGORIES[int(value)]
        g = shapely_shape(geom)
        if g.area < min_area:
            continue
        polys.append((name, g.simplify(grid.resolution * 0.5, preserve_topology=True)))
    polys.sort(key=lambda t: (t[0], round(t[1].centroid.x, 1), round(t[1].centroid.y, 1)))
    counters: dict[str, int] = {}
    out: list[VegetationRegion] = []
    for name, g in polys:
        k = counters.get(name, 0)
        counters[name] = k + 1
        mask = rfeatures.rasterize([(g, 1)], out_shape=grid.shape, transform=grid.transform(), fill=0,
                                   dtype="uint8").astype(bool) & (grid.category == CAT[name])
        if not mask.any():
            continue
        h = grid.canopy_height[mask]
        cov = grid.coverage[mask]
        src = grid.source[mask]
        props = {
            "category": name,
            "area_m2": round(float(g.area), 1),
            "coverage_fraction": round(float(np.mean(cov)), 3),
            "canopy_height_mean": round(float(np.nanmean(h)), 2),
            "canopy_height_p10": round(float(np.nanpercentile(h, 10)), 2),
            "canopy_height_p90": round(float(np.nanpercentile(h, 90)), 2),
            "canopy_height_max": round(float(np.nanmax(h)), 2),
            "sources": {SOURCE_NAMES[int(s)]: round(float(np.mean(src == s)), 3) for s in np.unique(src)},
        }
        rc = max(0.8, 0.32 * max(props["canopy_height_mean"], 3.0) ** 0.85)
        props["density_stems_per_ha"] = round(props["coverage_fraction"] / (math.pi * rc * rc) * 1e4, 1)
        canopy_obs = float(np.mean(grid.canopy_observed[mask]))
        props["canopy_height_observed_fraction"] = round(canopy_obs, 3)
        weights = dict(cfg.categories[name].archetypes) if name in cfg.categories else {}
        lt = grid.leaf_type[mask]
        if (lt == 2).mean() > 0.5:
            weights = {"conifer": 0.9, "broadleaf": 0.1}
        elif (lt == 1).mean() > 0.5:
            weights = {"broadleaf": 0.95, "conifer": 0.05}
        reg = VegetationRegion(id=f"vegetation:{name}:{k}", geometry=g, properties=props,
                               source="derived", source_id="+".join(sorted(props["sources"])),
                               confidence=round(float(np.mean(grid.confidence[mask])), 3))
        reg.infer("archetype_distribution", weights, "category_defaults" if not (lt > 0).any() else "osm_leaf_type", 0.4)
        reg.mark_missing("species_distribution")
        if canopy_obs < 0.5:
            reg.uncertain = True
        out.append(reg)
    return out


def environmental_fields(grid: VegetationGrid, trees: TreeTable) -> dict[str, np.ndarray]:
    """Application-independent vegetation descriptors on the vegetation grid."""
    rows, cols = grid.shape
    cell_area = grid.resolution ** 2
    stems = np.zeros(grid.shape)
    basal = np.zeros(grid.shape)
    if len(trees):
        big = np.array([a in ("broadleaf", "conifer", "palm") for a in trees.archetype])
        rr, cc, inside = grid.cell_of(trees.x, trees.y)
        sel = big & inside
        np.add.at(stems, (rr[sel], cc[sel]), 1.0)
        np.add.at(basal, (rr[sel], cc[sel]), math.pi * (trees.trunk_radius[sel] + 0.4) ** 2)
    density = stems / cell_area * 1e4
    shrub = np.isin(grid.category, [CAT["shrubland"], CAT["wetland"]]) * grid.coverage
    obstruction = np.clip(shrub + basal / cell_area, 0.0, 1.0)
    h = np.minimum(np.nan_to_num(grid.canopy_height), 25.0)
    occlusion = 1.0 - np.exp(-1.5 * grid.coverage * h / 10.0)
    return {
        "canopy_height_m": grid.canopy_height.astype(np.float32),
        "canopy_coverage": grid.coverage.astype(np.float32),
        "vegetation_density": density.astype(np.float32),
        "ground_obstruction": obstruction.astype(np.float32),
        "visual_occlusion": occlusion.astype(np.float32),
        "confidence": grid.confidence.astype(np.float32),
        "vegetation_category": grid.category.astype(np.float32),
    }


def region_outline_pieces(regions: list[VegetationRegion], terrain: Heightfield, grid: TileGrid,
                          wanted: set[str] | None = None) -> list[Piece]:
    pieces = []
    for reg in regions:
        for poly in polygons_of(reg.geometry):
            for ring in [poly.exterior, *poly.interiors]:
                xy = np.asarray(ring.segmentize(4.0).coords)[:, :2]
                xy, = grid.insert_tile_crossings(xy)
                tiles = grid.segment_tiles(xy)
                start = 0
                for k in range(1, len(tiles) + 1):
                    if k == len(tiles) or tiles[k] != tiles[start]:
                        t = tiles[start]
                        if wanted is None or t in wanted:
                            seg = xy[start:k + 1]
                            inside = terrain.contains(seg[:, 0], seg[:, 1])
                            if inside.all():
                                z = terrain.sample(seg[:, 0], seg[:, 1]) + 0.6
                                pieces.append(Piece(t, "vegetation_regions", f"vegetation_{reg.properties['category']}",
                                                    reg.id, line_strip_mesh([np.column_stack([seg, z])])))
                        start = k
    return pieces


# --------------------------------------------------------------------------------------
# archetype meshes (unit space: base at origin, height 1, crown radius 1)

QUALITY = {"low": (0, 5), "medium": (1, 8), "high": (2, 12)}


def _scaled(m: Mesh, sx: float, sy: float, sz: float, dz: float) -> Mesh:
    p = m.positions * np.array([sx, sy, sz]) + np.array([0.0, 0.0, dz])
    n = None
    if m.normals is not None:
        n = m.normals / np.array([sx, sy, sz])
        n /= np.linalg.norm(n, axis=1, keepdims=True)
    return Mesh(p, m.indices, n, m.uvs, m.colors)


def archetype_meshes(quality: str) -> dict[str, dict[str, Mesh]]:
    """{archetype: {material: mesh}} in unit space."""
    sub, seg = QUALITY[quality]
    sphere = icosphere(sub)
    out: dict[str, dict[str, Mesh]] = {}
    out["broadleaf"] = {
        "tree_trunk": cylinder(0.07, 0.5, seg, radius_top=0.045),
        "foliage_broadleaf": _scaled(sphere, 1.0, 1.0, 0.33, 0.66),
    }
    cones = [cone(1.0, 0.5, seg + 1, 0.12), _scaled(cone(1.0, 0.45, seg + 1, 0.0), 0.72, 0.72, 1.0, 0.4),
             _scaled(cone(1.0, 0.42, seg + 1, 0.0), 0.45, 0.45, 1.0, 0.58)]
    out["conifer"] = {"tree_trunk": cylinder(0.06, 0.3, seg, radius_top=0.04), "foliage_conifer": merge_meshes(cones)}
    fronds = []
    for k in range(7):
        a = 2 * math.pi * k / 7
        leaf = _scaled(cone(1.0, 1.0, 4, 0.0), 0.12, 0.12, 1.0, 0.0)
        # Lay the cone down along +x, droop it, then rotate around the trunk.
        p = leaf.positions.copy()
        x, y, z = p[:, 2], p[:, 1], -p[:, 0]
        z = z - 0.25 * x * x + 0.0
        ca, sa = math.cos(a), math.sin(a)
        p = np.column_stack([ca * x - sa * y, sa * x + ca * y, z + 0.9])
        m = Mesh(p, leaf.indices)
        m.ensure_normals()
        fronds.append(m)
    out["palm"] = {"tree_trunk": cylinder(0.05, 0.92, seg, radius_top=0.035), "foliage_palm": merge_meshes(fronds)}
    out["shrub"] = {"foliage_shrub": _scaled(sphere, 1.0, 1.0, 0.5, 0.45)}
    quads = []
    for k in range(3):
        a = math.pi * k / 3
        ca, sa = math.cos(a), math.sin(a)
        p = np.array([[-ca, -sa, 0.0], [ca, sa, 0.0], [ca, sa, 1.0], [-ca, -sa, 1.0]])
        quads.append(Mesh(p, np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64),
                          normals=np.tile([0.0, 0.0, 1.0], (4, 1)),
                          uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)))
    out["grass"] = {"foliage_grass": merge_meshes(quads)}
    return out


def instance_transforms(trees: TreeTable, idx: np.ndarray, origin: tuple[float, float]) -> dict[str, np.ndarray]:
    """glTF-space TRANSLATION / ROTATION / SCALE arrays for EXT_mesh_gpu_instancing."""
    ox, oy = origin
    x = trees.x[idx] - ox
    y = trees.y[idx] - oy
    sink = 0.15 + trees.trunk_radius[idx]
    z = trees.ground_z[idx] - sink
    yaw = trees.yaw[idx]
    t = np.column_stack([x, z, -y]).astype(np.float32)
    r = np.column_stack([np.zeros_like(yaw), np.sin(yaw / 2), np.zeros_like(yaw), np.cos(yaw / 2)]).astype(np.float32)
    s = np.column_stack([trees.crown_radius[idx], trees.height[idx] + sink, trees.crown_radius[idx]]).astype(np.float32)
    return {"TRANSLATION": t, "ROTATION": r, "SCALE": s}


ARCHETYPE_ORDER = ARCHETYPES
