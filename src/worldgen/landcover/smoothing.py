"""Post-processing: natural borders for land cover, vegetation and shorelines.

Land cover reaches the compiler as source pixels: ESA WorldCover (10 m) is sampled
nearest-neighbour onto the terrain and vegetation grids, OSM polygons are rasterised
node by node and water is carved per grid node. Drawn as they are, every border is a
staircase of source pixels, single misclassified pixels show up as square specks and
the waterline zig-zags along the terrain triangles.

This stage takes the grid out of those borders without moving them:

* **Class borders** (:func:`smooth_classes`): every class becomes a membership field
  (Gaussian of its indicator, normalised over the participating nodes), the fields are
  displaced by smooth world-space noise (a domain warp) and each node takes the class
  with the largest membership. Borders become smooth curves with a natural
  irregularity; specks smaller than about ``2 * border_radius_m`` vanish.
* **Terrain colours** (:func:`smooth_landcover`): class colours are blended over
  ``blend_width_m`` across borders instead of switching between two grid nodes.
  WorldCover water just outside mapped water (a registration mismatch that paints blue
  staircases on the bank) takes the surrounding land class, and submerged nodes take
  the wet colour of the nearby bank.
* **Vegetation** (:func:`refine_vegetation`): the vegetation grid is refined, its
  category borders are smoothed the same way (same noise, so trees and ground colours
  agree) and tree density thins out towards forest edges.
* **Shorelines** (:func:`shape_shorelines`, run during terrain shaping because roads
  and buildings must follow it): near the water outline the terrain is made linear in
  the signed distance to the outline, so the water surface meets the terrain on the
  outline instead of on a grid staircase.

Everything is deterministic: the noise is a hash of world coordinates and the seed.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace

import numpy as np
import shapely
from affine import Affine
from scipy import ndimage
from shapely.geometry.base import BaseGeometry
from shapely.strtree import STRtree

from worldgen.config import SmoothingConfig
from worldgen.landcover.landcover import CODE_COLORS
from worldgen.noise import _lattice, value_noise, warp_offsets  # noqa: F401  (re-exported)
from worldgen.terrain.build import carve_polygons, rasterize_mask
from worldgen.terrain.heightfield import Heightfield
from worldgen.vegetation.classify import CAT, TREE_CATEGORIES, VegetationGrid, _rasterize

WATER_CODE = 7
# Bank colours darken where they are wet (just below the waterline).
WET_FACTOR = 0.72


# World-space noise (value_noise, warp_offsets) lives in worldgen.noise, shared with the
# vegetation classification.


# --------------------------------------------------------------------------------------
# class borders


@dataclass
class _Warp:
    """Sampling positions (fractional row, col) of a warped grid; None = identity."""

    rows: np.ndarray | None
    cols: np.ndarray | None

    def apply(self, f: np.ndarray) -> np.ndarray:
        if self.rows is None:
            return f
        return ndimage.map_coordinates(f, [self.rows, self.cols], order=1, mode="nearest").astype(f.dtype)


def _grid_warp(xmin: float, ymin: float, res: float, shape: tuple[int, int], centred: bool,
               cfg: SmoothingConfig, seed: int) -> _Warp:
    """Warped sampling positions of a grid whose nodes (or cell centres) are world points."""
    if cfg.warp_amplitude_m <= 0:
        return _Warp(None, None)
    rows, cols = shape
    off = 0.5 if centred else 0.0
    ci = np.arange(cols, dtype=float)
    xs = xmin + (ci + off) * res
    R = np.empty(shape, dtype=np.float32)
    C = np.empty(shape, dtype=np.float32)
    step = 256  # row blocks keep the noise temporaries small on large grids
    for r0 in range(0, rows, step):
        ri = np.arange(r0, min(r0 + step, rows), dtype=float)
        X, Y = np.meshgrid(xs, ymin + (ri + off) * res)
        dx, dy = warp_offsets(X, Y, cfg.warp_amplitude_m, cfg.warp_wavelength_m, seed)
        R[r0:r0 + len(ri)] = ri[:, None] + dy / res
        C[r0:r0 + len(ri)] = ci[None, :] + dx / res
    return _Warp(R, C)


def smooth_classes(codes: np.ndarray, participate: np.ndarray, target: np.ndarray, res: float,
                   sigma_m: float, warp: _Warp | None = None, track: set[int] | None = None
                   ) -> tuple[np.ndarray, np.ndarray | None]:
    """Class of the largest smoothed (and warped) membership.

    ``participate`` marks the nodes whose class counts; ``target`` the nodes that receive a
    new class (everything else keeps its code). Target nodes that do not participate (fixed
    classes elsewhere, or holes to fill) take the classes around them. When ``track`` is
    given, the summed membership of those classes is returned as well (0..1).
    """
    out = codes.copy()
    warp = warp or _Warp(None, None)
    present = np.unique(codes[participate])
    if not target.any() or len(present) == 0:
        return out, None
    sigma = sigma_m / res
    w = participate.astype(np.float32)

    def blur(a: np.ndarray) -> np.ndarray:
        return ndimage.gaussian_filter(a, sigma, mode="nearest") if sigma >= 0.3 else a

    den = warp.apply(blur(w))
    ok = den > 1e-6
    inv = np.where(ok, 1.0 / np.maximum(den, 1e-6), 0.0).astype(np.float32)
    best = np.full(codes.shape, -1.0, dtype=np.float32)
    best_code = np.zeros(codes.shape, dtype=codes.dtype)
    tracked = np.zeros(codes.shape, dtype=np.float32) if track is not None else None
    for c in present:
        m = warp.apply(blur(((codes == c) & participate).astype(np.float32))) * inv
        better = m > best
        best = np.where(better, m, best)
        best_code = np.where(better, c, best_code)
        if tracked is not None and int(c) in track:
            tracked += m
    # Nodes beyond the reach of every participating node: nearest participating class.
    lost = target & ~ok
    if lost.any():
        _, (ir, ic) = ndimage.distance_transform_edt(~participate, return_indices=True)
        best_code[lost] = codes[ir[lost], ic[lost]]
    out[target] = best_code[target]
    if tracked is not None:
        tracked = np.clip(tracked, 0.0, 1.0)
    return out, tracked


def _smoothstep(t: np.ndarray) -> np.ndarray:
    t = np.clip(t, 0.0, 1.0)
    return t * t * (3.0 - 2.0 * t)


# --------------------------------------------------------------------------------------
# terrain land cover


@dataclass
class SmoothedLandcover:
    codes: np.ndarray                 # (rows, cols) uint8 land-cover codes
    colors: np.ndarray                # (rows, cols, 4) float32 RGBA vertex colours
    stats: dict = field(default_factory=dict)


def code_lut() -> np.ndarray:
    lut = np.zeros((256, 3), dtype=np.float32)
    lut[:] = CODE_COLORS[0]
    for k, c in CODE_COLORS.items():
        lut[k] = c
    return lut


def smooth_landcover(codes: np.ndarray, hf: Heightfield, water_areas: list[BaseGeometry],
                     waterways: list[BaseGeometry], cfg: SmoothingConfig, seed: int) -> SmoothedLandcover:
    """Smooth land-cover borders on the terrain grid and blend the terrain colours.

    ``water_areas`` are the mapped water polygons (they decide water exactly);
    ``waterways`` are mapped water lines already buffered to their width.
    """
    res = hf.resolution
    codes = codes.astype(np.uint8)
    osm_water = rasterize_mask(hf, water_areas)
    water = codes == WATER_CODE
    fill = np.zeros(codes.shape, dtype=bool)
    stray = water & ~osm_water
    if stray.any() and cfg.water_snap_m > 0:
        mapped = osm_water | rasterize_mask(hf, waterways)
        if mapped.any():
            # A WorldCover water patch that reaches this close to mapped water is the same water
            # body seen differently (registration, tide): the whole patch is reassigned, so no
            # strip of it is left behind. Patches far from mapped water are kept.
            dist = ndimage.distance_transform_edt(~mapped) * res
            labels, _ = ndimage.label(stray, structure=np.ones((3, 3), dtype=bool))
            hit = np.unique(labels[stray & (dist <= cfg.water_snap_m)])
            fill = np.isin(labels, hit[hit > 0])
            water &= ~fill
    land = ~water & ~fill
    warp = _grid_warp(hf.xmin, hf.ymin, res, codes.shape, False, cfg, seed)
    new, _ = smooth_classes(codes, land, land | fill, res, cfg.border_radius_m, warp)
    stats = {"landcover_nodes_changed": int((new != codes).sum()), "water_mismatch_nodes": int(fill.sum())}

    lut = code_lut()
    colors = lut[new]
    land = new != WATER_CODE
    sigma = cfg.blend_width_m / 4.0 / res
    if sigma >= 0.3:
        wl = land.astype(np.float32)
        den = ndimage.gaussian_filter(wl, sigma, mode="nearest")
        blended = np.stack([ndimage.gaussian_filter(colors[..., k] * wl, sigma, mode="nearest")
                            for k in range(3)], axis=-1)
        colors = np.where(land[..., None], blended / np.maximum(den, 1e-6)[..., None], colors)
    sub = ~land
    if sub.any() and land.any():
        # Submerged nodes: the wet bank colour near the shore, the bed colour further out.
        s2 = max(2.0, sigma)
        wl = land.astype(np.float32)
        den = ndimage.gaussian_filter(wl, s2, mode="nearest")
        bank = np.stack([ndimage.gaussian_filter(colors[..., k] * wl, s2, mode="nearest") for k in range(3)],
                        axis=-1) / np.maximum(den, 1e-6)[..., None]
        depth = ndimage.distance_transform_edt(sub) * res
        t = _smoothstep(depth / (3.0 * res))[..., None]
        t = np.where((den > 1e-3)[..., None], t, 1.0)
        wet = bank * WET_FACTOR * (1.0 - t) + lut[WATER_CODE] * t
        colors = np.where(sub[..., None], wet, colors)
    rgba = np.concatenate([colors, np.ones(codes.shape + (1,), dtype=np.float32)], axis=-1).astype(np.float32)
    return SmoothedLandcover(new, rgba, stats)


# --------------------------------------------------------------------------------------
# vegetation grid


def refine_vegetation(grid: VegetationGrid, water_areas: list[BaseGeometry], cfg: SmoothingConfig, seed: int
                      ) -> VegetationGrid:
    """Refined vegetation grid with smooth, irregular category borders and thinned forest edges.

    Cells keep the channels (canopy height, coverage, source, ...) of their coarse cell; a
    cell whose category changed takes them from the nearest cell of its new category.
    """
    f = max(1, int(cfg.vegetation_upsample))
    res = grid.resolution / f

    def up(a: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(a, f, axis=0), f, axis=1) if f > 1 else a.copy()

    cat0 = up(grid.category)
    shape = cat0.shape
    water = _rasterize(shape, Affine(res, 0.0, grid.xmin, 0.0, res, grid.ymin), water_areas)
    # Coarse cells cleared because their centre was in water: their fine cells off the water
    # take the vegetation around them.
    coarse_water = up(_rasterize(grid.shape, grid.transform(), water_areas))
    fill = coarse_water & ~water
    participate = ~water & ~fill
    warp = _grid_warp(grid.xmin, grid.ymin, res, shape, True, cfg, seed)
    tree_codes = {CAT[c] for c in TREE_CATEGORIES}
    cat, m_tree = smooth_classes(cat0, participate, participate | fill, res, cfg.border_radius_m, warp,
                                 track=tree_codes)
    cat[water] = 0

    chans = {name: up(getattr(grid, name)) for name in
             ("source", "canopy_height", "canopy_observed", "coverage", "confidence", "tree_fraction_wc",
              "leaf_type")}
    changed = cat != cat0
    pristine = {k: v.copy() for k, v in chans.items()} if changed.any() else chans
    for c in np.unique(cat[changed]):
        sel = changed & (cat == c)
        donors = (cat0 == c) & participate
        if not donors.any():
            continue
        _, (ir, ic) = ndimage.distance_transform_edt(~donors, return_indices=True)
        for k, a in chans.items():
            a[sel] = pristine[k][ir[sel], ic[sel]]
    if changed.any():
        none = changed & (cat == 0)
        chans["canopy_height"][none] = 0.0
        chans["coverage"][none] = 0.0
    trees = np.isin(cat, list(tree_codes))
    if m_tree is not None and cfg.forest_edge_thinning > 0:
        keep = 1.0 - cfg.forest_edge_thinning * (1.0 - _smoothstep((m_tree - 0.5) / 0.4))
        chans["coverage"] = np.where(trees, chans["coverage"] * keep, chans["coverage"]).astype(np.float32)
    meta = dict(grid.meta)
    meta["smoothing"] = {"upsample": f, "source_resolution": grid.resolution,
                         "cells_changed": int(changed.sum()), "border_radius_m": cfg.border_radius_m}
    return replace(grid, resolution=res, category=cat.astype(np.uint8), meta=meta, **chans)


# --------------------------------------------------------------------------------------
# shorelines


def _segments(geom: BaseGeometry) -> np.ndarray:
    """(n, 2, 2) array of the straight segments of a (multi)line geometry."""
    out = []
    for part in shapely.get_parts(geom):
        if part.is_empty or part.geom_type not in ("LineString", "LinearRing"):
            continue
        c = np.asarray(part.coords)[:, :2]
        if len(c) >= 2:
            out.append(np.stack([c[:-1], c[1:]], axis=1))
    return np.concatenate(out) if out else np.zeros((0, 2, 2))


def shape_shorelines(hf: Heightfield, items: list[tuple[BaseGeometry, float, float]],
                     extent: BaseGeometry | None, slope_min: float, slope_max: float) -> int:
    """Carve water areas so the terrain meets the water surface exactly on their outline.

    ``items`` are ``(area, surface elevation, bed elevation)``. Inside the areas the terrain
    is lowered to the bed (as plain carving does). Within about 1.5 cells of the outline
    it is set to ``surface + g * d`` with ``d`` the signed distance to the outline (negative
    in the water); the bank gradient ``g`` is estimated from the original terrain just
    outside and smoothed along the shore, so gentle banks stay gentle and steep ones steep.
    Because the terrain is then linear in ``d`` on both sides, the terrain triangles cross
    the water surface on the outline instead of on a grid staircase. Further out the
    terrain blends back to its original shape. Returns the number of nodes changed.
    """
    items = [(g, float(s), float(b)) for g, s, b in items if g is not None and not g.is_empty]
    if not items:
        return 0
    r = hf.resolution
    z0 = hf.z.copy()
    geoms = [g for g, _, _ in items]
    union = shapely.union_all(geoms)
    inside = rasterize_mask(hf, [union])
    carve_polygons(hf, [(g, b) for g, _, b in items])
    shore = union.boundary
    if extent is not None:
        # The region edge cuts water areas (the sea); that cut is not a shoreline.
        shore = shore.difference(extent.exterior.buffer(0.25 * r))
    segs = _segments(shore)
    if len(segs) == 0:
        return int((hf.z != z0).sum())
    band, ramp, shoulder = 1.5 * r, 3.0 * r, 2.0 * r
    reach = band + max(ramp, shoulder)
    grid_d = np.where(inside, ndimage.distance_transform_edt(inside), ndimage.distance_transform_edt(~inside)) * r
    rr, cc = np.nonzero(grid_d <= reach + 1.5 * r)
    if len(rr) == 0:
        return int((hf.z != z0).sum())
    px = hf.xmin + cc * r
    py = hf.ymin + rr * r
    pts = shapely.points(px, py)
    (pi, si), dist = STRtree(shapely.linestrings(segs)).query_nearest(
        pts, max_distance=reach + 2.0 * r, return_distance=True, all_matches=False)
    d = np.full(len(px), reach + r)
    d[pi] = dist
    d = np.minimum(d, reach + r)
    # Nodes on an outline count as water (the rasteriser keeps only some of them), so the
    # outer node row of a sea cut by the region edge is carved too.
    wet = shapely.intersects_xy(union, px, py)
    sd = np.where(wet, -d, d)
    # Surface and bed of the nearest (containing) water area.
    (qi, ai) = STRtree(geoms).query_nearest(pts, all_matches=False)
    k = np.zeros(len(px), dtype=np.int64)
    k[qi] = ai
    surf = np.array([s for _, s, _ in items])[k]
    bed = np.array([b for _, _, b in items])[k]

    zo = z0[rr, cc]
    # Bank gradient from the original terrain just off the shore, smoothed along it.
    est = (sd > 0.5 * r) & (sd <= band + shoulder)
    graw = np.clip((zo - surf) / np.maximum(sd, 1e-6), slope_min, slope_max)
    G = np.zeros(hf.z.shape)
    W = np.zeros(hf.z.shape)
    G[rr[est], cc[est]] = graw[est]
    W[rr[est], cc[est]] = 1.0
    G = ndimage.gaussian_filter(G, 2.0, mode="nearest")
    W = ndimage.gaussian_filter(W, 2.0, mode="nearest")
    default = float(np.clip(0.15, slope_min, slope_max))
    g = np.where(W[rr, cc] > 1e-3, G[rr, cc] / np.maximum(W[rr, cc], 1e-9), default)
    g = np.clip(g, slope_min, slope_max)

    lin = surf + g * sd
    out = hf.z[rr, cc].copy()
    m = np.abs(sd) <= band
    out[m] = lin[m]
    m = sd < -band
    edge = surf - g * band
    t = _smoothstep((-sd - band) / ramp)
    out[m] = np.minimum(zo, edge + (bed - edge) * t)[m]
    m = sd > band
    t = _smoothstep((sd - band) / shoulder)
    out[m] = (lin * (1.0 - t) + zo * t)[m]
    hf.z[rr, cc] = out
    return int((hf.z != z0).sum())
