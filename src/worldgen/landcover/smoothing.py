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
from worldgen.spatial.windows import gauss_radius, grid_windows
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
    """Displacement (rows, cols) of each node's sampling position, in cells; None = identity.

    Sampling is bilinear with the weights taken from the displacement alone (its fractional
    part), so a node gets bit-identical values in any window of the grid that holds the
    nodes around its sampling position."""

    rows: np.ndarray | None
    cols: np.ndarray | None

    def apply(self, f: np.ndarray) -> np.ndarray:
        if self.rows is None:
            return f
        nr, nc = f.shape
        fr, fc = np.floor(self.rows), np.floor(self.cols)
        tr, tc = (self.rows - fr).astype(f.dtype), (self.cols - fc).astype(f.dtype)
        r0 = np.arange(nr)[:, None] + fr.astype(np.int64)
        c0 = np.arange(nc)[None, :] + fc.astype(np.int64)
        r1, c1 = np.clip(r0 + 1, 0, nr - 1), np.clip(c0 + 1, 0, nc - 1)
        r0, c0 = np.clip(r0, 0, nr - 1), np.clip(c0, 0, nc - 1)
        one = f.dtype.type(1)
        top = f[r0, c0] * (one - tc) + f[r0, c1] * tc
        bottom = f[r1, c0] * (one - tc) + f[r1, c1] * tc
        return (top * (one - tr) + bottom * tr).astype(f.dtype)


def _grid_warp(xmin: float, ymin: float, res: float, shape: tuple[int, int], centred: bool,
               cfg: SmoothingConfig, seed: int, rows: slice | None = None, cols: slice | None = None) -> _Warp:
    """Warp of a grid whose nodes (or cell centres) are world points, for the nodes
    ``[rows, cols]`` (default: all): world-space noise, so any window gets the same values."""
    if cfg.warp_amplitude_m <= 0:
        return _Warp(None, None)
    rows = rows or slice(0, shape[0])
    cols = cols or slice(0, shape[1])
    off = 0.5 if centred else 0.0
    xs = xmin + (np.arange(cols.start, cols.stop, dtype=float) + off) * res
    R = np.empty((rows.stop - rows.start, len(xs)), dtype=np.float32)
    C = np.empty_like(R)
    step = 256  # row blocks keep the noise temporaries small
    for r0 in range(rows.start, rows.stop, step):
        ri = np.arange(r0, min(r0 + step, rows.stop), dtype=float)
        X, Y = np.meshgrid(xs, ymin + (ri + off) * res)
        dx, dy = warp_offsets(X, Y, cfg.warp_amplitude_m, cfg.warp_wavelength_m, seed)
        R[r0 - rows.start:r0 - rows.start + len(ri)] = dy / res
        C[r0 - rows.start:r0 - rows.start + len(ri)] = dx / res
    return _Warp(R, C)


# Nodes no participating node reaches (no smoothed membership) take the class of the nearest
# participating node up to this far; farther ones keep their own class.
MAX_FILL_M = 64.0
# Nodes per side of the windows the smoothing works in (memory follows the window).
WINDOW_NODES = 512


def _class_reach(res: float, sigma_m: float, cfg: SmoothingConfig) -> int:
    """Cells around a node that its smoothed class (and a changed cell's donor) depends on."""
    warp = int(np.ceil(cfg.warp_amplitude_m / res)) + 1 if cfg.warp_amplitude_m > 0 else 0
    reach = gauss_radius(sigma_m / res) + warp + 1
    return max(int(np.ceil(reach * 1.5)), int(np.ceil(MAX_FILL_M / res))) + 1


def _nearest(mask: np.ndarray, sel: np.ndarray, max_cells: float | None = None) -> tuple[np.ndarray, ...]:
    """For each True cell of ``sel``: (rows, cols, sel rows, sel cols) of the nearest True cell
    of ``mask`` (Euclidean), ties broken by the smallest row offset, then column offset, so any
    window holding both cells finds the same one. Cells with none within ``max_cells`` are
    left out."""
    rs, cs = np.nonzero(sel)
    empty = np.zeros(0, dtype=np.int64)
    if not len(rs) or not mask.any():
        return empty, empty, empty, empty
    d2 = np.rint(ndimage.distance_transform_edt(~mask)[rs, cs] ** 2).astype(np.int64)
    nr, nc = mask.shape
    found = np.zeros(len(rs), dtype=bool)
    out_r = np.zeros(len(rs), dtype=np.int64)
    out_c = np.zeros(len(rs), dtype=np.int64)
    for D in np.unique(d2):
        if max_cells is not None and D > max_cells * max_cells:
            continue
        todo = np.nonzero(d2 == D)[0]
        r = int(np.sqrt(D)) + 1
        for dy in range(-r, r + 1):
            rest = D - dy * dy
            if rest < 0:
                continue
            dx0 = int(round(np.sqrt(rest)))
            for dx in sorted({-dx0, dx0}):
                if dx * dx != rest or not len(todo):
                    continue
                rr, cc = rs[todo] + dy, cs[todo] + dx
                hit = (rr >= 0) & (rr < nr) & (cc >= 0) & (cc < nc)
                hit[hit] = mask[rr[hit], cc[hit]]
                k = todo[hit]
                out_r[k], out_c[k], found[k] = rr[hit], cc[hit], True
                todo = todo[~hit]
    return out_r[found], out_c[found], rs[found], cs[found]


def smooth_classes(codes: np.ndarray, participate: np.ndarray, target: np.ndarray, res: float,
                   sigma_m: float, warp: _Warp | None = None, track: set[int] | None = None
                   ) -> tuple[np.ndarray, np.ndarray | None]:
    """Class of the largest smoothed (and warped) membership.

    ``participate`` marks the nodes whose class counts; ``target`` the nodes that receive a
    new class (everything else keeps its code). Target nodes that do not participate (fixed
    classes elsewhere, or holes to fill) take the classes around them; beyond the smoothing's
    reach, the class of the nearest participating node within ``MAX_FILL_M``, else their own.
    When ``track`` is given, the summed membership of those classes is returned as well (0..1).
    Each node depends only on the nodes within ``_class_reach`` cells.
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
    lost = target & ~ok
    if lost.any():
        best_code[lost] = codes[lost]
        ir, ic, lr, lc = _nearest(participate, lost, MAX_FILL_M / res)
        best_code[lr, lc] = codes[ir, ic]
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


def _within(mask: np.ndarray, radius_cells: float) -> np.ndarray:
    """Nodes within ``radius_cells`` (Euclidean) of a True node of ``mask``."""
    r = int(np.floor(radius_cells))
    if r <= 0:
        return mask.copy()
    yy, xx = np.mgrid[-r:r + 1, -r:r + 1]
    return ndimage.binary_dilation(mask, structure=(yy * yy + xx * xx) <= radius_cells * radius_cells)


def smooth_landcover(codes: np.ndarray, hf: Heightfield, water_areas: list[BaseGeometry],
                     waterways: list[BaseGeometry], cfg: SmoothingConfig, seed: int,
                     window: int = WINDOW_NODES) -> SmoothedLandcover:
    """Smooth land-cover borders on the terrain grid and blend the terrain colours.

    ``water_areas`` are the mapped water polygons (they decide water exactly);
    ``waterways`` are mapped water lines already buffered to their width. Masks are made
    for the whole grid; the smoothing and the colours are computed window by window
    (``window`` nodes per side, with the halo each needs), which gives the same values as
    one whole-grid window.
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
            labels, _ = ndimage.label(stray, structure=np.ones((3, 3), dtype=bool))
            hit = np.unique(labels[stray & _within(mapped, cfg.water_snap_m / res)])
            fill = np.isin(labels, hit[hit > 0])
            del labels
            water &= ~fill
    land = ~water & ~fill
    target = land | fill
    new = codes.copy()
    for win, core_w, core in grid_windows(codes.shape, window, _class_reach(res, cfg.border_radius_m, cfg)):
        warp = _grid_warp(hf.xmin, hf.ymin, res, codes.shape, False, cfg, seed, *win)
        out, _ = smooth_classes(codes[win], land[win], target[win], res, cfg.border_radius_m, warp)
        new[core] = out[core_w]
    stats = {"landcover_nodes_changed": int((new != codes).sum()), "water_mismatch_nodes": int(fill.sum())}

    sigma = cfg.blend_width_m / 4.0 / res
    # Bank colours blur the blended land colours: the two reaches add up.
    halo = max(gauss_radius(sigma) + gauss_radius(max(2.0, sigma)), 4) + 1
    rgba = np.empty(codes.shape + (4,), dtype=np.float32)
    for win, core_w, core in grid_windows(codes.shape, window, halo):
        rgba[core] = _colors(new[win], res, sigma)[core_w]
    return SmoothedLandcover(new, rgba, stats)


def _colors(new: np.ndarray, res: float, sigma: float) -> np.ndarray:
    """RGBA of a window of smoothed codes (exact from the colour halo inwards)."""
    lut = code_lut()
    colors = lut[new]
    land = new != WATER_CODE
    if sigma >= 0.3:
        wl = land.astype(np.float32)
        den = ndimage.gaussian_filter(wl, sigma, mode="nearest")
        blended = np.stack([ndimage.gaussian_filter(colors[..., k] * wl, sigma, mode="nearest")
                            for k in range(3)], axis=-1)
        colors = np.where(land[..., None], blended / np.maximum(den, 1e-6)[..., None], colors)
    sub = ~land
    if sub.any() and land.any():
        # Submerged nodes: the wet bank colour near the shore, the bed colour further out (it
        # no longer changes beyond three cells from the shore).
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
    return np.concatenate([colors, np.ones(new.shape + (1,), dtype=np.float32)], axis=-1).astype(np.float32)


# --------------------------------------------------------------------------------------
# vegetation grid

VEGETATION_CHANNELS = ("source", "canopy_height", "canopy_observed", "coverage", "confidence", "tree_fraction_wc",
                       "leaf_type")


def refine_vegetation(grid: VegetationGrid, water_areas: list[BaseGeometry], cfg: SmoothingConfig, seed: int,
                      window: int = WINDOW_NODES) -> VegetationGrid:
    """Refined vegetation grid with smooth, irregular category borders and thinned forest edges.

    Cells keep the channels (canopy height, coverage, source, ...) of their coarse cell; a
    cell the smoothing moved to another category takes them from the nearest cell of that
    category (one always lies within the smoothing's reach); cells on water become "none". Computed window by window like
    ``smooth_landcover``, with the same result as one whole-grid window.
    """
    f = max(1, int(cfg.vegetation_upsample))
    res = grid.resolution / f

    def up(a: np.ndarray) -> np.ndarray:
        return np.repeat(np.repeat(a, f, axis=0), f, axis=1) if f > 1 else a.copy()

    shape = (grid.shape[0] * f, grid.shape[1] * f)
    water = _rasterize(shape, Affine(res, 0.0, grid.xmin, 0.0, res, grid.ymin), water_areas)
    # Coarse cells cleared because their centre was in water: their fine cells off the water
    # take the vegetation around them.
    coarse_water = _rasterize(grid.shape, grid.transform(), water_areas)
    tree_codes = {CAT[c] for c in TREE_CATEGORIES}
    out = {"category": np.empty(shape, dtype=np.uint8)}
    for name in VEGETATION_CHANNELS:
        out[name] = np.empty(shape, dtype=getattr(grid, name).dtype)
    changed_total = 0
    for win, core_w, core in grid_windows(shape, window, _class_reach(res, cfg.border_radius_m, cfg), align=f):
        cwin = (slice(win[0].start // f, -(-win[0].stop // f)), slice(win[1].start // f, -(-win[1].stop // f)))
        cat0 = up(grid.category[cwin])
        water_w = water[win]
        fill = up(coarse_water[cwin]) & ~water_w
        participate = ~water_w & ~fill
        warp = _grid_warp(grid.xmin, grid.ymin, res, shape, True, cfg, seed, *win)
        cat, m_tree = smooth_classes(cat0, participate, participate | fill, res, cfg.border_radius_m, warp,
                                     track=tree_codes)
        chans = {name: up(getattr(grid, name)[cwin]) for name in VEGETATION_CHANNELS}
        # Cells the smoothing moved to another category take that category's channels from its
        # nearest cell (within the smoothing's reach); cells on water become "none" and keep
        # their own channels.
        moved = (cat != cat0) & ~water_w
        pristine = {k: v.copy() for k, v in chans.items()} if moved.any() else chans
        for c in np.unique(cat[moved]):
            ir, ic, sr, sc = _nearest((cat0 == c) & participate, moved & (cat == c))
            for k, a in chans.items():
                a[sr, sc] = pristine[k][ir, ic]
        cat[water_w] = 0
        changed = cat != cat0
        if changed.any():
            none = changed & (cat == 0)
            chans["canopy_height"][none] = 0.0
            chans["coverage"][none] = 0.0
        if m_tree is not None and cfg.forest_edge_thinning > 0:
            trees = np.isin(cat, list(tree_codes))
            keep = 1.0 - cfg.forest_edge_thinning * (1.0 - _smoothstep((m_tree - 0.5) / 0.4))
            chans["coverage"] = np.where(trees, chans["coverage"] * keep, chans["coverage"]).astype(np.float32)
        out["category"][core] = cat[core_w]
        for name in VEGETATION_CHANNELS:
            out[name][core] = chans[name][core_w]
        changed_total += int(changed[core_w].sum())
    meta = dict(grid.meta)
    meta["smoothing"] = {"upsample": f, "source_resolution": grid.resolution,
                         "cells_changed": changed_total, "border_radius_m": cfg.border_radius_m}
    return replace(grid, resolution=res, meta=meta, **out)


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


def _near_other_side(inside: np.ndarray, radius_cells: float) -> np.ndarray:
    """Nodes within ``radius_cells`` (Euclidean) of a node on the other side of ``inside``.

    The nearest node of the other side always has a neighbour (8-connected) on this side, so
    marking a disk around the boundary nodes of each side finds exactly those nodes, with
    work and memory following the outline rather than the grid."""
    rows, cols = inside.shape
    near = np.zeros(inside.shape, dtype=bool)
    k = int(np.floor(radius_cells))
    offsets = [(dy, dx) for dy in range(-k, k + 1) for dx in range(-k, k + 1)
               if dy * dy + dx * dx <= radius_cells * radius_cells]
    ring = np.ones((3, 3), dtype=bool)
    for side in (inside, ~inside):
        br, bc = np.nonzero(side & ndimage.binary_dilation(~side, structure=ring))
        for dy, dx in offsets:
            r, c = br + dy, bc + dx
            ok = (r >= 0) & (r < rows) & (c >= 0) & (c < cols)
            r, c = r[ok], c[ok]
            near[r[~side[r, c]], c[~side[r, c]]] = True
    return near


def _shore_blur(shape: tuple[int, int], rr: np.ndarray, cc: np.ndarray, est: np.ndarray, value: np.ndarray,
                sigma: float) -> tuple[np.ndarray, np.ndarray]:
    """At nodes (rr, cc): the Gaussian blur of ``value`` placed at the ``est`` nodes (zero
    elsewhere) and of their indicator, as whole-grid ``gaussian_filter``s would give, computed
    over windows around the nodes only."""
    halo = gauss_radius(sigma)
    G = np.zeros(len(rr))
    W = np.zeros(len(rr))
    er, ec, ev = rr[est], cc[est], value[est]
    for win, core_w, core in grid_windows(shape, WINDOW_NODES, halo):
        here = (rr >= core[0].start) & (rr < core[0].stop) & (cc >= core[1].start) & (cc < core[1].stop)
        if not here.any():
            continue
        r0, c0 = win[0].start, win[1].start
        inw = (er >= r0) & (er < win[0].stop) & (ec >= c0) & (ec < win[1].stop)
        g = np.zeros((win[0].stop - r0, win[1].stop - c0))
        w = np.zeros_like(g)
        g[er[inw] - r0, ec[inw] - c0] = ev[inw]
        w[er[inw] - r0, ec[inw] - c0] = 1.0
        g = ndimage.gaussian_filter(g, sigma, mode="nearest")
        w = ndimage.gaussian_filter(w, sigma, mode="nearest")
        G[here] = g[rr[here] - r0, cc[here] - c0]
        W[here] = w[rr[here] - r0, cc[here] - c0]
    return G, W


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
    # Nodes within reach + 1.5 cells of the other side of the outline (on the grid).
    rr, cc = np.nonzero(_near_other_side(inside, (reach + 1.5 * r) / r))
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
    Gs, Ws = _shore_blur(hf.z.shape, rr, cc, est, graw, 2.0)
    default = float(np.clip(0.15, slope_min, slope_max))
    g = np.where(Ws > 1e-3, Gs / np.maximum(Ws, 1e-9), default)
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
