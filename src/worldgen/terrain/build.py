"""Terrain construction: DEM resampling, DSM correction, vertical datum and terrain shaping."""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from affine import Affine
from rasterio import features as rfeatures
from scipy import ndimage
from shapely.geometry.base import BaseGeometry

from worldgen.config import WorldConfig
from worldgen.errors import SourceError
from worldgen.log import get_logger
from worldgen.spatial.raster import GeoRaster, rasterize_window
from worldgen.spatial.region import Region
from worldgen.spatial.windows import gauss_radius, grid_windows
from worldgen.terrain.heightfield import Heightfield, dsm_reach, dsm_to_dtm, fill_nodata

log = get_logger("terrain")


# Nodes per side of the windows the terrain is built in (memory follows the window).
WINDOW_NODES = 512


class _DemSampler:
    """Bilinear DEM heights at pixel centres, like ``GeoRaster.sample_bilinear``, from a copy
    of the DEM whose nodata pixels hold the nearest valid pixel's height and whose edge
    pixels extend outwards, so every node gets a height. ``valid`` marks the nodes all four
    of whose source pixels are valid data inside the raster (the source coverage)."""

    def __init__(self, dem: GeoRaster, src_crs: Any) -> None:
        self.dem = dem
        self.src_crs = src_crs
        ok = dem.valid_mask()
        data = dem.data.astype(np.float64)
        self.data = fill_nodata(np.where(ok, data, 0.0), ok, default=0.0)
        self.ok = ok

    def sample(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        cols, rows = self.dem._pixel_coords(x.ravel(), y.ravel(), self.src_crs)
        cols, rows = cols - 0.5, rows - 0.5
        h, w = self.data.shape
        c0, r0 = np.floor(cols).astype(np.int64), np.floor(rows).astype(np.int64)
        fc, fr = cols - c0, rows - r0
        c0c, c1c = np.clip(c0, 0, w - 1), np.clip(c0 + 1, 0, w - 1)
        r0c, r1c = np.clip(r0, 0, h - 1), np.clip(r0 + 1, 0, h - 1)
        d = self.data
        z = (d[r0c, c0c] * (1 - fc) * (1 - fr) + d[r0c, c1c] * fc * (1 - fr) + d[r1c, c0c] * (1 - fc) * fr
             + d[r1c, c1c] * fc * fr)
        ok = self.ok
        valid = ok[r0c, c0c] & ok[r0c, c1c] & ok[r1c, c0c] & ok[r1c, c1c]
        valid &= ~((cols < -0.5) | (rows < -0.5) | (cols > w - 0.5) | (rows > h - 0.5))
        return z.reshape(x.shape), valid.reshape(x.shape)


def build_heightfield(
    region: Region,
    cfg: WorldConfig,
    dem: GeoRaster | None,
    landcover: GeoRaster | None,
    dem_datum: str = "EGM2008",
    geoid_offset: Callable[[np.ndarray, np.ndarray], np.ndarray] | None = None,
    window: int = WINDOW_NODES,
) -> Heightfield:
    """The terrain heightfield from the DEM: bilinear heights, a light smoothing scaled to the
    DEM pixel, DSM-to-bare-earth correction and the vertical datum.

    Built window by window (``window`` nodes per side plus the halo the filters read); each
    node depends only on nodes within that halo, so any window size gives the same heights.
    DEM nodata is filled on the DEM's own grid, with the nearest valid pixel.
    """
    xmin, ymin, xmax, ymax = region.extent
    res = cfg.terrain.resolution
    cols = int(round((xmax - xmin) / res)) + 1
    rows = int(round((ymax - ymin) / res)) + 1
    hf = Heightfield(xmin, ymin, res, np.zeros((rows, cols)), vertical_datum=dem_datum)
    if dem is None:
        hf.valid = np.zeros((rows, cols), dtype=bool)
        hf.meta["source"] = "none (flat terrain at datum zero)"
        log.warning("no DEM available: terrain is flat and marked as missing coverage")
        return hf
    local_crs = region.frame.local_crs()
    sampler = _DemSampler(dem, local_crs)
    # Pixel-centre bilinear sampling of a coarse DEM leaves visible facets at fine
    # resolutions; a light smoothing scaled to the DEM pixel size removes them.
    dem_px = abs(dem.transform.a)
    if dem.crs.is_geographic:
        dem_px *= 111_320.0
    sigma = 0.35 * dem_px / res
    smooth = sigma > 0.5
    dc = cfg.terrain.dsm_correction
    dsm = dc.enabled and landcover is not None
    dsm_cells = int(round(dc.window_m / res))
    target = cfg.terrain.output_vertical_datum
    convert = target not in ("source", dem_datum)
    halo = (gauss_radius(sigma) if smooth else 0) + (dsm_reach(dsm_cells) if dsm else 0)
    z_out = np.empty((rows, cols))
    valid_out = np.empty((rows, cols), dtype=bool)
    masked = 0
    for win, core_w, core in grid_windows((rows, cols), window, halo):
        gx, gy = np.meshgrid(xmin + np.arange(win[1].start, win[1].stop) * res,
                             ymin + np.arange(win[0].start, win[0].stop) * res)
        z, valid = sampler.sample(gx, gy)
        valid_out[core] = valid[core_w]
        if smooth:
            z = ndimage.gaussian_filter(z, sigma=sigma, mode="nearest")
        if dsm:
            lc = landcover.sample_nearest(gx.ravel(), gy.ravel(), src_crs=local_crs).reshape(gx.shape)
            mask = np.isin(lc, dc.worldcover_classes)
            masked += int(mask[core_w].sum())
            z = dsm_to_dtm(z, mask, dsm_cells)
        if convert:
            z = convert_vertical_datum(z, gx, gy, region, dem_datum, target, geoid_offset)
        z_out[core] = z[core_w]
    hf.valid = valid_out
    hf.meta["source_coverage"] = float(valid_out.mean())
    if dsm:
        hf.meta["dsm_correction"] = {
            "method": "morphological_opening", "window_m": dc.window_m,
            "classes": dc.worldcover_classes, "masked_fraction": round(masked / (rows * cols), 4),
        }
    if convert:
        hf.vertical_datum = target
    hf.z = z_out
    return hf


def convert_vertical_datum(
    z: np.ndarray, gx: np.ndarray, gy: np.ndarray, region: Region, source: str, target: str,
    geoid_offset: Callable[[np.ndarray, np.ndarray], np.ndarray] | None,
) -> np.ndarray:
    """Convert orthometric (geoid) heights to/from ellipsoidal heights.

    ``geoid_offset(lon, lat)`` returns the geoid undulation N (metres). When not
    given, PROJ's EGM2008 grid is used (requires PROJ network access or the
    ``us_nga_egm08_25.tif`` grid installed).
    """
    lon, lat = region.frame.to_wgs84(gx.ravel(), gy.ravel())
    if geoid_offset is None:
        try:
            import pyproj
            from pyproj import Transformer

            pyproj.network.set_network_enabled(True)
            t = Transformer.from_crs("EPSG:4326+3855", "EPSG:4979", always_xy=True)
            _, _, h = t.transform(lon, lat, np.zeros_like(lon))
            n = np.asarray(h)
        except Exception as exc:  # pyproj raises several error types for missing grids
            raise SourceError(f"vertical datum conversion {source}->{target} unavailable: {exc}") from exc
    else:
        n = geoid_offset(lon, lat)
    n = np.asarray(n).reshape(z.shape)
    if source == "EGM2008" and target == "WGS84_ellipsoid":
        return z + n
    if source == "WGS84_ellipsoid" and target == "EGM2008":
        return z - n
    raise SourceError(f"unsupported vertical datum conversion {source} -> {target}")


# --------------------------------------------------------------------------------------
# terrain shaping


def _grid_transform(hf: Heightfield) -> Affine:
    """Affine mapping grid (col, row) pixel *centres* onto nodes, with row 0 = south."""
    r = hf.resolution
    return Affine(r, 0.0, hf.xmin - r / 2.0, 0.0, r, hf.ymin - r / 2.0)


def rasterize_mask(hf: Heightfield, geoms: list[BaseGeometry]) -> np.ndarray:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(hf.z.shape, dtype=bool)
    return rfeatures.rasterize(
        [(g, 1) for g in geoms], out_shape=hf.z.shape, transform=_grid_transform(hf), fill=0,
        all_touched=False, dtype="uint8",
    ).astype(bool)


def node_mask_window(hf: Heightfield, geom: BaseGeometry | None
                     ) -> tuple[tuple[slice, slice], np.ndarray] | None:
    """Grid nodes inside one geometry, over its bounding-box window only (see rasterize_window).

    Same nodes as ``rasterize_mask(hf, [geom])``; use it when rasterising many polygons one by
    one, where a full-grid mask per polygon would cost polygons x grid nodes.
    """
    return rasterize_window(geom, hf.z.shape, _grid_transform(hf))


@dataclass
class Corridor:
    """A centreline with per-vertex target elevation used to shape terrain."""

    xy: np.ndarray       # (n, 2)
    z: np.ndarray        # (n,)
    half_width: float
    shoulder: float
    mode: str = "set"    # "set": conform terrain; "max": only lower terrain to z


def _nearest_segment(px: np.ndarray, py: np.ndarray, xy: np.ndarray, z: np.ndarray
                     ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Squared distance to the nearest segment (first one on ties), its index and the z there."""
    a = xy[:-1]
    b = xy[1:]
    ab = b - a
    ab2 = np.maximum((ab ** 2).sum(axis=1), 1e-12)
    apx = px[:, None] - a[None, :, 0]
    apy = py[:, None] - a[None, :, 1]
    t = np.clip((apx * ab[None, :, 0] + apy * ab[None, :, 1]) / ab2[None, :], 0.0, 1.0)
    cx = a[None, :, 0] + t * ab[None, :, 0]
    cy = a[None, :, 1] + t * ab[None, :, 1]
    d2 = (px[:, None] - cx) ** 2 + (py[:, None] - cy) ** 2
    k = np.argmin(d2, axis=1)
    rows = np.arange(len(px))
    tz = z[:-1][k] + t[rows, k] * (z[1:][k] - z[:-1][k])
    return d2[rows, k], k, tz


# Segments projected together in corridor_nodes; their bounding box bounds the work per piece.
CORRIDOR_PIECE_SEGMENTS = 16


def corridor_nodes(hf: Heightfield, xy: np.ndarray, z: np.ndarray, reach: float
                   ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Grid nodes within ``reach`` of a polyline: (rows, cols, distance, z at the closest point).

    The polyline is projected in short pieces, each against the nodes of its own bounding box. A
    single box around a long diagonal way holds far more nodes than the corridor, and projecting
    each of them onto every segment grows with the cube of the length. Per node the nearest
    segment of the whole polyline wins (the first one on ties), so the result is the same as one
    projection onto the whole polyline.
    """
    rows, cols = hf.z.shape
    r = hf.resolution
    n_seg = len(xy) - 1
    parts: list[tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]] = []
    for s0 in range(0, n_seg, CORRIDOR_PIECE_SEGMENTS):
        s1 = min(s0 + CORRIDOR_PIECE_SEGMENTS, n_seg)
        pxy = xy[s0:s1 + 1]
        x0, y0 = pxy.min(axis=0) - reach
        x1, y1 = pxy.max(axis=0) + reach
        c0 = max(int(np.floor((x0 - hf.xmin) / r)), 0)
        c1 = min(int(np.ceil((x1 - hf.xmin) / r)), cols - 1)
        r0 = max(int(np.floor((y0 - hf.ymin) / r)), 0)
        r1 = min(int(np.ceil((y1 - hf.ymin) / r)), rows - 1)
        if c1 < c0 or r1 < r0:
            continue
        cc, rr = np.meshgrid(np.arange(c0, c1 + 1), np.arange(r0, r1 + 1))
        cc, rr = cc.ravel(), rr.ravel()
        d2, k, tz = _nearest_segment(hf.xmin + cc * r, hf.ymin + rr * r, pxy, z[s0:s1 + 1])
        near = np.sqrt(d2) <= reach
        if near.any():
            parts.append((rr[near] * cols + cc[near], d2[near], k[near] + s0, tz[near]))
    if not parts:
        empty = np.zeros(0, dtype=np.int64)
        return empty, empty, np.zeros(0), np.zeros(0)
    flat, d2, seg, tz = (np.concatenate(p) for p in zip(*parts))
    if len(parts) > 1:
        # A node reached by several pieces keeps its nearest segment (lowest index on ties).
        order = np.lexsort((seg, d2, flat))
        flat, d2, tz = flat[order], d2[order], tz[order]
        first = np.r_[True, flat[1:] != flat[:-1]]
        flat, d2, tz = flat[first], d2[first], tz[first]
    ri, ci = np.divmod(flat, cols)
    return ri, ci, np.sqrt(d2), tz


def apply_corridors(hf: Heightfield, corridors: list[Corridor]) -> int:
    """Shape terrain along corridors. Returns the number of modified grid nodes.

    Inside the half width the terrain takes the corridor elevation; in the shoulder band it
    blends back to the original (smoothstep). Where "set" corridors overlap (a road and its
    separately mapped sidewalks, dual carriageways) their targets are averaged with their
    blend weights, so the terrain never steps between neighbouring roads. "max" corridors
    only lower terrain; the most restrictive one wins.
    """
    z0 = hf.z.copy()
    acc_wt = np.zeros(hf.z.shape)
    acc_w = np.zeros(hf.z.shape)
    max_w = np.zeros(hf.z.shape)
    lowest = np.full(hf.z.shape, np.inf)
    touched = np.zeros(hf.z.shape, dtype=bool)
    for cor in corridors:
        if len(cor.xy) < 2:
            continue
        ext = cor.half_width + cor.shoulder
        ri, ci, d, tz = corridor_nodes(hf, cor.xy, cor.z, ext)
        if not len(ri):
            continue
        hw = cor.half_width
        w = np.where(d <= hw, 1.0, np.clip((ext - d) / max(ext - hw, 1e-6), 0.0, 1.0))
        w = w * w * (3 - 2 * w)  # smoothstep blend
        touched[ri, ci] = True
        if cor.mode == "max":
            base = z0[ri, ci]
            cand = base * (1 - w) + np.minimum(tz, base) * w
            np.minimum.at(lowest, (ri, ci), cand)
        else:
            ww = np.maximum(w, 1e-6)
            np.add.at(acc_wt, (ri, ci), ww * tz)
            np.add.at(acc_w, (ri, ci), ww)
            np.maximum.at(max_w, (ri, ci), w)
    if not touched.any():
        return 0
    has_set = acc_w > 0
    blended = np.where(has_set, acc_wt / np.where(has_set, acc_w, 1.0), z0)
    z = z0 * (1 - max_w) + blended * max_w
    z = np.minimum(z, lowest)
    hf.z[:] = z
    return int(touched.sum())


def carve_polygons(hf: Heightfield, items: list[tuple[BaseGeometry, float]]) -> int:
    """Lower terrain inside polygons to at most the given level. Returns nodes changed."""
    changed = 0
    for geom, level in items:
        hit = node_mask_window(hf, geom)
        if hit is None or not hit[1].any():
            continue
        win, mask = hit
        zw = hf.z[win]
        before = zw[mask]
        after = np.minimum(before, level)
        changed += int((after < before).sum())
        zw[mask] = after
    return changed


def polygon_boundary_samples(hf: Heightfield, geom: BaseGeometry, spacing: float) -> np.ndarray:
    """Terrain samples along the boundary of a polygon."""
    pts = []
    for poly in getattr(geom, "geoms", [geom]):
        ring = poly.exterior.segmentize(spacing)
        pts.append(np.asarray(ring.coords)[:, :2])
    if not pts:
        return np.zeros(0)
    p = np.concatenate(pts)
    inside = hf.contains(p[:, 0], p[:, 1])
    p = p[inside]
    return hf.sample(p[:, 0], p[:, 1]) if len(p) else np.zeros(0)


def landcover_grid(hf: Heightfield, rasters: list[tuple[list[BaseGeometry], int]]) -> np.ndarray:
    """uint8 class grid on terrain nodes from (geometries, code) layers, later layers win."""
    out = np.zeros(hf.z.shape, dtype=np.uint8)
    for geoms, code in rasters:
        m = rasterize_mask(hf, geoms)
        out[m] = code
    return out
