"""In-memory georeferenced rasters and point sampling."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from affine import Affine
from pyproj import CRS, Transformer


@dataclass
class GeoRaster:
    """A single-band raster. ``data`` row 0 is the *top* row (GDAL convention).

    Nodata cells are NaN for float rasters; for integer rasters ``nodata`` gives
    the sentinel value.
    """

    data: np.ndarray
    transform: Affine
    crs: CRS
    nodata: float | int | None = None
    name: str = ""
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.data.shape  # type: ignore[return-value]

    def valid_mask(self) -> np.ndarray:
        if np.issubdtype(self.data.dtype, np.floating):
            m = np.isfinite(self.data)
            if self.nodata is not None and not np.isnan(self.nodata):
                m &= self.data != self.nodata
            return m
        if self.nodata is None:
            return np.ones(self.data.shape, dtype=bool)
        return self.data != self.nodata

    def _pixel_coords(self, xs: np.ndarray, ys: np.ndarray, src_crs: CRS | None) -> tuple[np.ndarray, np.ndarray]:
        if src_crs is not None and not CRS.from_user_input(src_crs).equals(self.crs):
            t = Transformer.from_crs(src_crs, self.crs, always_xy=True)
            xs, ys = t.transform(xs, ys)
        inv = ~self.transform
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        cols = inv.a * xs + inv.b * ys + inv.c
        rows = inv.d * xs + inv.e * ys + inv.f
        return cols, rows

    def sample_bilinear(self, xs: Any, ys: Any, src_crs: CRS | None = None) -> np.ndarray:
        """Bilinear sample at pixel centres; NaN outside the raster or where any neighbour is nodata."""
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        cols, rows = self._pixel_coords(xs, ys, src_crs)
        cols = cols - 0.5
        rows = rows - 0.5
        h, w = self.data.shape
        data = self.data.astype(np.float64)
        valid = self.valid_mask()
        data = np.where(valid, data, np.nan)
        c0 = np.floor(cols).astype(np.int64)
        r0 = np.floor(rows).astype(np.int64)
        fc = cols - c0
        fr = rows - r0
        c0c = np.clip(c0, 0, w - 1)
        c1c = np.clip(c0 + 1, 0, w - 1)
        r0c = np.clip(r0, 0, h - 1)
        r1c = np.clip(r0 + 1, 0, h - 1)
        v00 = data[r0c, c0c]
        v01 = data[r0c, c1c]
        v10 = data[r1c, c0c]
        v11 = data[r1c, c1c]
        out = (v00 * (1 - fc) * (1 - fr) + v01 * fc * (1 - fr) + v10 * (1 - fc) * fr + v11 * fc * fr)
        outside = (cols < -0.5) | (rows < -0.5) | (cols > w - 0.5) | (rows > h - 0.5)
        out = np.where(outside, np.nan, out)
        return out

    def sample_nearest(self, xs: Any, ys: Any, src_crs: CRS | None = None, fill: Any = None) -> np.ndarray:
        xs = np.asarray(xs, dtype=float)
        ys = np.asarray(ys, dtype=float)
        cols, rows = self._pixel_coords(xs, ys, src_crs)
        h, w = self.data.shape
        c = np.floor(cols).astype(np.int64)
        r = np.floor(rows).astype(np.int64)
        inside = (c >= 0) & (c < w) & (r >= 0) & (r < h)
        out = np.empty(xs.shape, dtype=self.data.dtype)
        fill_value = fill if fill is not None else (self.nodata if self.nodata is not None else 0)
        out[...] = fill_value
        out[inside] = self.data[r[inside], c[inside]]
        return out

    def coverage_fraction(self) -> float:
        return float(self.valid_mask().mean()) if self.data.size else 0.0


# --------------------------------------------------------------------------------------
# windowed rasterisation


def raster_window(transform: Affine, shape: tuple[int, int], bounds: tuple[float, float, float, float],
                  pad: int = 1) -> tuple[slice, slice] | None:
    """(rows, cols) of an axis-aligned grid whose pixels can meet ``bounds`` (x0, y0, x1, y1), padded."""
    cols = sorted(((bounds[0] - transform.c) / transform.a, (bounds[2] - transform.c) / transform.a))
    rows = sorted(((bounds[1] - transform.f) / transform.e, (bounds[3] - transform.f) / transform.e))
    c0 = max(int(np.floor(cols[0])) - pad, 0)
    c1 = min(int(np.ceil(cols[1])) + pad, shape[1])
    r0 = max(int(np.floor(rows[0])) - pad, 0)
    r1 = min(int(np.ceil(rows[1])) + pad, shape[0])
    if c1 <= c0 or r1 <= r0:
        return None
    return slice(r0, r1), slice(c0, c1)


def rasterize_window(geom: Any, shape: tuple[int, int], transform: Affine
                     ) -> tuple[tuple[slice, slice], np.ndarray] | None:
    """Pixel-centre mask of one geometry, rasterised only over its bounding-box window.

    Returns ``((rows, cols), mask)`` with ``mask`` covering ``grid[rows, cols]``, or None when the
    geometry is empty or outside the grid. The window transform is the grid transform shifted by
    whole pixels, so every pixel centre is the same point as in a full-grid rasterisation: one
    polygon costs its own area instead of the whole grid.
    """
    from rasterio import features as rfeatures

    if geom is None or geom.is_empty:
        return None
    win = raster_window(transform, shape, geom.bounds)
    if win is None:
        return None
    rows, cols = win
    mask = rfeatures.rasterize([(geom, 1)], out_shape=(rows.stop - rows.start, cols.stop - cols.start),
                               transform=transform @ Affine.translation(cols.start, rows.start), fill=0,
                               all_touched=False, dtype="uint8").astype(bool)
    return win, mask
