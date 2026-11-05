"""Regular-grid heightfield in the local frame, and its triangle-consistent sampler.

Row 0 of ``z`` is the southern edge (y = ymin), column 0 the western edge.
Each grid cell is split into two triangles along the (c, r)-(c+1, r+1)
diagonal; :meth:`Heightfield.sample` interpolates on exactly those triangles,
so sampled heights lie on the generated terrain mesh.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from affine import Affine
from scipy import ndimage

from worldgen.spatial.windows import box_filter

from worldgen.geometry.mesh import Mesh
from worldgen.spatial.windows import box_filter


@dataclass
class Heightfield:
    xmin: float
    ymin: float
    resolution: float
    z: np.ndarray                     # (rows, cols) float64, metres
    vertical_datum: str = "EGM2008"
    valid: np.ndarray | None = None   # (rows, cols) bool, source coverage
    meta: dict[str, Any] = field(default_factory=dict)

    @property
    def rows(self) -> int:
        return int(self.z.shape[0])

    @property
    def cols(self) -> int:
        return int(self.z.shape[1])

    @property
    def xmax(self) -> float:
        return self.xmin + (self.cols - 1) * self.resolution

    @property
    def ymax(self) -> float:
        return self.ymin + (self.rows - 1) * self.resolution

    def copy(self) -> Heightfield:
        return Heightfield(self.xmin, self.ymin, self.resolution, self.z.copy(), self.vertical_datum,
                           None if self.valid is None else self.valid.copy(), dict(self.meta))

    def node_coords(self) -> tuple[np.ndarray, np.ndarray]:
        xs = self.xmin + np.arange(self.cols) * self.resolution
        ys = self.ymin + np.arange(self.rows) * self.resolution
        return np.meshgrid(xs, ys)

    def geotransform(self) -> Affine:
        """North-up affine for GeoTIFF export (pixel-is-point aligned to grid nodes)."""
        r = self.resolution
        return Affine(r, 0.0, self.xmin - r / 2.0, 0.0, -r, self.ymax + r / 2.0)

    # -- sampling ------------------------------------------------------------------------
    def sample(self, x: Any, y: Any) -> np.ndarray:
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        fx = (x - self.xmin) / self.resolution
        fy = (y - self.ymin) / self.resolution
        c = np.clip(np.floor(fx).astype(np.int64), 0, self.cols - 2)
        r = np.clip(np.floor(fy).astype(np.int64), 0, self.rows - 2)
        u = np.clip(fx - c, 0.0, 1.0)
        v = np.clip(fy - r, 0.0, 1.0)
        z00 = self.z[r, c]
        z10 = self.z[r, c + 1]
        z01 = self.z[r + 1, c]
        z11 = self.z[r + 1, c + 1]
        lower = u >= v
        out = np.where(
            lower,
            z00 + u * (z10 - z00) + v * (z11 - z10),
            z00 + v * (z01 - z00) + u * (z11 - z01),
        )
        return out

    def slope_deg(self, x: Any, y: Any) -> np.ndarray:
        """Slope of the triangle containing each point, in degrees."""
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        fx = (x - self.xmin) / self.resolution
        fy = (y - self.ymin) / self.resolution
        c = np.clip(np.floor(fx).astype(np.int64), 0, self.cols - 2)
        r = np.clip(np.floor(fy).astype(np.int64), 0, self.rows - 2)
        u = fx - c
        v = fy - r
        z00 = self.z[r, c]
        z10 = self.z[r, c + 1]
        z01 = self.z[r + 1, c]
        z11 = self.z[r + 1, c + 1]
        lower = u >= v
        dzdx = np.where(lower, z10 - z00, z11 - z01) / self.resolution
        dzdy = np.where(lower, z11 - z10, z01 - z00) / self.resolution
        return np.degrees(np.arctan(np.hypot(dzdx, dzdy)))

    def contains(self, x: Any, y: Any) -> np.ndarray:
        x = np.asarray(x)
        y = np.asarray(y)
        return (x >= self.xmin) & (x <= self.xmax) & (y >= self.ymin) & (y <= self.ymax)

    def index_of(self, x: float, y: float) -> tuple[int, int]:
        return int(round((y - self.ymin) / self.resolution)), int(round((x - self.xmin) / self.resolution))

    def vertex_normals(self) -> np.ndarray:
        """Central-difference normals at every node, (rows, cols, 3)."""
        return _normals(self.z, self.resolution)

    def normals_window(self, r0: int, r1: int, c0: int, c1: int) -> np.ndarray:
        """``vertex_normals()[r0:r1 + 1, c0:c1 + 1]`` from a window with a one-node margin.

        Central differences only reach one node, and the window is clipped exactly where the
        grid ends (one-sided differences there, as in the whole grid), so the values are
        identical while the cost follows the window instead of the whole grid.
        """
        wr0, wr1 = max(r0 - 1, 0), min(r1 + 1, self.rows - 1)
        wc0, wc1 = max(c0 - 1, 0), min(c1 + 1, self.cols - 1)
        n = _normals(self.z[wr0:wr1 + 1, wc0:wc1 + 1], self.resolution)
        return n[r0 - wr0:r1 - wr0 + 1, c0 - wc0:c1 - wc0 + 1]

    # -- meshing -------------------------------------------------------------------------
    def tile_mesh(self, x0: float, y0: float, x1: float, y1: float, stride: int = 1,
                  colors: np.ndarray | None = None, normals: np.ndarray | None = None) -> Mesh:
        """Mesh of the grid nodes inside [x0, x1] x [y0, y1] (shared edges are identical)."""
        r = self.resolution
        c0 = int(round((x0 - self.xmin) / r))
        c1 = int(round((x1 - self.xmin) / r))
        r0 = int(round((y0 - self.ymin) / r))
        r1 = int(round((y1 - self.ymin) / r))
        c0, r0 = max(c0, 0), max(r0, 0)
        c1, r1 = min(c1, self.cols - 1), min(r1, self.rows - 1)
        cs = np.arange(c0, c1 + 1, stride)
        rs = np.arange(r0, r1 + 1, stride)
        if cs[-1] != c1:
            cs = np.append(cs, c1)
        if rs[-1] != r1:
            rs = np.append(rs, r1)
        nc, nr = len(cs), len(rs)
        if nc < 2 or nr < 2:
            from worldgen.geometry.mesh import empty_mesh

            return empty_mesh()
        cc, rr = np.meshgrid(cs, rs)
        xs = self.xmin + cc * r
        ys = self.ymin + rr * r
        zs = self.z[rr, cc]
        pos = np.column_stack([xs.ravel(), ys.ravel(), zs.ravel()])
        if normals is not None:
            nor = normals[rr, cc].reshape(-1, 3)
        else:
            nor = self.normals_window(r0, r1, c0, c1)[rr - r0, cc - c0].reshape(-1, 3)
        k = (np.arange(nr - 1)[:, None] * nc + np.arange(nc - 1)[None, :]).ravel()
        v00, v10, v01, v11 = k, k + 1, k + nc, k + nc + 1
        tri = np.concatenate([np.column_stack([v00, v10, v11]), np.column_stack([v00, v11, v01])])
        uvs = np.column_stack([(xs.ravel() - x0) / max(x1 - x0, 1e-9), (ys.ravel() - y0) / max(y1 - y0, 1e-9)])
        col = None
        if colors is not None:
            col = colors[rr, cc].reshape(-1, colors.shape[-1])
        return Mesh(pos, tri.astype(np.int64), normals=nor, uvs=uvs, colors=col)


def _normals(z: np.ndarray, resolution: float) -> np.ndarray:
    """Central-difference unit normals (one-sided at the array border), (rows, cols, 3)."""
    gy, gx = np.gradient(z, resolution)
    n = np.dstack([-gx, -gy, np.ones_like(z)])
    return n / np.linalg.norm(n, axis=2, keepdims=True)


def fill_nodata(z: np.ndarray, valid: np.ndarray, default: float = 0.0) -> np.ndarray:
    """Fill invalid cells with the nearest valid value (or ``default`` if nothing is valid)."""
    if valid.all():
        return z
    if not valid.any():
        return np.full_like(z, default)
    idx = ndimage.distance_transform_edt(~valid, return_distances=False, return_indices=True)
    return z[tuple(idx)]


def dsm_to_dtm(z: np.ndarray, mask: np.ndarray, window_cells: int) -> np.ndarray:
    """Approximate bare earth from a surface model with a morphological opening.

    Applied only where ``mask`` is true (built-up or tree-covered cells), then
    blended with a small smoothing so no steps appear at mask edges.
    """
    if window_cells < 3 or not mask.any():
        return z
    size = window_cells if window_cells % 2 else window_cells + 1
    opened = ndimage.grey_opening(z, size=(size, size))
    opened = box_filter(opened, max(3, size // 3))
    weight = box_filter(mask.astype(float), max(3, size // 2))
    out = z * (1.0 - weight) + np.minimum(opened, z) * weight
    return out


def dsm_reach(window_cells: int) -> int:
    """Cells around a node that ``dsm_to_dtm`` reads."""
    if window_cells < 3:
        return 0
    size = window_cells if window_cells % 2 else window_cells + 1
    return 2 * (size // 2) + max(3, size // 3) // 2 + 1
