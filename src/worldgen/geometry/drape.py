"""Draping flat polygons onto the terrain triangulation.

A polygon is cut along the heightfield's triangles (each grid cell is split
along its (c, r)-(c+1, r+1) diagonal, exactly like the terrain mesh). Every
resulting piece lies inside a single terrain triangle, so lifting its vertices
to ``terrain + offset`` places it on a plane parallel to that triangle: the
surface follows the terrain everywhere and terrain can never poke through.
All steps are vectorized with shapely 2.
"""

from __future__ import annotations

import numpy as np
import shapely
from shapely.geometry.base import BaseGeometry

from worldgen.geometry.mesh import Mesh, empty_mesh
from worldgen.terrain.heightfield import Heightfield


def _terrain_triangles(hf: Heightfield, rows: np.ndarray, cols: np.ndarray) -> np.ndarray:
    """Both triangles of every (row, col) cell as shapely polygons, shape (2n,)."""
    r = hf.resolution
    x0 = hf.xmin + cols * r
    y0 = hf.ymin + rows * r
    x1, y1 = x0 + r, y0 + r
    lower = np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y0]), np.column_stack([x1, y1]),
                      np.column_stack([x0, y0])], axis=1)
    upper = np.stack([np.column_stack([x0, y0]), np.column_stack([x1, y1]), np.column_stack([x0, y1]),
                      np.column_stack([x0, y0])], axis=1)
    return shapely.polygons(np.concatenate([lower, upper]))


def drape_polygons(polys: list[BaseGeometry], hf: Heightfield, offset: float | np.ndarray,
                   uv_scale: float = 8.0, per_polygon: bool = False) -> Mesh | dict[int, Mesh]:
    """Upward-facing mesh of ``polys`` cut along the terrain triangles at ``terrain + offset``.

    ``offset`` may be one value or one value per polygon. With ``per_polygon`` a mesh per
    polygon index is returned (batched draping).
    """
    offs = np.broadcast_to(np.asarray(offset, dtype=float), (len(polys),))
    coords, owner = _polygon_triangles(polys, hf)
    if per_polygon:
        return _split(coords, owner, owner, offs, hf, uv_scale)
    if len(coords) == 0:
        return empty_mesh()
    return _finish(coords, offs[owner], hf, uv_scale)


def _split(coords: np.ndarray, owner: np.ndarray, g: np.ndarray, offs: np.ndarray, hf: Heightfield,
           uv_scale: float) -> dict[int, Mesh]:
    out: dict[int, Mesh] = {}
    if len(coords) == 0:
        return out
    order = np.argsort(g, kind="stable")
    g_sorted = g[order]
    starts = np.flatnonzero(np.r_[True, g_sorted[1:] != g_sorted[:-1]])
    ends = np.r_[starts[1:], len(g_sorted)]
    for a_, b_ in zip(starts, ends):
        sel = order[a_:b_]
        out[int(g_sorted[a_])] = _finish(coords[sel], offs[owner[sel]], hf, uv_scale)
    return out


def _polygon_triangles(polys: list[BaseGeometry], hf: Heightfield) -> tuple[np.ndarray, np.ndarray]:
    """General path: triangles (m, 3, 2) of the polygons cut by terrain triangles, and owner index."""
    empty = (np.zeros((0, 3, 2)), np.zeros(0, dtype=np.int64))
    ok_idx = [i for i, p in enumerate(polys) if p is not None and not p.is_empty and p.area > 1e-6]
    if not ok_idx:
        return empty
    geoms = np.array([polys[i] for i in ok_idx], dtype=object)
    remap = np.array(ok_idx, dtype=np.int64)
    bounds = shapely.bounds(geoms)
    pi, cells_r, cells_c = _candidate_cells(hf, bounds)
    tris = _terrain_triangles(hf, cells_r, cells_c)
    pi2 = np.concatenate([pi, pi])
    shapely.prepare(geoms)
    hit = shapely.intersects(geoms[pi2], tris)
    if not hit.any():
        return empty
    pieces = shapely.intersection(geoms[pi2[hit]], tris[hit], grid_size=None)
    owner = pi2[hit]
    parts, idx = shapely.get_parts(pieces, return_index=True)
    is_poly = shapely.get_type_id(parts) == 3
    parts, owner = parts[is_poly], owner[idx[is_poly]]
    if len(parts) == 0:
        return empty
    bad = ~shapely.is_valid(parts)
    if bad.any():
        fixed, fidx = shapely.get_parts(shapely.make_valid(parts[bad]), return_index=True)
        fpoly = shapely.get_type_id(fixed) == 3
        parts = np.concatenate([parts[~bad], fixed[fpoly]])
        owner = np.concatenate([owner[~bad], owner[bad][fidx[fpoly]]])
    cdt = shapely.constrained_delaunay_triangles(parts)
    tri_parts, tidx = shapely.get_parts(cdt, return_index=True)
    ok = shapely.get_type_id(tri_parts) == 3
    tri_parts, tidx = tri_parts[ok], tidx[ok]
    if len(tri_parts) == 0:
        return empty
    coords = shapely.get_coordinates(tri_parts).reshape(-1, 4, 2)[:, :3]
    return coords, remap[owner[tidx]]


def _finish(coords: np.ndarray, tri_off: np.ndarray, hf: Heightfield, uv_scale: float) -> Mesh:
    """Triangles (m, 3, 2) -> CCW, welded mesh lifted to terrain + per-triangle offset."""
    a, b, c = coords[:, 0], coords[:, 1], coords[:, 2]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    keep = np.abs(cross) > 1e-9
    coords, cross, tri_off = coords[keep], cross[keep], tri_off[keep]
    if len(coords) == 0:
        return empty_mesh()
    flip = cross < 0
    coords[flip] = coords[flip][:, [0, 2, 1]]
    pts = coords.reshape(-1, 2)
    off_v = np.repeat(tri_off, 3)
    # Weld identical vertices (same position and offset) so normals are shared.
    key = np.column_stack([np.round(pts, 6), np.round(off_v, 4)])
    uniq, inverse = np.unique(key, axis=0, return_inverse=True)
    xy = uniq[:, :2]
    z = hf.sample(xy[:, 0], xy[:, 1]) + uniq[:, 2]
    mesh = Mesh(np.column_stack([xy, z]), inverse.reshape(-1, 3).astype(np.int64), uvs=xy / uv_scale)
    mesh.ensure_normals()
    return mesh


def _candidate_cells(hf: Heightfield, bounds: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(owner, row, col) for every grid cell overlapped by each bounding box (n, 4)."""
    r = hf.resolution
    c0 = np.clip(np.floor((bounds[:, 0] - hf.xmin) / r).astype(np.int64), 0, hf.cols - 2)
    c1 = np.clip(np.floor((bounds[:, 2] - hf.xmin) / r).astype(np.int64), 0, hf.cols - 2)
    r0 = np.clip(np.floor((bounds[:, 1] - hf.ymin) / r).astype(np.int64), 0, hf.rows - 2)
    r1 = np.clip(np.floor((bounds[:, 3] - hf.ymin) / r).astype(np.int64), 0, hf.rows - 2)
    counts = (c1 - c0 + 1) * (r1 - r0 + 1)
    owner = np.repeat(np.arange(len(bounds)), counts)
    local = np.arange(int(counts.sum())) - np.repeat(np.cumsum(counts) - counts, counts)
    width = np.repeat(c1 - c0 + 1, counts)
    return owner, np.repeat(r0, counts) + local // width, np.repeat(c0, counts) + local % width


MAXV = 8


def drape_quads(quads: np.ndarray, hf: Heightfield, offset: float | np.ndarray, uv_scale: float = 8.0,
                groups: np.ndarray | None = None) -> Mesh | dict[int, Mesh]:
    """Fast path for convex quads (n, 4, 2): vectorized Sutherland-Hodgman clipping of each
    terrain triangle against each quad (non-convex quads use the general shapely path).

    With ``groups`` (an int per quad) one mesh per group is returned, so many ribbons can be
    draped in a single batch.
    """
    quads = np.asarray(quads, dtype=float)
    n = len(quads)
    offs = np.broadcast_to(np.asarray(offset, dtype=float), (n,)).copy()
    coords_parts: list[np.ndarray] = []
    owner_parts: list[np.ndarray] = []
    if n:
        # Orient counter-clockwise and separate non-convex/degenerate quads.
        area2 = sum((quads[:, k, 0] * quads[:, (k + 1) % 4, 1] - quads[:, (k + 1) % 4, 0] * quads[:, k, 1])
                    for k in range(4))
        quads = np.where((area2 < 0)[:, None, None], quads[:, ::-1], quads)
        e = np.roll(quads, -1, axis=1) - quads
        e2 = np.roll(e, -1, axis=1)
        turn = e[:, :, 0] * e2[:, :, 1] - e[:, :, 1] * e2[:, :, 0]
        convex = (turn >= -1e-9).all(axis=1) & (np.abs(area2) > 1e-6)
        bad_idx = np.nonzero(~convex)[0]
        if len(bad_idx):
            bad = [shapely.make_valid(shapely.polygons(np.vstack([q, q[:1]]))) for q in quads[bad_idx]]
            c, o = _polygon_triangles(bad, hf)
            coords_parts.append(c)
            owner_parts.append(bad_idx[o])
        good_idx = np.nonzero(convex)[0]
        if len(good_idx):
            q = quads[good_idx]
            bounds = np.column_stack([q[:, :, 0].min(1), q[:, :, 1].min(1), q[:, :, 0].max(1), q[:, :, 1].max(1)])
            owner, rows, cols = _candidate_cells(hf, bounds)
            r = hf.resolution
            v00 = np.column_stack([hf.xmin + cols * r, hf.ymin + rows * r])
            v10, v11, v01 = v00 + [r, 0.0], v00 + [r, r], v00 + [0.0, r]
            tris = np.concatenate([np.stack([v00, v10, v11], 1), np.stack([v00, v11, v01], 1)])
            own = np.concatenate([owner, owner])
            P, cnt = _clip(tris, q[own])
            for k in range(1, MAXV - 1):
                m = cnt > k + 1
                if m.any():
                    coords_parts.append(np.stack([P[m, 0], P[m, k], P[m, k + 1]], 1))
                    owner_parts.append(good_idx[own[m]])
    coords = np.concatenate(coords_parts) if coords_parts else np.zeros((0, 3, 2))
    owner = np.concatenate(owner_parts) if owner_parts else np.zeros(0, dtype=np.int64)
    if groups is None:
        return _finish(coords, offs[owner], hf, uv_scale) if len(coords) else empty_mesh()
    return _split(coords, owner, np.asarray(groups)[owner], offs, hf, uv_scale)


def _clip(subject: np.ndarray, clip: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Clip triangles (n, 3, 2) by CCW convex quads (n, 4, 2). Returns (n, MAXV, 2) and counts."""
    n = len(subject)
    P = np.zeros((n, MAXV, 2))
    P[:, :3] = subject
    cnt = np.full(n, 3)
    rows = np.arange(n)
    for e in range(4):
        a = clip[:, e]
        d = clip[:, (e + 1) % 4] - a
        out = np.zeros_like(P)
        ocnt = np.zeros(n, dtype=np.int64)

        def side(q: np.ndarray) -> np.ndarray:
            return d[:, 0] * (q[:, 1] - a[:, 1]) - d[:, 1] * (q[:, 0] - a[:, 0])

        for i in range(MAXV):
            valid = i < cnt
            if not valid.any():
                break
            cur = P[:, i]
            prev = P[rows, (i - 1) % np.maximum(cnt, 1)]
            dc, dp = side(cur), side(prev)
            cin, pin = dc >= 0, dp >= 0
            add_i = valid & (cin != pin)
            with np.errstate(divide="ignore", invalid="ignore"):
                t = np.where(add_i, dp / (dp - dc), 0.0)
            inter = prev + (cur - prev) * t[:, None]
            idx = np.minimum(ocnt, MAXV - 1)
            out[rows[add_i], idx[add_i]] = inter[add_i]
            ocnt = ocnt + add_i
            add_c = valid & cin
            idx = np.minimum(ocnt, MAXV - 1)
            out[rows[add_c], idx[add_c]] = cur[add_c]
            ocnt = ocnt + add_c
        P, cnt = out, np.minimum(ocnt, MAXV)
    return P, cnt
