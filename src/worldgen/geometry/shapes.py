"""Mesh generators: polygon caps, extrusions, polyline sweeps and solid primitives."""

from __future__ import annotations

import math

import numpy as np
import shapely
from shapely.geometry import LineString, MultiPolygon, Polygon
from shapely.geometry.base import BaseGeometry
from shapely.geometry.polygon import orient

from worldgen.geometry.mesh import Mesh, empty_mesh, merge_meshes


def polygons_of(g: BaseGeometry) -> list[Polygon]:
    if g is None or g.is_empty:
        return []
    if isinstance(g, Polygon):
        return [g]
    if isinstance(g, MultiPolygon):
        return list(g.geoms)
    return [p for p in getattr(g, "geoms", []) if isinstance(p, Polygon)]


def triangulate_polygon(poly: Polygon) -> tuple[np.ndarray, np.ndarray]:
    """Constrained Delaunay triangulation of a polygon (holes respected, no Steiner points).

    Returns (vertices (n, 2), triangles (m, 3)) with counter-clockwise triangles.
    """
    if poly.is_empty or poly.area <= 0:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    tris = shapely.constrained_delaunay_triangles(poly)
    coords_list = [np.asarray(t.exterior.coords)[:3] for t in tris.geoms]
    if not coords_list:
        return np.zeros((0, 2)), np.zeros((0, 3), dtype=np.int64)
    pts = np.concatenate(coords_list)
    # Weld identical vertices (exact coordinates come from the polygon itself).
    keys = np.round(pts, 9)
    uniq, inverse = np.unique(keys, axis=0, return_inverse=True)
    tri = inverse.reshape(-1, 3).astype(np.int64)
    verts = uniq
    a, b, c = verts[tri[:, 0]], verts[tri[:, 1]], verts[tri[:, 2]]
    cross = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    flip = cross < 0
    tri[flip] = tri[flip][:, [0, 2, 1]]
    keep = np.abs(cross) > 1e-10
    return verts, tri[keep]


def flat_polygon_mesh(poly: Polygon, z: float | np.ndarray, uv_scale: float = 10.0) -> Mesh:
    """Upward-facing polygon at constant z (or per-vertex z from a callable array)."""
    v2, tri = triangulate_polygon(poly)
    if len(tri) == 0:
        return empty_mesh()
    zz = np.full(len(v2), z) if np.isscalar(z) else np.asarray(z)
    pos = np.column_stack([v2, zz])
    return Mesh(pos, tri, normals=np.tile([0.0, 0.0, 1.0], (len(v2), 1)), uvs=v2 / uv_scale)


def draped_polygon_mesh(poly: Polygon, zfun: "callable", offset: float, uv_scale: float = 10.0,
                        densify: float | None = None) -> Mesh:
    if densify:
        poly = poly.segmentize(densify)
    v2, tri = triangulate_polygon(poly)
    if len(tri) == 0:
        return empty_mesh()
    z = zfun(v2[:, 0], v2[:, 1]) + offset
    m = Mesh(np.column_stack([v2, z]), tri, uvs=v2 / uv_scale)
    m.ensure_normals()
    return m


def _ring_walls(ring: np.ndarray, zb: np.ndarray, zt: np.ndarray, uv_scale: float) -> Mesh:
    """Vertical quads along a closed ring (first point not repeated). zb/zt per ring vertex.

    Normals point to the right of the ring direction: outward for a CCW exterior,
    and into the hole (away from the solid) for a CW interior ring.
    """
    n = len(ring)
    pos, nor, uvs, idx = [], [], [], []
    s = 0.0
    base = 0
    for i in range(n):
        j = (i + 1) % n
        p0, p1 = ring[i], ring[j]
        d = p1 - p0
        length = float(math.hypot(d[0], d[1]))
        if length < 1e-6:
            continue
        nx, ny = d[1] / length, -d[0] / length
        quad = [
            (p0[0], p0[1], zb[i]), (p1[0], p1[1], zb[j]),
            (p1[0], p1[1], zt[j]), (p0[0], p0[1], zt[i]),
        ]
        pos.extend(quad)
        nor.extend([(nx, ny, 0.0)] * 4)
        uvs.extend([(s / uv_scale, zb[i] / uv_scale), ((s + length) / uv_scale, zb[j] / uv_scale),
                    ((s + length) / uv_scale, zt[j] / uv_scale), (s / uv_scale, zt[i] / uv_scale)])
        idx.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
        base += 4
        s += length
    if not pos:
        return empty_mesh()
    return Mesh(np.array(pos), np.array(idx, dtype=np.int64), np.array(nor), np.array(uvs))


def extrude_polygon(poly: Polygon, z_bottom: float, z_top: float, uv_scale: float = 4.0,
                    bottom_cap: bool = True, top_cap: bool = True) -> Mesh:
    """Closed prism: walls, top cap (up) and bottom cap (down)."""
    poly = orient(poly, sign=1.0)
    parts: list[Mesh] = []
    for ring in [poly.exterior, *poly.interiors]:
        r = np.asarray(ring.coords)[:-1, :2]
        parts.append(_ring_walls(r, np.full(len(r), z_bottom), np.full(len(r), z_top), uv_scale))
    if top_cap:
        parts.append(flat_polygon_mesh(poly, z_top, uv_scale))
    if bottom_cap:
        bot = flat_polygon_mesh(poly, z_bottom, uv_scale)
        if not bot.is_empty():
            bot.indices = bot.indices[:, [0, 2, 1]]
            bot.normals = -bot.normals
            parts.append(bot)
    return merge_meshes(parts)


def walls_to_roof(poly: Polygon, z_bottom: float, roof_z: "callable", uv_scale: float = 4.0) -> Mesh:
    """Walls whose top follows ``roof_z(x, y)`` at every ring vertex."""
    poly = orient(poly, sign=1.0)
    parts = []
    for ring in [poly.exterior, *poly.interiors]:
        r = np.asarray(ring.coords)[:-1, :2]
        zt = roof_z(r[:, 0], r[:, 1])
        parts.append(_ring_walls(r, np.full(len(r), z_bottom), zt, uv_scale))
    return merge_meshes(parts)


# --------------------------------------------------------------------------------------
# polyline sweeps


def polyline_frames(xy: np.ndarray, miter_limit: float = 3.0) -> tuple[np.ndarray, np.ndarray]:
    """Per-vertex left-pointing offset directions and miter scale factors."""
    n = len(xy)
    d = np.diff(xy, axis=0)
    seg_len = np.linalg.norm(d, axis=1)
    seg_len[seg_len < 1e-9] = 1e-9
    t = d / seg_len[:, None]
    left_seg = np.column_stack([-t[:, 1], t[:, 0]])
    left = np.zeros((n, 2))
    scale = np.ones(n)
    left[0] = left_seg[0]
    left[-1] = left_seg[-1]
    if n > 2:
        avg = left_seg[:-1] + left_seg[1:]
        norm = np.linalg.norm(avg, axis=1)
        ok = norm > 1e-6
        avg[ok] /= norm[ok, None]
        avg[~ok] = left_seg[:-1][~ok]
        left[1:-1] = avg
        cos_half = np.einsum("ij,ij->i", avg, left_seg[1:])
        scale[1:-1] = np.minimum(1.0 / np.maximum(cos_half, 1e-3), miter_limit)
    return left, scale


def ribbon(xy: np.ndarray, z: np.ndarray, width: float | np.ndarray, s0: float = 0.0,
           uv_scale: float | None = None, start_offset: np.ndarray | None = None,
           end_offset: np.ndarray | None = None) -> Mesh:
    """Flat upward strip of ``width`` centred on the polyline (cross-section horizontal).

    ``start_offset``/``end_offset`` replace the left-edge offset vector at the first/last
    station (used for mitred joints so adjacent ribbons share their end vertices).
    """
    n = len(xy)
    if n < 2:
        return empty_mesh()
    left, scale = polyline_frames(xy)
    half = (np.asarray(width) / 2.0) * scale if np.ndim(width) else (width / 2.0) * scale
    offset = left * np.reshape(half, (-1, 1))
    if start_offset is not None:
        offset[0] = start_offset
    if end_offset is not None:
        offset[-1] = end_offset
    lpos = xy + offset
    rpos = xy - offset
    pos = np.empty((2 * n, 3))
    pos[0::2, :2] = lpos
    pos[1::2, :2] = rpos
    pos[0::2, 2] = z
    pos[1::2, 2] = z
    s = s0 + np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    w = float(np.max(width)) if np.ndim(width) else float(width)
    sc = uv_scale or max(w, 1.0)
    uvs = np.empty((2 * n, 2))
    uvs[0::2, 0] = 0.0
    uvs[1::2, 0] = 1.0
    uvs[0::2, 1] = s / sc
    uvs[1::2, 1] = s / sc
    k = np.arange(n - 1) * 2
    # left_i, right_i, right_{i+1} / left_i, right_{i+1}, left_{i+1}: CCW from above.
    idx = np.concatenate([np.column_stack([k, k + 1, k + 3]), np.column_stack([k, k + 3, k + 2])])
    m = Mesh(pos, idx.astype(np.int64), uvs=uvs)
    m.ensure_normals()
    return m


def swept_box(xy: np.ndarray, z_top: np.ndarray, width: float, thickness: float,
              inward: bool = False) -> Mesh:
    """Closed rectangular sweep (bridge deck, railing). ``inward`` flips normals (tunnel tube)."""
    n = len(xy)
    if n < 2:
        return empty_mesh()
    left, scale = polyline_frames(xy)
    half = width / 2.0 * scale
    lxy = xy + left * half[:, None]
    rxy = xy - left * half[:, None]
    zt = np.asarray(z_top, dtype=float)
    zb = zt - thickness
    # Four profile corners: left-top, right-top, right-bottom, left-bottom.
    corners = [(lxy, zt), (rxy, zt), (rxy, zb), (lxy, zb)]
    faces = []
    for a, b in [(0, 1), (1, 2), (2, 3), (3, 0)]:
        (pa, za), (pb, zb_) = corners[a], corners[b]
        pos = np.empty((2 * n, 3))
        pos[0::2, :2] = pa
        pos[0::2, 2] = za
        pos[1::2, :2] = pb
        pos[1::2, 2] = zb_
        k = np.arange(n - 1) * 2
        idx = np.concatenate([np.column_stack([k, k + 1, k + 3]), np.column_stack([k, k + 3, k + 2])])
        s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
        uvs = np.empty((2 * n, 2))
        uvs[0::2, 0] = 0.0
        uvs[1::2, 0] = 1.0
        uvs[0::2, 1] = s / 4.0
        uvs[1::2, 1] = s / 4.0
        faces.append(Mesh(pos, idx.astype(np.int64), uvs=uvs))
    # End caps.
    for end, flip in ((0, True), (n - 1, False)):
        quad = np.array([[*corners[c][0][end], corners[c][1][end]] for c in range(4)])
        idx = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int64)
        if flip:
            idx = idx[:, [0, 2, 1]]
        faces.append(Mesh(quad, idx, uvs=np.array([[0, 0], [1, 0], [1, 1], [0, 1]], dtype=float)))
    if inward:
        for f in faces:
            f.indices = f.indices[:, [0, 2, 1]]
    m = merge_meshes(faces)
    # Flat-ish normals per face strip are computed per strip; recompute per strip only.
    return m


def box(cx: float, cy: float, z0: float, z1: float, sx: float, sy: float, yaw: float = 0.0) -> Mesh:
    hx, hy = sx / 2.0, sy / 2.0
    c, s = math.cos(yaw), math.sin(yaw)
    pts = np.array([[-hx, -hy], [hx, -hy], [hx, hy], [-hx, hy]])
    rot = pts @ np.array([[c, s], [-s, c]])
    poly = Polygon(rot + [cx, cy])
    return extrude_polygon(poly, z0, z1, uv_scale=2.0)


def cylinder(radius: float, height: float, segments: int, z0: float = 0.0, cap: bool = True,
             radius_top: float | None = None) -> Mesh:
    rt = radius if radius_top is None else radius_top
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    ca, sa = np.cos(ang), np.sin(ang)
    pos, nor, idx = [], [], []
    slope = (radius - rt) / max(height, 1e-9)
    for i in range(segments):
        j = (i + 1) % segments
        b = len(pos)
        for a_ in (i, j):
            n = np.array([ca[a_], sa[a_], slope])
            n /= np.linalg.norm(n)
            pos.append((radius * ca[a_], radius * sa[a_], z0))
            nor.append(n)
        for a_ in (j, i):
            n = np.array([ca[a_], sa[a_], slope])
            n /= np.linalg.norm(n)
            pos.append((rt * ca[a_], rt * sa[a_], z0 + height))
            nor.append(n)
        idx.extend([(b, b + 1, b + 2), (b, b + 2, b + 3)])
    if cap and rt > 0:
        c0 = len(pos)
        pos.append((0.0, 0.0, z0 + height))
        nor.append((0.0, 0.0, 1.0))
        for i in range(segments):
            pos.append((rt * ca[i], rt * sa[i], z0 + height))
            nor.append((0.0, 0.0, 1.0))
        for i in range(segments):
            idx.append((c0, c0 + 1 + i, c0 + 1 + (i + 1) % segments))
    return Mesh(np.array(pos, dtype=float), np.array(idx, dtype=np.int64), np.array(nor, dtype=float))


def cone(radius: float, height: float, segments: int, z0: float = 0.0) -> Mesh:
    ang = np.linspace(0, 2 * math.pi, segments, endpoint=False)
    pos, nor, idx = [], [], []
    k = radius / max(height, 1e-9)
    for i in range(segments):
        j = (i + 1) % segments
        b = len(pos)
        for a_ in (ang[i], ang[j]):
            n = np.array([math.cos(a_), math.sin(a_), k])
            n /= np.linalg.norm(n)
            pos.append((radius * math.cos(a_), radius * math.sin(a_), z0))
            nor.append(n)
        mid = (ang[i] + ang[j]) / 2
        n = np.array([math.cos(mid), math.sin(mid), k])
        n /= np.linalg.norm(n)
        pos.append((0.0, 0.0, z0 + height))
        nor.append(n)
        idx.append((b, b + 1, b + 2))
    # Bottom disc (downward).
    c0 = len(pos)
    pos.append((0.0, 0.0, z0))
    nor.append((0.0, 0.0, -1.0))
    for a_ in ang:
        pos.append((radius * math.cos(a_), radius * math.sin(a_), z0))
        nor.append((0.0, 0.0, -1.0))
    for i in range(segments):
        idx.append((c0, c0 + 1 + (i + 1) % segments, c0 + 1 + i))
    return Mesh(np.array(pos, dtype=float), np.array(idx, dtype=np.int64), np.array(nor, dtype=float))


def icosphere(subdivisions: int = 1) -> Mesh:
    t = (1.0 + 5 ** 0.5) / 2.0
    v = [(-1, t, 0), (1, t, 0), (-1, -t, 0), (1, -t, 0), (0, -1, t), (0, 1, t), (0, -1, -t), (0, 1, -t),
         (t, 0, -1), (t, 0, 1), (-t, 0, -1), (-t, 0, 1)]
    f = [(0, 11, 5), (0, 5, 1), (0, 1, 7), (0, 7, 10), (0, 10, 11), (1, 5, 9), (5, 11, 4), (11, 10, 2),
         (10, 7, 6), (7, 1, 8), (3, 9, 4), (3, 4, 2), (3, 2, 6), (3, 6, 8), (3, 8, 9), (4, 9, 5), (2, 4, 11),
         (6, 2, 10), (8, 6, 7), (9, 8, 1)]
    verts = [np.array(p, dtype=float) / np.linalg.norm(p) for p in v]
    faces = f
    for _ in range(subdivisions):
        cache: dict[tuple[int, int], int] = {}

        def mid(a: int, b: int) -> int:
            key = (min(a, b), max(a, b))
            if key not in cache:
                m = verts[a] + verts[b]
                verts.append(m / np.linalg.norm(m))
                cache[key] = len(verts) - 1
            return cache[key]

        nf = []
        for a, b, c in faces:
            ab, bc, ca = mid(a, b), mid(b, c), mid(c, a)
            nf += [(a, ab, ca), (b, bc, ab), (c, ca, bc), (ab, bc, ca)]
        faces = nf
    pos = np.array(verts)
    return Mesh(pos, np.array(faces, dtype=np.int64), normals=pos.copy())


def line_strip_mesh(lines: list[np.ndarray]) -> Mesh:
    """glTF LINES mesh from polylines with xyz vertices."""
    pos, idx = [], []
    base = 0
    for ln in lines:
        if len(ln) < 2:
            continue
        pos.append(ln)
        k = np.arange(len(ln) - 1) + base
        idx.append(np.column_stack([k, k + 1]))
        base += len(ln)
    if not pos:
        return empty_mesh("lines")
    return Mesh(np.concatenate(pos), np.concatenate(idx).astype(np.int64), mode="lines")


def resample_polyline(line: LineString, spacing: float) -> tuple[np.ndarray, np.ndarray]:
    """Keep all original vertices and insert points so no segment exceeds ``spacing``.

    Returns (xy (n, 2), s (n,) arc length).
    """
    c = np.asarray(line.coords)[:, :2]
    out = [c[0]]
    for a, b in zip(c[:-1], c[1:]):
        d = float(np.hypot(*(b - a)))
        k = max(1, int(math.ceil(d / spacing)))
        for i in range(1, k + 1):
            out.append(a + (b - a) * (i / k))
    xy = np.array(out)
    s = np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(xy, axis=0), axis=1))])
    return xy, s
