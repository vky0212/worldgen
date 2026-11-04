"""Heightfield sampling, mesh generators, GLB writing and merging."""

from __future__ import annotations

import numpy as np
import pytest
import trimesh
from shapely.geometry import Polygon

from worldgen.export.gltf import GltfBuilder, enu_to_gltf, read_glb_json
from worldgen.export.merge import merge_glbs
from worldgen.geometry.shapes import extrude_polygon, resample_polyline, ribbon, swept_box, triangulate_polygon
from worldgen.terrain.heightfield import Heightfield
from worldgen.tiling.grid import TileGrid


def _hf() -> Heightfield:
    rng = np.random.default_rng(0)
    return Heightfield(-64.0, -64.0, 4.0, rng.uniform(0, 30, (33, 33)))


def test_sampling_lies_on_mesh_triangles():
    hf = _hf()
    mesh = hf.tile_mesh(-64, -64, 64, 64)
    tri = mesh.positions[mesh.indices]                      # (m, 3, 3)
    rng = np.random.default_rng(1)
    pts = rng.uniform(-63.9, 63.9, (300, 2))
    a, b, c = tri[:, 0, :2], tri[:, 1, :2], tri[:, 2, :2]
    for p, z in zip(pts, hf.sample(pts[:, 0], pts[:, 1])):
        v0, v1, v2 = b - a, c - a, p - a
        den = v0[:, 0] * v1[:, 1] - v1[:, 0] * v0[:, 1]
        u = (v2[:, 0] * v1[:, 1] - v1[:, 0] * v2[:, 1]) / den
        v = (v0[:, 0] * v2[:, 1] - v2[:, 0] * v0[:, 1]) / den
        k = np.nonzero((u >= -1e-9) & (v >= -1e-9) & (u + v <= 1 + 1e-9))[0][0]
        zt = tri[k, 0, 2] + u[k] * (tri[k, 1, 2] - tri[k, 0, 2]) + v[k] * (tri[k, 2, 2] - tri[k, 0, 2])
        assert z == pytest.approx(zt, abs=1e-9)
    # Grid nodes are reproduced exactly.
    assert hf.sample(-64 + 8.0, -64 + 12.0) == pytest.approx(hf.z[3, 2])


def test_terrain_tiles_share_edges_and_face_up():
    hf = _hf()
    a = hf.tile_mesh(-64, -64, 0, 0)
    b = hf.tile_mesh(0, -64, 64, 0)
    ea = a.positions[np.isclose(a.positions[:, 0], 0.0)]
    eb = b.positions[np.isclose(b.positions[:, 0], 0.0)]
    assert np.array_equal(ea[np.argsort(ea[:, 1])], eb[np.argsort(eb[:, 1])])
    from worldgen.geometry.mesh import triangle_normals

    assert (triangle_normals(a.positions, a.indices)[:, 2] > 0).all()


def test_extrusion_with_hole_is_closed_and_outward():
    poly = Polygon([(0, 0), (20, 0), (20, 20), (0, 20)], [[(5, 5), (15, 5), (15, 15), (5, 15)]])
    m = extrude_polygon(poly, 0.0, 10.0)
    tm = trimesh.Trimesh(m.positions, m.indices, process=True)
    tm.merge_vertices()
    assert tm.is_watertight
    assert tm.volume == pytest.approx(poly.area * 10.0, rel=1e-6)


def test_triangulation_respects_holes():
    poly = Polygon([(0, 0), (10, 0), (10, 10), (0, 10)], [[(3, 3), (7, 3), (7, 7), (3, 7)]])
    v, t = triangulate_polygon(poly)
    a = v[t]
    area = 0.5 * np.abs((a[:, 1, 0] - a[:, 0, 0]) * (a[:, 2, 1] - a[:, 0, 1]) -
                        (a[:, 1, 1] - a[:, 0, 1]) * (a[:, 2, 0] - a[:, 0, 0]))
    assert area.sum() == pytest.approx(poly.area)


def test_ribbon_and_box():
    xy = np.array([[0.0, 0.0], [10.0, 0.0], [20.0, 5.0]])
    r = ribbon(xy, np.array([1.0, 1.0, 2.0]), 6.0)
    assert (r.normals[:, 2] > 0.9).all()
    left = r.positions[0::2, :2]
    right = r.positions[1::2, :2]
    assert np.allclose(np.linalg.norm(left[0] - right[0]), 6.0)
    b = swept_box(xy, np.array([5.0, 5.0, 5.0]), 4.0, 1.0)
    tm = trimesh.Trimesh(b.positions, b.indices, process=True)
    tm.merge_vertices()
    assert tm.is_watertight and tm.volume > 0
    s_xy, s = resample_polyline(__import__("shapely").geometry.LineString(xy), 3.0)
    assert np.max(np.diff(s)) <= 3.0 + 1e-9 and len(s_xy) >= 9


def test_tile_crossings_insert_exact_boundaries():
    g = TileGrid(256.0)
    xy = np.array([[-300.0, 10.0], [300.0, 20.0]])
    out, z = g.insert_tile_crossings(xy, np.array([0.0, 6.0]))
    assert np.any(np.isclose(out[:, 0], 0.0)) and np.any(np.isclose(out[:, 0], 256.0))
    assert np.any(np.isclose(out[:, 0], -256.0))
    assert g.segment_tiles(out) == ["-2_0", "-1_0", "0_0", "1_0"]


def test_glb_roundtrip_and_merge():
    from worldgen.geometry.mesh import Mesh

    m = Mesh(np.array([[0, 0, 0], [1, 0, 0], [0, 1, 0]], dtype=float), np.array([[0, 1, 2]]),
             feature=np.zeros(3, dtype=np.int64))
    files = []
    for k in range(2):
        gb = GltfBuilder()
        mi = gb.add_mesh("roads", {"road_minor": m}, extras={"feature_ids": [f"road:{k}"]})
        gb.add_node(f"tile_{k}", mesh=mi, root=True, translation=(k * 10.0, 0, 0))
        files.append(gb.to_glb())
    doc = read_glb_json(files[0])
    prim = doc["meshes"][0]["primitives"][0]
    assert "_FEATURE_ID_0" in prim["attributes"] and doc["meshes"][0]["extras"]["feature_ids"] == ["road:0"]
    merged = merge_glbs(files)
    scene = trimesh.load(trimesh.util.wrap_as_stream(merged), file_type="glb", force="scene")
    assert len(scene.geometry) == 2
    # ENU -> glTF: north (+y) becomes -z, up (+z) becomes +y.
    assert np.array_equal(enu_to_gltf(np.array([[1.0, 2.0, 3.0]])), np.array([[1.0, 3.0, -2.0]]))


def test_limit_grade_keeps_ends_and_bounds_slope():
    from worldgen.transportation.elevation import limit_grade

    s = np.linspace(0, 100, 26)
    z = 10 + 30 * np.exp(-((s - 50) / 10) ** 2)            # a 30 m spike on a 100 m road
    z[0], z[-1] = 10.0, 12.0
    out = limit_grade(z, s, 0.18)
    assert out[0] == 10.0 and out[-1] == 12.0
    assert np.max(np.abs(np.diff(out) / np.diff(s))) <= 0.18 + 1e-9
    steep = limit_grade(np.array([0.0, 5.0, 40.0]), np.array([0.0, 50.0, 100.0]), 0.1)
    assert steep[-1] == 40.0                              # ends win over the grade limit
