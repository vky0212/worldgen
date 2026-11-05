"""Coarser levels of detail (export/pyramid.py, heightmap level sections): nesting, colours, errors."""

from __future__ import annotations

import numpy as np
import pytest

from worldgen.export.container import decode_grid, decode_heights, read_container, write_container
from worldgen.export.heightmap import _halve, level_colors, level_node_range, terrain_sections
from worldgen.export.pyramid import finish_levels, plan_levels, terrain_error, top_level
from worldgen.export.tileset import data_index, level_paths, subtree_of
from worldgen.terrain.heightfield import Heightfield
from worldgen.tiling.grid import tile_id

T = 64.0        # tile size of the synthetic world
R = 4.0         # spacing: 16 cells per tile
Q = 0.01


def _world(tiles_x: int = 4, tiles_y: int = 3, x0: float = -2 * T, y0: float = -T) -> tuple[Heightfield, np.ndarray, np.ndarray]:
    nx, ny = int(tiles_x * T / R) + 1, int(tiles_y * T / R) + 1
    yy, xx = np.mgrid[0:ny, 0:nx].astype(float)
    z = 50.0 + 6.0 * np.sin(xx * 0.31) * np.cos(yy * 0.23) + 0.4 * xx - 0.2 * yy + 0.01 * xx * yy
    z = np.round(z / Q) * Q
    hf = Heightfield(x0, y0, R, z)
    rgba = np.zeros((ny, nx, 4), dtype=np.float32)
    rgba[..., 0] = (0.5 + 0.5 * np.sin(xx * 0.9)).astype(np.float32)
    rgba[..., 1] = (0.5 + 0.5 * np.cos(yy * 0.7)).astype(np.float32)
    rgba[..., 2] = ((xx + yy) % 3 / 3.0).astype(np.float32)
    rgba[..., 3] = 1.0
    codes = ((xx * 3 + yy) % 11).astype(np.uint8)
    return hf, rgba, codes


def test_top_level_needs_the_lattice_on_every_tile_corner():
    assert top_level(4, 256.0, 4.0) == 4            # 64 nodes per tile
    assert top_level(2, 256.0, 4.0) == 2
    assert top_level(4, 256.0, 16.0) == 4           # 16 nodes per tile
    assert top_level(4, 256.0, 32.0) == 3           # 8 nodes: level 4 would skip tile corners
    assert top_level(4, 96.0, 4.0) == 3             # 24 nodes = 8 * 3
    assert top_level(0, 256.0, 4.0) == 0


def test_plan_levels_groups_tiles_and_stops_when_a_level_does_not_shrink():
    tiles = [tile_id(i, j) for j in range(-2, 2) for i in range(-3, 3)]          # 6 x 4, across the origin
    plan = plan_levels(tiles, 4)
    assert sorted(plan) == [1, 2]          # level 2: 2 x 2 tiles; level 3 would have 2 x 2 again
    assert len(plan[1]) == 8 and len(plan[2]) == 4
    assert plan[1][(-2, -1)] == [(-3, -2), (-3, -1)]                  # floor shifts: -3 >> 1 == -2
    assert sorted(t for kids in plan[1].values() for t in kids) == sorted((i, j) for j in range(-2, 2) for i in range(-3, 3))
    assert plan[2][(-1, -1)] == sorted([(i, j) for j in range(-2, 0) for i in range(-3, 0)], key=lambda t: (t[1], t[0]))
    assert plan_levels([tile_id(0, 0)], 4) == {}
    assert subtree_of(-1, -1, 1) == (-1, -1) and subtree_of(7, 8, 1) == (0, 1) and subtree_of(3, -2, 4) == (3, -2)
    assert level_paths(2, -1, 3)["content"] == "tiles/2/-1_0/-1_3.wgt"
    assert level_paths(2, -1, 3)["ortho"] == "tiles/2/-1_0/-1_3.ortho.jpg"


@pytest.mark.parametrize("level,i,j", [(1, -1, 0), (1, 0, -1), (2, -1, -1), (2, 0, 0)])
def test_level_heights_are_the_level0_heights_at_every_other_node(level, i, j):
    hf, rgba, codes = _world()
    k = 1 << level
    s = T * k
    bounds = (i * s, j * s, (i + 1) * s, (j + 1) * s)
    c = read_container(write_container(level, i, j, terrain_sections(hf, bounds, Q, rgba, codes, level)))
    g = decode_heights(c.sections["HGT"])
    r0, r1, c0, c1 = level_node_range(hf, bounds, level)
    assert g.spacing == R * k and g.apron == 1
    assert g.x0 == hf.xmin + (c0 - k) * R and g.y0 == hf.ymin + (r0 - k) * R
    inner = g.z[1:-1, 1:-1]
    assert np.array_equal(inner, hf.z[r0:r1 + 1:k, c0:c1 + 1:k])               # nested, bit for bit
    # The apron is the next coarse node where the grid has it, else extrapolated linearly.
    if r0 >= k:
        assert np.array_equal(g.z[0, 1:-1], hf.z[r0 - k, c0:c1 + 1:k])
    else:
        assert np.allclose(g.z[0, 1:-1], 2 * inner[0] - inner[1])
    assert np.array_equal(decode_grid(c.sections["LCC"], 1), codes[r0:r1 + 1:k, c0:c1 + 1:k])
    rgb = decode_grid(c.sections["RGB"], 3)
    assert rgb.shape == inner.shape + (3,)


def test_level0_sections_unchanged_by_the_level_parameter():
    hf, rgba, codes = _world()
    b = (0.0, 0.0, T, T)
    assert terrain_sections(hf, b, Q, rgba, codes) == terrain_sections(hf, b, Q, rgba, codes, 0)


def test_level_colors_do_not_depend_on_the_tile_window():
    hf, rgba, _ = _world(8, 8, 0.0, 0.0)
    full = (0, hf.rows - 1, 0, hf.cols - 1)
    for level in (1, 2, 3):
        whole = rgba[..., :3].astype(np.float64)
        for _ in range(level):
            whole = _halve(whole)
        k = 1 << level
        assert np.array_equal(level_colors(rgba, full, level), whole)
        s = T * k
        for i in range(8 // k):
            for j in range(8 // k):
                rng = level_node_range(hf, (i * s, j * s, (i + 1) * s, (j + 1) * s), level)
                part = level_colors(rgba, rng, level)
                r0, r1, c0, c1 = rng
                assert np.array_equal(part, whole[r0 // k:r1 // k + 1, c0 // k:c1 // k + 1]), (level, i, j)


def test_level_colors_keep_flat_colours_and_average_detail():
    flat = np.full((33, 33, 4), 0.25, dtype=np.float32)
    assert np.allclose(level_colors(flat, (0, 32, 0, 32), 2), 0.25)
    stripes = np.zeros((33, 33, 4), dtype=np.float32)
    stripes[:, ::2, :3] = 1.0                      # alternating columns: 1, 0, 1, 0, ...
    out = level_colors(stripes, (0, 32, 0, 32), 1)
    assert np.allclose(out[:, 1:-1], 0.5)          # the binomial filter averages them out


def test_terrain_error_is_the_largest_difference_at_level0_nodes():
    hf, _, _ = _world()
    level, i, j = 2, -1, -1
    k = 1 << level
    s = T * k
    bounds = (i * s, j * s, (i + 1) * s, (j + 1) * s)
    leaves = [(a * T, b * T, (a + 1) * T, (b + 1) * T) for a in range(-2, 0) for b in range(-1, 0)]
    got = terrain_error(hf, bounds, level, leaves)
    r0, r1, c0, c1 = level_node_range(hf, bounds, level)
    coarse = Heightfield(hf.xmin + c0 * R, hf.ymin + r0 * R, R * k, hf.z[r0:r1 + 1:k, c0:c1 + 1:k])
    worst = 0.0
    for lb in leaves:
        for y in np.arange(lb[1], lb[3] + 1e-9, R):
            for x in np.arange(lb[0], lb[2] + 1e-9, R):
                worst = max(worst, abs(float(hf.sample(x, y)) - float(coarse.sample(x, y))))
    assert worst > 0.1 and worst <= got < worst + 0.0011
    # A plane is drawn exactly by any level.
    yy, xx = np.mgrid[0:hf.rows, 0:hf.cols].astype(float)
    plane = Heightfield(hf.xmin, hf.ymin, R, np.round((10 + 0.25 * xx - 0.5 * yy) / Q) * Q)
    assert terrain_error(plane, bounds, level, leaves) <= 0.001


def test_finish_levels_errors_grow_up_the_tree_and_cover_children():
    tiles = [tile_id(i, j) for j in range(0, 4) for i in range(0, 4)]
    plan = plan_levels(tiles, 4)
    assert sorted(plan) == [1, 2]
    state = {t: {"hash": "h", "zrange": [10.0, 20.0], "lift": 0.65} for t in tiles}
    state[tile_id(3, 3)] = {"hash": "h", "zrange": [10.0, 55.0], "lift": 31.5}
    level_state = {1: {k: {"hash": "c", "terr": 0.4, "zrange": [11.0, 19.0], "ortho": False} for k in plan[1]},
                   2: {k: {"hash": "d", "terr": 2.5, "zrange": [12.0, 18.0], "ortho": True} for k in plan[2]}}
    level_state[1][(0, 0)]["terr"] = 6.0
    out = finish_levels(plan, state, level_state, R)
    assert out[1][(0, 0)]["err"] == 6.0                  # terrain error
    assert out[1][(1, 0)]["err"] == 4.0                  # the floor: half the level-1 spacing (8 m)
    assert out[1][(1, 1)]["err"] == 31.5                 # the lift of tile 3_3
    assert out[1][(1, 1)]["zrange"] == [10.0, 55.0]      # children's content included
    assert out[2][(0, 0)]["err"] == 31.5 and out[2][(0, 0)]["terr"] == 2.5
    assert out[2][(0, 0)]["ortho"] is True
    tileset, docs = data_index(tiles, state, 64.0, R, Q, ["terrain"], None, {}, out, {1: {"pixels": 64, "m_per_px": 2.0}})
    assert [lv["level"] for lv in tileset["levels"]] == [0, 1, 2]
    assert tileset["levels"][1] == {"level": 1, "size": 128.0, "spacing": 8.0, "geometric_error": 31.5, "tiles": 4,
                                    "ortho": {"pixels": 64, "m_per_px": 2.0}}
    sub = docs["tiles/sub/0_0.json"]
    l1 = sub["levels"][1]
    assert l1["level"] == 1 and l1["i"] == [0, 1, 0, 1] and l1["j"] == [0, 0, 1, 1]
    assert l1["err"] == [6.0, 4.0, 4.0, 31.5] and l1["terr"] == [6.0, 0.4, 0.4, 0.4] and l1["flags"] == [0, 0, 0, 0]
    assert sub["levels"][2]["flags"] == [2]
    assert tileset["subtrees"][0]["zrange"] == [10.0, 55.0]


def test_coarse_colours_show_the_water_surface_level0_draws():
    from worldgen.export.gltf import MATERIALS
    from worldgen.export.heightmap import WATER_CODE, surface_colors

    rgba = np.full((9, 9, 4), 0.5, dtype=np.float32)
    codes = np.zeros((9, 9), dtype=np.uint8)
    codes[:, :4] = WATER_CODE
    w = MATERIALS["water"].color
    seen = surface_colors(rgba[..., :3].astype(np.float64), codes)
    assert np.allclose(seen[:, :4], w[3] * np.array(w[:3]) + (1 - w[3]) * 0.5) and np.allclose(seen[:, 4:], 0.5)
    out = level_colors(rgba, (0, 8, 0, 8), 1, codes)
    assert np.allclose(out[:, 0], seen[0, 0]) and np.allclose(out[:, -1], 0.5)
    assert np.allclose(level_colors(rgba, (0, 8, 0, 8), 1), 0.5)           # without codes: the terrain colour
