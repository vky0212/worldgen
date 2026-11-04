"""Scaling helpers must give exactly the results of the whole-region computations they replace."""

from __future__ import annotations

import numpy as np
import pytest
import shapely
from affine import Affine
from rasterio import features
from scipy.spatial import cKDTree

from worldgen.export.tiles import PieceSelector, generate_pieces, tile_blocks, world_tiles
from worldgen.spatial.raster import rasterize_window
from worldgen.terrain.heightfield import Heightfield
from worldgen.vegetation.placement import _matern_ii
from worldgen.world.model import TreeTable


def _random_polygon(rng: np.random.Generator, cx: float, cy: float, radius: float) -> shapely.Polygon:
    n = int(rng.integers(3, 14))
    ang = np.sort(rng.uniform(0, 2 * np.pi, n))
    r = radius * rng.uniform(0.3, 1.0, n)
    poly = shapely.Polygon(np.column_stack([cx + r * np.cos(ang), cy + r * np.sin(ang)])).buffer(0)
    if rng.random() < 0.3:
        poly = poly.difference(shapely.Point(cx, cy).buffer(radius * 0.2))
    return poly


@pytest.mark.parametrize("transform", [
    Affine(4.0, 0.0, -1026.0, 0.0, 4.0, -1026.0),    # heightfield nodes, row 0 south
    Affine(8.0, 0.0, -1024.0, 0.0, 8.0, -1024.0),    # vegetation cells, row 0 south
    Affine(4.0, 0.0, -1026.0, 0.0, -4.0, 1026.0),    # north-up
])
def test_rasterize_window_matches_full_grid(transform):
    rng = np.random.default_rng(7)
    shape = (260, 230)
    for _ in range(60):
        cx, cy = transform @ (rng.uniform(-30, shape[1] + 30), rng.uniform(-30, shape[0] + 30))
        poly = _random_polygon(rng, cx, cy, rng.uniform(2.0, 400.0))
        full = features.rasterize([(poly, 1)], out_shape=shape, transform=transform, fill=0,
                                  all_touched=False, dtype="uint8").astype(bool)
        got = np.zeros(shape, dtype=bool)
        hit = rasterize_window(poly, shape, transform)
        if hit is not None:
            (rows, cols), mask = hit
            got[rows, cols] = mask
        assert np.array_equal(got, full)
    assert rasterize_window(shapely.Polygon(), shape, transform) is None


def test_window_normals_match_whole_grid():
    rng = np.random.default_rng(3)
    z = rng.normal(size=(37, 41)).cumsum(0).cumsum(1)
    hf = Heightfield(-64.0, -64.0, 4.0, z)
    full = hf.vertex_normals()
    for r0, r1, c0, c1 in [(0, 36, 0, 40), (0, 5, 0, 5), (10, 20, 3, 30), (30, 36, 35, 40), (5, 5, 7, 7)]:
        assert np.array_equal(hf.normals_window(r0, r1, c0, c1), full[r0:r1 + 1, c0:c1 + 1])
    for stride in (1, 2):
        a = hf.tile_mesh(-64.0, -64.0, 0.0, 16.0, stride=stride)
        b = hf.tile_mesh(-64.0, -64.0, 0.0, 16.0, stride=stride, normals=full)
        assert np.array_equal(a.normals, b.normals) and np.array_equal(a.positions, b.positions)


def _matern_reference(x, y, mark, delta, obs_xy):  # noqa: ANN001, ANN202
    """The single-radius implementation _matern_ii replaced."""
    keep = np.ones(len(x), dtype=bool)
    pts = np.column_stack([x, y])
    pairs = cKDTree(pts).query_pairs(float(delta.max()), output_type="ndarray")
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        d = np.hypot(x[a] - x[b], y[a] - y[b])
        close = d < np.maximum(delta[a], delta[b])
        a, b = a[close], b[close]
        keep[np.where(mark[a] > mark[b], a, b)] = False
    if len(obs_xy):
        otree = cKDTree(obs_xy)
        r = np.maximum(delta, 1.5)
        for k in np.nonzero(keep)[0]:
            if otree.query_ball_point(pts[k], r[k]):
                keep[k] = False
    return keep


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_matern_matches_single_radius_reference(seed):
    rng = np.random.default_rng(seed)
    n = 6000
    x = rng.uniform(-700, 900, n)
    y = rng.uniform(-500, 600, n)
    x[:40], y[:40] = x[40:80], y[40:80]                   # coincident candidates
    mark = rng.random(n)
    mark[100:140] = mark[140:180]                          # equal marks
    delta = np.clip(rng.lognormal(1.3, 0.6, n), 1.0, 18.0)
    obs_xy = np.column_stack([rng.uniform(-700, 900, 80), rng.uniform(-500, 600, 80)])
    obs = TreeTable(ids=[f"o{k}" for k in range(80)], x=obs_xy[:, 0], y=obs_xy[:, 1])
    assert np.array_equal(_matern_ii(x, y, mark, delta, obs), _matern_reference(x, y, mark, delta, obs_xy))
    assert np.array_equal(_matern_ii(x, y, mark, delta, TreeTable()),
                          _matern_reference(x, y, mark, delta, np.zeros((0, 2))))


def _same_pieces(a, b) -> bool:  # noqa: ANN001
    if [(p.layer, p.material, p.feature_id) for p in a] != [(p.layer, p.material, p.feature_id) for p in b]:
        return False
    return all(np.array_equal(p.mesh.positions, q.mesh.positions) and np.array_equal(p.mesh.indices, q.mesh.indices)
               for p, q in zip(a, b))


@pytest.mark.parametrize("block", [1, 2, 3])
def test_blockwise_pieces_match_whole_world(compiled, block):
    world = compiled("demo_town")
    tiles = world_tiles(world)
    whole = generate_pieces(world, set(tiles))
    selector = PieceSelector(world)
    seen = set()
    for group in tile_blocks(tiles, block):
        part = generate_pieces(world, set(group), selector)
        assert set(part) <= set(group)
        for t in group:
            assert _same_pieces(part.get(t, []), whole.get(t, [])), t
            seen.add(t)
    assert seen == set(tiles)
    assert any(len(whole.get(t, [])) > 3 for t in tiles)  # the fixture really has content
