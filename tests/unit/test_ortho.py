"""Per-tile orthophoto bake: pixel size, north-up orientation, seams, colour levels, build wiring."""

from __future__ import annotations

import dataclasses
import json
import warnings

import numpy as np
import pytest
from pyproj import CRS, Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_origin

from worldgen.config import ImagerySourceConfig
from worldgen.export.ortho import (
    HIGHLIGHT_GENTLE,
    HIGHLIGHT_KNEE,
    HIGHLIGHT_TARGET,
    IDENTITY_LEVELS,
    ColourLevels,
    OrthoBaker,
    compute_levels,
    highlight_shoulder,
    imagery_hash,
    tile_pixels,
)
from worldgen.export.tiles import tile_input_hash
from worldgen.spatial.frame import GeoPoint, LocalFrame
from worldgen.spatial.raster import GeoRaster
from worldgen.tiling.grid import TileGrid

FRAME = LocalFrame(GeoPoint(25.17667, 121.537628))
GRID = TileGrid(256.0)
EXTENT = (-512.0, -512.0, 512.0, 512.0)


def _local_raster(data: np.ndarray, res: float = 1.0, half: float = 600.0) -> GeoRaster:
    """North-up RGB raster in the local frame CRS covering [-half, half]^2 (row 0 = north)."""
    return GeoRaster(data, from_origin(-half, half, res, res), FRAME.local_crs(), 0, "imagery")


def _blank(value: int = 40, res: float = 1.0, half: float = 600.0) -> np.ndarray:
    n = int(round(2 * half / res))
    return np.full((3, n, n), value, dtype=np.uint8)


def _decode(jpeg: bytes) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with MemoryFile(jpeg) as mf, mf.open() as ds:
            assert ds.driver == "JPEG"
            return ds.read().transpose(1, 2, 0)


@pytest.mark.parametrize("tile,mpp,cap,expected", [
    (256.0, 10.0, 1024, 32),       # Sentinel-2: 25.6 px -> 32
    (256.0, 100.0, 1024, 32),      # never below 32
    (256.0, 0.54, 1024, 512),      # XYZ zoom 18 at 25 N
    (256.0, 0.27, 1024, 1024),
    (256.0, 0.25, 1024, 1024),     # exactly 1024
    (256.0, 0.05, 1024, 1024),     # capped by tile_px_max
    (256.0, 0.5, 256, 256),
    (512.0, 3.0, 2048, 256),
])
def test_tile_pixels_power_of_two(tile, mpp, cap, expected):
    n = tile_pixels(tile, mpp, cap)
    assert n == expected and n & (n - 1) == 0


def test_bake_size_and_north_up_orientation():
    data = _blank()
    # Bright square in the north-east corner of tile 0_0 (x, y in [232, 256)); local row = 600 - y.
    data[:, 600 - 256:600 - 232, 600 + 232:600 + 256] = 250
    # Dim marker in the south-west corner (x, y in [0, 24)).
    data[:, 600 - 24:600, 600:600 + 24] = 120
    cfg = ImagerySourceConfig(enabled=True, tile_px_max=64)
    baker = OrthoBaker(_local_raster(data), FRAME, EXTENT, 256.0, cfg)
    assert baker.px == 64 and baker.m_per_px == 4.0 and baker.prefilter == 4
    baker.levels = IDENTITY_LEVELS             # orientation only; colour levels are tested below
    img = _decode(baker.bake(GRID.bounds("0_0")))
    assert img.shape == (64, 64, 3)
    ne, nw, se, sw = img[:5, -5:].mean(), img[:5, :5].mean(), img[-5:, -5:].mean(), img[-5:, :5].mean()
    assert ne > 230 and 100 < sw < 140 and nw < 50 and se < 50       # bright top-right, dim bottom-left


def test_orientation_through_a_different_source_crs():
    """Source in UTM 51N (rotated grid relative to the local frame): north stays up."""
    utm = CRS.from_epsg(32651)
    to_utm = Transformer.from_crs(FRAME.local_crs(), utm, always_xy=True)
    cx, cy = to_utm.transform(128.0, 128.0)
    x0, y1 = cx - 400, cy + 400
    data = np.full((3, 400, 400), 40, dtype=np.uint8)
    # Mark the NE corner of tile 0_0 at its UTM position.
    ex, ey = to_utm.transform(244.0, 244.0)
    c, r = int((ex - x0) / 2), int((y1 - ey) / 2)
    data[:, r - 6:r + 6, c - 6:c + 6] = 250
    raster = GeoRaster(data, from_origin(x0, y1, 2.0, 2.0), utm, 0, "imagery")
    baker = OrthoBaker(raster, FRAME, (0.0, 0.0, 256.0, 256.0), 256.0, ImagerySourceConfig(enabled=True))
    assert baker.px == 128
    baker.levels = IDENTITY_LEVELS
    img = baker.sample(GRID.bounds("0_0"))
    rows, cols = np.nonzero(img[..., 0] > 145)
    # Centroid of the marker in local metres (pixel centres are 2 m apart) vs. its true centre.
    mx, my = Transformer.from_crs(utm, FRAME.local_crs(), always_xy=True).transform(x0 + 2 * c, y1 - 2 * r)
    assert 236 < mx < 250 and 236 < my < 250
    assert abs((cols.mean() + 0.5) * 2.0 - mx) < 1.0
    assert abs(256.0 - (rows.mean() + 0.5) * 2.0 - my) < 1.0
    assert rows.max() < 16 and cols.min() > 110                       # top-right of the image


def test_adjacent_tiles_are_continuous_at_seams():
    # Linear ramps: red grows to the east, green grows to the north.
    n = 1200
    x = np.arange(n) + 0.5 - 600.0
    y = 600.0 - (np.arange(n) + 0.5)
    data = np.zeros((3, n, n), dtype=np.uint8)
    data[0] = np.rint(30 + 0.15 * (x[None, :] + 600))
    data[1] = np.rint(30 + 0.15 * (y[:, None] + 600))
    data[2] = 60
    baker = OrthoBaker(_local_raster(data), FRAME, EXTENT, 256.0, ImagerySourceConfig(enabled=True, tile_px_max=64))
    baker.levels = IDENTITY_LEVELS
    a = baker.sample(GRID.bounds("0_0")).astype(float)
    east = baker.sample(GRID.bounds("1_0")).astype(float)
    north = baker.sample(GRID.bounds("0_1")).astype(float)
    step = 0.15 * baker.m_per_px                                       # 0.6 DN per output pixel
    # Across the vertical seam the red ramp continues with the same step as inside a tile.
    across = east[:, 0, 0] - a[:, -1, 0]
    inside = a[:, -1, 0] - a[:, -2, 0]
    assert np.abs(across - step).max() <= 1.0 and np.abs(inside - step).max() <= 1.0
    assert np.abs(east[:, 0, 1] - a[:, -1, 1]).max() <= 1.0           # green unchanged along x
    # Across the horizontal seam (tile 0_1 lies north, so its bottom row meets our top row).
    across_n = north[-1, :, 1] - a[0, :, 1]
    assert np.abs(across_n - step).max() <= 1.0
    # And the tile is a continuous sample of the ramp: first/last pixel centres sit half a pixel
    # inside the tile edge (x = 2, x = 254 for 4 m pixels).
    assert a[10, 0, 0] == pytest.approx(30 + 0.15 * (2 + 600), abs=1.0)
    assert a[10, -1, 0] == pytest.approx(30 + 0.15 * (254 + 600), abs=1.0)


def test_levels_stretch_dark_hazy_imagery_within_0_255():
    rng = np.random.default_rng(1)
    px = np.clip(rng.normal([35, 45, 40], [8, 9, 6], size=(20000, 3)), 0, 255)   # dark, hazy, green-ish
    lv = compute_levels(px.astype(np.uint8))
    out = lv.apply(px)
    assert out.dtype == np.uint8 and out.min() >= 0 and out.max() <= 255
    assert out.max() >= 200 and out.min() <= 25                        # stretched (highlights soft)
    assert out[:, 1].mean() > out[:, 0].mean()                         # green stays greener than red
    assert all(0.95 <= g <= 1.05 for g in lv.gain) and lv.gamma > 1.0
    full = np.stack([np.arange(256)] * 3, axis=1).astype(np.uint8)
    ramp = lv.apply(full)
    assert ramp[0].tolist() == [0, 0, 0] and ramp[-1].max() <= 255 * (lv.knee + lv.shoulder) + 0.5
    assert (np.diff(ramp.astype(int), axis=0) >= 0).all()             # monotonic, no wrap-around
    assert IDENTITY_LEVELS.apply(full).tolist() == full.tolist()
    assert compute_levels(np.zeros((0, 3), dtype=np.uint8)) == IDENTITY_LEVELS
    assert set(lv.to_dict()) == {"black", "white", "gain", "gamma", "knee", "shoulder"}
    assert isinstance(lv, ColourLevels)


def test_highlight_shoulder_is_smooth_and_fitted():
    k = HIGHLIGHT_KNEE
    assert highlight_shoulder(0.5) == HIGHLIGHT_GENTLE                # nothing bright: gentlest shoulder
    for ref in (0.9, 1.2, 2.0, 5.0):
        s = highlight_shoulder(ref)
        lv = dataclasses.replace(IDENTITY_LEVELS, knee=k, shoulder=s)
        assert 0 < s <= HIGHLIGHT_GENTLE
        assert float(lv.tone(np.float32(ref))) <= HIGHLIGHT_TARGET + 1e-6
    assert highlight_shoulder(5.0) < highlight_shoulder(1.2) <= highlight_shoulder(0.9)
    lv = dataclasses.replace(IDENTITY_LEVELS, knee=k, shoulder=highlight_shoulder(2.0))
    v = np.linspace(0, 3, 3001, dtype=np.float32)
    t = lv.tone(v)
    assert np.array_equal(t[v <= k], v[v <= k])                       # identity below the knee
    assert (np.diff(t) >= 0).all() and t.max() < 1.0                  # monotonic, never clipped to white
    assert float(lv.tone(np.float32(2.0)) - lv.tone(np.float32(1.0))) > 0.01   # highlights keep their order
    eps = np.float32(1e-4)
    assert float(lv.tone(np.float32(k) + eps) - k) == pytest.approx(float(eps), rel=0.01)  # slope 1 at the knee


def test_bright_ground_is_compressed_not_blown_and_midtones_keep():
    """A bright tile (40 % bare ground / roofs) next to dark forest: < 0.5 % blown pixels per tile."""
    rng = np.random.default_rng(3)
    data = np.clip(rng.normal([30, 42, 32], [6, 7, 5], size=(1200, 1200, 3)), 1, 255)
    bright = np.zeros((1200, 1200), dtype=bool)
    bright[600 - 230:600 - 30, 600 + 300:600 + 480] = True            # inside tile 1_0
    data[bright] = np.clip(rng.normal([170, 165, 160], 20, size=(int(bright.sum()), 3)), 1, 255)
    data = np.ascontiguousarray(data.transpose(2, 0, 1).astype(np.uint8))
    baker = OrthoBaker(_local_raster(data), FRAME, EXTENT, 256.0, ImagerySourceConfig(enabled=True, tile_px_max=64))
    lv = baker.levels
    assert lv.knee == HIGHLIGHT_KNEE and 0 < lv.shoulder < HIGHLIGHT_GENTLE
    hard = dataclasses.replace(lv, knee=1.0, shoulder=0.0)            # the old hard white point
    blown, blown_hard = {}, {}
    for tid in ("-2_-2", "-1_-1", "0_0", "1_0", "1_1", "-2_1"):
        img = _decode(baker.bake(GRID.bounds(tid)))
        blown[tid] = float((img.min(axis=2) > 225).mean())
        baker.levels = hard
        blown_hard[tid] = float((baker.sample(GRID.bounds(tid)).min(axis=2) > 225).mean())
        baker.levels = lv
    assert blown_hard["1_0"] > 0.2                                    # clipping would burn it out
    assert max(blown.values()) < 0.005
    # Mid-tones are untouched: every value below the knee is identical with and without shoulder.
    full = np.stack([np.arange(256)] * 3, axis=1).astype(np.uint8)
    below = lv.display(full).max(axis=1) <= HIGHLIGHT_KNEE
    assert below.sum() > 50 and np.array_equal(lv.apply(full)[below], hard.apply(full)[below])
    # The bright patch stays light (compressed, not darkened into the mid-tones).
    img = baker.sample(GRID.bounds("1_0"))
    assert img[10:40, 20:50].mean() > 200


def test_baked_tiles_use_the_full_range_and_fill_holes():
    rng = np.random.default_rng(2)
    data = np.clip(rng.normal(50, 12, size=(3, 1200, 1200)), 1, 255).astype(np.uint8)
    data[:, :200, :] = 0                      # northern strip without imagery
    baker = OrthoBaker(_local_raster(data), FRAME, EXTENT, 256.0, ImagerySourceConfig(enabled=True, tile_px_max=32))
    assert baker.valid_fraction == pytest.approx(1000 / 1200, abs=1e-3)
    img = baker.sample(GRID.bounds("0_1"))   # y in [256, 512]: partly inside the hole (y > 400)
    assert img.dtype == np.uint8 and img.min() < 60 and img.max() > 190
    assert img[:4].min() > 60                # the hole is filled with the mean colour, not black


def test_imagery_hash_and_tile_hash(compiled):
    world = compiled("simple_intersection")
    cfg = world.config
    data = _blank(70)
    w1 = dataclasses.replace(world, imagery=_local_raster(data), imagery_hash=None)
    data2 = data.copy()
    data2[0, 5, 5] = 71
    w2 = dataclasses.replace(world, imagery=_local_raster(data2), imagery_hash=None)
    assert imagery_hash(world) is None
    h1, h2 = imagery_hash(w1), imagery_hash(w2)
    assert h1 and h2 and h1 != h2 and imagery_hash(w1) == h1
    cfg3 = dataclasses.replace(cfg, sources=dataclasses.replace(
        cfg.sources, imagery=dataclasses.replace(cfg.sources.imagery, jpeg_quality=70)))
    w3 = dataclasses.replace(world, config=cfg3, imagery=_local_raster(data), imagery_hash=None)
    assert imagery_hash(w3) != h1
    tid = f"{world.region.tiles()[0][0]}_{world.region.tiles()[0][1]}"
    base = tile_input_hash(world, tid)
    assert len({base, tile_input_hash(w1, tid), tile_input_hash(w2, tid)}) == 3


def test_build_writes_ortho_tiles_index_and_manifest(tmp_path, monkeypatch):
    import worldgen.build as build_mod
    from worldgen.build import BuildOptions, build, fixture_region
    from worldgen.config import load_config
    from worldgen.sources.base import SourceRecord

    real = build_mod.bundle_from_fixture

    def with_imagery(fx):  # noqa: ANN001, ANN202
        b = real(fx)
        data = _blank(60, res=2.0, half=700.0)
        data[1, : data.shape[1] // 2] = 140          # northern half greener
        b.imagery = GeoRaster(data, from_origin(-700, 700, 2.0, 2.0), fx.dem.crs, 0, "imagery")
        b.records.append(SourceRecord("imagery", "local_file", "CC0-1.0", "synthetic imagery",
                                      coverage={"m_per_px": 2.0}))
        return b

    name = "simple_intersection"
    cfg = load_config("basic", overrides=["sources.imagery.enabled=true", "sources.imagery.tile_px_max=64"],
                      region=fixture_region(name))
    monkeypatch.setattr(build_mod, "bundle_from_fixture", with_imagery)
    res = build(cfg, BuildOptions(tmp_path, fixture=name))
    index = json.loads((tmp_path / "tiles" / "index.json").read_text())
    manifest = json.loads((tmp_path / "manifest.json").read_text())
    assert index["ortho"]["pixels"] == 64 and index["ortho"]["m_per_px"] == 4.0
    for t in index["tiles"]:
        assert t["ortho"] == f"tiles/{t['id']}.ortho.jpg"
        img = _decode((tmp_path / t["ortho"]).read_bytes())
        assert img.shape == (64, 64, 3)
        assert t["ortho"] in manifest["outputs"]
    assert any(s["name"] == "imagery" for s in manifest["sources"])
    assert manifest["statistics"]["ortho"]["pixels"] == 64
    assert len(res.tiles_written) == len(index["tiles"])
    # Unchanged imagery: nothing to rebuild.
    res2 = build(cfg, BuildOptions(tmp_path, fixture=name))
    assert res2.tiles_written == [] and len(res2.tiles_skipped) == len(index["tiles"])
    # A deleted ortho file makes its tile dirty again.
    first = index["tiles"][0]
    (tmp_path / first["ortho"]).unlink()
    res3 = build(cfg, BuildOptions(tmp_path, fixture=name))
    assert res3.tiles_written == [first["id"]] and (tmp_path / first["ortho"]).is_file()
    # No imagery (enabled by default, but fixtures carry none): every tile rebuilds without ortho,
    # and the source is recorded as missing instead of failing the build.
    monkeypatch.setattr(build_mod, "bundle_from_fixture", real)
    cfg_default = load_config("basic", region=fixture_region(name))
    assert cfg_default.sources.imagery.enabled
    res4 = build(cfg_default, BuildOptions(tmp_path, fixture=name))
    index4 = json.loads((tmp_path / "tiles" / "index.json").read_text())
    assert len(res4.tiles_written) == len(index4["tiles"]) and index4["ortho"] is None
    assert all(t["ortho"] is None for t in index4["tiles"])
    assert not list((tmp_path / "tiles").glob("*.ortho.jpg"))
    rec = next(s for s in res4.manifest["sources"] if s["name"] == "imagery")
    assert rec["status"] == "missing" and "ortho" not in res4.manifest["statistics"]
