"""Config, coordinate frames, regions, hashing and raster helpers."""

from __future__ import annotations

import numpy as np
import pytest

from worldgen.config import load_config, parse_override
from worldgen.errors import ConfigError
from worldgen.hashing import rng_for, sha256_json, stable_u64
from worldgen.sources.rasters import chm_quadkeys, copernicus_tile_name, quadkey, worldcover_tile
from worldgen.spatial.frame import GeoPoint, LocalFrame
from worldgen.spatial.region import Region

REGION = {"center_lat": 25.033, "center_lon": 121.5654, "radius_m": 1000.0}


def test_profiles_load_and_extend():
    basic = load_config("basic", region=REGION)
    proc = load_config("procedural", region=REGION)
    high = load_config("high_detail", region=REGION)
    assert basic.buildings.procedural_variation == 0.0
    assert proc.buildings.procedural_variation == 0.25
    assert high.buildings.procedural_variation == 0.25          # inherited through two levels
    assert high.terrain.resolution == 2.0
    assert basic.hash() != proc.hash()
    assert basic.hash() == load_config("basic", region=REGION).hash()


def test_overrides_and_validation():
    cfg = load_config("basic", overrides=["roads.lane_width=3.5", "tiling.tile_size=128"], region=REGION)
    assert cfg.roads.lane_width == 3.5 and cfg.tiling.tile_size == 128.0
    assert parse_override("a.b=1") == {"a": {"b": 1}}
    with pytest.raises(ConfigError):
        load_config("basic", overrides=["roads.no_such_key=1"], region=REGION)
    with pytest.raises(ConfigError):
        load_config("basic", overrides=["terrain.resolution=3"], region=REGION)  # 256 / 3 not integral
    with pytest.raises(ConfigError):
        load_config("basic", overrides=["roads.lane_width=wide"], region=REGION)
    with pytest.raises(ConfigError):
        load_config("basic")  # no region


@pytest.mark.parametrize("lat,lon", [(25.033, 121.5654), (-33.86, 151.21), (51.5, -0.12), (64.1, -21.9),
                                     (0.0, 179.99), (-54.8, -68.3)])
@pytest.mark.parametrize("method", ["tmerc", "utm"])
def test_frame_roundtrip(lat: float, lon: float, method: str):
    f = LocalFrame(GeoPoint(lat, lon), method)
    x0, y0 = f.point_to_local(GeoPoint(lat, lon))
    assert abs(x0) < 1e-6 and abs(y0) < 1e-6  # origin maps to (0, 0)
    xs = np.linspace(-5000, 5000, 11)
    gx, gy = np.meshgrid(xs, xs)
    lo, la = f.to_wgs84(gx.ravel(), gy.ravel())
    bx, by = f.to_local(lo, la)
    assert np.max(np.hypot(bx - gx.ravel(), by - gy.ravel())) < 1e-6


def test_frame_metric_and_north_up():
    f = LocalFrame(GeoPoint(25.033, 121.5654))
    # 0.01 degree of latitude is ~1108 m at 25N; local y must point north.
    x, y = f.point_to_local(GeoPoint(25.043, 121.5654))
    assert abs(x) < 1e-6 and 1100 < y < 1115
    x, y = f.point_to_local(GeoPoint(25.033, 121.5754))
    assert x > 1000 and abs(y) < 1.0
    assert abs(f.scale_factor(1000, 1000) - 1.0) < 1e-6


def test_region_extent_and_tiles():
    cfg = load_config("basic", region=REGION)
    r = Region.from_config(cfg)
    assert r.extent == (-1024.0, -1024.0, 1024.0, 1024.0)
    tiles = r.tiles()
    assert len(tiles) == 60  # 8x8 minus the four corners outside the 1 km circle
    assert (0, 0) in tiles and (-4, -4) not in tiles
    w, s, e, n = r.wgs84_bbox()
    assert w < 121.5654 < e and s < 25.033 < n
    bb = load_config("basic", region={"bbox": [25.02, 121.55, 25.04, 121.57]})
    rb = Region.from_config(bb)
    assert rb.radius_m is None and rb.footprint.area > 0


def test_hashing_is_stable():
    assert sha256_json({"b": 1, "a": [1.0, 2]}) == sha256_json({"a": [1.0, 2], "b": 1})
    assert stable_u64("x", 1) == stable_u64("x", 1) != stable_u64("x", 2)
    assert np.array_equal(rng_for(1, "t", 3).random(5), rng_for(1, "t", 3).random(5))


def test_tile_names():
    assert copernicus_tile_name(25, 121) == "Copernicus_DSM_COG_10_N25_00_E121_00_DEM"
    assert copernicus_tile_name(-6, -76) == "Copernicus_DSM_COG_10_S06_00_W076_00_DEM"
    assert worldcover_tile(24, 120) == "N24E120"
    assert worldcover_tile(-3, -63) == "S03W063"
    assert quadkey(3, 5, 3) == "213"
    qk = chm_quadkeys((121.55, 25.02, 121.58, 25.05), 7)
    assert qk and all(len(q) == 7 for q in qk)
