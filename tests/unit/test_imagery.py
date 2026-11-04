"""Imagery source: configuration, Sentinel-2 SCL masking and median composite, XYZ tile math, caching."""

from __future__ import annotations

import datetime as dt

import numpy as np
import pytest
from pyproj import CRS, Transformer
from rasterio.transform import from_origin

from worldgen.config import ImagerySourceConfig, load_config, parse_date_range, validate_imagery
from worldgen.errors import ConfigError, OfflineCacheMiss, SourceError
from worldgen.export.ortho import encode_jpeg
from worldgen.sources import imagery
from worldgen.sources.base import SourceCache
from worldgen.sources.imagery import (
    GROWING_SEASON_NORTH,
    GROWING_SEASON_SOUTH,
    HOLE_FILL_SCENES,
    MERC_HALF,
    Scene,
    fallback_count,
    month_intervals,
    preferred_months,
    decode_tile,
    fetch_imagery,
    median_composite,
    mosaic_tiles,
    scenes_from_items,
    scl_clear_mask,
    select_scenes,
    xyz_tile_bounds,
    xyz_tile_range,
    xyz_tile_url,
    xyz_tile_xy,
)
from worldgen.sources.synthetic import ORIGIN
from worldgen.spatial.raster import GeoRaster

REGION = {"center_lat": ORIGIN.lat, "center_lon": ORIGIN.lon, "radius_m": 300.0}
XYZ = ["sources.imagery.provider=xyz", "sources.imagery.url_template=https://tiles.example/{z}/{x}/{y}.jpg"]
BBOX = (120.998, 23.998, 121.002, 24.002)
MERC = CRS.from_epsg(3857)
WGS84 = CRS.from_epsg(4326)


# --------------------------------------------------------------------------------------
# configuration


def test_imagery_is_enabled_by_default():
    cfg = load_config("basic", region=REGION)
    im = cfg.sources.imagery
    assert im.enabled is True and im.provider == "sentinel2_l2a" and im.season == "growing"


@pytest.mark.parametrize("extra", [
    [],                                                       # neither
    ["sources.imagery.license=CC-BY-4.0"],                    # no attribution
    ["sources.imagery.attribution=© Someone"],                # no licence
    ["sources.imagery.license=' '", "sources.imagery.attribution=© Someone"],   # blank licence
])
def test_xyz_requires_license_and_attribution(extra):
    with pytest.raises(ConfigError, match="license and attribution"):
        load_config("basic", overrides=XYZ + extra, region=REGION)


def test_xyz_with_terms_is_valid():
    cfg = load_config("basic", overrides=XYZ + ["sources.imagery.license=CC-BY-4.0",
                                                "sources.imagery.attribution=© Someone"], region=REGION)
    assert cfg.sources.imagery.provider == "xyz"


@pytest.mark.parametrize("change,match", [
    ({"provider": "bing"}, "provider"),
    ({"tile_px_max": 1000}, "power of two"),
    ({"tile_px_max": 16}, "power of two"),
    ({"jpeg_quality": 0}, "jpeg_quality"),
    ({"max_m_per_px": 0.0}, "max_m_per_px"),
    ({"max_scenes": 0}, "max_scenes"),
    ({"max_cloud_pct": 0.0}, "max_cloud_pct"),
    ({"date_range": "last year"}, "date range"),
    ({"provider": "xyz", "url_template": "https://t/{z}/{x}.png", "license": "x", "attribution": "y"}, "{y}"),
    ({"provider": "local"}, "local_path"),
    ({"season": "summer"}, "season"),
    ({"season": "custom"}, "season_months"),
    ({"season": "custom", "season_months": [6, 13]}, "1-12"),
])
def test_invalid_imagery_config(change, match):
    with pytest.raises(ConfigError, match=match):
        validate_imagery(ImagerySourceConfig(**change))


def test_nlsc_example_profile_loads():
    cfg = load_config("imagery_taiwan_nlsc", region=REGION)
    im = cfg.sources.imagery
    assert im.enabled and im.provider == "xyz"
    assert im.url_template.endswith("/GoogleMapsCompatible/{z}/{y}/{x}")
    assert "Open Government Data License" in im.license and "NLSC" not in im.license
    assert "National Land Surveying and Mapping Center" in im.attribution
    assert cfg.terrain.resolution == 4.0          # extends basic


def test_date_ranges():
    today = dt.date(2026, 11, 4)
    assert parse_date_range("P3Y", today) == (dt.date(2023, 11, 4), today)
    assert parse_date_range("p18m", today) == (dt.date(2025, 5, 4), today)
    assert parse_date_range("P1M", dt.date(2026, 3, 31)) == (dt.date(2026, 2, 28), dt.date(2026, 3, 31))
    assert parse_date_range("2024-01-01/2024-12-31", today) == (dt.date(2024, 1, 1), dt.date(2024, 12, 31))
    for bad in ("P0Y", "2024-12-31/2024-01-01", "2024-01-01", "2024-13-01/2025-01-01"):
        with pytest.raises(ConfigError):
            parse_date_range(bad, today)


# --------------------------------------------------------------------------------------
# Sentinel-2 composite


def test_scl_mask_classes():
    scl = np.arange(12, dtype=np.uint8)
    assert scl_clear_mask(scl).tolist() == [False, False, True, False, True, True, True, True,
                                            False, False, False, False]


def _px(r: int, g: int, b: int) -> list[int]:
    return [r, g, b]


def test_median_composite_masks_clouds_and_fills_holes():
    # Four pixels in a row, three scenes ordered least-cloudy first.
    #   p0: clear everywhere            -> median of the three scenes
    #   p1: cloud (SCL 9) in scene 0    -> median of scenes 1 and 2 (mean of the two)
    #   p2: masked in every scene       -> raw value of scene 0 (the least cloudy)
    #   p3: no data in any scene        -> nodata (all zero), invalid
    rgb = np.zeros((3, 3, 1, 4), dtype=np.uint8)
    scl = np.full((3, 1, 4), 4, dtype=np.uint8)
    vals = [
        [_px(10, 20, 30), _px(250, 250, 250), _px(77, 77, 77), _px(0, 0, 0)],
        [_px(12, 22, 32), _px(40, 50, 60), _px(88, 88, 88), _px(0, 0, 0)],
        [_px(90, 90, 90), _px(44, 54, 64), _px(99, 99, 99), _px(0, 0, 0)],
    ]
    for s in range(3):
        for p in range(4):
            rgb[s, :, 0, p] = vals[s][p]
    scl[0, 0, 1] = 9            # cloud over p1 in scene 0
    scl[:, 0, 2] = [8, 3, 10]   # p2: cloud, shadow, cirrus
    scl[:, 0, 3] = 0
    out, valid, stats = median_composite(rgb, scl_clear_mask(scl))
    assert out.dtype == np.uint8 and out.shape == (3, 1, 4)
    assert out[:, 0, 0].tolist() == [12, 22, 32]
    assert out[:, 0, 1].tolist() == [42, 52, 62]
    assert out[:, 0, 2].tolist() == [77, 77, 77]
    assert out[:, 0, 3].tolist() == [0, 0, 0]
    assert valid.tolist() == [[True, True, True, False]]
    assert stats["composited_fraction"] == 0.5 and stats["filled_fraction"] == 0.25
    assert stats["valid_fraction"] == 0.75


def test_median_composite_ignores_scene_nodata_and_keeps_valid_pixels_nonzero():
    rgb = np.zeros((2, 3, 1, 2), dtype=np.uint8)
    rgb[0, :, 0, 0] = 0               # scene 0 has no data at p0 (outside its swath)
    rgb[1, :, 0, 0] = [60, 70, 80]
    clear = np.ones((2, 1, 2), dtype=bool)
    out, valid, _ = median_composite(rgb, clear)
    assert out[:, 0, 0].tolist() == [60, 70, 80]
    assert not valid[0, 1] and out[:, 0, 1].tolist() == [0, 0, 0]


def _item(iid: str, date: str, cc: float, mgrs: str = "51RUH", assets: bool = True) -> dict:
    a = {"visual": {"href": f"https://x/{iid}/TCI.tif"}, "scl": {"href": f"https://x/{iid}/SCL.tif"}} if assets else {}
    return {"id": iid, "properties": {"datetime": f"{date}T02:40:00Z", "eo:cloud_cover": cc, "proj:epsg": 32651,
                                      "s2:mgrs_tile": mgrs}, "assets": a}


def test_scenes_from_items_dedupes_reprocessed_and_filters():
    items = [
        _item("S2B_51RUH_20240214_1_L2A", "2024-02-14", 0.0045),
        _item("S2B_51RUH_20240214_0_L2A", "2024-02-14", 0.0061),     # same acquisition, reprocessed
        _item("S2B_51RUG_20240214_0_L2A", "2024-02-14", 0.5, "51RUG"),  # same date, other tile: kept
        _item("S2A_51RUH_20250114_0_L2A", "2025-01-14", 25.0),       # above the cloud limit
        _item("S2A_51RUH_20250301_0_L2A", "2025-03-01", 3.0, assets=False),
        _item("S2A_51RUH_20250401_0_L2A", "2025-04-01", 1.0),
    ]
    scenes = scenes_from_items(items, 20.0)
    assert [s.id for s in scenes] == ["S2B_51RUH_20240214_1_L2A", "S2B_51RUG_20240214_0_L2A",
                                      "S2A_51RUH_20250401_0_L2A"]
    assert scenes[0].epsg == 32651 and scenes[0].visual.endswith("TCI.tif") and scenes[0].date == "2024-02-14"


def test_select_scenes_prefers_locally_clear_one_per_date():
    def sc(iid: str, date: str, cc: float, clear: float) -> Scene:
        return Scene(iid, f"{date}T00:00:00Z", cc, 32651, "51RUH", "v", "s", clear)

    cands = [sc("a", "2024-01-01", 0.1, 0.30), sc("a2", "2024-01-01", 0.2, 0.90), sc("b", "2024-02-01", 1.0, 1.0),
             sc("c", "2024-03-01", 2.0, 0.02), sc("d", "2024-04-01", 5.0, 0.60)]
    assert [s.id for s in select_scenes(cands, 5)] == ["b", "a2", "d"]   # c is below the clear minimum
    assert [s.id for s in select_scenes(cands, 2)] == ["b", "a2"]
    assert [s.id for s in select_scenes([sc("x", "2024-01-01", 1, 0.01)], 3)] == ["x"]   # better than nothing
    assert select_scenes([sc("y", "2024-01-01", 1, 0.0)], 3) == []


def _cfg(**kw) -> ImagerySourceConfig:  # noqa: ANN003
    c = ImagerySourceConfig(enabled=True, **kw)
    validate_imagery(c)
    return c


def test_sentinel_composite_is_cached_and_offline_rebuilds_from_cache(tmp_path, monkeypatch):
    utm = CRS.from_epsg(32651)
    data = np.full((3, 20, 20), 50, dtype=np.uint8)
    raster = GeoRaster(data, from_origin(300000, 2800020, 10, 10), utm, 0, "imagery")
    calls = []

    def fake(cfg, bbox):  # noqa: ANN001, ANN202
        calls.append(bbox)
        return raster, {"date_range": ["2023-11-04", "2026-11-04"], "composite": {"valid_fraction": 1.0},
                        "scenes": [{"id": "S2B_x_20240214_0_L2A", "datetime": "2024-02-14T02:40:51Z",
                                    "cloud_cover": 0.1, "clear_fraction": 1.0},
                                   {"id": "S2A_x_20260114_0_L2A", "datetime": "2026-01-14T02:40:47Z",
                                    "cloud_cover": 0.2, "clear_fraction": 0.9}]}

    monkeypatch.setattr(imagery, "_sentinel_composite", fake)
    cfg = _cfg()
    r1, rec1 = fetch_imagery(cfg, BBOX, SourceCache(tmp_path))
    r2, rec2 = fetch_imagery(cfg, BBOX, SourceCache(tmp_path, offline=True))
    assert len(calls) == 1
    assert np.array_equal(r1.data, r2.data) and r2.crs.equals(utm) and r2.data.shape == (3, 20, 20)
    assert rec1.snapshot_hash == rec2.snapshot_hash and rec2.status == "ok"
    assert rec2.name == "imagery" and rec2.provider == "sentinel2_l2a"
    assert rec2.license == "Copernicus Sentinel data licence (free, full and open)"
    assert rec2.attribution == "Contains modified Copernicus Sentinel data 2024-2026"
    assert [s["id"] for s in rec2.coverage["scenes"]] == ["S2B_x_20240214_0_L2A", "S2A_x_20260114_0_L2A"]
    assert rec2.coverage["m_per_px"] == pytest.approx(10.0, abs=0.05)
    assert rec2.coverage["valid_fraction"] == 1.0
    with pytest.raises(OfflineCacheMiss):
        fetch_imagery(_cfg(max_scenes=3), BBOX, SourceCache(tmp_path, offline=True))


def test_disabled_imagery_record():
    raster, rec = fetch_imagery(ImagerySourceConfig(enabled=False), BBOX, None)  # type: ignore[arg-type]
    assert raster is None and rec.status == "disabled" and rec.name == "imagery"


# --------------------------------------------------------------------------------------
# XYZ tiles


def test_xyz_tile_xy_known_values():
    assert xyz_tile_xy(-180.0, 85.05112878, 3) == pytest.approx((0.0, 0.0), abs=1e-6)
    assert xyz_tile_xy(0.0, 0.0, 1) == pytest.approx((1.0, 1.0))
    x, y = xyz_tile_xy(121.537628, 25.17667, 17)
    assert (int(x), int(y)) == (109786, 56059)


def test_xyz_tile_range_covers_bbox():
    bbox = (121.5148, 25.1559, 121.5605, 25.1975)
    x0, y0, x1, y1 = xyz_tile_range(bbox, 17)
    assert x0 <= x1 and y0 <= y1
    to_merc = Transformer.from_crs(WGS84, MERC, always_xy=True)
    w, s = to_merc.transform(bbox[0], bbox[1])
    e, n = to_merc.transform(bbox[2], bbox[3])
    bx0, _, _, by1 = xyz_tile_bounds(x0, y0, 17)          # north-west tile
    _, by0, bx1, _ = xyz_tile_bounds(x1, y1, 17)          # south-east tile
    assert bx0 <= w and by1 >= n and bx1 >= e and by0 <= s
    # Tight: dropping an outer row or column would leave part of the bbox uncovered.
    assert xyz_tile_bounds(x0 + 1, y0, 17)[0] > w and xyz_tile_bounds(x1 - 1, y1, 17)[2] < e
    assert xyz_tile_bounds(x0, y0 + 1, 17)[3] < n and xyz_tile_bounds(x1, y1 - 1, 17)[1] > s


def test_xyz_tile_range_edges_and_world():
    assert xyz_tile_range((-180, -85, 180, 85), 0) == (0, 0, 0, 0)
    assert xyz_tile_range((-180, -85, 180, 85), 2) == (0, 0, 3, 3)
    # A bbox that ends exactly on a tile edge does not pull in the next tile.
    to_geo = Transformer.from_crs(MERC, WGS84, always_xy=True)
    x0, y0, x1, y1 = xyz_tile_bounds(5, 9, 4)
    w, s = to_geo.transform(x0 + 1.0, y0 + 1.0)
    e, n = to_geo.transform(x1, y1)
    assert xyz_tile_range((w, s, e, n), 4) == (5, 9, 5, 9)


def test_xyz_tile_bounds_and_url():
    assert xyz_tile_bounds(0, 0, 0) == pytest.approx((-MERC_HALF, -MERC_HALF, MERC_HALF, MERC_HALF))
    assert xyz_tile_bounds(1, 0, 1) == pytest.approx((0.0, 0.0, MERC_HALF, MERC_HALF))     # north-east quadrant
    assert xyz_tile_url("https://w/PHOTO2/default/GoogleMapsCompatible/{z}/{y}/{x}", 17, 109786, 56059) == \
        "https://w/PHOTO2/default/GoogleMapsCompatible/17/56059/109786"


def _tile(rgb: tuple[int, int, int], px: int = 16) -> np.ndarray:
    return np.broadcast_to(np.array(rgb, dtype=np.uint8)[:, None, None], (3, px, px)).copy()


def test_mosaic_is_north_up_and_georeferenced():
    tiles = {(10, 20): _tile((255, 1, 1)), (11, 20): _tile((1, 255, 1)), (10, 21): _tile((1, 1, 255)), (11, 21): None}
    r = mosaic_tiles(tiles, 10, 20, 11, 21, 6)
    assert r.crs.equals(MERC) and r.data.shape == (3, 32, 32)
    assert r.data[:, 0, 0].tolist() == [255, 1, 1]          # north-west
    assert r.data[:, 0, 31].tolist() == [1, 255, 1]         # north-east
    assert r.data[:, 31, 0].tolist() == [1, 1, 255]         # south-west
    assert r.data[:, 31, 31].tolist() == [0, 0, 0]          # missing tile = nodata
    xmin, _, _, ymax = xyz_tile_bounds(10, 20, 6)
    assert (r.transform.c, r.transform.f) == pytest.approx((xmin, ymax))
    assert r.transform.a == pytest.approx(2 * MERC_HALF / 64 / 16) and r.transform.e < 0


def test_decode_tile_roundtrip():
    img = np.zeros((32, 32, 3), dtype=np.uint8)
    img[:16] = (200, 120, 40)
    a = decode_tile(encode_jpeg(img, 95))
    assert a.shape == (3, 32, 32) and a.dtype == np.uint8
    assert abs(int(a[0, 4, 4]) - 200) < 8 and a[:, 28, 28].max() <= 8
    assert a.any(axis=0).all()                    # black is valid imagery, not nodata
    assert decode_tile(b"") is None and decode_tile(b"not an image") is None


def test_xyz_fetch_guard_cache_and_offline(tmp_path, monkeypatch):
    calls: list[str] = []
    jpg = encode_jpeg(np.full((256, 256, 3), 90, dtype=np.uint8), 90)

    def fake_download(session, url):  # noqa: ANN001, ANN202
        calls.append(url)
        return jpg, "ok"

    monkeypatch.setattr(imagery, "_download_tile", fake_download)
    tpl = "https://tiles.example/{z}/{x}/{y}"
    cfg = _cfg(provider="xyz", url_template=tpl, zoom=17, license="Test licence", attribution="© Test",
               max_tiles=4)
    with pytest.raises(SourceError, match="max_tiles"):
        fetch_imagery(cfg, (121.5148, 25.1559, 121.5605, 25.1975), SourceCache(tmp_path))
    assert calls == []
    bbox = (121.5370, 25.1760, 121.5380, 25.1770)
    n = (lambda r: (r[2] - r[0] + 1) * (r[3] - r[1] + 1))(xyz_tile_range(bbox, 17))
    r1, rec1 = fetch_imagery(cfg, bbox, SourceCache(tmp_path))
    assert len(calls) == n and r1.crs.equals(MERC) and r1.data.shape[0] == 3
    assert rec1.provider == "xyz" and rec1.license == "Test licence" and rec1.attribution == "© Test"
    assert rec1.coverage["zoom"] == 17 and rec1.coverage["tiles"] == n
    assert rec1.coverage["m_per_px"] == pytest.approx(1.08, abs=0.01)
    r2, rec2 = fetch_imagery(cfg, bbox, SourceCache(tmp_path, offline=True))
    assert len(calls) == n and np.array_equal(r1.data, r2.data)
    assert rec2.snapshot_hash == rec1.snapshot_hash
    with pytest.raises(OfflineCacheMiss):
        fetch_imagery(cfg, (121.60, 25.20, 121.601, 25.201), SourceCache(tmp_path, offline=True))


# --------------------------------------------------------------------------------------
# season preference


def test_preferred_months_follow_the_hemisphere():
    north, south = (121.5, 25.1, 121.6, 25.2), (151.1, -33.9, 151.2, -33.8)
    growing = ImagerySourceConfig()
    assert preferred_months(growing, north) == GROWING_SEASON_NORTH == (5, 6, 7, 8, 9, 10)
    assert preferred_months(growing, south) == GROWING_SEASON_SOUTH == (11, 12, 1, 2, 3, 4)
    assert preferred_months(growing, (10.0, -0.1, 10.1, 0.1)) == GROWING_SEASON_NORTH    # equator: north
    assert preferred_months(ImagerySourceConfig(season="custom", season_months=[9, 3, 3]), south) == (3, 9)
    assert preferred_months(ImagerySourceConfig(season="any"), north) == ()


def test_month_intervals():
    d = dt.date
    start, end = d(2023, 11, 4), d(2026, 11, 4)
    assert month_intervals(GROWING_SEASON_NORTH, start, end) == [
        (d(2024, 5, 1), d(2024, 10, 31)), (d(2025, 5, 1), d(2025, 10, 31)), (d(2026, 5, 1), d(2026, 10, 31))]
    assert month_intervals(GROWING_SEASON_SOUTH, start, end) == [
        (d(2023, 11, 4), d(2024, 4, 30)), (d(2024, 11, 1), d(2025, 4, 30)), (d(2025, 11, 1), d(2026, 4, 30)),
        (d(2026, 11, 1), d(2026, 11, 4))]
    assert month_intervals((2, 7), d(2024, 1, 15), d(2024, 12, 31)) == [
        (d(2024, 2, 1), d(2024, 2, 29)), (d(2024, 7, 1), d(2024, 7, 31))]
    assert month_intervals((), start, end) == [(start, end)]


def test_fallback_count():
    assert fallback_count(5, 5, 1.0) == 0                     # enough clear preferred scenes
    assert fallback_count(2, 5, 1.0) == 3                     # too few: fill up with other months
    assert fallback_count(0, 5, 0.0) == 5
    assert fallback_count(5, 5, 0.9) == HOLE_FILL_SCENES      # enough, but clouded holes remain


def test_tiered_composite_prefers_season_and_falls_back_per_pixel():
    # Scenes: two preferred (tier 0), one other-season (tier 1). Pixels:
    #   p0: clear in both preferred and the other   -> median of the preferred only
    #   p1: clouded in both preferred, clear in other -> the other-season value
    rgb = np.zeros((3, 3, 1, 2), dtype=np.uint8)
    rgb[0, :, 0, :] = 100
    rgb[1, :, 0, :] = 110
    rgb[2, :, 0, :] = 200
    clear = np.ones((3, 1, 2), dtype=bool)
    clear[0, 0, 1] = clear[1, 0, 1] = False
    out, valid, stats = median_composite(rgb, clear, [0, 0, 1])
    assert out[:, 0, 0].tolist() == [105, 105, 105]
    assert out[:, 0, 1].tolist() == [200, 200, 200]
    assert valid.all() and stats["preferred_fraction"] == 0.5 and stats["other_season_fraction"] == 0.5
    untiered, _, _ = median_composite(rgb, clear)
    assert untiered[:, 0, 0].tolist() == [110, 110, 110]


class FakeS2:
    """Fake STAC + COG layer: items per (id, date, cloud), SCL class and TCI colour per scene."""

    def __init__(self, scenes: list[tuple[str, str, float, int, int]]) -> None:
        self.scenes = {s[0]: s for s in scenes}
        self.searches: list[tuple[str, str]] = []

    def search(self, cfg, bbox, start, end):  # noqa: ANN001, ANN201
        self.searches.append((start, end))
        return [_item(i, d, cc) for i, d, cc, _, _ in self.scenes.values() if start <= d <= end]

    def read(self, url, crs, transform, width, height, bands):  # noqa: ANN001, ANN201
        sid = url.split("/")[-2]
        _, _, _, scl, colour = self.scenes[sid]
        if url.endswith("SCL.tif"):
            return np.full((1, height, width), scl, dtype=np.uint8)
        return np.full((3, height, width), colour, dtype=np.uint8)


WINTER = [("W1", "2025-01-10", 0.1, 4, 30), ("W2", "2025-02-10", 0.2, 4, 31), ("W3", "2024-12-10", 0.3, 4, 32),
          ("W4", "2026-01-12", 0.4, 4, 33), ("W5", "2026-03-02", 0.5, 4, 34)]


def _run_s2(monkeypatch, scenes, bbox, **cfg):  # noqa: ANN001, ANN003, ANN202
    fake = FakeS2(scenes)
    monkeypatch.setattr(imagery, "stac_search", fake.search)
    monkeypatch.setattr(imagery, "_read_on_grid", fake.read)
    monkeypatch.setattr(imagery, "_today", lambda: dt.date(2026, 11, 4))
    raster, meta = imagery._sentinel_composite(_cfg(**{"max_scenes": 5, "composite": "median", **cfg}), bbox)
    return fake, raster, meta


NORTH_BBOX = (121.53, 25.17, 121.54, 25.18)
SOUTH_BBOX = (151.20, -33.87, 151.21, -33.86)


def test_growing_season_scenes_are_preferred(monkeypatch):
    summer = [("S1", "2025-06-01", 9.0, 4, 120), ("S2", "2025-07-03", 12.0, 5, 130), ("S3", "2024-08-09", 15.0, 4, 140),
              ("S4", "2026-09-12", 18.0, 6, 150), ("S5", "2024-05-20", 19.0, 4, 160)]
    fake, raster, meta = _run_s2(monkeypatch, WINTER + summer, NORTH_BBOX)
    assert len(fake.searches) == 4                         # three May-Oct intervals, then the full range
    assert sorted(s["id"] for s in meta["scenes"]) == ["S1", "S2", "S3", "S4", "S5"]
    assert all(s["preferred_season"] for s in meta["scenes"])
    assert meta["season"]["months_used"] == [5, 6, 7, 8, 9] and meta["season"]["fallback"] is False
    assert meta["season"]["hemisphere"] == "north" and meta["season"]["preferred_months"] == [5, 6, 7, 8, 9, 10]
    assert raster.data[:, 0, 0].tolist() == [140, 140, 140]   # median of the summer colours only
    _, raster, meta = _run_s2(monkeypatch, WINTER + summer, NORTH_BBOX, composite="clear_quantile")
    assert raster.data[:, 0, 0].tolist() == [130, 130, 130]   # lower third by blue of the summer scenes
    assert meta["composite"]["method"] == "clear_quantile" and meta["composite"]["cloud_buffer_m"] == 30.0


def test_other_months_fill_in_when_the_season_is_cloudy(monkeypatch):
    summer = [("S1", "2025-06-01", 9.0, 4, 120), ("S2", "2025-07-03", 12.0, 9, 250),   # S2 clouded (SCL 9)
              ("S3", "2024-08-09", 15.0, 4, 140)]
    fake, raster, meta = _run_s2(monkeypatch, WINTER + summer, NORTH_BBOX)
    ids = [s["id"] for s in meta["scenes"]]
    assert "S2" not in ids and {"S1", "S3"} <= set(ids) and len(ids) == 5
    assert {s["id"] for s in meta["scenes"] if not s["preferred_season"]} == {"W1", "W2", "W3", "W4", "W5"} & set(ids)
    assert meta["season"]["fallback"] is True
    assert meta["season"]["months_used"] == sorted({int(s["datetime"][5:7]) for s in meta["scenes"]})
    assert raster.data[:, 0, 0].tolist() == [130, 130, 130]   # preferred pixels: median of S1 and S3 only
    assert meta["composite"]["preferred_fraction"] == 1.0


def test_southern_hemisphere_prefers_november_to_april(monkeypatch):
    summer_north = [("S1", "2025-06-01", 0.05, 4, 120), ("S2", "2025-07-03", 0.06, 4, 130)]
    fake, raster, meta = _run_s2(monkeypatch, WINTER + summer_north, SOUTH_BBOX)
    assert meta["season"]["hemisphere"] == "south"
    assert sorted(s["id"] for s in meta["scenes"]) == ["W1", "W2", "W3", "W4", "W5"]   # the austral summer
    assert meta["season"]["months_used"] == [1, 2, 3, 12] and meta["season"]["fallback"] is False


class _PassThroughCache(SourceCache):
    """Never caches: every request calls ``fetch`` (no files written)."""

    def __init__(self) -> None:
        self.offline = False
        self.last_hit = False

    def get_bytes(self, namespace, key_parts, fetch, suffix=".bin"):  # noqa: ANN001, ANN201
        return fetch()


def test_unexpected_errors_become_source_errors(monkeypatch):
    def boom(cfg, bbox):  # noqa: ANN001, ANN202
        raise KeyError("assets")

    monkeypatch.setattr(imagery, "_sentinel_composite", boom)
    with pytest.raises(SourceError, match="KeyError"):
        fetch_imagery(_cfg(), BBOX, _PassThroughCache())


def test_offline_without_cached_imagery_does_not_fail_acquisition(tmp_path):
    from worldgen.pipeline import acquire
    from worldgen.spatial.region import Region

    cfg = load_config("basic", region=REGION)
    assert cfg.sources.imagery.enabled
    bundle = acquire(Region.from_config(cfg), cfg, tmp_path / "cache", offline=True)
    rec = next(r for r in bundle.records if r.name == "imagery")
    assert bundle.imagery is None and rec.status == "failed"
    assert any("offline" in n for n in rec.notes)


def test_cloud_buffer_grows_clouds_and_shadows_only():
    scl = np.full((11, 11), 4, dtype=np.uint8)
    scl[5, 5] = 9                     # one cloud pixel
    scl[0, 0] = 7                     # unclassified (often a cloud fringe)
    scl[10, 10] = 6                   # water: never grown
    plain = scl_clear_mask(scl)
    assert (~plain).sum() == 1 and plain[0, 0] and plain[10, 9]
    buf = scl_clear_mask(scl, 3)
    assert not buf[5, 5] and not buf[5, 8] and not buf[2, 5] and not buf[6, 7]   # within 3 px (diamond)
    assert buf[5, 9]
    assert not buf[0, 3] and buf[0, 4]  # the unclassified pixel is masked and grown too
    assert buf[10, 10] and buf[10, 9]


def test_clear_quantile_rejects_haze_clouds_and_shadows():
    # One pixel, six observations all "clear" by SCL: a missed shadow, two clear views, two hazy
    # ones and a missed cloud. The per-channel median lands on haze; the clear pick does not.
    obs = [(20, 25, 30), (60, 90, 70), (64, 92, 75), (120, 130, 160), (125, 135, 170), (230, 232, 240)]
    rgb = np.array(obs, dtype=np.uint8)[:, :, None, None]
    clear = np.ones((6, 1, 1), dtype=bool)
    pick, _, _ = imagery.composite(rgb, clear, method="clear_quantile")
    med, _, _ = imagery.composite(rgb, clear, method="median")
    assert pick[:, 0, 0].tolist() == [64, 92, 75]              # an actual clear observation, unmixed
    assert med[:, 0, 0].tolist() != [64, 92, 75] and med[2, 0, 0] > 100
    # With only shadowed observations left the pick still returns one of them.
    pick2, _, _ = imagery.composite(rgb[:1], clear[:1], method="clear_quantile")
    assert pick2[:, 0, 0].tolist() == [20, 25, 30]
