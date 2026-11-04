"""Raster source adapters: elevation (DEM), land cover (ESA WorldCover) and canopy height.

All three read windows from public Cloud-Optimized GeoTIFFs over HTTP, so only
the bytes covering the region are transferred. Windows are cached as GeoTIFF.
A tile that does not exist upstream (e.g. open ocean for Copernicus DEM) is
reported as missing coverage rather than an error.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
import requests
from pyproj import CRS, Transformer
from rasterio.merge import merge
from rasterio.transform import from_origin

from worldgen.config import CanopySourceConfig, DemSourceConfig, WorldCoverSourceConfig
from worldgen.errors import SourceError
from worldgen.log import get_logger
from worldgen.sources.base import SourceCache, SourceRecord, read_geotiff
from worldgen.spatial.raster import GeoRaster

log = get_logger("sources.rasters")

GDAL_ENV = {
    "GDAL_DISABLE_READDIR_ON_OPEN": "EMPTY_DIR",
    "CPL_VSIL_CURL_ALLOWED_EXTENSIONS": ".tif",
    "GDAL_HTTP_MAX_RETRY": "4",
    "GDAL_HTTP_RETRY_DELAY": "1",
    "VSI_CACHE": "TRUE",
}

WGS84 = CRS.from_epsg(4326)


def _url_exists(url: str) -> bool:
    try:
        r = requests.head(url, timeout=30, allow_redirects=True)
    except requests.RequestException as exc:
        raise SourceError(f"cannot reach {url}: {exc}") from exc
    if r.status_code in (403, 404):
        return False
    if r.status_code >= 400:
        raise SourceError(f"{url}: HTTP {r.status_code}")
    return True


def read_window_mosaic(
    urls: list[str], bbox_wgs84: tuple[float, float, float, float], dtype: str, nodata: float | int | None
) -> tuple[GeoRaster | None, list[str], list[str]]:
    """Mosaic the window covering ``bbox_wgs84`` from several COGs sharing a CRS.

    Returns (raster or None, urls used, urls missing upstream).
    """
    present = [u for u in urls if _url_exists(u)]
    missing = [u for u in urls if u not in present]
    if not present:
        return None, [], missing
    with rasterio.Env(**GDAL_ENV):
        srcs = [rasterio.open(f"/vsicurl/{u}") for u in present]
        try:
            crs = CRS.from_user_input(srcs[0].crs)
            w, s, e, n = bbox_wgs84
            if not crs.equals(WGS84):
                t = Transformer.from_crs(WGS84, crs, always_xy=True)
                xs, ys = t.transform([w, e, w, e, (w + e) / 2], [s, s, n, n, (s + n) / 2])
                bounds = (min(xs), min(ys), max(xs), max(ys))
            else:
                bounds = (w, s, e, n)
            res = srcs[0].res
            pad = 2
            bounds = (bounds[0] - pad * res[0], bounds[1] - pad * res[1], bounds[2] + pad * res[0], bounds[3] + pad * res[1])
            src_nodata = srcs[0].nodata
            if src_nodata is not None:
                fill: float | int = src_nodata
            elif np.issubdtype(np.dtype(srcs[0].dtypes[0]), np.floating):
                fill = float("nan")
            else:
                fill = nodata if nodata is not None else 0
            arr, transform = merge(srcs, bounds=bounds, res=res, nodata=fill)
        finally:
            for ds in srcs:
                ds.close()
    data = arr[0]
    if np.issubdtype(np.dtype(dtype), np.floating):
        data = data.astype(np.float32)
        if src_nodata is not None:
            data = np.where(data == src_nodata, np.nan, data)
        out_nodata: float | int | None = float("nan")
    else:
        data = data.astype(dtype)
        out_nodata = nodata
    return GeoRaster(data, transform, crs, out_nodata), present, missing


def _read_local(path: str, name: str) -> GeoRaster:
    p = Path(path)
    if not p.is_file():
        raise SourceError(f"{name}: local file {p} not found")
    return read_geotiff(p, name=name)


# --------------------------------------------------------------------------------------
# Copernicus GLO-30 DEM

COP_LICENSE = "Copernicus DEM licence (free use with attribution)"
COP_ATTRIBUTION = (
    "Copernicus DEM GLO-30 © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 "
    "provided under COPERNICUS by the European Union and ESA; all rights reserved"
)


def copernicus_tile_name(lat_floor: int, lon_floor: int) -> str:
    ns = "N" if lat_floor >= 0 else "S"
    ew = "E" if lon_floor >= 0 else "W"
    return f"Copernicus_DSM_COG_10_{ns}{abs(lat_floor):02d}_00_{ew}{abs(lon_floor):03d}_00_DEM"


def _degree_tiles(bbox: tuple[float, float, float, float], step: int) -> list[tuple[int, int]]:
    w, s, e, n = bbox
    lats = range(math.floor(s / step) * step, math.floor(n / step) * step + 1, step)
    lons = range(math.floor(w / step) * step, math.floor(e / step) * step + 1, step)
    return [(la, lo) for la in lats for lo in lons]


@dataclass
class DemResult:
    raster: GeoRaster | None
    record: SourceRecord


def fetch_dem(cfg: DemSourceConfig, bbox: tuple[float, float, float, float], cache: SourceCache) -> DemResult:
    if not cfg.enabled:
        return DemResult(None, SourceRecord("dem", cfg.provider, "-", "-", status="disabled"))
    if cfg.local_path:
        r = _read_local(cfg.local_path, "dem")
        rec = SourceRecord("dem", "local_file", "see file", f"local DEM {cfg.local_path}",
                           url=cfg.local_path, vertical_datum=cfg.vertical_datum)
        return DemResult(r, rec)
    if cfg.provider != "copernicus_glo30":
        raise SourceError(f"unsupported DEM provider {cfg.provider!r}")
    names = [copernicus_tile_name(la, lo) for la, lo in _degree_tiles(bbox, 1)]
    urls = [cfg.url_template.format(name=nm) for nm in names]

    def fetch() -> tuple[GeoRaster | None, dict[str, Any]]:
        r, used, missing = read_window_mosaic(urls, bbox, "float32", None)
        return r, {"urls": used, "missing_tiles": missing}

    raster, meta = cache.get_raster("dem", ["copernicus_glo30", [round(v, 7) for v in bbox], urls], fetch)
    rec = SourceRecord(
        name="dem", provider="copernicus_glo30", license=COP_LICENSE, attribution=COP_ATTRIBUTION,
        dataset_version="GLO-30 (2021 release, COG on AWS Open Data)", url=", ".join(meta.get("urls", [])) or None,
        retrieved_utc=meta.get("retrieved_utc"), snapshot_hash=meta.get("sha256"),
        vertical_datum="EGM2008",
    )
    if meta.get("missing_tiles"):
        rec.notes.append(f"upstream tiles absent (usually open water): {len(meta['missing_tiles'])}")
    if raster is None:
        rec.status = "missing"
    else:
        rec.coverage["valid_fraction"] = round(raster.coverage_fraction(), 4)
        if raster.coverage_fraction() < 0.999:
            rec.status = "partial"
    return DemResult(raster, rec)


# --------------------------------------------------------------------------------------
# ESA WorldCover

WC_LICENSE = "CC-BY-4.0"
WC_ATTRIBUTION = "© ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium"

WORLDCOVER_CLASSES = {
    10: "tree_cover", 20: "shrubland", 30: "grassland", 40: "cropland", 50: "built_up",
    60: "bare_sparse", 70: "snow_ice", 80: "permanent_water", 90: "herbaceous_wetland",
    95: "mangroves", 100: "moss_lichen",
}


def worldcover_tile(lat_floor3: int, lon_floor3: int) -> str:
    ns = "N" if lat_floor3 >= 0 else "S"
    ew = "E" if lon_floor3 >= 0 else "W"
    return f"{ns}{abs(lat_floor3):02d}{ew}{abs(lon_floor3):03d}"


def fetch_worldcover(
    cfg: WorldCoverSourceConfig, bbox: tuple[float, float, float, float], cache: SourceCache
) -> tuple[GeoRaster | None, SourceRecord]:
    if not cfg.enabled:
        return None, SourceRecord("worldcover", "esa_worldcover", "-", "-", status="disabled")
    if cfg.local_path:
        r = _read_local(cfg.local_path, "worldcover")
        return r, SourceRecord("worldcover", "local_file", "see file", f"local land cover {cfg.local_path}",
                               url=cfg.local_path)
    tiles = [worldcover_tile(la, lo) for la, lo in _degree_tiles(bbox, 3)]
    urls = [cfg.url_template.format(version=cfg.version, year=cfg.year, tile=t) for t in tiles]

    def fetch() -> tuple[GeoRaster | None, dict[str, Any]]:
        r, used, missing = read_window_mosaic(urls, bbox, "uint8", 0)
        return r, {"urls": used, "missing_tiles": missing}

    raster, meta = cache.get_raster("worldcover", ["esa_worldcover", cfg.version, cfg.year,
                                                   [round(v, 7) for v in bbox], urls], fetch)
    rec = SourceRecord(
        name="worldcover", provider="esa_worldcover", license=WC_LICENSE, attribution=WC_ATTRIBUTION,
        dataset_version=f"{cfg.version} ({cfg.year})", url=", ".join(meta.get("urls", [])) or None,
        retrieved_utc=meta.get("retrieved_utc"), snapshot_hash=meta.get("sha256"),
        coverage={"observation_year": cfg.year},
    )
    if raster is None:
        rec.status = "missing"
    else:
        if raster.nodata is None:
            raster.nodata = 0
        rec.coverage["valid_fraction"] = round(raster.coverage_fraction(), 4)
        vals, counts = np.unique(raster.data[raster.valid_mask()], return_counts=True)
        tot = counts.sum() or 1
        rec.coverage["class_fractions"] = {
            WORLDCOVER_CLASSES.get(int(v), str(int(v))): round(float(c) / tot, 4) for v, c in zip(vals, counts)
        }
    return raster, rec


# --------------------------------------------------------------------------------------
# Meta / WRI canopy height (CHM v2)

CHM_LICENSE = "CC-BY-4.0"
CHM_ATTRIBUTION = "Meta and World Resources Institute (WRI), Global Canopy Height Map v2 (CHMv2), imagery 2019"


def quadkey(tx: int, ty: int, level: int) -> str:
    digits = []
    for i in range(level, 0, -1):
        d = 0
        mask = 1 << (i - 1)
        if tx & mask:
            d += 1
        if ty & mask:
            d += 2
        digits.append(str(d))
    return "".join(digits)


def _webmercator_tile(lon: float, lat: float, level: int) -> tuple[int, int]:
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 1 << level
    x = int((lon + 180.0) / 360.0 * n)
    s = math.sin(math.radians(lat))
    y = int((0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n)
    return min(max(x, 0), n - 1), min(max(y, 0), n - 1)


def chm_quadkeys(bbox: tuple[float, float, float, float], level: int) -> list[str]:
    w, s, e, n = bbox
    x0, y0 = _webmercator_tile(w, n, level)
    x1, y1 = _webmercator_tile(e, s, level)
    return [quadkey(x, y, level) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]


def fetch_canopy_height(
    cfg: CanopySourceConfig, bbox: tuple[float, float, float, float], cache: SourceCache
) -> tuple[GeoRaster | None, SourceRecord]:
    if not cfg.enabled:
        return None, SourceRecord("canopy_height", cfg.provider, "-", "-", status="disabled")
    if cfg.local_path:
        r = _read_local(cfg.local_path, "canopy_height")
        return r, SourceRecord("canopy_height", "local_file", "see file", f"local CHM {cfg.local_path}",
                               url=cfg.local_path)
    qks = chm_quadkeys(bbox, cfg.quadkey_level)
    urls = [cfg.url_template.format(quadkey=q) for q in qks]

    def fetch() -> tuple[GeoRaster | None, dict[str, Any]]:
        r, used, missing = read_window_mosaic(urls, bbox, "uint8", 255)
        return r, {"urls": used, "missing_tiles": missing}

    raster, meta = cache.get_raster("canopy_height", ["meta_chm_v2", [round(v, 7) for v in bbox], urls], fetch)
    rec = SourceRecord(
        name="canopy_height", provider=cfg.provider, license=CHM_LICENSE, attribution=CHM_ATTRIBUTION,
        dataset_version="CHMv2 10 m mean (2026-08 derivative)", url=", ".join(meta.get("urls", [])) or None,
        retrieved_utc=meta.get("retrieved_utc"), snapshot_hash=meta.get("sha256"),
        coverage={"observation_year": 2019, "statistic": "mean canopy height (m above ground)"},
    )
    if raster is None:
        rec.status = "missing"
        return None, rec
    data = raster.data.astype(np.float32)
    valid = raster.data != 255
    data = np.where(valid, data, np.nan)
    out = GeoRaster(data, raster.transform, raster.crs, float("nan"), "canopy_height")
    rec.coverage["valid_fraction"] = round(float(valid.mean()), 4)
    return out, rec


def synthetic_geo_raster(data: np.ndarray, xmin: float, ymax: float, res: float, crs: CRS,
                         nodata: float | int | None = None, name: str = "") -> GeoRaster:
    """Helper for fixtures: build a north-up GeoRaster from an array (row 0 = north)."""
    return GeoRaster(data, from_origin(xmin, ymax, res, res), crs, nodata, name)
