"""Orthoimagery source: a Sentinel-2 L2A cloud-free composite, XYZ/WMTS tiles or a local GeoTIFF.

Every provider returns a north-up RGB :class:`GeoRaster` in the source CRS whose ``data`` has
shape ``(3, rows, cols)`` and dtype uint8. A pixel whose three bands are all 0 is nodata; valid
pixels are never all-zero. ``export/ortho.py`` bakes it per tile into the local frame.

* ``sentinel2_l2a``: STAC search (Element 84 Earth Search by default) for scenes over the bbox
  below a scene cloud-cover limit, preferring the growing season (May-Oct in the northern
  hemisphere, Nov-Apr in the southern; ``season``), which is searched interval by interval.
  The SCL (scene classification, 20 m) window of the least-cloudy candidates is read first,
  resampled nearest to the 10 m grid, to rank scenes by their *local* clear fraction
  (scene-level cloud cover says little about a 5 km region). The best ``max_scenes``
  preferred-season acquisitions (one per date) contribute their TCI (``visual``) window; other
  months are added only when too few preferred scenes are clear, and only show where every
  preferred scene is clouded. Pixels flagged nodata, saturated, cloud shadow, cloud, cirrus or
  snow are masked (clouds, shadows and unclassified pixels grown by 30 m) and each pixel takes
  the clearest observation (``composite: clear_quantile``) or the per-channel median. Pixels
  masked in every scene are filled from the least-cloudy scene. Only the composite is cached
  (a GeoTIFF keyed by the request).
* ``xyz``: EPSG:3857 tiles at ``zoom``, fetched politely (generic User-Agent, retries, a tile
  count guard). Each raw tile is cached, so interrupted downloads resume and ``--offline``
  rebuilds read exactly the bytes first fetched.
* ``local`` (or ``local_path`` with any provider): an RGB GeoTIFF read as is.
"""

from __future__ import annotations

import datetime as _dt
import math
import time
import warnings
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from typing import Any
from urllib.parse import urlparse

import numpy as np
import rasterio
import requests
from pyproj import CRS, Geod, Transformer
from rasterio.enums import Resampling
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from rasterio.transform import from_origin
from rasterio.vrt import WarpedVRT
from rasterio.warp import transform_bounds
from scipy.ndimage import binary_dilation

from worldgen import __version__
from worldgen.config import ImagerySourceConfig, parse_date_range
from worldgen.errors import SourceError
from worldgen.hashing import sha256_json
from worldgen.log import get_logger
from worldgen.sources.base import SourceCache, SourceRecord, utc_now
from worldgen.sources.rasters import GDAL_ENV, WGS84
from worldgen.spatial.raster import GeoRaster

log = get_logger("sources.imagery")

USER_AGENT = f"worldgen/{__version__} (geospatial world compiler)"
SENTINEL_LICENSE = "Copernicus Sentinel data licence (free, full and open)"
SENTINEL_ATTRIBUTION = "Contains modified Copernicus Sentinel data {years}"
# Sentinel-2 L2A scene classification classes that never contribute to the composite:
# 0 no data, 1 saturated/defective, 3 cloud shadow, 8/9 cloud medium/high probability,
# 10 thin cirrus, 11 snow or ice.
SCL_MASKED = (0, 1, 3, 8, 9, 10, 11)
SCL_BUFFERED = (3, 7, 8, 9, 10)  # grown by CLOUD_BUFFER_PX: shadow, unclassified, cloud, cirrus
CLOUD_BUFFER_PX = 3            # 30 m at 10 m
CLEAR_QUANTILE = 1.0 / 3.0     # clear_quantile composite: rank by blue among non-shadow observations
SHADOW_FRACTION = 0.6          # ... observations darker than this x the pixel's median are shadows
S2_RES = 10.0
COMPOSITE_VERSION = 3          # part of the cache key; bump when the composite changes
MIN_CLEAR_FRACTION = 0.05      # scenes clearer than this over the region take part
GROWING_SEASON_NORTH = (5, 6, 7, 8, 9, 10)
GROWING_SEASON_SOUTH = (11, 12, 1, 2, 3, 4)
MIN_PREFERRED_COVERAGE = 0.995  # below this clear coverage, other-season scenes fill the holes
HOLE_FILL_SCENES = 2
WEB_MERCATOR = CRS.from_epsg(3857)
MERC_HALF = 20037508.342789244
WORKERS = 4


def rgb_valid_mask(data: np.ndarray) -> np.ndarray:
    """Valid pixels of a (3, rows, cols) imagery array (nodata = all bands 0)."""
    return data.any(axis=0)


def _mark_nodata(rgb: np.ndarray, valid: np.ndarray) -> np.ndarray:
    rgb[:, valid & ~rgb.any(axis=0)] = 1
    rgb[:, ~valid] = 0
    return rgb


def ground_m_per_px(r: GeoRaster) -> float:
    """Geodesic ground size of one pixel at the raster centre (geometric mean of both axes)."""
    rows, cols = r.data.shape[-2:]
    c, rr = cols / 2.0, rows / 2.0
    t = r.transform
    pts = [t @ (c, rr), t @ (c + 1, rr), t @ (c, rr + 1)]
    to = Transformer.from_crs(r.crs, WGS84, always_xy=True)
    lon, lat = to.transform([p[0] for p in pts], [p[1] for p in pts])
    geod = Geod(ellps="WGS84")
    _, _, dx = geod.inv(lon[0], lat[0], lon[1], lat[1])
    _, _, dy = geod.inv(lon[0], lat[0], lon[2], lat[2])
    return float(math.sqrt(dx * dy))


def raster_to_tiff_bytes(r: GeoRaster) -> bytes:
    data = r.data if r.data.ndim == 3 else r.data[None, ...]
    with MemoryFile() as mf:
        with mf.open(driver="GTiff", width=data.shape[2], height=data.shape[1], count=data.shape[0],
                     dtype=str(data.dtype), crs=r.crs, transform=r.transform, compress="deflate",
                     tiled=data.shape[1] >= 256 and data.shape[2] >= 256) as ds:
            ds.write(data)
        return bytes(mf.read())


def tiff_bytes_to_raster(raw: bytes, name: str = "imagery") -> GeoRaster:
    with MemoryFile(raw) as mf, mf.open() as ds:
        data = ds.read()
        crs = CRS.from_user_input(ds.crs)
        transform = ds.transform
    return GeoRaster(data, transform, crs, 0, name)


def _describe(rec: SourceRecord, raster: GeoRaster | None, cfg: ImagerySourceConfig) -> None:
    if raster is None:
        rec.status = "missing"
        return
    mpp = ground_m_per_px(raster)
    valid = float(rgb_valid_mask(raster.data).mean())
    rec.coverage.update({"m_per_px": round(mpp, 3), "effective_m_per_px": round(max(mpp, cfg.max_m_per_px), 3),
                         "valid_fraction": round(valid, 4)})
    if valid < 0.999 and rec.status == "ok":
        rec.status = "partial"


def fetch_imagery(
    cfg: ImagerySourceConfig, bbox: tuple[float, float, float, float], cache: SourceCache
) -> tuple[GeoRaster | None, SourceRecord]:
    """Imagery covering ``bbox`` (west, south, east, north) and its provenance record.

    Imagery is optional: every failure is raised as :class:`SourceError` (``OfflineCacheMiss``
    when offline and uncached), which the pipeline records without failing the build.
    """
    if not cfg.enabled:
        return None, SourceRecord("imagery", cfg.provider, "-", "-", status="disabled")
    try:
        return _fetch_imagery(cfg, bbox, cache)
    except SourceError:
        raise
    except Exception as exc:  # noqa: BLE001 - unexpected upstream data must not fail a build
        raise SourceError(f"imagery: {type(exc).__name__}: {exc}") from exc


def _fetch_imagery(
    cfg: ImagerySourceConfig, bbox: tuple[float, float, float, float], cache: SourceCache
) -> tuple[GeoRaster | None, SourceRecord]:
    if cfg.local_path:
        r = read_rgb_geotiff(cfg.local_path)
        rec = SourceRecord("imagery", "local_file", cfg.license or "see file",
                           cfg.attribution or f"local imagery {cfg.local_path}", url=cfg.local_path)
        _describe(rec, r, cfg)
        return r, rec
    if cfg.provider == "sentinel2_l2a":
        return fetch_sentinel2(cfg, bbox, cache)
    if cfg.provider == "xyz":
        return fetch_xyz(cfg, bbox, cache)
    raise SourceError(f"imagery: provider {cfg.provider!r} needs sources.imagery.local_path")


# --------------------------------------------------------------------------------------
# Local GeoTIFF


def read_rgb_geotiff(path: str) -> GeoRaster:
    try:
        ds = rasterio.open(path)
    except rasterio.RasterioIOError as exc:
        raise SourceError(f"imagery: cannot read local file {path}: {exc}") from exc
    with ds:
        data = ds.read([1, 2, 3]) if ds.count >= 3 else np.repeat(ds.read(1)[None], 3, axis=0)
        valid = ds.dataset_mask() > 0
        crs = CRS.from_user_input(ds.crs) if ds.crs else None
        transform = ds.transform
    if crs is None:
        raise SourceError(f"imagery: local file {path} has no CRS")
    if data.dtype != np.uint8:
        vals = data[:, valid].astype(np.float64)
        hi = float(np.percentile(vals, 99.9)) if vals.size else 1.0
        lo = min(0.0, float(vals.min())) if vals.size else 0.0
        data = np.clip((data.astype(np.float64) - lo) / max(hi - lo, 1e-9) * 255.0, 0, 255).round().astype(np.uint8)
    return GeoRaster(_mark_nodata(np.ascontiguousarray(data), valid), transform, crs, 0, "imagery")


# --------------------------------------------------------------------------------------
# Sentinel-2 L2A composite


@dataclass
class Scene:
    id: str
    datetime: str
    cloud_cover: float
    epsg: int | None
    mgrs: str
    visual: str
    scl: str
    clear_fraction: float = 0.0

    @property
    def date(self) -> str:
        return self.datetime[:10]

    @property
    def month(self) -> int:
        return int(self.datetime[5:7])


def preferred_months(cfg: ImagerySourceConfig, bbox: tuple[float, float, float, float]) -> tuple[int, ...]:
    """Months to prefer (empty = no preference). "growing" follows the hemisphere of the bbox."""
    if cfg.season == "any":
        return ()
    if cfg.season == "custom":
        return tuple(sorted(set(cfg.season_months)))
    lat = (bbox[1] + bbox[3]) / 2.0
    return GROWING_SEASON_NORTH if lat >= 0 else GROWING_SEASON_SOUTH


def month_intervals(months: tuple[int, ...] | list[int], start: _dt.date, end: _dt.date
                    ) -> list[tuple[_dt.date, _dt.date]]:
    """Contiguous date intervals inside [start, end] whose months are all in ``months``."""
    if not months:
        return [(start, end)]
    wanted = set(months)
    out: list[tuple[_dt.date, _dt.date]] = []
    run: list[_dt.date] | None = None
    d = _dt.date(start.year, start.month, 1)
    while d <= end:
        nxt = _dt.date(d.year + d.month // 12, d.month % 12 + 1, 1)
        if d.month in wanted:
            a, b = max(d, start), min(nxt - _dt.timedelta(days=1), end)
            if run is None:
                run = [a, b]
            else:
                run[1] = b
        elif run is not None:
            out.append((run[0], run[1]))
            run = None
        d = nxt
    if run is not None:
        out.append((run[0], run[1]))
    return out


def fallback_count(chosen_preferred: int, max_scenes: int, preferred_coverage: float) -> int:
    """How many other-season scenes to add: fill up to ``max_scenes``, or fill clouded holes."""
    if chosen_preferred < max_scenes:
        return max_scenes - chosen_preferred
    return HOLE_FILL_SCENES if preferred_coverage < MIN_PREFERRED_COVERAGE else 0


def scl_clear_mask(scl: np.ndarray, buffer_px: int = 0) -> np.ndarray:
    """True where a Sentinel-2 SCL class may contribute to the composite.

    With ``buffer_px`` > 0, cloud shadow, unclassified (often cloud edges), cloud and cirrus
    pixels are also grown by that many pixels: SCL misses the fringes of small clouds.
    """
    clear = ~np.isin(scl, SCL_MASKED)
    if buffer_px > 0:
        clear &= ~binary_dilation(np.isin(scl, SCL_BUFFERED), iterations=buffer_px)
    return clear


def _median_pick(rgb: np.ndarray, use: np.ndarray) -> np.ndarray:
    stack = np.where(use[:, None], rgb.astype(np.float32), np.float32(np.nan))
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)       # all-NaN pixels are not used
        return np.nanmedian(stack, axis=0)


def _clear_quantile_pick(rgb: np.ndarray, use: np.ndarray) -> np.ndarray:
    """Per pixel, the whole RGB of one observation: drop shadows (darker than
    ``SHADOW_FRACTION`` of the pixel's median brightness), then take the observation at rank
    ``CLEAR_QUANTILE`` by blue, which haze and missed clouds raise."""
    bright = rgb.astype(np.float32).sum(axis=1)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        med = np.nanmedian(np.where(use, bright, np.float32(np.nan)), axis=0)
    keep = use & (bright >= SHADOW_FRACTION * med)
    keep = np.where(keep.any(axis=0), keep, use)
    order = np.argsort(np.where(keep, rgb[:, 2].astype(np.float32), np.float32(np.inf)), axis=0, kind="stable")
    k = np.clip(np.floor((keep.sum(axis=0) - 1) * CLEAR_QUANTILE).astype(np.int64), 0, None)
    idx = np.take_along_axis(order, k[None], axis=0)[0]
    return np.take_along_axis(rgb, np.repeat(idx[None, None], 3, axis=1), axis=0)[0].astype(np.float32)


COMPOSITE_METHODS = {"clear_quantile": _clear_quantile_pick, "median": _median_pick}


def median_composite(rgb: np.ndarray, clear: np.ndarray, tiers: np.ndarray | list[int] | None = None
                     ) -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """:func:`composite` with the per-pixel median."""
    return composite(rgb, clear, tiers, "median")


def composite(rgb: np.ndarray, clear: np.ndarray, tiers: np.ndarray | list[int] | None = None,
              method: str = "clear_quantile") -> tuple[np.ndarray, np.ndarray, dict[str, float]]:
    """Per-pixel composite of clear observations, by preference tier.

    ``rgb``: (scenes, 3, rows, cols) uint8 with all-zero pixels as nodata, scenes ordered
    least-cloudy first; ``clear``: (scenes, rows, cols) bool (e.g. :func:`scl_clear_mask`);
    ``tiers``: per scene, 0 = preferred season, 1 = other months (default all 0). Each pixel is
    reduced (``method``: ``median`` per channel, or ``clear_quantile``, see
    :func:`_clear_quantile_pick`) over the clear observations of the lowest tier that has any
    there, so other-season scenes only show where every preferred scene is clouded. Pixels
    without any clear observation are filled from the first scene that has data there.
    Returns (composite (3, rows, cols) uint8, valid mask, statistics).
    """
    reduce = COMPOSITE_METHODS[method]
    present = rgb.any(axis=1)
    use = clear & present
    count = use.sum(axis=0)
    tier = np.zeros(rgb.shape[0], dtype=np.int64) if tiers is None else np.asarray(tiers, dtype=np.int64)
    out = np.zeros(rgb.shape[1:], dtype=np.uint8)
    got = np.zeros(count.shape, dtype=bool)
    by_tier: dict[int, float] = {}
    for t in sorted(set(tier.tolist())):
        use_t = use & (tier == t)[:, None, None]
        todo = ~got & use_t.any(axis=0)
        if not todo.any():
            by_tier[t] = 0.0
            continue
        val = reduce(rgb, use_t)
        out[:, todo] = np.clip(np.rint(val[:, todo]), 0, 255).astype(np.uint8)
        got |= todo
        by_tier[t] = float(todo.mean())
    filled = np.zeros(got.shape, dtype=bool)
    for k in range(rgb.shape[0]):
        hole = ~got & ~filled & present[k]
        out[:, hole] = rgb[k][:, hole]
        filled |= hole
    valid = got | filled
    _mark_nodata(out, valid)
    stats = {
        "composited_fraction": round(float(got.mean()), 4),
        "preferred_fraction": round(by_tier.get(0, 0.0), 4),
        "other_season_fraction": round(sum(v for k, v in by_tier.items() if k > 0), 4),
        "filled_fraction": round(float(filled.mean()), 4),
        "valid_fraction": round(float(valid.mean()), 4),
        "mean_observations": round(float(count[got].mean()), 2) if got.any() else 0.0,
    }
    return out, valid, stats


def stac_search(cfg: ImagerySourceConfig, bbox: tuple[float, float, float, float], start: str, end: str
                ) -> list[dict[str, Any]]:
    body = {
        "collections": [cfg.collection],
        "bbox": [round(v, 7) for v in bbox],
        "datetime": f"{start}T00:00:00Z/{end}T23:59:59Z",
        "query": {"eo:cloud_cover": {"lt": cfg.max_cloud_pct}},
        "sortby": [{"field": "properties.eo:cloud_cover", "direction": "asc"}],
        "limit": min(100, max(6 * cfg.max_scenes, 24)),
    }
    url = cfg.stac_url.rstrip("/") + "/search"
    err: Exception | str | None = None
    for attempt in range(3):
        try:
            r = requests.post(url, json=body, timeout=60, headers={"User-Agent": USER_AGENT})
        except requests.RequestException as exc:
            err = exc
        else:
            if r.status_code == 429 or r.status_code >= 500:
                err = f"HTTP {r.status_code}"
            elif r.status_code >= 400:
                raise SourceError(f"STAC search {url}: HTTP {r.status_code}: {r.text[:300]}")
            else:
                try:
                    return list(r.json().get("features", []))
                except ValueError as exc:
                    err = exc
        log.warning("STAC search failed (attempt %d): %s", attempt + 1, err)
        time.sleep(2.0 * (attempt + 1))
    raise SourceError(f"STAC search {url} failed: {err}")


def scenes_from_items(items: list[dict[str, Any]], max_cloud_pct: float) -> list[Scene]:
    """Usable scenes, least cloudy first; reprocessed duplicates (same date and tile) collapse."""
    out: dict[tuple[str, str], Scene] = {}
    for it in items:
        p = it.get("properties", {})
        a = it.get("assets", {})
        cc = p.get("eo:cloud_cover")
        if "visual" not in a or "scl" not in a or cc is None or float(cc) >= max_cloud_pct:
            continue
        epsg = p.get("proj:epsg")
        if epsg is None and str(p.get("proj:code", "")).upper().startswith("EPSG:"):
            epsg = int(str(p["proj:code"]).split(":")[1])
        mgrs = p.get("s2:mgrs_tile") or "".join(
            str(p.get(k, "")) for k in ("mgrs:utm_zone", "mgrs:latitude_band", "mgrs:grid_square"))
        s = Scene(str(it.get("id")), str(p.get("datetime", "")), float(cc), int(epsg) if epsg else None,
                  mgrs, a["visual"]["href"], a["scl"]["href"])
        key = (s.date, s.mgrs or s.id)
        if key not in out or (s.cloud_cover, s.id) < (out[key].cloud_cover, out[key].id):
            out[key] = s
    return sorted(out.values(), key=lambda s: (s.cloud_cover, s.id))


def s2_grid(bbox: tuple[float, float, float, float], crs: CRS) -> tuple[Any, int, int]:
    """10 m grid in ``crs`` covering ``bbox``, aligned to 20 m so SCL pixels map onto 2x2 blocks."""
    xmin, ymin, xmax, ymax = transform_bounds(WGS84, crs, *bbox, densify_pts=21)
    a = 2 * S2_RES
    x0, y0 = math.floor(xmin / a) * a, math.floor(ymin / a) * a
    x1, y1 = math.ceil(xmax / a) * a, math.ceil(ymax / a) * a
    return from_origin(x0, y1, S2_RES, S2_RES), int(round((x1 - x0) / S2_RES)), int(round((y1 - y0) / S2_RES))


def _read_on_grid(url: str, crs: CRS, transform: Any, width: int, height: int, bands: list[int]) -> np.ndarray:
    """Read ``bands`` of a remote COG resampled (nearest) onto the target grid; outside = 0."""
    with rasterio.Env(**GDAL_ENV):
        try:
            with rasterio.open(f"/vsicurl/{url}") as src, WarpedVRT(
                src, crs=crs, transform=transform, width=width, height=height,
                resampling=Resampling.nearest, nodata=0,
            ) as vrt:
                return vrt.read(bands)
        except rasterio.RasterioIOError as exc:
            raise SourceError(f"cannot read {url}: {exc}") from exc


def _parallel(fn: Any, items: list[Any]) -> list[Any]:
    with ThreadPoolExecutor(max_workers=WORKERS) as ex:
        return list(ex.map(fn, items))


def select_scenes(candidates: list[Scene], max_scenes: int) -> list[Scene]:
    """At most one scene per acquisition date, clearest over the region first."""
    best: dict[str, Scene] = {}
    for s in candidates:
        if s.date not in best or s.clear_fraction > best[s.date].clear_fraction:
            best[s.date] = s
    ranked = sorted(best.values(), key=lambda s: (-s.clear_fraction, s.cloud_cover, s.id))
    keep = [s for s in ranked if s.clear_fraction >= MIN_CLEAR_FRACTION][:max_scenes]
    if not keep and ranked and ranked[0].clear_fraction > 0:
        keep = ranked[:1]
    return keep


def _sentinel_composite(cfg: ImagerySourceConfig, bbox: tuple[float, float, float, float]
                        ) -> tuple[GeoRaster | None, dict[str, Any]]:
    start, end = parse_date_range(cfg.date_range, _today())
    months = preferred_months(cfg, bbox)
    items: list[dict[str, Any]] = []
    if months:   # the least-cloudy scenes overall are often all from the dry season: search each season too
        for a, b in month_intervals(months, start, end):
            items += stac_search(cfg, bbox, a.isoformat(), b.isoformat())
    items += stac_search(cfg, bbox, start.isoformat(), end.isoformat())
    scenes = scenes_from_items(items, cfg.max_cloud_pct)
    season: dict[str, Any] = {"mode": cfg.season, "preferred_months": list(months),
                              "hemisphere": "north" if (bbox[1] + bbox[3]) / 2.0 >= 0 else "south"}
    meta: dict[str, Any] = {"stac_url": cfg.stac_url, "collection": cfg.collection,
                            "date_range": [start.isoformat(), end.isoformat()],
                            "items_found": len({it.get("id") for it in items}), "season": season}
    if not scenes:
        log.warning("imagery: no Sentinel-2 scene below %.0f%% cloud in %s..%s", cfg.max_cloud_pct, start, end)
        return None, meta
    epsgs = [s.epsg for s in scenes if s.epsg]
    crs = CRS.from_epsg(max(set(epsgs), key=epsgs.count)) if epsgs else CRS.from_epsg(32600 + _utm_zone(bbox))
    transform, width, height = s2_grid(bbox, crs)
    preferred = [s for s in scenes if not months or s.month in months]
    others = [s for s in scenes if months and s.month not in months]
    n_screen = max(2 * cfg.max_scenes, 12)
    clear_by_id: dict[str, np.ndarray] = {}
    screened: list[Scene] = []

    def screen(group: list[Scene]) -> list[Scene]:
        log.info("imagery: screening %d Sentinel-2 scenes on a %dx%d grid (%s)", len(group), width, height,
                 crs.to_string())
        scls = _parallel(lambda s: _read_on_grid(s.scl, crs, transform, width, height, [1])[0], group)
        for s, scl in zip(group, scls):
            clear = scl_clear_mask(scl, CLOUD_BUFFER_PX)
            s.clear_fraction = round(float(clear.mean()), 4)
            clear_by_id[s.id] = clear
        screened.extend(group)
        return group

    chosen_pref = select_scenes(screen(preferred[:n_screen]), cfg.max_scenes) if preferred else []
    coverage = float(np.logical_or.reduce([clear_by_id[s.id] for s in chosen_pref]).mean()) if chosen_pref else 0.0
    n_other = fallback_count(len(chosen_pref), cfg.max_scenes, coverage) if others else 0
    chosen_other = select_scenes(screen(others[:n_screen]), n_other) if n_other else []
    meta["scenes_screened"] = [{"id": s.id, "clear_fraction": s.clear_fraction} for s in screened]
    pref_ids = {s.id for s in chosen_pref}
    chosen = sorted(chosen_pref + chosen_other, key=lambda s: (-s.clear_fraction, s.cloud_cover, s.id))
    season.update({"preferred_coverage": round(coverage, 4), "fallback": bool(chosen_other),
                   "months_used": sorted({s.month for s in chosen})})
    if not chosen:
        log.warning("imagery: every candidate scene is clouded over the region")
        return None, meta
    log.info("imagery: compositing %s", ", ".join(
        f"{s.id} ({s.clear_fraction:.0%} clear{'' if s.id in pref_ids else ', other season'})" for s in chosen))
    rgbs = _parallel(lambda s: _read_on_grid(s.visual, crs, transform, width, height, [1, 2, 3]), chosen)
    comp, _, stats = composite(np.stack(rgbs).astype(np.uint8), np.stack([clear_by_id[s.id] for s in chosen]),
                               [0 if s.id in pref_ids else 1 for s in chosen], cfg.composite)
    meta.update({
        "scenes": [{**{k: v for k, v in asdict(s).items() if k in ("id", "datetime", "cloud_cover", "clear_fraction")},
                    "preferred_season": s.id in pref_ids} for s in chosen],
        "composite": {"method": cfg.composite, "cloud_buffer_m": CLOUD_BUFFER_PX * S2_RES, **stats},
        "crs": crs.to_string(), "resolution_m": S2_RES,
    })
    return GeoRaster(comp, transform, crs, 0, "imagery"), meta


def _today() -> _dt.date:
    return _dt.datetime.now(_dt.UTC).date()


def _utm_zone(bbox: tuple[float, float, float, float]) -> int:
    lon = (bbox[0] + bbox[2]) / 2.0
    return int((lon + 180.0) // 6.0) + 1


def _years_label(scenes: list[dict[str, Any]]) -> str:
    years = sorted({int(s["datetime"][:4]) for s in scenes if s.get("datetime")})
    if not years:
        return ""
    return str(years[0]) if len(years) == 1 else f"{years[0]}-{years[-1]}"


def fetch_sentinel2(
    cfg: ImagerySourceConfig, bbox: tuple[float, float, float, float], cache: SourceCache
) -> tuple[GeoRaster | None, SourceRecord]:
    def fetch() -> tuple[bytes, dict[str, Any]]:
        r, meta = _sentinel_composite(cfg, bbox)
        if r is None:
            return b"", {**meta, "empty": True}
        return raster_to_tiff_bytes(r), meta

    key = ["sentinel2_l2a", COMPOSITE_VERSION, cfg.stac_url, cfg.collection, cfg.date_range, cfg.max_cloud_pct,
           cfg.max_scenes, list(preferred_months(cfg, bbox)), cfg.composite, [round(v, 7) for v in bbox]]
    raw, meta = cache.get_bytes("imagery", key, fetch, suffix=".tif")
    raster = None if meta.get("empty") or not raw else tiff_bytes_to_raster(raw)
    scenes = meta.get("scenes", [])
    years = _years_label(scenes)
    dr = meta.get("date_range") or ["?", "?"]
    rec = SourceRecord(
        name="imagery", provider="sentinel2_l2a", license=SENTINEL_LICENSE,
        attribution=SENTINEL_ATTRIBUTION.format(years=years).strip(),
        dataset_version=(f"Sentinel-2 L2A TCI {meta.get('composite', {}).get('method', 'median')} composite "
                         f"of {len(scenes)} scene(s), "
                         f"{min(s['datetime'][:10] for s in scenes)}..{max(s['datetime'][:10] for s in scenes)}")
        if scenes else f"Sentinel-2 L2A ({dr[0]}..{dr[1]})",
        url=cfg.stac_url, retrieved_utc=meta.get("retrieved_utc"), snapshot_hash=meta.get("sha256"),
        coverage={"collection": cfg.collection, "date_range": dr, "max_cloud_pct": cfg.max_cloud_pct,
                  "months_used": meta.get("season", {}).get("months_used", []), "season": meta.get("season", {}),
                  "scenes": scenes, "composite": meta.get("composite", {})},
    )
    if raster is None:
        rec.notes.append(f"no usable scene below {cfg.max_cloud_pct}% cloud in {dr[0]}..{dr[1]}")
    _describe(rec, raster, cfg)
    return raster, rec


# --------------------------------------------------------------------------------------
# XYZ / WMTS (GoogleMapsCompatible) tiles


def xyz_tile_xy(lon: float, lat: float, zoom: int) -> tuple[float, float]:
    """Fractional Web-Mercator tile coordinates (x right/east, y down/south)."""
    lat = max(min(lat, 85.05112878), -85.05112878)
    n = 1 << zoom
    s = math.sin(math.radians(lat))
    return (lon + 180.0) / 360.0 * n, (0.5 - math.log((1 + s) / (1 - s)) / (4 * math.pi)) * n


def xyz_tile_range(bbox: tuple[float, float, float, float], zoom: int) -> tuple[int, int, int, int]:
    """Inclusive tile range (x0, y0, x1, y1) covering ``bbox`` (west, south, east, north)."""
    w, s, e, n = bbox
    n_tiles = 1 << zoom
    fx0, fy0 = xyz_tile_xy(w, n, zoom)
    fx1, fy1 = xyz_tile_xy(e, s, zoom)
    x0, y0 = math.floor(fx0), math.floor(fy0)
    x1, y1 = max(math.ceil(fx1) - 1, x0), max(math.ceil(fy1) - 1, y0)
    clamp = lambda v: min(max(v, 0), n_tiles - 1)  # noqa: E731
    return clamp(x0), clamp(y0), clamp(x1), clamp(y1)


def xyz_tile_bounds(x: int, y: int, zoom: int) -> tuple[float, float, float, float]:
    """EPSG:3857 bounds (xmin, ymin, xmax, ymax) of tile (x, y)."""
    size = 2 * MERC_HALF / (1 << zoom)
    xmin = -MERC_HALF + x * size
    ymax = MERC_HALF - y * size
    return xmin, ymax - size, xmin + size, ymax


def xyz_tile_url(template: str, zoom: int, x: int, y: int) -> str:
    return template.replace("{z}", str(zoom)).replace("{x}", str(x)).replace("{y}", str(y))


def _download_tile(session: requests.Session, url: str) -> tuple[bytes, str]:
    """(bytes, status) where status is 'ok' or 'missing' (no imagery served for this tile)."""
    err: Exception | str | None = None
    for attempt in range(4):
        try:
            r = session.get(url, timeout=30)
        except requests.RequestException as exc:
            err = exc
        else:
            if r.status_code in (204, 404):
                return b"", "missing"
            if r.status_code == 429 or r.status_code >= 500:
                err = f"HTTP {r.status_code}"
            elif r.status_code >= 400:
                raise SourceError(f"imagery tile {url}: HTTP {r.status_code}")
            else:
                ctype = r.headers.get("Content-Type", "").lower()
                if not r.content or not (ctype.startswith("image/") or ctype in ("", "application/octet-stream")):
                    return b"", "missing"
                return r.content, "ok"
        time.sleep(min(1.0 * 2 ** attempt, 8.0))
    raise SourceError(f"imagery tile {url}: {err}")


def decode_tile(raw: bytes) -> np.ndarray | None:
    """Decode an image tile (JPEG/PNG/WebP/TIFF via GDAL) into (3, h, w) uint8; None if unreadable."""
    if not raw:
        return None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        try:
            with rasterio.Env(GDAL_PAM_ENABLED="NO"), MemoryFile(raw) as mf, mf.open() as ds:
                alpha = None
                if ds.count >= 3:
                    rgb = ds.read([1, 2, 3])
                    if ds.count >= 4:
                        alpha = ds.read(4)
                else:
                    band = ds.read(1)
                    if ds.count == 2:
                        alpha = ds.read(2)
                    try:
                        cmap = ds.colormap(1)
                    except ValueError:
                        cmap = None
                    if cmap:
                        lut = np.zeros((256, 4), dtype=np.uint8)
                        for k, v in cmap.items():
                            if 0 <= k < 256:
                                lut[k] = v
                        rgb = lut[band][..., :3].transpose(2, 0, 1)
                        alpha = lut[band][..., 3] if alpha is None else alpha
                    else:
                        rgb = np.repeat(band[None], 3, axis=0)
        except rasterio.RasterioIOError:
            return None
    rgb = np.ascontiguousarray(rgb.astype(np.uint8))
    valid = np.ones(rgb.shape[1:], dtype=bool) if alpha is None else alpha > 0
    return _mark_nodata(rgb, valid)


def mosaic_tiles(tiles: dict[tuple[int, int], np.ndarray | None], x0: int, y0: int, x1: int, y1: int,
                 zoom: int) -> GeoRaster | None:
    """Mosaic decoded tiles of the inclusive range into one EPSG:3857 GeoRaster (row 0 = north)."""
    sizes = {a.shape[1:] for a in tiles.values() if a is not None}
    if not sizes:
        return None
    if len(sizes) > 1 or next(iter(sizes))[0] != next(iter(sizes))[1]:
        raise SourceError(f"imagery tiles have inconsistent sizes {sorted(sizes)}")
    px = next(iter(sizes))[0]
    out = np.zeros((3, (y1 - y0 + 1) * px, (x1 - x0 + 1) * px), dtype=np.uint8)
    for (x, y), a in tiles.items():
        if a is not None:
            r, c = (y - y0) * px, (x - x0) * px
            out[:, r:r + px, c:c + px] = a
    xmin, _, _, ymax = xyz_tile_bounds(x0, y0, zoom)
    res = 2 * MERC_HALF / (1 << zoom) / px
    return GeoRaster(out, from_origin(xmin, ymax, res, res), WEB_MERCATOR, 0, "imagery")


def fetch_xyz(
    cfg: ImagerySourceConfig, bbox: tuple[float, float, float, float], cache: SourceCache
) -> tuple[GeoRaster | None, SourceRecord]:
    template = cfg.url_template or ""
    z = cfg.zoom
    x0, y0, x1, y1 = xyz_tile_range(bbox, z)
    coords = [(x, y) for y in range(y0, y1 + 1) for x in range(x0, x1 + 1)]
    if len(coords) > cfg.max_tiles:
        raise SourceError(f"imagery: {len(coords)} tiles at zoom {z} exceed sources.imagery.max_tiles="
                          f"{cfg.max_tiles}; lower the zoom or raise max_tiles")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    def one(xy: tuple[int, int]) -> tuple[tuple[int, int], bytes, dict[str, Any]]:
        url = xyz_tile_url(template, z, *xy)

        def fetch() -> tuple[bytes, dict[str, Any]]:
            data, status = _download_tile(session, url)
            return data, {"url": url, "status": status}

        raw, meta = cache.get_bytes("imagery_xyz", ["xyz", template, z, xy[0], xy[1]], fetch, suffix=".img")
        return xy, raw, meta

    log.info("imagery: %d XYZ tiles at zoom %d from %s", len(coords), z, urlparse(template).netloc)
    try:
        results = _parallel(one, coords)
    finally:
        session.close()
    decoded = {xy: decode_tile(raw) for xy, raw, _ in results}
    missing = sum(1 for a in decoded.values() if a is None)
    if missing == len(coords):
        raise SourceError(f"imagery: none of the {len(coords)} tiles returned an image; check url_template "
                          f"({template}) and zoom {z}")
    raster = mosaic_tiles(decoded, x0, y0, x1, y1, z)
    metas = [m for _, _, m in results]
    rec = SourceRecord(
        name="imagery", provider="xyz", license=cfg.license, attribution=cfg.attribution,
        dataset_version=f"{urlparse(template).netloc} zoom {z}", url=template,
        retrieved_utc=max((m.get("retrieved_utc", "") for m in metas), default=None) or utc_now(),
        snapshot_hash=sha256_json(sorted(m.get("sha256", "") for m in metas)),
        coverage={"zoom": z, "tile_range": [x0, y0, x1, y1], "tiles": len(coords), "tiles_missing": missing},
    )
    if missing:
        rec.notes.append(f"{missing} of {len(coords)} tiles had no imagery")
    _describe(rec, raster, cfg)
    return raster, rec
