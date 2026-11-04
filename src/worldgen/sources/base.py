"""Source provenance records and the on-disk acquisition cache.

The cache makes builds reproducible and offline-capable: every remote response
or raster window is stored under a key derived from the request, together with
the retrieval time and a content hash. A rebuild with ``--offline`` reads only
from the cache and fails explicitly (``OfflineCacheMiss``) if anything is
missing. The content hash of each cached item is recorded in the manifest, so
source changes between builds are detectable.
"""

from __future__ import annotations

import datetime as _dt
import json
from collections.abc import Callable
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from pyproj import CRS

from worldgen.errors import OfflineCacheMiss
from worldgen.hashing import sha256_bytes, sha256_json
from worldgen.log import get_logger
from worldgen.spatial.raster import GeoRaster

log = get_logger("sources")


@dataclass
class SourceRecord:
    """Provenance of one dataset used in a build (written to the manifest)."""

    name: str
    provider: str
    license: str
    attribution: str
    status: str = "ok"                # ok | partial | missing | disabled | failed
    dataset_version: str | None = None
    url: str | None = None
    retrieved_utc: str | None = None  # when the cached snapshot was fetched
    snapshot_hash: str | None = None  # sha256 of the cached raw data
    coverage: dict[str, Any] = field(default_factory=dict)
    notes: list[str] = field(default_factory=list)
    vertical_datum: str | None = None

    def to_dict(self) -> dict[str, Any]:
        return {k: v for k, v in asdict(self).items() if v not in (None, [], {})}


def utc_now() -> str:
    return _dt.datetime.now(_dt.UTC).replace(microsecond=0).isoformat().replace("+00:00", "Z")


class SourceCache:
    def __init__(self, root: str | Path, offline: bool = False) -> None:
        self.root = Path(root)
        self.offline = offline
        self.root.mkdir(parents=True, exist_ok=True)

    def key(self, *parts: Any) -> str:
        return sha256_json(list(parts))[:24]

    def _paths(self, namespace: str, key: str, suffix: str) -> tuple[Path, Path]:
        d = self.root / namespace
        d.mkdir(parents=True, exist_ok=True)
        return d / f"{key}{suffix}", d / f"{key}.meta.json"

    def read_meta(self, namespace: str, key: str) -> dict[str, Any] | None:
        _, mp = self._paths(namespace, key, "")
        if mp.is_file():
            return json.loads(mp.read_text(encoding="utf-8"))
        return None

    def get_bytes(
        self,
        namespace: str,
        key_parts: list[Any],
        fetch: Callable[[], tuple[bytes, dict[str, Any]]],
        suffix: str = ".bin",
    ) -> tuple[bytes, dict[str, Any]]:
        key = self.key(*key_parts)
        dp, mp = self._paths(namespace, key, suffix)
        if dp.is_file() and mp.is_file():
            meta = json.loads(mp.read_text(encoding="utf-8"))
            return dp.read_bytes(), meta
        if self.offline:
            raise OfflineCacheMiss(f"{namespace}: no cached data for request {key} (offline mode)")
        data, meta = fetch()
        meta = dict(meta)
        meta.setdefault("retrieved_utc", utc_now())
        meta["sha256"] = sha256_bytes(data)
        meta["key"] = key
        dp.write_bytes(data)
        mp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        log.info("%s: cached %d bytes as %s", namespace, len(data), dp.name)
        return data, meta

    def get_raster(
        self,
        namespace: str,
        key_parts: list[Any],
        fetch: Callable[[], tuple[GeoRaster | None, dict[str, Any]]],
    ) -> tuple[GeoRaster | None, dict[str, Any]]:
        """Cache a raster window as GeoTIFF. ``fetch`` may return None for 'no coverage'."""
        key = self.key(*key_parts)
        dp, mp = self._paths(namespace, key, ".tif")
        if mp.is_file():
            meta = json.loads(mp.read_text(encoding="utf-8"))
            if meta.get("empty"):
                return None, meta
            if dp.is_file():
                return read_geotiff(dp, name=namespace), meta
        if self.offline:
            raise OfflineCacheMiss(f"{namespace}: no cached raster for request {key} (offline mode)")
        raster, meta = fetch()
        meta = dict(meta)
        meta.setdefault("retrieved_utc", utc_now())
        meta["key"] = key
        if raster is None:
            meta["empty"] = True
            meta["sha256"] = sha256_bytes(b"")
        else:
            write_geotiff(dp, raster)
            meta["sha256"] = sha256_bytes(np.ascontiguousarray(raster.data).tobytes())
        mp.write_text(json.dumps(meta, indent=2, sort_keys=True), encoding="utf-8")
        return raster, meta


def write_geotiff(path: Path, r: GeoRaster, band_names: list[str] | None = None) -> None:
    data = r.data if r.data.ndim == 3 else r.data[None, ...]
    profile = {
        "driver": "GTiff",
        "height": data.shape[1],
        "width": data.shape[2],
        "count": data.shape[0],
        "dtype": str(data.dtype),
        "crs": r.crs,
        "transform": r.transform,
        "compress": "deflate",
        "tiled": data.shape[1] >= 256 and data.shape[2] >= 256,
    }
    if r.nodata is not None:
        profile["nodata"] = r.nodata
    with rasterio.open(path, "w", **profile) as ds:
        ds.write(data)
        if band_names:
            for i, n in enumerate(band_names, start=1):
                ds.set_band_description(i, n)


def read_geotiff(path: Path, name: str = "") -> GeoRaster:
    with rasterio.open(path) as ds:
        data = ds.read(1)
        nodata = ds.nodata
        crs = CRS.from_user_input(ds.crs)
        transform: Affine = ds.transform
    if np.issubdtype(data.dtype, np.floating) and nodata is not None and not np.isnan(nodata):
        data = np.where(data == nodata, np.nan, data).astype(data.dtype)
        nodata = float("nan")
    return GeoRaster(data, transform, crs, nodata, name)
