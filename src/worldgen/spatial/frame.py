"""Coordinate reference frames.

All computation happens in a *local metric frame*: metres, x = east, y = north,
z = up, with the origin at the region centre. Geographic input and every source
reference stay in WGS84 (EPSG:4326, lon/lat order in code).

The default projection is a transverse Mercator centred on the origin. Unlike a
UTM zone it never has a zone boundary inside the region, and scale distortion
stays below 1e-5 within ~30 km of the origin and below 1e-4 within ~90 km. For
larger regions use tiles of separate worlds, or the ``utm`` method, and see
docs/coordinate-systems.md for the geocentric strategy.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
import shapely
from pyproj import CRS, Transformer
from shapely.geometry.base import BaseGeometry

WGS84 = CRS.from_epsg(4326)


@dataclass(frozen=True)
class GeoPoint:
    lat: float
    lon: float

    @staticmethod
    def parse(text: str) -> GeoPoint:
        parts = [p.strip() for p in text.split(",")]
        if len(parts) != 2:
            raise ValueError(f"expected 'lat,lon', got {text!r}")
        lat, lon = float(parts[0]), float(parts[1])
        if not (-90.0 <= lat <= 90.0 and -180.0 <= lon <= 180.0):
            raise ValueError(f"coordinate out of range: {text!r}")
        return GeoPoint(lat, lon)


@dataclass
class LocalFrame:
    """Bidirectional WGS84 <-> local ENU (metres) transformation."""

    origin: GeoPoint
    method: str = "tmerc"
    origin_height: float = 0.0
    crs: CRS = field(init=False)
    _fwd: Transformer = field(init=False, repr=False)
    _inv: Transformer = field(init=False, repr=False)
    _offset: tuple[float, float] = field(init=False, default=(0.0, 0.0))

    def __post_init__(self) -> None:
        if self.method == "tmerc":
            self.crs = CRS.from_proj4(self.proj_string)
            self._offset = (0.0, 0.0)
        elif self.method == "utm":
            zone = int((self.origin.lon + 180.0) // 6.0) + 1
            south = self.origin.lat < 0
            self.crs = CRS.from_dict({"proj": "utm", "zone": zone, "south": south, "ellps": "WGS84"})
            t = Transformer.from_crs(WGS84, self.crs, always_xy=True)
            ox, oy = t.transform(self.origin.lon, self.origin.lat)
            self._offset = (float(ox), float(oy))
        else:
            raise ValueError(f"unknown projection method {self.method!r}")
        self._fwd = Transformer.from_crs(WGS84, self.crs, always_xy=True)
        self._inv = Transformer.from_crs(self.crs, WGS84, always_xy=True)

    @property
    def proj_string(self) -> str:
        if self.method == "tmerc":
            return (
                f"+proj=tmerc +lat_0={self.origin.lat:.10f} +lon_0={self.origin.lon:.10f} "
                "+k=1 +x_0=0 +y_0=0 +ellps=WGS84 +units=m +no_defs"
            )
        return self.crs.to_proj4()

    @property
    def local_offset(self) -> tuple[float, float]:
        """Projected coordinates of the origin (non-zero for UTM); local = projected - offset."""
        return self._offset

    def local_crs(self) -> CRS:
        """CRS whose coordinates equal local x/y (used for GeoTIFF exports)."""
        if self.method == "tmerc":
            return self.crs
        ox, oy = self._offset
        p = self.crs.to_dict()
        p["x_0"] = p.get("x_0", 500000.0) - ox
        p["y_0"] = p.get("y_0", 0.0) - oy
        return CRS.from_dict(p)

    # -- point transforms ---------------------------------------------------------------
    def to_local(self, lon: Any, lat: Any) -> tuple[np.ndarray, np.ndarray]:
        x, y = self._fwd.transform(np.asarray(lon, dtype=float), np.asarray(lat, dtype=float))
        return np.asarray(x) - self._offset[0], np.asarray(y) - self._offset[1]

    def to_wgs84(self, x: Any, y: Any) -> tuple[np.ndarray, np.ndarray]:
        lon, lat = self._inv.transform(
            np.asarray(x, dtype=float) + self._offset[0], np.asarray(y, dtype=float) + self._offset[1]
        )
        return np.asarray(lon), np.asarray(lat)

    def point_to_local(self, p: GeoPoint) -> tuple[float, float]:
        x, y = self.to_local(p.lon, p.lat)
        return float(x), float(y)

    def point_to_wgs84(self, x: float, y: float) -> GeoPoint:
        lon, lat = self.to_wgs84(x, y)
        return GeoPoint(float(lat), float(lon))

    # -- geometry transforms -------------------------------------------------------------
    def geom_to_local(self, geom: BaseGeometry) -> BaseGeometry:
        def f(coords: np.ndarray) -> np.ndarray:
            x, y = self.to_local(coords[:, 0], coords[:, 1])
            return np.column_stack([x, y])

        return shapely.transform(geom, f)

    def geom_to_wgs84(self, geom: BaseGeometry) -> BaseGeometry:
        def f(coords: np.ndarray) -> np.ndarray:
            lon, lat = self.to_wgs84(coords[:, 0], coords[:, 1])
            return np.column_stack([lon, lat])

        return shapely.transform(geom, f)

    def scale_factor(self, x: float, y: float) -> float:
        """Point scale factor of the projection at a local position (1.0 = exact)."""
        lon, lat = self.to_wgs84(x, y)
        from pyproj import Proj

        factors = Proj(self.crs).get_factors(float(lon), float(lat))
        return float(factors.meridional_scale)

    def describe(self) -> dict[str, Any]:
        return {
            "method": self.method,
            "proj": self.proj_string,
            "wkt": self.local_crs().to_wkt(),
            "projected_offset": list(self._offset),
        }
