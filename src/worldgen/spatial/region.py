"""Geographic region definition and its relationship to the tile grid."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any

import shapely
from shapely.geometry import Point, Polygon, box

from worldgen.config import RegionConfig, WorldConfig
from worldgen.spatial.frame import GeoPoint, LocalFrame


@dataclass
class Region:
    """A region of interest: origin, local frame, footprint and tile-snapped extent."""

    center: GeoPoint
    frame: LocalFrame
    footprint: Polygon          # local metres; circle or projected bbox
    extent: tuple[float, float, float, float]   # tile-snapped local (xmin, ymin, xmax, ymax)
    tile_size: float
    radius_m: float | None
    name: str | None = None

    @staticmethod
    def from_config(cfg: WorldConfig) -> Region:
        rc: RegionConfig = cfg.region
        if rc.bbox is not None:
            s, w, n, e = rc.bbox
            if not (s < n and w < e):
                raise ValueError("bbox must be south,west,north,east with south<north and west<east")
            center = GeoPoint((s + n) / 2.0, (w + e) / 2.0)
        else:
            assert rc.center_lat is not None and rc.center_lon is not None
            center = GeoPoint(rc.center_lat, rc.center_lon)
        frame = LocalFrame(center, cfg.projection.method)
        if rc.bbox is not None:
            s, w, n, e = rc.bbox
            geo = box(w, s, e, n).segmentize(0.001)
            footprint = frame.geom_to_local(geo)
            radius = None
        else:
            assert rc.radius_m is not None
            footprint = Point(0.0, 0.0).buffer(rc.radius_m, quad_segs=32)
            radius = rc.radius_m
        t = cfg.tiling.tile_size
        xmin, ymin, xmax, ymax = footprint.bounds
        extent = (
            math.floor(xmin / t) * t,
            math.floor(ymin / t) * t,
            math.ceil(xmax / t) * t,
            math.ceil(ymax / t) * t,
        )
        return Region(center, frame, footprint, extent, t, radius, rc.name)

    @property
    def extent_box(self) -> Polygon:
        return box(*self.extent)

    def tiles(self) -> list[tuple[int, int]]:
        """Tiles of the extent that intersect the footprint, sorted (j, i)."""
        t = self.tile_size
        xmin, ymin, xmax, ymax = self.extent
        out = []
        shapely.prepare(self.footprint)
        for j in range(int(round(ymin / t)), int(round(ymax / t))):
            for i in range(int(round(xmin / t)), int(round(xmax / t))):
                if self.footprint.intersects(box(i * t, j * t, (i + 1) * t, (j + 1) * t)):
                    out.append((i, j))
        return sorted(out, key=lambda ij: (ij[1], ij[0]))

    def wgs84_bbox(self, margin_m: float = 0.0) -> tuple[float, float, float, float]:
        """(west, south, east, north) covering the extent plus a margin."""
        g = self.extent_box.buffer(margin_m, join_style="mitre").segmentize(50.0)
        w, s, e, n = self.frame.geom_to_wgs84(g).bounds
        return (w, s, e, n)

    def describe(self) -> dict[str, Any]:
        w, s, e, n = self.wgs84_bbox()
        fw, fs, fe, fn = self.frame.geom_to_wgs84(self.footprint).bounds
        return {
            "name": self.name,
            "center": {"lat": self.center.lat, "lon": self.center.lon},
            "radius_m": self.radius_m,
            "bbox_wgs84": [fw, fs, fe, fn],
            "extent_bbox_wgs84": [w, s, e, n],
            "extent_local": list(self.extent),
        }
