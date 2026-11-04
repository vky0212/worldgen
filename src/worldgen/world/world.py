"""The canonical geographic world: every semantic layer of a compiled region."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any

import numpy as np
from shapely.geometry import LineString

from worldgen.config import WorldConfig
from worldgen.sources.base import SourceRecord
from worldgen.spatial.raster import GeoRaster
from worldgen.spatial.region import Region
from worldgen.terrain.heightfield import Heightfield
from worldgen.world.model import (
    Bridge,
    Building,
    Entity,
    Intersection,
    LandcoverArea,
    ObservedTree,
    Road,
    TreeTable,
    TurnRestriction,
    Tunnel,
    VegetationRegion,
    Waterbody,
)


@dataclass
class SourceBundle:
    """Raw-but-parsed inputs from all adapters (or a synthetic fixture)."""

    osm: Any = None                      # OsmData
    dem: GeoRaster | None = None
    worldcover: GeoRaster | None = None
    canopy: GeoRaster | None = None
    overture_buildings: list[Building] = field(default_factory=list)
    # Orthoimagery: RGB uint8, data shape (3, rows, cols), source CRS; all-zero pixels = nodata.
    imagery: GeoRaster | None = None
    records: list[SourceRecord] = field(default_factory=list)
    dem_datum: str = "EGM2008"


@dataclass
class CanonicalWorld:
    region: Region
    config: WorldConfig
    sources: list[SourceRecord] = field(default_factory=list)
    roads: dict[str, Road] = field(default_factory=dict)
    buildings: list[Building] = field(default_factory=list)
    water: list[Waterbody] = field(default_factory=list)
    landcover: list[LandcoverArea] = field(default_factory=list)
    observed_trees: list[ObservedTree] = field(default_factory=list)
    restrictions: list[TurnRestriction] = field(default_factory=list)
    coastlines: list[LineString] = field(default_factory=list)
    terrain: Heightfield | None = None
    terrain_base: Heightfield | None = None
    terrain_codes: np.ndarray | None = None
    terrain_rgba: np.ndarray | None = None   # (rows, cols, 4) blended vertex colours (smoothing stage)
    graph: Any = None                    # RoadGraph
    bridges: list[Bridge] = field(default_factory=list)
    tunnels: list[Tunnel] = field(default_factory=list)
    intersections: list[Intersection] = field(default_factory=list)
    crossings: list[Any] = field(default_factory=list)
    water_crossings: list[Any] = field(default_factory=list)
    profiles: dict[str, Any] = field(default_factory=dict)
    junctions: Any = None                # JunctionLayout (trims, mitres, junction polygons)
    water_levels: dict[str, float] = field(default_factory=dict)
    waterway_profiles: dict[str, Any] = field(default_factory=dict)
    covered_waterways: set[str] = field(default_factory=set)
    vegetation_grid: Any = None
    imagery: GeoRaster | None = None     # orthoimagery baked per tile by export/ortho.py
    imagery_hash: str | None = None      # memoised content hash (export.ortho.imagery_hash)
    vegetation_regions: list[VegetationRegion] = field(default_factory=list)
    trees: TreeTable = field(default_factory=TreeTable)
    stage_issues: list[dict[str, Any]] = field(default_factory=list)
    stats: dict[str, Any] = field(default_factory=dict)
    timings: dict[str, float] = field(default_factory=dict)

    def entities(self) -> list[Entity]:
        """Every vector entity (trees excluded; they are columnar)."""
        out: list[Entity] = []
        out += [self.roads[k] for k in sorted(self.roads)]
        out += self.intersections + self.bridges + self.tunnels + self.buildings + self.water
        out += self.landcover + self.vegetation_regions
        return out

    def entity_index(self) -> dict[str, Entity]:
        return {e.id: e for e in self.entities()}
