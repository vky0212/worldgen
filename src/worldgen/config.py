"""Typed build configuration.

Configuration is layered: dataclass defaults (the ``basic`` behaviour) <- profile
YAML (``configs/<profile>.yaml``) <- user YAML <- ``--set key=value`` overrides
<- CLI region arguments. The fully resolved configuration is written to the
manifest together with its hash, so every output is traceable to its settings.
"""

from __future__ import annotations

import calendar
import copy
import dataclasses
import datetime as _dt
import json
import re
import types
import typing
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import yaml

from worldgen.errors import ConfigError
from worldgen.hashing import sha256_json


@dataclass
class RegionConfig:
    """Geographic region. Either center + radius or a bbox (south, west, north, east)."""

    center_lat: float | None = None
    center_lon: float | None = None
    radius_m: float | None = None
    bbox: list[float] | None = None
    name: str | None = None


@dataclass
class ProjectionConfig:
    # "tmerc": transverse Mercator centred on the region origin (default, any zone).
    # "utm": the UTM zone of the origin, translated so the origin is (0, 0).
    method: str = "tmerc"


@dataclass
class TilingConfig:
    tile_size: float = 256.0


@dataclass
class OsmSourceConfig:
    enabled: bool = True
    endpoints: list[str] = field(
        default_factory=lambda: [
            "https://overpass-api.de/api/interpreter",
            "https://overpass.private.coffee/api/interpreter",
        ]
    )
    timeout_s: int = 180
    # Extra margin fetched around the terrain extent so ways entering the region keep context.
    margin_m: float = 50.0


@dataclass
class OvertureSourceConfig:
    enabled: bool = False
    release: str = "latest"
    base_uri: str = "s3://overturemaps-us-west-2/release"
    # Optional local GeoParquet file/directory with Overture "building" rows (offline use, tests).
    local_path: str | None = None
    min_overlap_to_skip: float = 0.3


@dataclass
class DemSourceConfig:
    enabled: bool = True
    provider: str = "copernicus_glo30"
    url_template: str = (
        "https://copernicus-dem-30m.s3.amazonaws.com/{name}/{name}.tif"
    )
    # Optional local GeoTIFF overriding the remote provider.
    local_path: str | None = None
    vertical_datum: str = "EGM2008"


@dataclass
class WorldCoverSourceConfig:
    enabled: bool = True
    url_template: str = (
        "https://esa-worldcover.s3.eu-central-1.amazonaws.com/{version}/{year}/map/"
        "ESA_WorldCover_10m_{year}_{version}_{tile}_Map.tif"
    )
    version: str = "v200"
    year: int = 2021
    local_path: str | None = None


@dataclass
class CanopySourceConfig:
    enabled: bool = True
    provider: str = "meta_chm_v2"
    url_template: str = (
        "https://dataforgood-fb-data.s3.amazonaws.com/forests/v2/global/"
        "dinov3_global_chm_v2_ml3_10m_mean/chm/{quadkey}.tif"
    )
    quadkey_level: int = 7
    local_path: str | None = None


IMAGERY_PROVIDERS = ("sentinel2_l2a", "xyz", "local")
IMAGERY_SEASONS = ("growing", "custom", "any")
IMAGERY_COMPOSITES = ("clear_quantile", "median")


@dataclass
class ImagerySourceConfig:
    """Orthoimagery baked per tile into ``tiles/<id>.ortho.jpg`` (see docs/output-format.md).

    On by default; a failed or (offline) uncached fetch never fails the build, the tiles then
    have no orthophoto. ``sentinel2_l2a`` is an open 10 m cloud-free composite; ``xyz`` reads any
    EPSG:3857 XYZ/WMTS tile service (licensed, higher resolution) and requires ``license`` and
    ``attribution``; ``local`` (or any ``local_path``) reads an RGB GeoTIFF.
    """

    enabled: bool = True
    provider: str = "sentinel2_l2a"
    # Finest ground resolution kept (m/px); finer imagery is averaged down to it.
    max_m_per_px: float = 0.25
    # Largest per-tile image edge in pixels (a power of two).
    tile_px_max: int = 1024
    jpeg_quality: int = 88
    # -- sentinel2_l2a: STAC search (Element 84 Earth Search; COGs on AWS Open Data) -----------
    stac_url: str = "https://earth-search.aws.element84.com/v1"
    collection: str = "sentinel-2-l2a"
    # "YYYY-MM-DD/YYYY-MM-DD", or a period "P<n>Y" / "P<n>M" ending on the retrieval date.
    date_range: str = "P3Y"
    max_cloud_pct: float = 20.0      # scene-level eo:cloud_cover filter
    max_scenes: int = 10             # scenes in the per-pixel composite
    # "clear_quantile": per pixel, the lower-third-by-blue of the non-shadow clear observations
    # (rejects the haze and cloud fringes SCL misses); "median": per-channel median.
    composite: str = "clear_quantile"
    # Preferred acquisition months. "growing": May-Oct north of the equator, Nov-Apr south of it
    # (from the bbox latitude); "custom": season_months; "any": no preference. Scenes from other
    # months are used only where the preferred ones are too few or leave pixels clouded.
    season: str = "growing"
    season_months: list[int] = field(default_factory=list)
    # -- xyz: Web-Mercator tiles; the template must contain {z}, {x} and {y} -------------------
    url_template: str | None = None
    zoom: int = 18
    license: str = ""
    attribution: str = ""
    max_tiles: int = 2500            # refuse requests above this many tiles
    # Local RGB GeoTIFF overriding the remote provider (any CRS).
    local_path: str | None = None


def parse_date_range(text: str, today: _dt.date) -> tuple[_dt.date, _dt.date]:
    """``"2023-01-01/2025-12-31"`` or a period ``"P3Y"`` / ``"P18M"`` ending on ``today``."""
    t = text.strip()
    m = re.fullmatch(r"P(\d+)([YM])", t, flags=re.IGNORECASE)
    if m:
        n = int(m.group(1))
        months = n * 12 if m.group(2).upper() == "Y" else n
        if months <= 0:
            raise ConfigError(f"date range {text!r} must be a positive period")
        y, mo = divmod(today.year * 12 + today.month - 1 - months, 12)
        day = min(today.day, calendar.monthrange(y, mo + 1)[1])
        return _dt.date(y, mo + 1, day), today
    parts = t.split("/")
    if len(parts) != 2:
        raise ConfigError(f"date range {text!r} must be 'YYYY-MM-DD/YYYY-MM-DD' or a period like 'P3Y'")
    try:
        start, end = (_dt.date.fromisoformat(p.strip()) for p in parts)
    except ValueError as exc:
        raise ConfigError(f"date range {text!r}: {exc}") from exc
    if start > end:
        raise ConfigError(f"date range {text!r} starts after it ends")
    return start, end


@dataclass
class SourcesConfig:
    osm: OsmSourceConfig = field(default_factory=OsmSourceConfig)
    overture: OvertureSourceConfig = field(default_factory=OvertureSourceConfig)
    dem: DemSourceConfig = field(default_factory=DemSourceConfig)
    worldcover: WorldCoverSourceConfig = field(default_factory=WorldCoverSourceConfig)
    canopy_height: CanopySourceConfig = field(default_factory=CanopySourceConfig)
    imagery: ImagerySourceConfig = field(default_factory=ImagerySourceConfig)


@dataclass
class DsmCorrectionConfig:
    """Approximate bare-earth terrain from a surface model (Copernicus is a DSM)."""

    enabled: bool = True
    window_m: float = 90.0
    # ESA WorldCover classes where the DSM is corrected (10 = trees, 50 = built-up).
    worldcover_classes: list[int] = field(default_factory=lambda: [10, 50])


@dataclass
class TerrainConfig:
    resolution: float = 4.0
    dsm_correction: DsmCorrectionConfig = field(default_factory=DsmCorrectionConfig)
    road_conformance: bool = True
    road_shoulder_m: float = 3.0
    water_carve_depth: float = 1.5
    # "source" keeps the DEM datum; "EGM2008" or "WGS84_ellipsoid" convert when possible.
    output_vertical_datum: str = "source"
    lod_strides: list[int] = field(default_factory=lambda: [1])


@dataclass
class RoadsConfig:
    default_widths: dict[str, float] = field(
        default_factory=lambda: {
            "motorway": 14.0, "trunk": 12.0, "primary": 11.0, "secondary": 9.0,
            "tertiary": 8.0, "unclassified": 6.0, "residential": 6.0, "service": 4.0,
            "living_street": 5.0, "pedestrian": 5.0, "track": 3.0, "busway": 7.0,
            "motorway_link": 7.0, "trunk_link": 7.0, "primary_link": 7.0,
            "secondary_link": 6.5, "tertiary_link": 6.0, "road": 6.0,
            "footway": 2.0, "path": 1.8, "cycleway": 2.5, "steps": 2.0, "bridleway": 2.0,
            "corridor": 2.0,
        }
    )
    lane_width: float = 3.25
    # "drape": ground roads and junctions are cut along the terrain triangles and follow the
    # terrain exactly (never pierced); "ribbon": flat cross-sections on the centreline profile.
    surface_mode: str = "drape"
    surface_offset: float = 0.12
    # Ground roads never exceed this grade within an edge (cuttings/embankments shape the terrain).
    max_grade: float = 0.18
    grade_exempt_classes: list[str] = field(
        default_factory=lambda: ["steps", "path", "footway", "track", "bridleway", "corridor", "cycleway",
                                 "pedestrian", "via_ferrata"]
    )
    sample_spacing: float = 4.0
    intersections: bool = True
    collision: bool = True
    collision_spacing: float = 12.0
    exclude_classes: list[str] = field(
        default_factory=lambda: [
            "proposed", "construction", "abandoned", "platform", "raceway", "razed",
            "bus_stop", "elevator", "via_ferrata", "disused", "planned", "emergency_bay",
        ]
    )
    # Classes drawn as meshes; topology always keeps every imported class.
    mesh_exclude_classes: list[str] = field(default_factory=list)


@dataclass
class BridgesConfig:
    deck_thickness: float = 1.2
    min_clearance_road: float = 5.0
    min_clearance_water: float = 4.0
    # Vertical separation assumed per OSM layer step at crossings (relative, never absolute).
    layer_spacing: float = 6.0
    default_deck_height: float = 6.0
    short_bridge_length: float = 40.0
    short_bridge_height: float = 1.5
    max_grade: float = 0.07
    supports: str = "simple"
    pier_spacing: float = 30.0
    pier_width: float = 1.6
    railings: bool = True
    railing_height: float = 1.0


@dataclass
class TunnelsConfig:
    # Soil cover kept above the tunnel roof wherever the grade allows.
    min_cover: float = 1.0
    clearance_height: float = 5.0
    max_grade: float = 0.07
    wall_thickness: float = 0.6
    portals: bool = True


@dataclass
class BudgetConfig:
    """Rendering budget. Above it, smaller items are dropped first, at random (deterministic).

    Dropped items stay in the semantic exports (``budget_dropped``) but get no geometry.
    0 disables a limit.
    """

    enabled: bool = True
    max_per_tile: int = 0
    max_total: int = 0
    keep_observed: bool = True   # trees: mapped (observed) trees are dropped last


@dataclass
class BuildingsConfig:
    floor_height: float = 3.0
    default_height: float = 9.0
    min_height: float = 2.5
    class_levels: dict[str, float] = field(
        default_factory=lambda: {
            "house": 2, "detached": 2, "semidetached_house": 2, "terrace": 3, "bungalow": 1,
            "residential": 5, "apartments": 7, "dormitory": 5, "hotel": 8,
            "commercial": 5, "retail": 2, "office": 10, "industrial": 2, "warehouse": 2,
            "school": 4, "university": 5, "hospital": 6, "church": 3, "temple": 2,
            "public": 4, "civic": 4, "government": 5, "train_station": 2,
            "garage": 1, "garages": 1, "shed": 1, "hut": 1, "roof": 1, "carport": 1,
            "kiosk": 1, "toilets": 1, "parking": 4, "construction": 3, "service": 1,
        }
    )
    roofs: bool = True
    max_roof_pitch_deg: float = 30.0
    # Deterministic +/- fraction applied to estimated heights (procedural profiles).
    procedural_variation: float = 0.0
    # Assign estimated gabled roofs to small residential buildings without roof data.
    procedural_roofs: bool = False
    building_parts: bool = True
    foundation_depth: float = 0.5
    budget: BudgetConfig = field(default_factory=lambda: BudgetConfig(True, 250, 0))


@dataclass
class WaterConfig:
    default_widths: dict[str, float] = field(
        default_factory=lambda: {
            "river": 20.0, "stream": 3.0, "canal": 8.0, "drain": 2.0, "ditch": 1.5,
            "tidal_channel": 6.0, "brook": 2.0,
        }
    )
    lake_level_percentile: float = 10.0
    surface_offset: float = 0.05


@dataclass
class VegetationCategoryConfig:
    """Defaults used where sources do not provide canopy structure.

    ``spacing_factor`` is the hard-core distance between generated stems in units
    of crown radius (1.0 lets neighbouring crowns overlap by half).
    """

    canopy_height: float = 0.0
    coverage: float = 0.0
    spacing_factor: float = 1.6
    archetypes: dict[str, float] = field(default_factory=dict)


def _default_categories() -> dict[str, VegetationCategoryConfig]:
    c = VegetationCategoryConfig
    return {
        "dense_forest": c(16.0, 0.85, 1.1, {"broadleaf": 0.7, "conifer": 0.3}),
        "sparse_woodland": c(10.0, 0.35, 1.6, {"broadleaf": 0.7, "conifer": 0.3}),
        "shrubland": c(2.0, 0.4, 1.2, {"shrub": 0.9, "broadleaf": 0.1}),
        "grassland": c(0.4, 0.0, 2.0, {"grass": 1.0}),
        "agricultural": c(0.8, 0.0, 2.0, {"grass": 1.0}),
        "urban_trees": c(9.0, 0.3, 2.0, {"broadleaf": 0.85, "palm": 0.15}),
        "wetland": c(1.0, 0.15, 1.6, {"shrub": 0.6, "grass": 0.4}),
        "mangrove": c(6.0, 0.7, 1.1, {"broadleaf": 1.0}),
    }


@dataclass
class GroundCoverConfig:
    enabled: bool = True
    shrub_density: float = 0.01   # instances per m2 in shrubland/wetland
    grass_density: float = 0.004  # clusters per m2 in grassland/agricultural/parks
    max_instances: int = 20000


@dataclass
class VegetationConfig:
    enabled: bool = True
    resolution: float = 8.0
    seed: int = 1337
    cell_size: float = 16.0
    # Source priority for classification conflicts; earlier wins where it has data.
    priority: list[str] = field(default_factory=lambda: ["osm", "worldcover"])
    # OSM forest polygons whose WorldCover tree fraction is below this become sparse woodland.
    forest_conflict_threshold: float = 0.3
    categories: dict[str, VegetationCategoryConfig] = field(default_factory=_default_categories)
    use_canopy_height: bool = True
    max_slope_deg: float = 42.0
    tree_row_spacing: float = 8.0
    exclusion_buffer: float = 1.0
    ground_cover: GroundCoverConfig = field(default_factory=GroundCoverConfig)
    quality: str = "low"
    # Trees, shrubs and grass clusters rendered per tile / in the whole world.
    budget: BudgetConfig = field(default_factory=lambda: BudgetConfig(True, 2500, 200000))


@dataclass
class SmoothingConfig:
    """Post-processing of grid-derived borders (``landcover/smoothing.py``).

    Land cover arrives as source pixels (ESA WorldCover is 10 m) sampled onto the terrain and
    vegetation grids, and water is carved node by node, so every border is a staircase. This
    stage turns them into smooth, slightly irregular natural borders. Distances in metres.
    """

    enabled: bool = True
    # Gaussian sigma of the class-membership smoothing. Removes source-pixel stairs; specks
    # smaller than about twice this vanish into their surroundings.
    border_radius_m: float = 6.0
    # Width of the colour transition across land-cover borders (0: no extra blending).
    blend_width_m: float = 10.0
    # Natural irregularity: borders are displaced by smooth world-space noise (0: off).
    warp_amplitude_m: float = 5.0
    warp_wavelength_m: float = 60.0
    # A WorldCover water patch outside mapped (OSM) water that comes this close to it is a
    # registration/tide mismatch: the whole patch takes the surrounding land class instead of
    # painting the bank blue. Patches farther away (unmapped ponds) are kept.
    water_snap_m: float = 30.0
    # Shorelines: the terrain crosses the water surface exactly on the mapped outline. The bank
    # gradient there follows the original terrain, clamped to these limits (m/m).
    shorelines: bool = True
    shore_slope_min: float = 0.03
    shore_slope_max: float = 1.0
    # The vegetation grid is refined by this integer factor so forest edges are not cell stairs.
    vegetation_upsample: int = 2
    # Fraction of tree density removed at the very edge of forests (fades in over the edge zone).
    forest_edge_thinning: float = 0.5


@dataclass
class ExportConfig:
    combined_glb: bool = True
    # world.glb (every tile in one file) is only written up to this many tiles (0: no limit).
    # Large worlds are streamed tile by tile, and GLB files cannot exceed 4 GiB.
    combined_glb_max_tiles: int = 400
    geoparquet: bool = True
    collision: bool = True
    heightfield: bool = True
    fields: bool = True


@dataclass
class ValidationConfig:
    enabled: bool = True
    fail_on_error: bool = False
    max_issues_per_check: int = 200


@dataclass
class WorldConfig:
    profile: str = "basic"
    region: RegionConfig = field(default_factory=RegionConfig)
    projection: ProjectionConfig = field(default_factory=ProjectionConfig)
    tiling: TilingConfig = field(default_factory=TilingConfig)
    sources: SourcesConfig = field(default_factory=SourcesConfig)
    terrain: TerrainConfig = field(default_factory=TerrainConfig)
    roads: RoadsConfig = field(default_factory=RoadsConfig)
    bridges: BridgesConfig = field(default_factory=BridgesConfig)
    tunnels: TunnelsConfig = field(default_factory=TunnelsConfig)
    buildings: BuildingsConfig = field(default_factory=BuildingsConfig)
    water: WaterConfig = field(default_factory=WaterConfig)
    vegetation: VegetationConfig = field(default_factory=VegetationConfig)
    smoothing: SmoothingConfig = field(default_factory=SmoothingConfig)
    export: ExportConfig = field(default_factory=ExportConfig)
    validation: ValidationConfig = field(default_factory=ValidationConfig)

    def to_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)

    def hash(self) -> str:
        return sha256_json(self.to_dict())

    def section_hash(self, *names: str, exclude: dict[str, tuple[str, ...]] | None = None) -> str:
        """Hash of the named sections; ``exclude`` drops keys of a section (section -> keys)."""
        data = self.to_dict()
        for section, keys in (exclude or {}).items():
            for k in keys:
                data[section].pop(k, None)
        return sha256_json({n: data[n] for n in names})

    def validate(self) -> None:
        t = self.tiling.tile_size
        r = self.terrain.resolution
        if t <= 0 or r <= 0:
            raise ConfigError("tiling.tile_size and terrain.resolution must be positive")
        cells = t / r
        if abs(cells - round(cells)) > 1e-9:
            raise ConfigError(
                f"terrain.resolution ({r}) must divide tiling.tile_size ({t}) exactly "
                "so that adjacent terrain tiles share vertices"
            )
        if self.projection.method not in ("tmerc", "utm"):
            raise ConfigError(f"unknown projection method {self.projection.method!r}")
        if self.roads.surface_mode not in ("drape", "ribbon"):
            raise ConfigError("roads.surface_mode must be drape or ribbon")
        if self.vegetation.quality not in ("low", "medium", "high"):
            raise ConfigError("vegetation.quality must be low, medium or high")
        sm = self.smoothing
        if min(sm.border_radius_m, sm.blend_width_m, sm.warp_amplitude_m, sm.water_snap_m) < 0:
            raise ConfigError("smoothing distances must not be negative")
        if sm.warp_wavelength_m <= 0:
            raise ConfigError("smoothing.warp_wavelength_m must be positive")
        if not 0 < sm.shore_slope_min <= sm.shore_slope_max:
            raise ConfigError("smoothing requires 0 < shore_slope_min <= shore_slope_max")
        if int(sm.vegetation_upsample) != sm.vegetation_upsample or not 1 <= sm.vegetation_upsample <= 8:
            raise ConfigError("smoothing.vegetation_upsample must be an integer from 1 to 8")
        if not 0 <= sm.forest_edge_thinning <= 1:
            raise ConfigError("smoothing.forest_edge_thinning must be in [0, 1]")
        reg = self.region
        if reg.bbox is None and (reg.center_lat is None or reg.center_lon is None):
            raise ConfigError("region requires center_lat/center_lon (+radius_m) or bbox")
        if reg.bbox is None and (reg.radius_m is None or reg.radius_m <= 0):
            raise ConfigError("region.radius_m must be positive")
        validate_imagery(self.sources.imagery)


def validate_imagery(im: ImagerySourceConfig) -> None:
    p = "sources.imagery"
    if im.provider not in IMAGERY_PROVIDERS:
        raise ConfigError(f"{p}.provider must be one of {', '.join(IMAGERY_PROVIDERS)}, got {im.provider!r}")
    if im.tile_px_max < 32 or im.tile_px_max > 8192 or im.tile_px_max & (im.tile_px_max - 1):
        raise ConfigError(f"{p}.tile_px_max must be a power of two between 32 and 8192")
    if not 1 <= im.jpeg_quality <= 100:
        raise ConfigError(f"{p}.jpeg_quality must be between 1 and 100")
    if im.max_m_per_px <= 0:
        raise ConfigError(f"{p}.max_m_per_px must be positive")
    if im.provider == "sentinel2_l2a":
        if im.max_scenes < 1:
            raise ConfigError(f"{p}.max_scenes must be at least 1")
        if not 0 < im.max_cloud_pct <= 100:
            raise ConfigError(f"{p}.max_cloud_pct must be in (0, 100]")
        if im.composite not in IMAGERY_COMPOSITES:
            raise ConfigError(f"{p}.composite must be one of {', '.join(IMAGERY_COMPOSITES)}, got {im.composite!r}")
        if im.season not in IMAGERY_SEASONS:
            raise ConfigError(f"{p}.season must be one of {', '.join(IMAGERY_SEASONS)}, got {im.season!r}")
        if im.season == "custom" and not im.season_months:
            raise ConfigError(f"{p}: season 'custom' requires season_months")
        if any(not 1 <= m <= 12 for m in im.season_months):
            raise ConfigError(f"{p}.season_months must be month numbers 1-12")
        parse_date_range(im.date_range, _dt.date.today())
    elif im.provider == "xyz":
        tpl = im.url_template or ""
        if not all(k in tpl for k in ("{z}", "{x}", "{y}")):
            raise ConfigError(f"{p}.url_template must contain {{z}}, {{x}} and {{y}} for provider 'xyz'")
        if not im.license.strip() or not im.attribution.strip():
            raise ConfigError(f"{p}: provider 'xyz' requires a non-empty license and attribution "
                              "(record the terms of the tile service you are allowed to use)")
        if not 0 <= im.zoom <= 24:
            raise ConfigError(f"{p}.zoom must be between 0 and 24")
        if im.max_tiles < 1:
            raise ConfigError(f"{p}.max_tiles must be at least 1")
    elif im.provider == "local" and not im.local_path:
        raise ConfigError(f"{p}: provider 'local' requires local_path")


# --------------------------------------------------------------------------------------
# dict <-> dataclass plumbing


def _coerce(tp: Any, value: Any, path: str, default: Any = None) -> Any:
    """``default`` is the value the field would have had, used to fill a partial dataclass."""
    origin = typing.get_origin(tp)
    if dataclasses.is_dataclass(tp):
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping, got {type(value).__name__}")
        return from_dict(tp, value, path, default if isinstance(default, tp) else None)
    if origin in (typing.Union, types.UnionType):
        args = [a for a in typing.get_args(tp) if a is not type(None)]
        if value is None:
            return None
        return _coerce(args[0], value, path, default)
    if origin is dict:
        _, vt = typing.get_args(tp)
        if not isinstance(value, dict):
            raise ConfigError(f"{path}: expected a mapping")
        return {str(k): _coerce(vt, v, f"{path}.{k}") for k, v in value.items()}
    if origin is list:
        (it,) = typing.get_args(tp)
        if not isinstance(value, (list, tuple)):
            raise ConfigError(f"{path}: expected a list")
        return [_coerce(it, v, f"{path}[]") for v in value]
    if tp is float:
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ConfigError(f"{path}: expected a number, got {value!r}")
        return float(value)
    if tp is int:
        if isinstance(value, bool) or not isinstance(value, int):
            raise ConfigError(f"{path}: expected an integer, got {value!r}")
        return value
    if tp is bool:
        if not isinstance(value, bool):
            raise ConfigError(f"{path}: expected true/false, got {value!r}")
        return value
    if tp is str:
        if not isinstance(value, str):
            raise ConfigError(f"{path}: expected a string, got {value!r}")
        return value
    return value


def _field_default(f: dataclasses.Field) -> Any:
    if f.default_factory is not dataclasses.MISSING:
        return f.default_factory()
    if f.default is not dataclasses.MISSING:
        return f.default
    return None


def from_dict(cls: type, data: dict[str, Any], path: str = "config", default: Any = None) -> Any:
    """Build ``cls`` from a (possibly partial) mapping.

    Keys missing from ``data`` come from ``default`` -- the instance the enclosing field
    would have had (e.g. ``vegetation.budget``'s ``default_factory``) -- or, without one,
    from ``cls``'s own defaults. Nested dataclass sections are filled the same way, so
    setting one key of a section never resets its siblings to the class defaults. Plain
    dict and list values replace the default wholesale.
    """
    hints = typing.get_type_hints(cls)
    fields = {f.name: f for f in dataclasses.fields(cls)}
    unknown = set(data) - set(fields)
    if unknown:
        raise ConfigError(f"{path}: unknown keys {sorted(unknown)}")
    kwargs: dict[str, Any] = {}
    for k, v in data.items():
        base = getattr(default, k) if default is not None else _field_default(fields[k])
        kwargs[k] = _coerce(hints[k], v, f"{path}.{k}", base)
    if default is not None:
        for k, f in fields.items():
            if k not in kwargs and f.init:
                kwargs[k] = copy.deepcopy(getattr(default, k))
    return cls(**kwargs)


def deep_merge(base: dict[str, Any], over: dict[str, Any]) -> dict[str, Any]:
    out = dict(base)
    for k, v in over.items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = v
    return out


def parse_override(expr: str) -> dict[str, Any]:
    """``a.b.c=value`` -> nested dict; the value is parsed as YAML."""
    if "=" not in expr:
        raise ConfigError(f"override {expr!r} must look like key.path=value")
    key, raw = expr.split("=", 1)
    value = yaml.safe_load(raw)
    out: dict[str, Any] = {}
    cur = out
    parts = key.strip().split(".")
    for p in parts[:-1]:
        cur = cur.setdefault(p, {})
    cur[parts[-1]] = value
    return out


def profile_search_paths(name: str) -> list[Path]:
    here = Path(__file__).resolve()
    return [
        Path(name),
        Path.cwd() / "configs" / f"{name}.yaml",
        here.parents[2] / "configs" / f"{name}.yaml",
        here.parent / "_profiles" / f"{name}.yaml",
    ]


def load_profile(name: str) -> dict[str, Any]:
    for p in profile_search_paths(name):
        if p.is_file():
            with p.open("r", encoding="utf-8") as fh:
                data = yaml.safe_load(fh) or {}
            if not isinstance(data, dict):
                raise ConfigError(f"profile {p} must contain a mapping")
            return data
    raise ConfigError(f"profile {name!r} not found (looked in configs/ and package profiles)")


def load_config(
    profile: str = "basic",
    config_file: str | Path | None = None,
    overrides: list[str] | None = None,
    region: dict[str, Any] | None = None,
) -> WorldConfig:
    data: dict[str, Any] = {}
    prof = load_profile(profile)
    # Profiles may extend another profile.
    parent = prof.pop("extends", None)
    if parent:
        data = deep_merge(data, load_profile(parent))
        data.pop("extends", None)
    data = deep_merge(data, prof)
    data["profile"] = Path(profile).stem if profile.endswith(".yaml") else profile
    if config_file:
        with open(config_file, "r", encoding="utf-8") as fh:
            data = deep_merge(data, yaml.safe_load(fh) or {})
    for o in overrides or []:
        data = deep_merge(data, parse_override(o))
    if region:
        data = deep_merge(data, {"region": region})
    cfg: WorldConfig = from_dict(WorldConfig, data)
    cfg.validate()
    return cfg


def dump_config(cfg: WorldConfig) -> str:
    return json.dumps(cfg.to_dict(), indent=2, sort_keys=True)
