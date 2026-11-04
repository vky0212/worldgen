"""Vegetation classification and canopy field on a regular local grid.

Inputs are OSM land-cover polygons (explicit, vector) and ESA WorldCover
(observed raster), combined with explicit, configurable priority rules, plus an
optional canopy height raster (Meta/WRI CHM). The resulting
:class:`VegetationGrid` holds the category per cell, the source that decided
it, and the canopy field channels. Canopy height is always height *above
ground*, never an elevation.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
from affine import Affine
from rasterio import features as rfeatures
from scipy import ndimage

from worldgen.config import VegetationConfig
from worldgen.noise import value_noise
from worldgen.spatial.raster import GeoRaster, rasterize_window
from worldgen.spatial.region import Region
from worldgen.world.model import LandcoverArea, Waterbody

CATEGORIES = ["none", "dense_forest", "sparse_woodland", "shrubland", "grassland", "agricultural",
              "urban_trees", "wetland", "mangrove"]
CAT = {name: i for i, name in enumerate(CATEGORIES)}
TREE_CATEGORIES = {"dense_forest", "sparse_woodland", "urban_trees", "mangrove"}

SRC_NONE, SRC_OSM, SRC_WORLDCOVER, SRC_BOTH = 0, 1, 2, 3
SOURCE_NAMES = {SRC_NONE: "none", SRC_OSM: "osm", SRC_WORLDCOVER: "worldcover", SRC_BOTH: "osm+worldcover"}

WORLDCOVER_CATEGORY = {10: "dense_forest", 20: "shrubland", 30: "grassland", 40: "agricultural",
                       90: "wetland", 95: "mangrove", 100: "grassland"}

OSM_CATEGORY = {"forest": "dense_forest", "scrub": "shrubland", "heath": "shrubland", "grass": "grassland",
                "farmland": "agricultural", "orchard": "sparse_woodland", "wetland": "wetland",
                "park": "park"}


@dataclass
class VegetationGrid:
    xmin: float
    ymin: float
    resolution: float
    category: np.ndarray            # (rows, cols) uint8 codes into CATEGORIES; row 0 = south
    source: np.ndarray              # uint8 SRC_*
    canopy_height: np.ndarray       # float32 metres above ground, NaN = no data
    canopy_observed: np.ndarray     # bool: canopy height from a measured source
    coverage: np.ndarray            # float32 0..1
    confidence: np.ndarray          # float32 0..1
    tree_fraction_wc: np.ndarray    # float32 WorldCover tree-cover fraction (3x3), NaN without data
    leaf_type: np.ndarray           # uint8: 0 unknown, 1 broadleaved, 2 needleleaved, 3 mixed
    meta: dict = field(default_factory=dict)

    @property
    def shape(self) -> tuple[int, int]:
        return self.category.shape  # type: ignore[return-value]

    def transform(self) -> Affine:
        """Affine with row 0 = south (cell edges)."""
        return Affine(self.resolution, 0.0, self.xmin, 0.0, self.resolution, self.ymin)

    def cell_of(self, x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        c = np.floor((np.asarray(x) - self.xmin) / self.resolution).astype(np.int64)
        r = np.floor((np.asarray(y) - self.ymin) / self.resolution).astype(np.int64)
        inside = (c >= 0) & (c < self.shape[1]) & (r >= 0) & (r < self.shape[0])
        return np.clip(r, 0, self.shape[0] - 1), np.clip(c, 0, self.shape[1] - 1), inside

    def cell_centres(self) -> tuple[np.ndarray, np.ndarray]:
        rows, cols = self.shape
        xs = self.xmin + (np.arange(cols) + 0.5) * self.resolution
        ys = self.ymin + (np.arange(rows) + 0.5) * self.resolution
        return np.meshgrid(xs, ys)


def _rasterize(grid_shape: tuple[int, int], transform: Affine, geoms: list, value: int = 1) -> np.ndarray:
    geoms = [g for g in geoms if g is not None and not g.is_empty]
    if not geoms:
        return np.zeros(grid_shape, dtype=bool)
    return rfeatures.rasterize([(g, value) for g in geoms], out_shape=grid_shape, transform=transform,
                               fill=0, dtype="uint8").astype(bool)


# Feature size of the canopy-height variation where no canopy-height raster covers a cell.
CANOPY_NOISE_WAVELENGTH_M = 100.0


def classify_vegetation(
    region: Region, cfg: VegetationConfig, worldcover: GeoRaster | None, canopy: GeoRaster | None,
    landcover: list[LandcoverArea], water: list[Waterbody],
) -> VegetationGrid:
    xmin, ymin, xmax, ymax = region.extent
    res = cfg.resolution
    cols = int(np.ceil((xmax - xmin) / res))
    rows = int(np.ceil((ymax - ymin) / res))
    shape = (rows, cols)
    transform = Affine(res, 0.0, xmin, 0.0, res, ymin)
    xs = xmin + (np.arange(cols) + 0.5) * res
    ys = ymin + (np.arange(rows) + 0.5) * res
    gx, gy = np.meshgrid(xs, ys)
    local_crs = region.frame.local_crs()

    # WorldCover layer.
    wc_cat = np.zeros(shape, dtype=np.uint8)
    tree_frac = np.full(shape, np.nan, dtype=np.float32)
    built_frac = np.zeros(shape, dtype=np.float32)
    has_wc = worldcover is not None
    if has_wc:
        # Supersample 3x3 inside each cell for class fractions.
        sub = np.array([-1 / 3, 0.0, 1 / 3]) * res
        counts_tree = np.zeros(shape)
        counts_built = np.zeros(shape)
        center = worldcover.sample_nearest(gx.ravel(), gy.ravel(), src_crs=local_crs).reshape(shape)
        for dx in sub:
            for dy in sub:
                v = worldcover.sample_nearest((gx + dx).ravel(), (gy + dy).ravel(), src_crs=local_crs).reshape(shape)
                counts_tree += v == 10
                counts_built += v == 50
        tree_frac = (counts_tree / 9.0).astype(np.float32)
        built_frac = (counts_built / 9.0).astype(np.float32)
        for code, name in WORLDCOVER_CATEGORY.items():
            wc_cat[center == code] = CAT[name]
        # Tree cover embedded in built-up surroundings is urban tree canopy.
        built_ctx = ndimage.uniform_filter(built_frac, size=5)
        urban = (wc_cat == CAT["dense_forest"]) & (built_ctx > 0.35)
        wc_cat[urban] = CAT["urban_trees"]

    # OSM layer (larger polygons first so nested features win).
    osm_cat = np.zeros(shape, dtype=np.uint8)
    leaf = np.zeros(shape, dtype=np.uint8)
    for lc in sorted(landcover, key=lambda a: -a.geometry.area):
        cls = str(lc.properties.get("class"))
        name = OSM_CATEGORY.get(cls)
        if name is None:
            continue
        # Rasterised over the polygon's own window: a full-grid mask per polygon costs polygons x cells.
        hit = rasterize_window(lc.geometry, shape, transform)
        if hit is None or not hit[1].any():
            continue
        win, m = hit
        cat_w = osm_cat[win]
        if name == "park":
            # Parks: trees where WorldCover sees tree cover, lawn elsewhere.
            treeish = (tree_frac[win] >= 0.34) if has_wc else np.zeros(m.shape, dtype=bool)
            cat_w[m & treeish] = CAT["urban_trees"]
            cat_w[m & ~treeish] = CAT["grassland"]
        elif name == "dense_forest" and has_wc:
            frac = float(np.nanmean(tree_frac[win][m]))
            cat_w[m] = CAT["dense_forest"] if frac >= cfg.forest_conflict_threshold else CAT["sparse_woodland"]
        elif name == "wetland" and lc.properties.get("wetland") == "mangrove":
            cat_w[m] = CAT["mangrove"]
        else:
            cat_w[m] = CAT[name]
        lt = str(lc.properties.get("leaf_type", ""))
        if lt:
            leaf[win][m] = {"broadleaved": 1, "needleleaved": 2, "mixed": 3}.get(lt, 0)

    category = np.zeros(shape, dtype=np.uint8)
    source = np.zeros(shape, dtype=np.uint8)
    for src in reversed(cfg.priority):   # lower priority first, higher overwrites
        if src == "osm":
            sel = osm_cat > 0
            category[sel] = osm_cat[sel]
            source[sel] = SRC_OSM
        elif src == "worldcover":
            sel = wc_cat > 0
            category[sel] = wc_cat[sel]
            source[sel] = SRC_WORLDCOVER
    agree = (osm_cat > 0) & (osm_cat == wc_cat)
    source[agree] = SRC_BOTH

    # Water is never vegetated.
    water_mask = _rasterize(shape, transform, [w.geometry for w in water if w.is_area])
    category[water_mask] = 0
    source[water_mask] = SRC_NONE

    # Canopy field.
    cats = cfg.categories
    default_h = np.zeros(shape, dtype=np.float32)
    default_cov = np.zeros(shape, dtype=np.float32)
    for name, cc in cats.items():
        if name in CAT:
            sel = category == CAT[name]
            default_h[sel] = cc.canopy_height
            default_cov[sel] = cc.coverage
    # World-space noise: the same at a position whatever region or window is classified.
    noise = value_noise(gx, gy, CANOPY_NOISE_WAVELENGTH_M, cfg.seed * 7919 + 17)
    est_h = default_h * (1.0 + 0.2 * noise)
    canopy_h = est_h.astype(np.float32)
    observed = np.zeros(shape, dtype=bool)
    if canopy is not None and cfg.use_canopy_height:
        ch = canopy.sample_bilinear(gx.ravel(), gy.ravel(), src_crs=local_crs).reshape(shape)
        ok = np.isfinite(ch) & (category > 0)
        canopy_h = np.where(ok, ch, canopy_h).astype(np.float32)
        observed = ok
    coverage = default_cov.copy()
    if has_wc:
        treeish_cat = np.isin(category, [CAT[c] for c in TREE_CATEGORIES])
        coverage = np.where(treeish_cat, np.maximum(np.nan_to_num(tree_frac, nan=0.0), 0.15 * default_cov),
                            default_cov).astype(np.float32)
        coverage = np.where(category == CAT["sparse_woodland"], np.minimum(coverage, 0.5), coverage)
    if observed.any():
        # Very low measured canopy means no trees regardless of class.
        low = observed & (canopy_h < 2.0) & np.isin(category, [CAT[c] for c in TREE_CATEGORIES])
        coverage = np.where(low, coverage * 0.2, coverage)
    coverage = np.clip(coverage, 0.0, 1.0).astype(np.float32)
    canopy_h = np.where(category > 0, canopy_h, 0.0).astype(np.float32)

    conf = np.zeros(shape, dtype=np.float32)
    conf[source == SRC_WORLDCOVER] = 0.55
    conf[source == SRC_OSM] = 0.6
    conf[source == SRC_BOTH] = 0.8
    conf = np.where(observed, np.minimum(conf + 0.1, 1.0), conf)

    return VegetationGrid(xmin, ymin, res, category, source, canopy_h, observed, coverage, conf,
                          tree_frac, leaf, meta={"worldcover": has_wc, "canopy_observed_fraction":
                                                 float(observed.mean()) if observed.size else 0.0,
                                                 "priority": list(cfg.priority)})
