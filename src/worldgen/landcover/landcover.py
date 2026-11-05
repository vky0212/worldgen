"""Terrain-aligned land-cover classification (mask on the terrain grid) and terrain colours."""

from __future__ import annotations

from typing import Any

import numpy as np

from worldgen import parallel
from worldgen.spatial.raster import GeoRaster
from worldgen.spatial.region import Region
from worldgen.terrain.build import node_mask_window, rasterize_mask
from worldgen.spatial.windows import row_block
from worldgen.terrain.heightfield import Heightfield
from worldgen.world.model import LandcoverArea, Waterbody

# Terrain land-cover codes (uint8) used by terrain/landcover.tif and terrain colours.
LANDCOVER_CODES: dict[int, str] = {
    0: "unknown", 1: "forest", 2: "shrub", 3: "grass", 4: "farmland", 5: "built", 6: "bare",
    7: "water", 8: "wetland", 9: "park", 10: "sand", 11: "snow_ice", 12: "mangrove",
}

WORLDCOVER_TO_CODE = {10: 1, 20: 2, 30: 3, 40: 4, 50: 5, 60: 6, 70: 11, 80: 7, 90: 8, 95: 12, 100: 3}

OSM_CLASS_TO_CODE = {
    "forest": 1, "scrub": 2, "heath": 2, "grass": 3, "nature_reserve": 3, "farmland": 4, "orchard": 4,
    "residential": 5, "commercial": 5, "industrial": 5, "construction": 6, "bare": 6, "wetland": 8,
    "park": 9, "sand": 10,
}

CODE_COLORS = {
    0: (0.55, 0.53, 0.47), 1: (0.23, 0.38, 0.20), 2: (0.42, 0.47, 0.27), 3: (0.50, 0.60, 0.33),
    4: (0.62, 0.62, 0.38), 5: (0.58, 0.56, 0.53), 6: (0.62, 0.57, 0.48), 7: (0.30, 0.40, 0.45),
    8: (0.38, 0.48, 0.38), 9: (0.40, 0.55, 0.30), 10: (0.80, 0.75, 0.58), 11: (0.92, 0.94, 0.96),
    12: (0.20, 0.35, 0.25),
}


def classify_terrain(hf: Heightfield, region: Region, worldcover: GeoRaster | None,
                     landcover: list[LandcoverArea], water: list[Waterbody]) -> np.ndarray:
    """Per terrain node land-cover code. OSM polygons override WorldCover; water wins last."""
    codes = np.zeros(hf.z.shape, dtype=np.uint8)
    if worldcover is not None:
        # Sampled in row blocks (each node on its own), by the build's worker processes when
        # there are any: no whole-grid coordinate arrays.
        step = row_block()
        blocks = [(r0, min(r0 + step, hf.rows)) for r0 in range(0, hf.rows, step)]
        ctx = (worldcover, hf.xmin + np.arange(hf.cols) * hf.resolution, hf.ymin, hf.resolution,
               region.frame.local_crs())
        for (r0, r1), block in zip(blocks, parallel.tasks(_worldcover_codes, ctx, blocks)):
            codes[r0:r1] = block
    # Larger polygons first so nested small features (a park in a residential zone) win.
    for lc in sorted(landcover, key=lambda a: -a.geometry.area):
        code = OSM_CLASS_TO_CODE.get(str(lc.properties.get("class")))
        if code is None:
            continue
        hit = node_mask_window(hf, lc.geometry)
        if hit is not None:
            win, m = hit
            codes[win][m] = code
    areas = [w.geometry for w in water if w.is_area]
    if areas:
        codes[rasterize_mask(hf, areas)] = 7
    return codes


def _worldcover_codes(ctx: tuple[GeoRaster, np.ndarray, float, float, Any], rows: tuple[int, int]) -> np.ndarray:
    """Land-cover codes from WorldCover for a row block of terrain nodes."""
    worldcover, xs, ymin, res, crs = ctx
    gx, gy = np.meshgrid(xs, ymin + np.arange(rows[0], rows[1]) * res)
    wc = worldcover.sample_nearest(gx.ravel(), gy.ravel(), src_crs=crs).reshape(gx.shape)
    block = np.zeros(gx.shape, dtype=np.uint8)
    for k, v in WORLDCOVER_TO_CODE.items():
        block[wc == k] = v
    return block


def terrain_colors(codes: np.ndarray) -> np.ndarray:
    lut = np.zeros((256, 4))
    lut[:, :3] = CODE_COLORS[0]
    lut[:, 3] = 1.0
    for k, c in CODE_COLORS.items():
        lut[k, :3] = c
    return lut[codes]
