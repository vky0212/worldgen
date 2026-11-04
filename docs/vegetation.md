# Vegetation & Forest Generation

Implementation of `plan_ext.md` in `src/worldgen/vegetation/`. Rendering-engine
and gameplay independent.

## Sources

| Source | Used for | Notes |
|---|---|---|
| ESA WorldCover 10 m (2021) | tree / shrub / grass / crop / wetland / mangrove classes, tree-cover fraction | 3×3 sub-samples per cell give class fractions |
| OpenStreetMap | forest, scrub, heath, grass, farmland, orchard, wetland, park polygons; `leaf_type`; individual trees (`natural=tree`) and tree rows | explicit, observed |
| Meta/WRI CHM v2 (10 m mean) | canopy height above ground | `sources.canopy_height`, observation imagery 2019 |
| Local LiDAR / inventories | optional: set `sources.canopy_height.local_path` (a canopy-height GeoTIFF) | |

All records (version, license, observation date, coverage) land in the
manifest.

## Pipeline

```text
OSM + WorldCover + CHM ─► VegetationGrid (classify.py)
                            category, source, canopy height, coverage, confidence
                        ─► refined grid, smooth borders, thinned forest edges
                            (landcover/smoothing.py, smoothing stage)
                        ─► VegetationRegion polygons + statistics (regions.py)
                        ─► tree placement (placement.py) ─► terrain conformance
                        ─► TreeTable ─► GLB instancing + GeoParquet + fields GeoTIFF
```

### Classification (`classify.py`)

Categories: `dense_forest`, `sparse_woodland`, `shrubland`, `grassland`,
`agricultural`, `urban_trees`, `wetland`, `mangrove`.

* WorldCover tree cover surrounded by built-up land (5×5 neighbourhood > 35 %)
  becomes `urban_trees`.
* OSM parks are split into `urban_trees` where WorldCover sees tree cover and
  `grassland` elsewhere.
* An OSM forest polygon whose WorldCover tree fraction is below
  `vegetation.forest_conflict_threshold` becomes `sparse_woodland`.
* `vegetation.priority` (default `[osm, worldcover]`) decides which source
  wins where both have data; cells where both agree are attributed to both.
* Water areas are never vegetated.
* Canopy height comes from the CHM where available (`canopy_observed`), else
  from the category default modulated by smooth deterministic noise and is
  then inferred. Measured canopy below 2 m in a tree category cuts coverage.

### Smoothing (`landcover/smoothing.py`)

The classified grid inherits the 10 m WorldCover pixels, so forest edges would
be cell staircases. The smoothing stage (on unless `smoothing.enabled=false`)
refines the grid by `smoothing.vegetation_upsample` (default 2: 8 m → 4 m),
smooths the category borders (Gaussian class memberships, `border_radius_m`)
and displaces them with the same world-space noise as the terrain colours, so
trees and ground agree. Cells keep the canopy height, coverage, source and
confidence of their source cell; a cell whose category changed takes them from
the nearest cell of its new category. Coverage in tree categories fades
towards the edge (`forest_edge_thinning`, the fraction removed right at the
edge), giving open, irregular forest margins. Exported fields and region
statistics are computed on the refined grid (`fields.json` has its resolution);
regions keep the minimum area `2 × vegetation.resolution²`.

### Placement (`placement.py`)

Deterministic and tile-independent:

1. Sampling cells of `vegetation.cell_size` metres on a global lattice; each
   cell's random stream is seeded only by `(seed, i, j)`.
2. Target stem density from coverage and crown radius (allometry from canopy
   height), converted to the candidate intensity of a **Matérn type-II
   hard-core process** (`λ0 = −ln(1 − λπδ²)/(πδ²)`), with the hard-core
   distance `δ = spacing_factor × crown radius` per category. This prevents
   regular grids and clumping alike.
3. Candidates outside vegetated cells, inside exclusions (buildings + buffer,
   road surfaces, water, intersections), outside the terrain or on slopes
   steeper than `max_slope_deg` are dropped.
4. A candidate is removed if a lower-mark candidate or an **observed** tree is
   within the hard-core distance. Each rule only looks at a bounded
   neighbourhood, so placing a single tile (with a halo) reproduces exactly
   the trees of a global run: stable across tile boundaries, no duplicates.
5. Heights follow the local canopy height with log-normal jitter; archetype is
   drawn from the category's archetype weights (overridden by OSM
   `leaf_type`); crown and trunk radii follow per-archetype allometry.

Mapped OSM trees keep their exact position and tagged attributes and are
`observed`; missing attributes are estimated and listed per tree. Trees along
mapped tree rows are placed at `tree_row_spacing` and marked inferred
(`position`). Shrubs and grass clusters come from a separate per-cell process
capped by `ground_cover.max_instances`.

### Terrain integration

`ground_z` is sampled on the shaped terrain with the triangle-consistent
sampler (so it lies exactly on the terrain mesh). Instances are sunk by
`0.15 m + trunk radius` to avoid floating trunks on slopes.

### Geometry (`regions.py`)

Archetypes `broadleaf`, `conifer`, `palm`, `shrub`, `grass` in unit space
(base at origin, height 1, crown radius 1) with quality profiles
`low|medium|high`. Each tile GLB has one node per archetype using
`EXT_mesh_gpu_instancing` (TRANSLATION / ROTATION / SCALE), with the tree ids
in `extras.instance_ids`.

## Environmental fields

`fields/vegetation_fields.tif` (see `fields/fields.json`): canopy height (m
above ground), canopy coverage, stem density (from the placed trees), ground
obstruction (shrubs + trunk footprints), approximate visual occlusion along a
10 m horizontal sight line, confidence and category. These describe the
environment only; movement costs, concealment and similar rules belong to
downstream applications.

## Validation

`vegetation_placement` and `vegetation_tile_independence` checks (see
`validation/checks.py`) and `tests/integration/test_vegetation.py`.
