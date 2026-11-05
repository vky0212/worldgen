# Architecture

WorldGen is an offline compiler. A build is a pure function of
`(region, configuration, source snapshots, compiler version)`; the source cache
pins snapshots so a rebuild reproduces the same bytes.

```text
Region ──► sources/ ──► normalization ──► CanonicalWorld ──► compilation ──► export/ ──► files
           (adapters)   (osm_normalize)   (world/)           (terrain,          (GLB, GeoParquet,
            OSM, DEM,                                         topology,          graph JSON,
            WorldCover,                                       transportation,    GeoTIFF, JSON)
            CHM, Overture                                     buildings, water,
                                                              vegetation, tiling)
                                                                   │
                                                              validation/ ──► diagnostics.json
```

## Packages

| Package | Responsibility | Key modules |
|---|---|---|
| `worldgen.sources` | Acquisition and caching; all source-specific parsing | `osm.py` (Overpass), `osm_normalize.py` (tags → entities), `rasters.py` (Copernicus DEM, ESA WorldCover, Meta CHM v2), `overture.py`, `synthetic.py` (test scenarios), `base.py` (cache, provenance) |
| `worldgen.spatial` | Coordinate frames, regions, raster sampling | `frame.py`, `region.py`, `raster.py` |
| `worldgen.world` | Canonical, source-independent model | `model.py` (entities), `world.py` (container) |
| `worldgen.topology` | Road graph, crossings, routing, graph schema | `graph.py`, `crossings.py` |
| `worldgen.terrain` | Heightfield, DSM→DTM, datum, terrain shaping | `heightfield.py`, `build.py` |
| `worldgen.transportation` | Elevation profiles, bridge/tunnel alignment, junction geometry, road/bridge/tunnel meshes | `elevation.py`, `junctions.py`, `meshing.py` |
| `worldgen.buildings` | Footprint repair, Overture conflation, heights, roofs, extrusion | `buildings.py` |
| `worldgen.landcover` | Water levels/surfaces/coastlines, terrain land-cover classes, border smoothing (post-processing) | `water.py`, `landcover.py`, `smoothing.py` |
| `worldgen.vegetation` | Classification, canopy field, regions, tree placement, archetypes, fields | `classify.py`, `placement.py`, `regions.py` |
| `worldgen.geometry` | Engine-independent meshes and generators | `mesh.py`, `shapes.py` |
| `worldgen.tiling` | Tile grid and ownership | `grid.py` |
| `worldgen.export` | GLB writer/merger, tile assets, GeoParquet | `gltf.py`, `merge.py`, `tiles.py`, `vectors.py` |
| `worldgen.validation` | In-memory checks and file-level re-validation | `checks.py`, `assets.py` |
| top level | Config, pipeline, build/export orchestration, progress events, CLI, server + build jobs | `config.py`, `pipeline.py`, `build.py`, `progress.py`, `cli.py`, `serve.py`, `studio.py` |

The viewer (`viewer/`, TypeScript + Three.js + Vite) only consumes the files
described in [output-format.md](output-format.md); it shares no code with the
compiler.

## Pipeline stages

1. **Acquire** (`pipeline.acquire`). Every adapter returns parsed data plus a
   `SourceRecord` (license, attribution, dataset version, snapshot hash,
   coverage, status). A failing source is recorded as `failed` and the build
   continues; missing coverage is explicit (`missing` / `partial`).
2. **Normalize** (`osm_normalize`). Tags become typed, unit-normalised
   properties; estimates go to `inferred_properties`; ways are clipped to the
   tile-snapped extent with synthetic boundary nodes; multipolygons are
   assembled. Coastlines become sea polygons.
3. **Terrain** (`terrain.build`). The DEM is sampled at every node of a grid in
   the local frame, smoothed to the DEM's native scale, corrected from surface
   model to approximate bare earth in built-up/tree cells, and optionally
   converted between vertical datums.
4. **Topology** (`topology.graph`). Graph nodes are shared source node ids,
   endpoints, boundary cuts and restriction via nodes. Turn restrictions are
   resolved onto edges. `topology.crossings` finds 2D crossings without a
   shared node and classifies them as grade separations or suspicious
   same-level crossings — neither creates a connection.
5. **Water levels** (`landcover.water`). Lakes take a low percentile of the
   shoreline terrain; waterways get a downstream-monotone surface profile.
6. **Road elevation** (`transportation.elevation`). Ground roads follow the
   terrain. Bridges and tunnels are grouped into structures and solved with a
   grade-limited envelope method against anchors, crossing clearances, water
   clearance and terrain. Uncertain results are flagged, never hidden.
7. **Terrain shaping**. Water is carved, terrain is kept below bridge decks,
   and terrain is conformed under ground roads (one full cell past the edge,
   blended over a shoulder). With `smoothing.shorelines`, water areas are
   carved by `landcover.smoothing.shape_shorelines`: within ~1.5 cells of the
   outline the terrain is set to `surface + g·d` (`d` = signed distance to the
   outline, `g` = bank gradient estimated from the original terrain and smoothed
   along the shore), so the terrain triangles cross the water surface on the
   outline instead of zig-zagging along grid nodes; further out it blends back
   to the original terrain. The region edge is never treated as a shore.
8. **Junctions** (`transportation.junctions`): two ends of similar width that
   continue each other get a mitred joint (both ribbons end on the same mitre
   line). Three or more ends get a corner polygon: neighbouring road edge lines
   are intersected (bevel when they diverge, chamfer when very acute), every
   road is trimmed back to its furthest corner, and the polygon's cut vertices
   are exactly the ribbon end vertices — no gaps, no overlapping surfaces.
9. **Buildings, land cover, vegetation** on the shaped terrain. Terrain
   shaping averages overlapping road corridors (weighted by their blend) so
   parallel ways never create steps; ground road and junction surfaces are
   then draped onto the terrain triangles (`geometry/drape.py`: convex ribbon
   quads are clipped in vectorized numpy, other shapes through shapely), and
   instance budgets (`budget.py`) decide which buildings and trees are
   rendered.
10. **Smoothing** (`landcover/smoothing.py`), a post-processing stage after
    land-cover classification (which also classifies the vegetation grid) and
    before tree placement.
    Each class becomes a Gaussian membership field (`smoothing.border_radius_m`)
    displaced by smooth world-space noise (`warp_amplitude_m`,
    `warp_wavelength_m`); every node takes the class of the largest membership,
    so source-pixel staircases become smooth, natural borders and specks vanish.
    Mapped water is kept exactly; a WorldCover water patch outside it that
    reaches within `water_snap_m` of mapped water takes the surrounding land
    class (whole patch; unmapped ponds farther away are kept) up to 64 m from
    land (`MAX_FILL_M`); farther out it stays water. Terrain
    vertex colours are blended over `blend_width_m`; submerged nodes take the
    wet bank colour. The vegetation grid is refined `vegetation_upsample` times
    with the same smoothing and noise (trees and ground colours agree) and tree
    coverage fades towards forest edges (`forest_edge_thinning`).
    Only the masks (mapped water, the water patches to snap) are made for the whole
    grid; the smoothing, colour blending and vegetation refinement run in windows of
    512 nodes with the halo each needs (`_class_reach`: blur + warp + the 64 m fill
    limit), and every node depends only on nodes within it - warp sampling takes its
    weights from the displacement alone, and nearest-cell searches break ties by
    offset - so any window size gives bit-identical grids
    (`tests/unit/test_smoothing_windows.py`) and memory follows the window.
11. **Validation** (`validation.checks`) on the in-memory world (the tree checks run
    during the export, chunk by chunk).
12. **Export** (`build.export_world`): tile hashes → only dirty tiles are
    regenerated → tile GLB/collision/meta, tile index, combined `world.glb`
    (binary merge of tile GLBs, only up to `export.combined_glb_max_tiles`),
    graph JSON, GeoParquet, GeoTIFFs, fields, diagnostics, attribution and
    finally the manifest with output hashes. Dirty tiles are generated and
    written in blocks of 8 × 8 tiles: `export.tiles.PieceSelector` hands each
    block only the entities whose bounding box can reach it (every piece is
    anchored inside its entity's box), and terrain normals are computed per
    tile window, so memory follows the block rather than the world while the
    files stay identical to a whole-world pass.

    **Trees are placed during the export, chunk by chunk.** `worldgen build` compiles
    with `compile_world(..., place_trees=False)`: the vegetation stage prepares the
    world-level inputs (regions, exclusions, the ground-cover density scale) but
    places nothing. The export walks squares of `export.chunk_tiles` × `chunk_tiles`
    tiles (default 8, 2 km): `pipeline.place_vegetation` places the chunk's trees and
    ground cover (`place_trees` pulls in every candidate that can thin a tree inside the
    window, so each window gets exactly the whole-world trees, in the same per-tile
    order), the per-tile instance budget applies, the chunk's tile hashes decide what
    to rebuild, its dirty tiles are written, and its trees go to the tree validation
    checks (`IndependenceSample` keeps the two most wooded tiles across chunks),
    `vectors/trees.parquet` (`TreesParquetWriter`), the vegetation field sums and the
    statistics before the next chunk replaces them. Tree memory therefore follows the
    chunk, not the world; every output file but `trees.parquet` (row order: by chunk)
    is identical whatever the chunk size (`tests/integration/test_chunked_export.py`).
    `vegetation.budget.max_total`, a world-wide cap, places the whole world at once.
    `compile_world` called directly still places every tree, and
    `BuildOptions(keep_trees=True)` keeps them in `BuildResult.world.trees`.

## Progress and browser builds

Every stage boundary, source acquisition, validation check and exported tile
is reported through `progress.current()` (a no-op by default).
`worldgen build --progress jsonl` streams these events to stdout;
`studio.JobManager` runs that command in a child process per job, re-sequences
the events and fans them out over Server-Sent Events (`serve.py`). See
[studio-api.md](studio-api.md).

## Determinism

* All iteration is over sorted ids; JSON is written in a fixed order.
* Every random value comes from `hashing.rng_for(seed, …)` keyed by stable
  identifiers (sampling-cell indices, entity ids), never from global state.
* The GLB writer is in-house and byte-stable.
* `tests/integration/test_build.py::test_deterministic_rebuild` builds twice
  and compares every output hash; `worldgen compare` does the same for any two
  worlds and reports which sources, config keys or files changed.

## Incremental builds

`build_state.json` stores a hash per tile covering the terrain window (with a
one-cell margin for normals), the land-cover window, every entity whose
geometry touches the tile (geometry, properties, elevations), the trees in the
tile, the relevant config sections and the compiler version. Only tiles whose
hash changed (or whose files are missing) are regenerated; `--tiles` limits a
build to chosen tiles and `--force` ignores the hashes.

## Extending

* **New source**: add an adapter in `sources/` returning canonical entities or
  a `GeoRaster` plus a `SourceRecord`; wire it in `pipeline.acquire`. Nothing
  downstream needs to know the source format.
* **Engine adapters** (Unreal, Unity, Godot) should read the tile GLBs, the
  tile metadata and the graph JSON; the core compiler has no engine
  dependency.
