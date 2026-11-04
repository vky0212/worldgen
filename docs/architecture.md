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
| `worldgen.transportation` | Elevation profiles, bridge/tunnel alignment, road/bridge/tunnel meshes | `elevation.py`, `meshing.py` |
| `worldgen.buildings` | Footprint repair, Overture conflation, heights, roofs, extrusion | `buildings.py` |
| `worldgen.landcover` | Water levels/surfaces/coastlines, terrain land-cover classes | `water.py`, `landcover.py` |
| `worldgen.vegetation` | Classification, canopy field, regions, tree placement, archetypes, fields | `classify.py`, `placement.py`, `regions.py` |
| `worldgen.geometry` | Engine-independent meshes and generators | `mesh.py`, `shapes.py` |
| `worldgen.tiling` | Tile grid and ownership | `grid.py` |
| `worldgen.export` | GLB writer/merger, tile assets, GeoParquet | `gltf.py`, `merge.py`, `tiles.py`, `vectors.py` |
| `worldgen.validation` | In-memory checks and file-level re-validation | `checks.py`, `assets.py` |
| top level | Config, pipeline, build/export orchestration, CLI, static server | `config.py`, `pipeline.py`, `build.py`, `cli.py`, `serve.py` |

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
   blended over a shoulder).
8. **Buildings, land cover, intersections, vegetation** on the shaped terrain.
9. **Validation** (`validation.checks`) on the in-memory world.
10. **Export** (`build.export_world`): tile hashes → only dirty tiles are
    regenerated → tile GLB/collision/meta, tile index, combined `world.glb`
    (binary merge of tile GLBs), graph JSON, GeoParquet, GeoTIFFs, fields,
    diagnostics, attribution and finally the manifest with output hashes.

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
