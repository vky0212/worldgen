# WorldGen — Geospatial World Compiler

WorldGen compiles a real-world region into a structured, geographically accurate,
engine-agnostic 3D world. Give it a centre and radius (or a bounding box) and it
produces:

* tiled **GLB** assets: terrain, roads, intersections, bridges (decks, railings,
  piers), tunnels (tubes and portals), buildings with roofs, water, vegetation
  outlines and GPU-instanced trees, plus simplified collision meshes;
* a routable, versioned **road graph** (JSON + GeoParquet) with one-way rules,
  turn restrictions and grade-separated crossings that are never falsely
  connected;
* **GeoParquet** layers for every semantic entity, with observed, inferred and
  missing properties kept apart;
* a **heightfield** (GeoTIFF + raw float32), a terrain land-cover mask and
  **vegetation fields** (canopy height, coverage, density, obstruction,
  occlusion);
* a **manifest** (region, CRS, local origin, resolved config + hash, source
  versions, licenses, snapshot hashes, output hashes) and a **diagnostics**
  report from automated validation;
* an interactive **Three.js viewer** to inspect all of it.

Sources: OpenStreetMap (Overpass), Copernicus DEM GLO-30, ESA WorldCover,
Meta/WRI canopy height (CHM v2) and, optionally, Overture Maps buildings.

## Quick start

Requirements: Python 3.12+, [uv](https://docs.astral.sh/uv/) (or pip), Node 20+
for the viewer.

```bash
uv sync --extra dev
```

```bash
uv run worldgen build --center 25.0330,121.5654 --radius 1000 --profile basic --output ./generated/taipei
```

```bash
cd viewer && npm ci && npm run build && cd ..
```

```bash
uv run worldgen serve --world ./generated/taipei --open
```

Or generate worlds from the browser: `uv run worldgen serve --open` opens
`generate.html`, where you pick a region on a world map, start the build and
watch its progress live (stages, sources, tiles on the map, log), then open the
result in the 3D viewer. The API behind it is documented in
[`docs/studio-api.md`](docs/studio-api.md).

The first build fetches the source data into `.worldgen-cache/` (about 7 MB for
the 1 km Taipei region; raster sources are read as small windows of
Cloud-Optimized GeoTIFFs) and takes about 1–2 minutes, mostly the Overpass query.
Rebuilding the same region with `--offline` reads only from that cache and
reproduces byte-identical outputs.

No network? Every test scenario is also a buildable world:

```bash
uv run worldgen build --fixture demo_town --output ./generated/demo
```

## CLI

| Command | Purpose |
|---|---|
| `worldgen build --center LAT,LON --radius M` / `--bbox S,W,N,E` / `--fixture NAME` | compile a world (`--profile basic\|procedural\|high_detail\|file.yaml`, `--config extra.yaml`, `--set key.path=value`, `--tile-size`, `--tiles 0_0,1_0`, `--force`, `--offline`, `--cache-dir`) |
| `worldgen route --world DIR --from LAT,LON --to LAT,LON [--mode motor_vehicle\|bicycle\|foot] [--json]` | shortest path on the compiled road graph (honours one-ways and turn restrictions) |
| `worldgen validate --world DIR` | re-check the written files: hashes against the manifest, graph integrity, feature-id ↔ metadata links, mesh normals and degenerate triangles |
| `worldgen compare DIR_A DIR_B` | report which sources, config keys and output files differ between two worlds |
| `worldgen transform --world DIR LAT,LON …` (`--inverse` for x,y) | convert between WGS84 and the world's local metres |
| `worldgen fixtures` | list the synthetic scenarios |
| `worldgen serve [--world DIR] [--output-root generated]` | serve the viewer, every world under the output root at `/worlds/<name>/`, the region picker at `/generate.html` and the generation API |
| `worldgen build … --progress jsonl` | emit machine-readable progress events (one JSON object per line) instead of the summary |

## Profiles

`configs/basic.yaml` (reference MVP), `configs/procedural.yaml` (deterministic
procedural enrichment of missing data: height variation, roofs, ground
cover), `configs/high_detail.yaml` (2 m terrain, finer sampling, Overture
buildings, better vegetation meshes). Every key and its default lives in
`src/worldgen/config.py`; the resolved configuration and its hash are written
to the manifest.

## What happens in a build

1. **Acquire** OSM (one Overpass query), DEM / WorldCover / canopy-height
   windows from Cloud-Optimized GeoTIFFs, and optional Overture buildings —
   all cached with retrieval time and content hash.
2. **Normalize** tags into the canonical model (`docs/data-model.md`): ways are
   clipped to the region with explicit boundary nodes, multipolygons assembled,
   invalid footprints repaired; estimates are stored as inferred properties.
3. **Terrain**: the DEM is resampled onto a local metric grid, corrected from
   surface model to approximate bare earth in built-up and forested cells.
4. **Topology**: nodes only where the source shares a node — 2D crossings
   without one are classified (grade separation vs. suspicious same-level
   crossing) but never connected. Turn restrictions are mapped onto edges.
5. **Vertical alignment**: ground roads follow the terrain within a grade
   limit; bridges and tunnels are solved per structure against their ground
   anchors, the roads and water they cross and the terrain, with a grade
   limit. OSM `layer` only orders crossing items; it is never an absolute
   height. Unresolvable cases are marked uncertain and reported.
6. **Terrain shaping**: carve water, keep terrain under bridge decks, conform
   terrain under roads.
7. **Buildings** (heights from tags → levels → class defaults, flat /
   gabled / pyramidal / skillion roofs, holes, building parts), **water**,
   **land cover**, **vegetation** (classification, canopy field, regions,
   deterministic tile-independent tree placement).
8. **Validate** and **export** per tile; unchanged tiles are skipped.

See `docs/architecture.md` for the module map.

## Outputs

The full contract is in [`docs/output-format.md`](docs/output-format.md).
Highlights:

* local frame: metres, x east, y north, z up, origin at the region centre;
  glTF: `(x, z, −y)` with +Y up, vertices relative to each tile's corner;
* each pickable vertex carries `_FEATURE_ID_0`, an index into the mesh's
  `extras.feature_ids`, resolved in `tiles/<tile>.meta.json`;
* trees use `EXT_mesh_gpu_instancing` (instance transforms separate from the
  archetype meshes).

## Viewer

`viewer/` (TypeScript + Three.js + Vite). `generate.html`: world map region
picker (circle or rectangle), live tile-grid preview, build progress (overall
bar, stage timeline, per-source status, tiles colouring in on the map as they
are exported, log), cancel, and links to the result. `index.html`: orbit and fly cameras, layer toggles,
click-to-inspect metadata (observed vs. inferred vs. missing), colour modes
(observed/inferred, confidence, source), road-topology overlay, client-side
routing, validation markers and issue list, lat/lon readout and go-to. See
[`viewer/README.md`](viewer/README.md).

## Validation

Run automatically on every build (`diagnostics.json`, also shown in the
viewer):

* geographic: coordinate round trip and projection scale, polygon validity,
  source coverage (missing DEM tiles, incomplete building data);
* topological: connectivity, unconnected same-level crossings,
  grade-separation clearance, bridge/tunnel anchoring and continuity, dead
  ends, turn-restriction resolution, cross-tile contiguity;
* geometric: road continuity and grades, deck grades and discontinuities,
  deck–terrain clearance outside abutments, terrain tile seams, building
  meshes (watertight, positive volume, normals);
* vegetation: placement inside vegetated regions, exclusion of roads,
  buildings and water, terrain elevation, duplicates, preservation of mapped
  trees, observed/inferred flags, canopy-height consistency, tile
  independence.

## Tests

```bash
uv run pytest
```

* `tests/unit` — config, frames, tag parsing, normalization, Overture
  conflation, cache, heightfield sampling, mesh generators, GLB writer.
* `tests/integration` — the eight plan scenarios (simple intersection,
  crossing without connection, bridge over a river, road under an elevated
  road, tunnel through a hill, curved hillside road, multi-tile road, building
  with holes), turn restrictions, vegetation, and end-to-end builds
  (determinism, incremental tiles, CLI).
* `tests/regression` — snapshot of the demo town
  (`WORLDGEN_UPDATE_GOLDEN=1` to refresh after an intended change).
* `WORLDGEN_NETWORK_TESTS=1 uv run pytest -m network` — a small real region
  from live sources, then an offline rebuild that must be byte-identical.

The viewer has its own tests: `cd viewer && npm test`.

## Repository layout

```text
configs/            profiles (basic, procedural, high_detail)
docs/               architecture, data model, coordinate systems, output format, sources
src/worldgen/       the compiler (see docs/architecture.md)
tests/              unit, integration, regression
viewer/             Three.js viewer
generated/          build outputs (git-ignored)
```

## Known limitations

* Bridge and tunnel heights are estimated from connectivity, crossings, water
  and terrain (OSM rarely records them); every such value is marked inferred
  with a confidence, and structures that cannot be constrained are flagged
  uncertain.
* Copernicus GLO-30 is a 30 m surface model; the bare-earth correction is
  approximate, and short features (tunnel portal cuttings) are below its
  resolution.
* Via-way turn restrictions are preserved in the graph but not enforced by
  the routers.
* Large regions should be split into several worlds (see
  `docs/coordinate-systems.md`).

## Licensing of outputs

Compiled worlds contain derived data from the sources above; keep
`ATTRIBUTION.txt` with any redistributed world (OSM data is ODbL). Details in
[`docs/source-attribution.md`](docs/source-attribution.md).
