# Geospatial World Compiler
## Project Plan & Technical Specification

**Project Codename:** WorldGen  
**Status:** Proposed  
**Primary Language:** Python  
**Architecture:** Offline Geospatial Compilation Pipeline  
**Target Outputs:** glTF/GLB, GeoParquet, Graph, Heightfield, JSON Metadata

---

## 1. Project Overview

### 1.1 Objective

Develop a standalone, reusable geospatial world-generation system capable of converting real-world geographic data into structured, geographically accurate, game-ready 3D environments.

The system shall accept a geographic region as input and automatically generate:

- Terrain geometry and elevation data
- Roads and intersections
- Bridges, overpasses, and tunnels
- Buildings and building footprints
- Rivers, lakes, and coastlines
- Vegetation distribution
- Transportation network topology
- Semantic metadata and source provenance
- Tiled, exportable 3D assets

The generated worlds should be reusable across multiple applications, including:

- Military real-time strategy games
- City-building and economic simulations
- First-person and third-person games
- Driving and cycling simulators
- Geographic visualization
- Synthetic environment generation

### 1.2 Design Principles

1. **Engine-agnostic:** No Unreal Engine or Unity dependency in the core compiler.
2. **Geographically grounded:** Preserve real-world geometry, coordinates, connectivity, and topology whenever reliable source data exists.
3. **Semantic-first:** Maintain meaningful geographic objects independently of their generated meshes.
4. **Deterministic:** Identical inputs, source snapshots, and configuration must produce equivalent outputs.
5. **Modular:** Data acquisition, processing, geometry generation, and export are separate components.
6. **Scalable:** Support small test areas initially and large tiled regions later.
7. **Traceable:** Preserve source attribution, confidence, and inferred versus observed properties.
8. **Testable:** Validate both geographic correctness and generated geometry through automated tests.

### 1.3 Non-goals for the Initial Release

The initial implementation will not include:

- Military combat simulation
- Vehicle physics
- AI unit behavior
- Destructible buildings
- Traffic simulation
- Detailed building interiors
- Photorealistic reconstruction
- Runtime procedural world generation

These capabilities may be implemented by downstream applications.

---

## 2. High-Level Architecture

The system shall use the following pipeline:

```text
Geographic Region
       |
       v
Data Acquisition
  - OpenStreetMap
  - Overture Maps
  - Digital Elevation Models
  - Optional Satellite Imagery
       |
       v
Data Normalization
  - Coordinate Projection
  - Geometry Validation
  - Attribute Normalization
  - Source Provenance
       |
       v
Canonical Geographic World
  - Terrain
  - Transportation Network
  - Buildings
  - Water
  - Landcover
       |
       v
World Compilation
  - Terrain Meshing
  - Road Generation
  - Bridge Generation
  - Building Generation
  - Spatial Tiling
       |
       v
Validation & Optimization
       |
       v
World Assets
  - GLB / glTF
  - GeoParquet
  - Graph
  - Heightfield
  - Metadata
       |
       v
Game Engines / Renderers
```

### 2.1 Core Components

| Module | Responsibility |
|---|---|
| `sources` | Geographic data acquisition |
| `spatial` | Coordinate systems and spatial indexing |
| `world` | Canonical geographic data model |
| `topology` | Road connectivity and graph construction |
| `terrain` | Heightfields and terrain geometry |
| `transportation` | Roads, intersections, bridges and tunnels |
| `buildings` | Building massing and procedural geometry |
| `landcover` | Water, vegetation and surface classification |
| `tiling` | Spatial partitioning and asset ownership |
| `export` | Engine-independent output formats |
| `validation` | Geographic and geometric correctness |
| `viewer` | Interactive visualization and debugging |

---

## 3. Technology Stack

### 3.1 Core

- Python 3.12+
- NumPy
- Shapely
- GeoPandas
- PyProj
- Rasterio
- PyArrow
- Trimesh
- NetworkX
- OSMnx

Use type annotations throughout the codebase. Prefer explicit data models and clear module boundaries.

### 3.2 Data Formats

| Data | Internal / Export Format |
|---|---|
| Geographic vectors | GeoParquet |
| Terrain elevation | GeoTIFF / NumPy arrays |
| Transportation graph | Typed nodes and edges, serialized to an explicit versioned schema |
| 3D geometry | glTF / GLB |
| Configuration | YAML |
| World manifest | JSON |
| Diagnostics | JSON |

Graph serialization must not rely on Python pickle for long-term interoperability.

### 3.3 Visualization

Implement an independent web-based viewer using:

- TypeScript
- Three.js
- Vite

The viewer should provide geographic navigation, layer visibility controls, object inspection, and topology visualization.

---

## 4. Canonical Geographic Data Model

The compiler must distinguish semantic geographic data from rendered geometry.

Core entities:

### Terrain

- Elevation
- Terrain coverage
- Coordinate reference system
- Resolution
- Source metadata

### Road

- Stable identifier
- Centerline geometry
- Road classification
- Width and lane information, when available
- Directionality
- Access restrictions
- Surface type
- Elevation profile
- Connected graph nodes

### Bridge

- Stable identifier
- Associated road segments
- Deck geometry
- Width
- Elevation profile
- Clearance, if known
- Structural classification, if known
- Connectivity information

### Building

- Stable identifier
- Footprint
- Height, if known
- Number of floors, if known
- Building classification
- Roof geometry/type
- Source metadata

### Waterbody

- Geographic boundary or centerline
- Water classification
- Surface elevation, if available
- Associated crossings

Every entity must preserve:

```text
id
geometry
properties
source
source_id
confidence
inferred_properties
```

Missing values must remain distinguishable from estimated values.

### Coordinate Systems

Use WGS84 for geographic input and source references.

Convert geometry to an appropriate local metric coordinate system for computation.

Export assets relative to an explicit local world origin to avoid floating-point precision problems.

For geographic regions spanning multiple projection zones, use a suitable local projection or geocentric transformation strategy.

---

## 5. Development Phases

### Phase 0: Project Foundation

**Goal:** Establish the architecture, package, configuration, and reproducible execution environment.

Tasks:

- Create Python package structure.
- Configure `pyproject.toml`.
- Implement CLI.
- Define geographic region configuration.
- Implement coordinate projection utilities.
- Define canonical entity schemas.
- Establish logging and error handling.
- Add unit tests and CI.
- Add source attribution and dataset version tracking.

Example CLI:

```bash
worldgen build \
  --center 25.0330,121.5654 \
  --radius 1000 \
  --profile basic \
  --output ./generated/taipei
```

**Acceptance criteria**

- CLI accepts a valid geographic region.
- Geographic coordinates can be transformed into local metric coordinates and back.
- Output manifest contains region bounds, CRS, local origin, configuration and source metadata.
- The project runs reproducibly in a clean environment.

### Phase 1: Geographic Data Acquisition

**Goal:** Acquire and normalize real-world geographic datasets.

Tasks:

- Implement OpenStreetMap adapter.
- Implement Overture Maps adapter.
- Implement DEM adapter.
- Support bounded geographic queries.
- Normalize geometries and attributes.
- Cache downloaded datasets.
- Record dataset versions and licenses.
- Detect missing or incomplete source coverage.

The data acquisition layer must be independent of geometry generation.

**Acceptance criteria**

- Download roads and buildings for a selected region.
- Acquire available terrain elevation data.
- Convert all inputs into a consistent geographic model.
- Rebuild from cached data without network access.
- Identify missing data explicitly.

### Phase 2: Transportation Topology

**Priority: Critical**

**Goal:** Build a geographically correct road network preserving real-world connectivity.

Tasks:

- Import directed road segments.
- Construct intersection and connector nodes.
- Preserve one-way restrictions.
- Represent bridges, tunnels and elevated roads.
- Prevent false connections at geometric crossings.
- Preserve turn restrictions where source data supports them.
- Handle road segments crossing tile boundaries.
- Construct a routable graph.
- Provide shortest-path queries.

Separate physical road geometry from graph topology.

**Acceptance criteria**

- Ordinary intersections connect correctly.
- Bridges crossing ground roads do not create false intersections.
- Tunnels and overpasses retain their respective topology.
- Dead ends remain dead ends.
- Shortest-path queries work across the complete test region.
- Tile partitioning does not disconnect roads.

### Phase 3: Terrain Generation

**Goal:** Generate geographically aligned terrain geometry.

Tasks:

- Import DEM raster data.
- Resample elevation to a configurable resolution.
- Generate terrain mesh and heightfield.
- Support multiple source resolutions.
- Implement terrain sampling at arbitrary coordinates.
- Address vertical datum differences where required.
- Prepare terrain to integrate with roads and structures.

**Acceptance criteria**

- Terrain aligns with the specified geographic coordinates.
- Elevation sampling is consistent.
- Meshes contain no unexpected gaps.
- Adjacent terrain tiles join continuously.
- Vertical scale is measured in meters.

### Phase 4: Road Geometry Generation

**Goal:** Convert road centerlines and road attributes into 3D geometry.

Tasks:

- Generate road surfaces from centerlines.
- Determine road width from source attributes or configurable defaults.
- Generate intersections.
- Generate elevation-following road geometry.
- Handle curved segments.
- Produce UV coordinates.
- Generate simplified collision geometry.
- Export road mesh separately from transportation topology.

**Acceptance criteria**

- Roads follow source centerlines.
- Road width is geographically plausible.
- Intersections are connected without major mesh artifacts.
- Road elevation transitions are continuous.
- Generated roads align with terrain.
- Routes remain associated with their source semantic entities.

### Phase 5: Bridge and Tunnel Generation

**Priority: Critical**

**Goal:** Generate 3D bridges, ramps and tunnels while preserving correct vertical relationships.

Tasks:

- Identify bridge and tunnel segments.
- Resolve crossing and connectivity topology.
- Estimate or acquire elevation profiles.
- Generate bridge decks.
- Generate ramps and road transitions.
- Implement configurable bridge support geometry.
- Represent tunnel portals.
- Handle terrain clearance and terrain modification.
- Store vertical clearance and source confidence when available.

Do not interpret OSM `layer` values as absolute elevation.

Where bridge geometry cannot be inferred reliably, mark it as uncertain rather than silently generating an apparently authoritative structure.

**Acceptance criteria**

- Bridges connect to roads at both ends.
- Bridge decks remain continuous.
- Crossings preserve appropriate vertical separation.
- Bridge geometry does not incorrectly intersect the terrain.
- Bridges and tunnels are distinguishable in exported semantic data.
- Unknown bridge heights are explicitly identified.

### Phase 6: Building Generation

**Goal:** Create geographically aligned buildings from footprints.

Tasks:

- Import building footprints.
- Repair invalid polygons.
- Handle interior holes and multipolygons.
- Extrude building geometry.
- Use known height or floor count when available.
- Apply configurable height estimation rules when necessary.
- Implement simple roof generators.
- Generate UVs and basic materials.
- Export building metadata and stable IDs.

Initial release should prioritize building shape and location rather than photorealistic appearance.

**Acceptance criteria**

- Building positions align with geographic sources.
- Footprints produce valid meshes.
- Buildings have plausible heights.
- Buildings do not contain invalid or inverted surfaces.
- Buildings preserve semantic identifiers.

### Phase 7: Water and Landcover

**Goal:** Add geographic environmental context.

Tasks:

- Import rivers, lakes, coastlines and landcover.
- Generate basic water surfaces.
- Add terrain-aligned landcover masks.
- Identify forests and vegetation areas.
- Generate deterministic vegetation placement.
- Preserve crossing relationships between waterways and bridges.

**Acceptance criteria**

- Major waterways appear at correct geographic locations.
- Water surfaces and terrain boundaries align reasonably.
- Bridges correctly cross water or terrain.
- Vegetation placement is reproducible.

### Phase 8: World Tiling and Optimization

**Goal:** Support larger regions through spatial partitioning.

Tasks:

- Implement configurable tile sizes.
- Assign stable ownership of entities across tile boundaries.
- Generate tile-local mesh assets.
- Build a spatial tile index.
- Implement geometry simplification.
- Support incremental tile compilation.
- Track source and configuration hashes.
- Avoid regenerating unaffected tiles.

Suggested initial tile size: 256 × 256 meters, configurable.

**Acceptance criteria**

- Adjacent tiles align without visible cracks.
- Cross-tile roads remain topologically connected.
- No duplicate buildings appear at tile boundaries.
- Individual tiles can be rebuilt independently.
- Compilation is deterministic for pinned inputs.

### Phase 9: Interactive Viewer

**Goal:** Enable inspection of compiled worlds without any game engine.

Tasks:

- Build Three.js viewer.
- Load generated GLB assets.
- Add orbit/fly camera.
- Toggle terrain, roads, bridges, buildings and water.
- Overlay transportation topology.
- Select geographic objects.
- Display entity metadata.
- Visualize missing or estimated properties.
- Highlight topology and geometry validation errors.

**Acceptance criteria**

- Users can navigate a generated world.
- Individual geographic layers can be enabled and disabled.
- Clicking a bridge or road displays its metadata.
- Transportation topology can be visually inspected.
- Validation errors are visible in the viewer.

---

## 6. Validation and Quality Assurance

Implement automated validation covering:

### Geographic correctness

- Coordinate conversion accuracy
- Geographic alignment
- Polygon validity
- Source coverage and completeness

### Topological correctness

- Road graph connectivity
- False intersection detection
- Bridge connectivity
- Tunnel connectivity
- Turn restriction preservation
- Cross-tile connectivity

### Geometric correctness

- Degenerate triangles
- Invalid normals
- Terrain intersections
- Road continuity
- Bridge elevation discontinuities
- Mesh boundary gaps

### Reproducibility

- Identical inputs generate equivalent geometry and metadata.
- Source changes are detectable.
- Configuration changes produce traceable outputs.
- Generated assets retain stable object identifiers.

### Automated Test Fixtures

Create small synthetic geographic scenarios:

1. Simple road intersection
2. Road crossing without connection
3. Bridge crossing a river
4. Road passing below an elevated road
5. Tunnel through elevated terrain
6. Curved hillside road
7. Road connecting multiple tiles
8. Building footprint containing holes

Use synthetic fixtures first, then evaluate against selected real-world areas.

---

## 7. Proposed Repository Structure

```text
worldgen/
├── pyproject.toml
├── README.md
├── configs/
│   ├── basic.yaml
│   ├── procedural.yaml
│   └── high_detail.yaml
├── src/worldgen/
│   ├── cli.py
│   ├── config.py
│   ├── sources/
│   ├── spatial/
│   ├── world/
│   ├── topology/
│   ├── terrain/
│   ├── transportation/
│   ├── buildings/
│   ├── landcover/
│   ├── tiling/
│   ├── geometry/
│   ├── export/
│   └── validation/
├── viewer/
│   ├── src/
│   └── package.json
├── tests/
│   ├── unit/
│   ├── integration/
│   ├── fixtures/
│   └── regression/
├── docs/
│   ├── architecture.md
│   ├── data-model.md
│   ├── coordinate-systems.md
│   └── source-attribution.md
└── generated/
```

---

## 8. Initial MVP

The first implementation should focus on a limited geographic region.

**Suggested test region:** Taipei, Taiwan  
**Coverage:** 1 × 1 km  
**Target output:** Basic 3D geographic scene with valid transportation topology.

### MVP Scope

Required:

- Geographic region input
- OSM road acquisition
- OSM building acquisition
- Terrain elevation acquisition
- Canonical entity normalization
- Road graph construction
- Simple road mesh generation
- Bridge connectivity and basic bridge mesh generation
- Simple terrain mesh
- Extruded building meshes
- GLB export
- JSON manifest
- Basic interactive viewer

Not required for MVP:

- Photorealistic textures
- Detailed buildings
- Detailed bridge supports
- Street furniture
- Traffic simulation
- Runtime streaming
- Advanced AI reconstruction

### MVP Definition of Done

A user can execute one command to generate a geographic region and then:

1. Open the result in the viewer.
2. Inspect the real-world terrain, roads and buildings in 3D.
3. Identify bridges and elevated roads.
4. Query routes between points on the road network.
5. Confirm that grade-separated crossings do not create false connections.
6. Inspect source data and inferred properties.
7. Export assets into another 3D application.

---

## 9. Future Research Directions

After the core pipeline is stable, investigate:

### 9.1 Satellite Image Reconstruction

- Building footprint segmentation
- Building height estimation
- Roof geometry reconstruction
- Surface material classification
- Multi-view reconstruction

### 9.2 Procedural Urban Detail

- Regional building grammars
- Facade generation
- Windows, balconies and architectural details
- Street furniture
- Vegetation

### 9.3 Transportation Enhancements

- Lane-level road graphs
- Advanced intersection geometry
- Traffic restrictions
- Road grade and curvature analysis
- More accurate bridge and tunnel reconstruction

### 9.4 Game Engine Integrations

Implement optional adapters for:

- Unreal Engine
- Unity
- Godot
- Custom rendering engines

These adapters must consume compiled assets without requiring modifications to the core geographic compiler.

### 9.5 Large-Scale World Generation

- Region-wide datasets
- Distributed compilation
- Incremental world updates
- Multi-resolution terrain
- Streaming-ready world assets
- On-demand geographic tile generation

---

## 10. Implementation Strategy

Development should proceed in vertical slices rather than implementing every module simultaneously.

**Milestone A: Geographic Foundation**

Region input → source acquisition → normalization → coordinate transformation → manifest.

**Milestone B: Transportation Network**

Road acquisition → canonical graph → connectivity validation → topology visualization.

**Milestone C: 3D World**

DEM → terrain → road geometry → bridges → basic buildings → GLB export.

**Milestone D: Viewer and Validation**

Interactive viewer → metadata inspection → automated validation → regression fixtures.

**Milestone E: Optimization and Reuse**

Tiling → caching → incremental compilation → additional application adapters.

Each milestone must leave the project in an executable and testable state.

---

## 11. Engineering Guidelines

- Keep source-specific parsing logic isolated from canonical geographic models.
- Never infer road connectivity solely from 2D line intersections.
- Never assume that geographic source attributes are complete or correct.
- Keep geometry generation deterministic.
- Use explicit units and coordinate conventions.
- Preserve stable IDs across compiler stages.
- Separate physical facts from application-specific simulation rules.
- Avoid introducing rendering-engine dependencies into the core package.
- Implement automated tests for topology and coordinates before optimizing performance.
- Track provenance, licenses and attribution for every imported dataset.
- Prefer clear, testable algorithms over premature optimization.
- Treat geographic uncertainty as explicit data, not an error to be hidden.

---

## 12. Success Criteria

The project succeeds when it can reliably convert a selected real-world geographic region into a reusable, structured 3D world with:

- Geographically aligned terrain
- Connected road networks
- Correct bridge and tunnel topology
- Buildings positioned using real-world footprints
- Exportable 3D geometry
- Persistent semantic metadata
- Automated validation
- Engine-independent outputs

**Long-term vision:** A reusable world compiler capable of transforming real geographic data into simulation-ready digital environments for multiple independent projects.
