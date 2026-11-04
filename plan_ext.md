# WorldGen Extension: Vegetation & Forest Generation

**Priority:** P1 (Core Geographic System)  
**Module:** `worldgen.vegetation`  
**Dependencies:** Spatial, Terrain, Landcover, Transportation  
**Output:** Vegetation regions, canopy fields, tree instances, procedural vegetation meshes

## 1. Objective

Develop a geographically grounded vegetation generation system capable of reconstructing approximate real-world forest distribution, vegetation density, canopy height, and individual tree placement.

The module must remain independent of rendering engines and application-specific simulation logic.

## 2. Data Sources

| Source | Information | Priority |
|---|---|---|
| ESA WorldCover | Forest, grassland, shrubland and cropland classification | P0 |
| OpenStreetMap | Forest boundaries, individual trees and tree rows | P0 |
| Meta/WRI CHM v2 | Canopy height estimates | P1 |
| Local LiDAR datasets | Tree height and terrain separation | Optional |
| Regional forest inventories | Tree species and forest type | Optional |

All datasets must retain observation dates, source metadata, licenses and geographic coverage.

## 3. Canonical Vegetation Data Model

### VegetationRegion

Represents a geographic area containing vegetation.

Properties:

- Geographic polygon
- Vegetation category
- Estimated coverage fraction
- Canopy height statistics
- Vegetation density
- Species distribution, if available
- Source attribution and confidence

### CanopyField

A georeferenced raster representing vegetation structure.

Channels:

- Canopy height
- Canopy presence/coverage
- Optional canopy density
- No-data and confidence masks

Canopy height must be explicitly distinguished from terrain elevation.

### TreeInstance

Represents an observed or procedurally generated tree.

Properties:

- Stable instance ID
- Geographic coordinates
- Ground elevation
- Tree height
- Crown radius
- Trunk radius, when available or estimated
- Species or vegetation archetype
- Placement confidence
- Observed versus generated flag
- Deterministic generation seed

## 4. Generation Pipeline

```text
OSM + ESA WorldCover + Canopy Height
                 |
                 v
       Spatial Alignment
                 |
                 v
      Vegetation Classification
                 |
                 v
       Canopy Field Generation
                 |
                 v
     Vegetation Region Extraction
                 |
                 v
       Procedural Tree Placement
                 |
                 v
         Terrain Conformance
                 |
                 v
     Tree Instances + Forest Masks
                 |
                 v
       Optional Mesh Generation
                 |
                 v
      GLB + GeoParquet + Raster
```

### 4.1 Vegetation Classification

Combine vector and raster data to identify:

- Dense forest
- Sparse woodland
- Shrubland
- Grassland
- Agricultural vegetation
- Urban trees
- Wetlands and mangroves

Source conflicts must be resolved using explicit, configurable priority rules.

### 4.2 Tree Placement

Use deterministic, spatially constrained sampling.

Requirements:

- Constrain generation to vegetation regions.
- Use canopy height to influence tree height distribution.
- Avoid placing trees inside buildings, roads and water surfaces.
- Preserve explicitly mapped trees when available.
- Support species-specific spacing and crown sizes.
- Prevent unwanted regular grid patterns.
- Maintain stable instances across tile boundaries.

Generated tree locations must be marked as inferred.

### 4.3 Terrain Integration

- Sample terrain elevation for each tree.
- Align trunks with the terrain surface.
- Use configurable slope constraints.
- Handle mountainous forest regions.
- Preserve forest distribution across tile boundaries.

### 4.4 Procedural Geometry

Provide configurable vegetation archetypes:

- Broadleaf tree
- Conifer
- Palm
- Shrub
- Grass cluster

Output tree instance transformations separately from source meshes to support GPU instancing.

Support multiple geometry quality profiles.

## 5. Application-Independent Environmental Fields

Export optional raster or vector fields describing:

- Vegetation coverage
- Canopy height
- Vegetation density
- Ground obstruction
- Approximate visual occlusion

These represent environmental characteristics, not game-specific movement or combat rules.

Downstream applications may derive movement costs, visibility penalties, concealment or other gameplay properties.

## 6. Validation

Implement tests for:

- Vegetation placement inside valid regions
- Exclusion of roads and buildings
- Correct terrain elevation
- Canopy height consistency
- Deterministic generation
- Tile-boundary continuity
- No duplicate trees across tiles
- Preservation of observed tree positions
- Correct distinction between observed and inferred data

## 7. MVP Scope

The first vegetation milestone must support:

1. Import ESA WorldCover forest masks.
2. Import OSM forest polygons and tree features.
3. Generate forest regions.
4. Generate deterministic tree positions.
5. Sample terrain elevation.
6. Produce simple 3D tree instances.
7. Export forest masks and tree metadata.
8. Display trees and forests in the WorldGen viewer.

Meta/WRI canopy height integration should follow once the initial vegetation pipeline is functional.

## 8. Future Extensions

- High-resolution tree crown segmentation
- LiDAR-based individual tree reconstruction
- Regional species classification
- Seasonal vegetation changes
- Wind animation metadata
- Forest growth simulation
- Dynamic vegetation destruction
- Understory and ground vegetation generation

## 9. Acceptance Criteria

A user can select a geographic region and compile a 3D environment containing geographically plausible forests, individual trees, and other vegetation.

The output must preserve both visual assets and the underlying geographic vegetation information.

The same compiled vegetation data must be reusable in military RTS games, driving simulators, environmental visualization and other applications.
