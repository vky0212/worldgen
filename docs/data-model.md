# Canonical Data Model

Defined in `src/worldgen/world/model.py`. Semantic entities exist independently
of any mesh; meshes reference them by id.

## Common fields (every entity)

| Field | Meaning |
|---|---|
| `id` | Stable id, unchanged across stages, e.g. `road:osm:way/123`, `building:overture:<gers>`, `bridge:osm:way/42`, `tree:gen:3_-1_7` |
| `geometry` | Shapely geometry in the local metric frame |
| `properties` | **Observed** values, normalised (metres, km/h, booleans) |
| `source`, `source_id` | Provenance of the observation (`osm` + `way/123`, `derived`, `procedural`, …) |
| `confidence` | 0–1 confidence in existence and geometry |
| `inferred_properties` | name → `{value, method, confidence}` for estimates |
| `missing` | names of properties known to be unknown |
| `uncertain` | geometry is an uncertain estimate (e.g. a bridge height that could not be constrained) |

An observed value and an estimate never share a map. `Entity.infer()` refuses
to overwrite an observed value. Consumers can therefore always tell *measured*,
*estimated* and *unknown* apart; the viewer's "Observed vs inferred" colour mode
and the `*_inferred` columns in GeoParquet expose exactly that.

## Entities

### Terrain (`terrain.heightfield.Heightfield`)
Regular grid in the local frame (row 0 = south), metres in
`vertical_datum`; source coverage mask; processing metadata (DSM correction,
datum conversion). The grid triangulation is fixed and shared by the sampler
and the meshes, so sampled heights lie exactly on the mesh.

### Road
Centreline per source way (or per in-region piece of a clipped way).
Observed: `class`, `name`, `ref`, `width`, `lanes`, `layer`, `bridge`,
`bridge_type`, `bridge_structure`, `tunnel`, `tunnel_type`, `surface`,
`maxspeed_kmh`, `maxheight`, `direction`, `modes` (legal access for
`motor_vehicle`/`bicycle`/`foot`), raw access tags.
Inferred: `width` (from lanes or class), `direction` (implied one-ways),
`layer` (OSM conventions), `deck_elevation` for structures.
`node_ids` holds the topology node id of every vertex; `elevation` the solved
height of every vertex. Graph edges reference their road via `road_id`.

### Bridge / Tunnel (`Structure`)
A connected component of bridge (or tunnel) road edges.
`road_ids`, `edge_ids`, `anchor_node_ids` (where it meets ground roads),
`crossings` (roads and waterbodies it passes over/under), `level` (max layer).
Observed: `name`, `bridge_type`, `structure`, `clearance` (from `maxheight` of
roads below), `width`, `length`. Inferred: `deck_elevation_range`,
`max_height_above_terrain` / `max_depth_below_terrain`,
`elevation_confidence`, `uncertainty_reasons`. Missing: e.g. `clearance`,
`structure`.

### Building
Footprint (Polygon/MultiPolygon with holes), `is_part`/`parent_id` for OSM
building parts (outline is then `rendered_by_parts`), `base_elevation`.
Observed: `class`, `height`, `min_height`, `levels`, `min_level`, `roof_shape`,
`roof_height`, `roof_levels`, colours, material, name, address.
Inferred: `height` (levels × floor height, class defaults, optional
deterministic variation), `levels`, `min_height`, `roof_shape` (procedural
profile), `ground_elevation`, `footprint_repaired`.

### Waterbody
Area (lake, river polygon, reservoir, sea from coastlines) or centreline
(river, stream, canal, …). Observed: `class`, `name`, `width`. Inferred:
`surface_elevation` (method recorded), `width`. `crossings` lists the bridges
over it.

### LandcoverArea
OSM land-use / natural / leisure polygon with a normalised `class` and the
original `tag`.

### Intersection
Derived junction patch at a ground graph node with degree ≥ 3 (`node_id`,
`degree`, `road_classes`, `streets`).

### VegetationRegion
Polygon of one vegetation category with `coverage_fraction`, canopy height
statistics (`mean`, `p10`, `p90`, `max`), `density_stems_per_ha`, per-source
fractions (`sources`), `canopy_height_observed_fraction`; inferred
`archetype_distribution`; `species_distribution` is missing unless mapped.

### CanopyField (`vegetation.classify.VegetationGrid`)
Raster channels: `canopy_height` (metres **above ground**, never an elevation),
`coverage`, `confidence`, category, source, `canopy_observed` mask.

### TreeInstance (`TreeTable`, columnar)
`ids`, position, `ground_z`, `height`, `crown_radius`, `trunk_radius`, `yaw`,
`archetype`, `species`, `category`, `confidence` (placement), `observed`,
`seed`, `source`, `source_id`, and per tree the list of inferred properties.
Mapped OSM trees are `observed`; trees interpolated along mapped tree rows and
all procedural trees are inferred.

### TurnRestriction
`restriction`, `from_way`, `via_node` / `via_ways`, `to_way`, resolved
`from_edge`, `to_edge`, `via_edges`, and `enforced`.
