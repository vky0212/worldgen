# Compiled World Output Format

This document is the contract between the compiler (`src/worldgen`) and every
consumer of a compiled world: the bundled viewer, engine adapters and external
tools. All JSON documents carry a `schema` name and a semantic `schema_version`;
consumers must reject major versions they do not understand.

## Directory layout

```text
<output>/
├── manifest.json                 # region, CRS, origin, config, sources, output hashes
├── diagnostics.json              # validation report (issues with locations)
├── build_state.json              # per-tile input hashes for incremental builds
├── world.glb                     # optional: every tile combined in one file (export.combined_glb,
│                                 #   only up to export.combined_glb_max_tiles tiles, default 400)
├── ATTRIBUTION.txt               # human-readable source attribution
├── graph/
│   └── road_graph.json           # routable transportation topology
├── tiles/
│   ├── index.json                # spatial tile index
│   ├── <tile>.glb                # tile geometry (all layers)
│   ├── <tile>.collision.glb      # simplified collision geometry
│   ├── <tile>.meta.json          # semantic metadata of every feature in the tile
│   └── <tile>.ortho.jpg          # optional: orthophoto of the tile (sources.imagery)
├── vectors/                      # GeoParquet, EPSG:4326 geometry
│   ├── roads.parquet  bridges.parquet  tunnels.parquet  buildings.parquet
│   ├── water.parquet  landcover.parquet  vegetation_regions.parquet
│   ├── trees.parquet  graph_nodes.parquet  graph_edges.parquet
├── terrain/
│   ├── heightfield.tif           # GeoTIFF, local CRS, float32 metres
│   ├── heightfield.f32           # raw little-endian float32, row 0 = south
│   ├── heightfield.json          # grid description for the raw file
│   └── landcover.tif             # uint8 landcover classes on the terrain grid (borders smoothed)
└── fields/
    ├── vegetation_fields.tif     # multi-band environmental fields (see below)
    └── fields.json               # band descriptions, legends
```

Tile ids are `"<i>_<j>"` (integers, may be negative), e.g. `"-4_3"`. Tile `(i, j)`
covers local `x ∈ [i·T, (i+1)·T)`, `y ∈ [j·T, (j+1)·T)` where `T` is the tile size.

## Coordinate conventions

* **Local frame (ENU):** metres, `x` = east, `y` = north, `z` = up, origin at the
  region centre (`manifest.origin`). The projection is given in
  `manifest.crs.local.proj` (a PROJ string, transverse Mercator centred on the
  origin by default). Heights are orthometric metres in
  `manifest.crs.vertical_datum`.
* **glTF frame:** glTF is +Y up. A local point `(x, y, z)` is written as glTF
  `(x, z, -y)`. So glTF +X = east, +Y = up, −Z = north.
* Every JSON document that contains local coordinates uses ENU `[x, y, z]`.

## manifest.json

```jsonc
{
  "schema": "worldgen.manifest", "schema_version": "1.0.0",
  "compiler": {"name": "worldgen", "version": "0.1.0"},
  "build": {"created_utc": "...", "duration_s": 12.3},          // not deterministic
  "region": {"center": {"lat": 25.033, "lon": 121.5654}, "radius_m": 1000,
             "bbox_wgs84": [west, south, east, north]},
  "crs": {"geographic": "EPSG:4326",
          "local": {"method": "tmerc", "proj": "+proj=tmerc ...", "wkt": "..."},
          "vertical_datum": "EGM2008", "units": "m", "axes": "x=east,y=north,z=up"},
  "origin": {"lat": 25.033, "lon": 121.5654, "height": 0.0},
  "gltf": {"up": "+Y", "mapping": "gltf = (x, z, -y)"},
  "local_bounds": {"xmin": -1024, "ymin": -1024, "xmax": 1024, "ymax": 1024,
                   "zmin": 3.1, "zmax": 180.4},
  "tiling": {"tile_size": 256, "tile_count": 64, "index": "tiles/index.json"},
  "layers": ["terrain", "roads", "bridges", "tunnels", "buildings", "water",
             "vegetation_regions", "trees"],
  "config": { ... }, "config_hash": "sha256...",
  "sources": [{"name": "osm", "dataset_version": "...", "license": "ODbL-1.0",
               "attribution": "...", "status": "ok|partial|missing|disabled", ...}],
  "outputs": {"tiles/0_0.glb": {"sha256": "...", "bytes": 1234}},
  "statistics": { ... },
  "validation": {"errors": 0, "warnings": 4, "report": "diagnostics.json"},
  "files": {"graph": "graph/road_graph.json", "diagnostics": "diagnostics.json",
            "combined_glb": "world.glb", "heightfield": "terrain/heightfield.json"}
}
```

`sources` always lists `imagery` (on by default; `"status": "disabled"` when
`sources.imagery.enabled` is false). A failed or, with `--offline`, uncached
fetch does not fail the build: the entry is `"status": "failed"` with the
reason in `notes` (`"missing"` when there is no usable imagery, e.g. synthetic
fixtures), and the world simply has no orthophotos. Otherwise its `coverage`
holds the effective resolution (`m_per_px`, `effective_m_per_px`),
`valid_fraction` and, for Sentinel-2, the composited `scenes` (`id`,
`datetime`, `cloud_cover`, `clear_fraction`, `preferred_season`), the
`months_used` and the `season` preference; every `tiles/<tile>.ortho.jpg` is
listed in `outputs`.

## tiles/index.json

```jsonc
{
  "schema": "worldgen.tile_index", "schema_version": "1.0.0",
  "tile_size": 256,
  "ortho": {                                    // null when no imagery was baked
    "format": "image/jpeg", "pixels": 32,       // every ortho image is pixels x pixels
    "m_per_px": 8.0,                            // tile_size / pixels
    "source_m_per_px": 10.0, "effective_source_m_per_px": 10.0,
    "levels": {"black": [..], "white": [..], "gain": [..], "gamma": 1.3},
    "orientation": "...", "uv": "...", "version": 1
  },
  "tiles": [{
    "id": "0_0", "i": 0, "j": 0,
    "bounds": [xmin, ymin, xmax, ymax],        // local ENU metres
    "zrange": [zmin, zmax],
    "glb": "tiles/0_0.glb",
    "collision": "tiles/0_0.collision.glb",     // may be null
    "meta": "tiles/0_0.meta.json",
    "ortho": "tiles/0_0.ortho.jpg",             // optional; null or absent without imagery
    "hash": "sha256 of the tile inputs",
    "counts": {"buildings": 12, "road_pieces": 30, "trees": 410, ...}
  }]
}
```

## tiles/<tile>.ortho.jpg

Orthophoto of one tile, written whenever imagery was acquired
(`sources.imagery`, on by default; see `src/worldgen/export/ortho.py`). When it
was not (disabled, failed, offline without cache), every tile has
`"ortho": null` and `index.ortho` is `null`. Provenance (provider, scenes or tile
service, licence, attribution, resolution) is the `imagery` entry of
`manifest.sources`; the image settings are `tiles/index.json` → `ortho`.

* Baseline JPEG, RGB, `N × N` pixels with `N` a power of two (32 … `tile_px_max`),
  the same `N` for every tile of a world: `N = min(tile_px_max, T / m_per_px)`
  rounded up to a power of two, where `m_per_px` is the source ground resolution
  but never finer than `max_m_per_px`.
* **Coverage:** exactly the tile square `bounds = [xmin, ymin, xmax, ymax]`,
  pixel-is-area, no border. Pixel `(column c, row r)` is centred at
  `x = xmin + (c + 0.5)·T/N`, `y = ymax − (r + 0.5)·T/N`.
* **Orientation:** north up. **Row 0 is the north edge** (`y = ymax`),
  **column 0 the west edge** (`x = xmin`).
* **Texture coordinates** of a local point `(x, y)` (ENU metres; for a glTF
  vertex `(gx, gy, gz)` in tile-local coordinates, `x − xmin = gx` and
  `y − ymin = −gz`):

  ```text
  u = (x − xmin) / T          0 at the west edge, 1 at the east edge
  v = (y − ymin) / T          0 at the south edge, 1 at the north edge (image row 0)
  ```

  This is the convention of WebGL with `UNPACK_FLIP_Y`, i.e. three.js
  `Texture.flipY = true` (the `TextureLoader` default): `v = 1` samples image
  row 0. With `flipY = false` (the glTF convention, image row 0 at `v = 0`) use
  `v' = 1 − v = (ymax − y) / T`. Use clamp-to-edge wrapping.
* **Seams:** all tiles sample one world-level image at their own pixel centres
  (bilinear), so edge pixels interpolate imagery up to half a pixel beyond the
  tile edge and adjacent tiles meet without a seam.
* **Colour:** normalised once per world (per-channel 1st/99th percentile
  levels, a weak gray-world white balance, gamma lift, and a soft highlight
  shoulder: display values above `knee` = 0.8 are compressed as
  `knee + a·ln(1 + (v − knee)/a)` instead of clipped, with `a` = `shoulder`
  fitted so the brightest 0.5 % of every tile stays at or below 0.86, i.e.
  219/255); the parameters are in `index.ortho.levels`. Holes without any
  source imagery get the mean colour.
* Tiles are rebuilt incrementally: the imagery content hash and bake settings
  are part of every tile's input `hash`.

## Tile GLB

* glTF 2.0 binary, one scene, one root node named `tile_<id>` whose
  `translation` is the tile's minimum corner in glTF axes
  (`[i·T, 0, −j·T]`). All vertex positions are relative to that corner.
* The root node's children are **layer nodes**, named exactly after the layer:
  `terrain`, `roads`, `bridges`, `tunnels`, `buildings`, `water`,
  `vegetation_regions`, `trees`. Each layer node has `extras.layer`. A layer
  node is omitted when the tile has nothing on that layer.
* Pickable meshes carry a custom vertex attribute **`_FEATURE_ID_0`** (scalar
  `FLOAT`). Its value is an index into `extras.feature_ids` of the same mesh
  (the list is also copied into the primitive's `extras`). The referenced string
  is the entity id; look it up in the tile's `meta.json`. Three.js' GLTFLoader
  exposes the attribute as `geometry.attributes._feature_id_0`, mesh extras as
  `mesh.userData`, primitive extras as `geometry.userData`.
* `terrain` uses `COLOR_0` vertex colours derived from the landcover classes,
  blended across class borders by the smoothing stage (`smoothing.blend_width_m`);
  submerged vertices carry the wet bank colour.
  Terrain feature id is the single entity `terrain:<tile>`.
* `vegetation_regions` uses glTF `LINES` primitives (draped outlines).
* `trees`: one child node per vegetation archetype (`broadleaf`, `conifer`,
  `palm`, `shrub`, `grass`) using **`EXT_mesh_gpu_instancing`** with
  `TRANSLATION`, `ROTATION`, `SCALE` attributes. The archetype mesh is defined
  in a unit space (base at origin, height 1, crown radius 1) and scaled per
  instance. Node `extras.instance_ids` lists the tree entity ids in instance
  order. The extension is listed in `extensionsUsed` (not required).

## tiles/<tile>.meta.json

```jsonc
{
  "schema": "worldgen.tile_meta", "schema_version": "1.0.0",
  "tile": "0_0",
  "entities": {
    "building:osm:way/123": {
      "kind": "building",              // terrain|road|intersection|bridge|tunnel|building|
                                       // water|landcover|vegetation_region|tree
      "layer": "buildings",            // GLB layer the feature is drawn on
      "source": "osm", "source_id": "way/123",
      "confidence": 0.9,               // 0..1
      "properties": {"height": 31.0, "building": "office", "name": "..."},   // observed
      "inferred": {"levels": {"value": 10, "method": "height/floor_height", "confidence": 0.5}},
      "missing": ["roof_shape"],       // known-unknown properties
      "uncertain": false,              // true when geometry is an uncertain estimate
      "centroid": [x, y, z]            // local ENU
    }
  }
}
```

Observed and inferred values are always kept in separate maps; a key that
appears only in `inferred` is an estimate, a key in `missing` is unknown.

## graph/road_graph.json

```jsonc
{
  "schema": "worldgen.road_graph", "schema_version": "1.0.0",
  "origin": {"lat": .., "lon": .., "height": 0}, "crs": {"local": "+proj=..."},
  "nodes": [{"id": "n:osm:123", "osm_id": 123, "kind": "intersection",   // intersection|dead_end|
             "x": 1.0, "y": 2.0, "z": 3.0, "lon": .., "lat": .., "degree": 3}],  // connector|boundary
  "edges": [{"id": "e:osm:way/42:0", "road_id": "road:osm:way/42",
             "from": "n:osm:1", "to": "n:osm:2",
             "direction": "both",             // both|forward|backward (relative to from→to)
             "length_m": 120.5, "class": "primary", "name": "...",
             "bridge": false, "tunnel": false, "layer": 0,
             "structure_implied": false,       // bridge/tunnel inferred from a non-zero layer only
             "structure_id": null,             // bridge:/tunnel: entity id when on a structure
             "modes": {"motor_vehicle": true, "bicycle": true, "foot": true},
             "maxspeed_kmh": 50,
             "geometry": [[x, y, z], ...],     // ENU, from-node to to-node
             "tiles": ["0_0", "1_0"]}],
  "restrictions": [{"id": "r:osm:relation/9", "restriction": "no_left_turn",
                    "kind": "prohibitory",     // prohibitory|mandatory
                    "from_edge": "e:..", "via_node": "n:..", "to_edge": "e:..",
                    "via_edges": [],           // via-way restrictions
                    "enforced": true}]         // false when routing cannot enforce it
}
```

Routing semantics: an edge with direction `both` may be traversed either way,
`forward` only from→to, `backward` only to→from; the `foot` mode ignores
direction (pedestrians may walk against one-way traffic). An edge is usable by
a mode when `modes[mode]` is true. A prohibitory restriction forbids moving from
`from_edge` through `via_node` onto `to_edge`; a mandatory restriction
(`only_*`) forbids every other continuation from `from_edge` at `via_node`.
Restrictions with `enforced: false` (via-way restrictions, or ones whose member
ways lie outside the region) are preserved but not applied. Immediate U-turns
back along the same edge are only allowed at dead ends. Bridge and tunnel edges
carry their densified deck profile in `geometry`.

## diagnostics.json

```jsonc
{
  "schema": "worldgen.diagnostics", "schema_version": "1.0.0",
  "summary": {"errors": 0, "warnings": 3, "info": 10,
              "checks": {"road_graph_connectivity": {"status": "pass", "issues": 0}}},
  "issues": [{"id": "false_intersection:0007", "check": "false_intersection",
              "severity": "warning",           // error|warning|info
              "message": "...", "entity_ids": ["road:osm:way/1"],
              "location": {"x": .., "y": .., "z": .., "lon": .., "lat": ..},  // may be null
              "tile": "0_0"}]
}
```

## terrain/heightfield.json

```jsonc
{"schema": "worldgen.heightfield", "schema_version": "1.0.0",
 "xmin": -1024, "ymin": -1024, "resolution": 4.0, "cols": 513, "rows": 513,
 "dtype": "float32", "byte_order": "little", "row_order": "south_to_north",
 "vertical_datum": "EGM2008", "units": "m", "file": "heightfield.f32",
 "triangulation": "quad (i,j)-(i+1,j+1) diagonal"}
```

Sample `(x, y)` → column `(x − xmin)/resolution`, row `(y − ymin)/resolution`.
Within a grid cell the surface is two triangles split along the
`(col, row)–(col+1, row+1)` diagonal, exactly as in the terrain meshes.

## fields/vegetation_fields.tif

Bands (float32, local CRS, nodata = NaN): 1 `canopy_height_m` (height above
ground, not elevation), 2 `canopy_coverage` (0–1), 3 `vegetation_density`
(stems per hectare), 4 `ground_obstruction` (0–1), 5 `visual_occlusion` (0–1),
6 `confidence` (0–1), 7 `vegetation_category` (code, see `fields.json`).
These are environmental descriptors only; gameplay rules are derived downstream.
