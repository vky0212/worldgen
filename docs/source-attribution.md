# Sources, Licenses and Attribution

Every build records, per source, a `SourceRecord` in `manifest.sources` and a
human-readable `ATTRIBUTION.txt` next to the assets. Redistributing a compiled
world requires carrying that attribution.

| Source | Adapter | What is used | License | Attribution |
|---|---|---|---|---|
| OpenStreetMap via Overpass API | `sources/osm.py` | roads, restrictions, buildings and parts, water, coastlines, land cover, trees, tree rows | ODbL 1.0 | © OpenStreetMap contributors |
| Copernicus DEM GLO-30 (AWS Open Data COGs) | `sources/rasters.py` | terrain elevation (EGM2008) | Copernicus DEM licence (free with attribution) | © DLR e.V. 2010-2014 and © Airbus Defence and Space GmbH 2014-2018 provided under COPERNICUS by the European Union and ESA |
| ESA WorldCover 10 m v200 (2021) | `sources/rasters.py` | land-cover classes, DSM correction mask, vegetation classification | CC BY 4.0 | © ESA WorldCover project 2021 / Contains modified Copernicus Sentinel data (2021) processed by ESA WorldCover consortium |
| Meta / WRI CHM v2 (10 m mean derivative) | `sources/rasters.py` | canopy height above ground | CC BY 4.0 | Meta and World Resources Institute, Global Canopy Height Map v2 |
| Overture Maps (buildings theme, optional) | `sources/overture.py` | non-OSM building footprints | ODbL 1.0 | © Overture Maps Foundation and contributors |
| Sentinel-2 L2A via Earth Search STAC (`sources.imagery`, on by default, optional) | `sources/imagery.py` | 10 m true-colour (TCI) cloud-free growing-season composite, baked into `tiles/<id>.ortho.jpg` | Copernicus Sentinel data licence (free, full and open) | Contains modified Copernicus Sentinel data `<years>` (years of the composited scenes) |
| XYZ / WMTS tile service (`sources.imagery.provider: xyz`, optional) | `sources/imagery.py` | higher-resolution orthophoto tiles, baked into `tiles/<id>.ortho.jpg` | as configured in `sources.imagery.license` (required) | as configured in `sources.imagery.attribution` (required) |

## What a SourceRecord contains

* `name`, `provider`, `license`, `attribution`
* `dataset_version` – OSM: `timestamp_osm_base` of the Overpass snapshot;
  rasters: product version; Overture: release
* `url`, `retrieved_utc` – when the cached snapshot was fetched
* `snapshot_hash` – SHA-256 of the cached raw data. Two worlds built from the
  same snapshots have identical hashes; `worldgen compare` reports changed
  sources by this hash.
* `coverage` – e.g. valid-data fraction, class fractions, observation year,
  element counts
* `status` – `ok`, `partial`, `missing`, `disabled`, `failed`, with `notes`
* `vertical_datum` for elevation sources
* imagery: `coverage.m_per_px` / `effective_m_per_px`, `valid_fraction`; for
  Sentinel-2 the composited `scenes` (id, acquisition time, scene cloud cover,
  clear fraction over the region), `date_range` and composite statistics; for
  XYZ the `zoom`, `tile_range` and missing tiles

## Imagery

Imagery is on by default (`sources.imagery.enabled: true`, Sentinel-2). It is
optional: a failed fetch, or `--offline` without cached imagery, is recorded
as `failed` in the manifest and the build continues without orthophotos. Set
`enabled: false` to skip it.

* `sentinel2_l2a` (default provider): scenes are searched on the STAC API
  (`stac_url`, `collection`) within `date_range` (`"P3Y"` = the three years
  before retrieval, or `"YYYY-MM-DD/YYYY-MM-DD"`) and below `max_cloud_pct`
  scene cloud cover. `season` picks the preferred months: `growing` (default)
  is May–October north of the equator and November–April south of it (from
  the region's latitude), `custom` uses `season_months`, `any` has no
  preference. Each preferred-season interval is searched separately, so the
  dry season's clearer scenes do not crowd them out. Candidates are ranked by
  their clear fraction over the region (SCL classes 0, 1, 3, 8, 9, 10, 11
  masked; cloud shadow, unclassified, cloud and cirrus pixels grown by 30 m
  because SCL misses cloud fringes), at most one per date, and the best
  `max_scenes` (default 10) preferred-season scenes are combined per pixel.
  `composite: clear_quantile` (default) drops observations darker than 0.6 ×
  the pixel's median brightness (missed shadows) and takes the whole RGB of
  the observation at the lower third by blue (haze and missed clouds raise
  blue); `composite: median` takes the per-channel median. Other months are used only when
  fewer than `max_scenes` preferred scenes are clear enough, or when the
  preferred ones leave more than 0.5 % of the region clouded, and then only
  for pixels no preferred scene sees clearly. Pixels clouded in every scene
  come from the least-cloudy one. The record lists `months_used`, the
  `season` settings and each scene's `preferred_season`; the attribution
  names the years of the composited scenes.
* `xyz`: any EPSG:3857 (GoogleMapsCompatible) tile service with `{z}`, `{x}`,
  `{y}` in `url_template`. **You are responsible for being allowed to use the
  service**: configuration fails unless `license` and `attribution` are set,
  and both are copied into the manifest and `ATTRIBUTION.txt`. See
  `configs/imagery_taiwan_nlsc.yaml` for an example.
* `local` / `local_path`: an RGB GeoTIFF; set `license`/`attribution` to record
  its terms.

## Cache and offline rebuilds

Raw Overpass responses and the exact raster windows are cached under
`--cache-dir` (default `.worldgen-cache/`) keyed by the request. `--offline`
rebuilds from the cache only and fails with an explicit `OfflineCacheMiss` if
something is not cached. Keep the cache with a world to make it reproducible.

## Observed vs. derived data in outputs

* Observed source attributes: `properties` (meta JSON) / `properties` column.
* Compiler estimates: `inferred` / `inferred_properties` with the method name.
* Known unknowns: `missing`.
* Procedural content (generated trees, estimated heights, default bridge
  heights) is always marked as inferred and carries `source: procedural` or
  `derived`.

## Fair-use notes

* Overpass API is a shared public service: the adapter sends one query per
  region, identifies itself with a User-Agent, retries politely and caches the
  response. For large-scale compilation use a local Overpass instance or OSM
  extracts and point `sources.osm.endpoints` at it.
* Raster windows are read with HTTP range requests; only the bytes covering
  the region are transferred.
* XYZ imagery tiles are fetched with a generic `worldgen/<version>`
  User-Agent, at most 4 requests in parallel, with retries and backoff on
  HTTP 429/5xx; a request above `max_tiles` tiles is refused. Every tile is
  cached, so an interrupted download resumes and rebuilds never re-download.
