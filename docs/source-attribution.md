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
