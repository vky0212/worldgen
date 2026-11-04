# Coordinate Systems

## Frames

| Frame | Use | Definition |
|---|---|---|
| WGS84 (EPSG:4326) | Input regions, source references, GeoParquet exports, graph `lon`/`lat` | longitude/latitude in degrees; code always passes **(lon, lat)** order to PROJ (`always_xy=True`) |
| Local ENU | All computation, JSON local coordinates | metres; `x` east, `y` north, `z` up; origin at the region centre |
| glTF | GLB assets | +Y up; `gltf = (x, z, −y)` — a proper rotation, so triangle winding is preserved |

## Local projection

Default `projection.method: tmerc`: a transverse Mercator centred on the
origin (`+proj=tmerc +lat_0=<lat> +lon_0=<lon> +k=1 +x_0=0 +y_0=0
+ellps=WGS84`). Properties:

* the origin maps to exactly (0, 0) and grid north equals true north on the
  central meridian;
* no projection-zone boundary can fall inside a region (unlike UTM);
* point scale error stays below 1e-5 within ~30 km and below 1e-4 within
  ~90 km of the origin — `validation` warns above 1e-4.

`projection.method: utm` uses the origin's UTM zone translated so the origin is
(0, 0); `manifest.crs.local.projected_offset` records the translation and the
GeoTIFF exports use a CRS whose coordinates equal local x/y.

The full PROJ string and WKT are written to `manifest.crs.local`, so any tool
can reproduce the transformation; `worldgen transform` converts points both
ways and reports the round-trip error (unit tests require < 1e-6 m across a
10 km square in both hemispheres and at high latitude).

## Large regions

A single world should stay within ~50 km of its origin. Larger areas are
compiled as several worlds (each with its own origin, ideally on a regular
geographic grid); a geocentric strategy is used to place them relative to each
other: convert each world origin to ECEF (`EPSG:4978`) and express neighbour
origins in the local ENU frame of the viewing world. The tile/asset format
needs no change for this because every asset is relative to its world origin.

## Precision

GLB vertex positions are float32 and relative to the **tile** origin (the tile
root node carries the tile's minimum corner as a float64 translation), so
vertex precision is better than 0.1 mm for 256 m tiles regardless of world
size.

## Vertical reference

Heights are orthometric metres in `manifest.crs.vertical_datum` (EGM2008 for
Copernicus GLO-30). Set `terrain.output_vertical_datum: WGS84_ellipsoid` to
convert to ellipsoidal heights (uses PROJ's EGM2008 grid; requires PROJ network
access or the installed grid). Mixed-datum inputs must be converted to the
target datum before compilation; the source datum is recorded in each
`SourceRecord.vertical_datum`.

OSM `layer` is **never** converted to an absolute height. It only orders items
at a crossing (higher layer above lower layer) and scales the minimum
separation the alignment solver enforces between them.
