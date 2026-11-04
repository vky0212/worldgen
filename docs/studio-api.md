# Studio API (world generation from the browser)

`worldgen serve` serves the viewer, compiled worlds and a small JSON API that
starts and monitors builds. It binds to `127.0.0.1` only.

```bash
worldgen serve --output-root ./generated            # API + every world under ./generated
worldgen serve --world ./generated/taipei            # additionally serves that world at /world/
```

## Static routes

| Path | Content |
|---|---|
| `/` | viewer (`viewer/dist/index.html`) |
| `/generate.html` | region picker + generation progress page |
| `/worlds/<name>/…` | world `<name>` inside the output root (`manifest.json`, `tiles/…`) |
| `/world/…` | the world given with `--world` (404 when none) |

## JSON endpoints

All responses are JSON (`application/json`). Errors: HTTP 4xx/5xx with
`{"error": "<message>"}`.

### `GET /api/health`
`{"ok": true, "version": "0.1.0", "output_root": "<abs path>", "default_world": "/world/" | null, "busy": false}`

### `GET /api/profiles`
`{"profiles": [{"name": "basic", "description": "Basic profile: …"}, …]}`

### `GET /api/plan`
Preview of a region without building it.

Query: `lat`, `lon`, `radius` (metres) **or** `bbox=south,west,north,east`;
optional `profile` (default `basic`), `tile_size`.

```jsonc
{
  "region": {"center": {"lat": 25.033, "lon": 121.5654}, "radius_m": 1000,
             "bbox_wgs84": [west, south, east, north]},
  "footprint": [[lat, lon], …],          // closed ring of the region outline
  "tile_size": 256,
  "tiles": [{"id": "0_0", "ring": [[lat, lon], [lat, lon], [lat, lon], [lat, lon]]}, …],
  "tile_count": 60,
  "area_km2": 3.14,
  "limits": {"max_radius_m": 5000, "max_tiles": 1600}
}
```

### `GET /api/worlds`
Compiled worlds under the output root (newest first):

```jsonc
{"worlds": [{
  "name": "taipei", "url": "/worlds/taipei/",
  "center": {"lat": 25.033, "lon": 121.5654}, "radius_m": 1000,
  "bbox_wgs84": [w, s, e, n], "footprint": [[lat, lon], …],
  "created_utc": "2026-11-04T04:05:22Z", "profile": "basic", "tile_count": 60,
  "statistics": {"roads": 2475, "buildings": 4038, "trees": 55704, "bridges": 13, "tunnels": 56},
  "validation": {"errors": 0, "warnings": 140, "info": 28}
}]}
```

### `POST /api/jobs`
Start a build. Body:

```jsonc
{
  "name": "taipei",                       // optional; [a-z0-9_-]{1,48}; default derived from the centre
  "center": {"lat": 25.033, "lon": 121.5654}, "radius_m": 1000,   // or:
  "bbox": [south, west, north, east],
  "profile": "basic",                     // optional
  "offline": false,                       // optional: use cached sources only
  "force": false,                         // optional: rebuild unchanged tiles
  "fixture": "demo_town",                 // optional: build a synthetic scenario instead (no network);
                                          // centre/radius are then taken from the fixture
  "budget": {                             // optional instance budgets (defaults: on, 250, 2500)
    "enabled": true,                      // drop smaller buildings/trees at random above the budget
    "buildings_per_tile": 250,
    "trees_per_tile": 2500                // trees + shrubs + grass clusters
  },
  "road_surface": "drape"                 // optional: "drape" (follow terrain) or "ribbon"
}
```

`201 {"job": Job}`. Only one build runs at a time; later jobs are `queued`.
Re-using an existing world name rebuilds it incrementally in place.

### `GET /api/jobs`  → `{"jobs": [Job, …]}` (newest first)
### `GET /api/jobs/<id>` → `{"job": Job, "events": [Event, …]}`
### `POST /api/jobs/<id>/cancel` → `{"job": Job}`

### `GET /api/jobs/<id>/events?after=<seq>`
Server-Sent Events. Every event of the job with `seq > after` is replayed, then
new ones are pushed live. Each message:

```text
id: <seq>
event: <type>
data: <Event JSON>
```

`Last-Event-ID` is honoured on reconnect. Comment heartbeats (`: ping`) are
sent every 15 s. The stream closes after the `end` event.

## Job

```jsonc
{
  "id": "a1b2c3d4", "name": "taipei",
  "state": "queued" | "running" | "succeeded" | "failed" | "cancelled",
  "created_utc": "…", "started_utc": "…" | null, "finished_utc": "…" | null,
  "request": { … POST body as normalised … },
  "world_url": "/worlds/taipei/",
  "progress": 0.42,                       // overall 0..1
  "stage": "vegetation" | null,           // current stage
  "message": "…" | null,                  // last error or notable log line
  "summary": { … } | null                 // the `done` event's summary once succeeded
}
```

## Events

Every event has `seq` (1, 2, …), `t` (seconds since the job was created, i.e.
including any time spent queued) and `type`. Events that change overall progress carry `overall` (0..1).

| `type` | Fields | Meaning |
|---|---|---|
| `job` | `state` | job state change (`queued`, `running`) |
| `plan` | `stages: [{name, label, weight}]`, `tiles: [{id, ring}]`, `region`, `sources: [name]` | emitted once at start: what will happen |
| `stage` | `name`, `state: "start"\|"end"\|"skipped"`, `duration_s` (on end), `overall` | pipeline stage boundaries |
| `source` | `name`, `state: "start"\|"ok"\|"partial"\|"missing"\|"failed"\|"disabled"`, `detail` | per-source acquisition (`osm`, `dem`, `worldcover`, `canopy_height`, `overture`) |
| `progress` | `stage`, `done`, `total`, `detail`, `overall` | fine-grained progress inside a stage (tiles during `export`) |
| `tile` | `id`, `state: "building"\|"built"\|"unchanged"` | per-tile export state |
| `log` | `level: "info"\|"warning"\|"error"`, `message` | compiler log lines |
| `done` | `summary: {statistics, validation, tiles_built, tiles_unchanged, duration_s}`, `world_url` | build succeeded |
| `error` | `message` | build failed |
| `end` | `state: "succeeded"\|"failed"\|"cancelled"` | last event of every job |

Stages (in order, with progress weights): `acquire` (Download sources, 0.25),
`normalize` (0.02), `terrain` (0.03), `topology` (0.03), `water_levels`
(0.01), `road_elevation` (0.05), `terrain_shaping` (0.03), `buildings`
(0.02), `landcover` (0.02), `smoothing` (0.02), `intersections` (0.03),
`vegetation` (0.10), `validate` (0.08), `export` (0.28), `manifest` (0.03).
`plan.stages` is
authoritative.

The same event stream is available without the server:
`worldgen build … --progress jsonl` writes one event per line to stdout.
