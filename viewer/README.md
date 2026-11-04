# WorldGen viewer

Interactive Three.js viewer for worlds compiled by WorldGen. It reads the output
contract in [`../docs/output-format.md`](../docs/output-format.md): `manifest.json`,
`tiles/index.json`, the tile `.glb` + `.meta.json` files,
`graph/road_graph.json` and `diagnostics.json`.

It also contains the **generate page** (`generate.html`): pick a region on a map, start a
WorldGen build through the `worldgen serve` API ([`../docs/studio-api.md`](../docs/studio-api.md))
and watch its progress live, then open the result in the viewer.

Stack: TypeScript, Three.js, Leaflet, Vite, proj4. There is no UI framework, only plain DOM and CSS.

## Requirements

- Node.js 22.18 or newer (Node 24 is recommended). `npm test` relies on Node's built-in
  TypeScript type stripping.
- npm 10 or newer.

```sh
npm install
```

## Development

The dev server serves `/world/*` from the directory in `WORLDGEN_WORLD`. The path can be
absolute, or relative to the directory you run npm from:

```sh
WORLDGEN_WORLD=../generated/taipei npm run dev      # bash / zsh
$env:WORLDGEN_WORLD='..\generated\taipei'; npm run dev   # PowerShell
```

Then open <http://localhost:5173/>. If `WORLDGEN_WORLD` is not set, `/world/*` returns 404
with a hint, and the viewer shows a "could not load the world manifest" message (or a world
picker, when the API below is running and lists worlds).
`npm run preview` (which serves the production build) uses the same middleware.

### With the WorldGen API (generate page, world selector)

Both `npm run dev` and `npm run preview` proxy `/api/*` and `/worlds/*` to `worldgen serve`,
at `WORLDGEN_API` (default `http://127.0.0.1:8765`). Server-Sent Events stream through the
proxy unbuffered. Start the backend, then the dev server:

```sh
worldgen serve --output-root ./generated              # repository root, port 8765
npm run dev                                           # viewer/ -> http://localhost:5173/generate.html
WORLDGEN_API=http://127.0.0.1:9000 npm run dev        # backend on another port
```

Without the dev server, `worldgen serve` serves the built pages itself at
<http://127.0.0.1:8765/> and <http://127.0.0.1:8765/generate.html>. When the API cannot be
reached, the generate page shows how to start it and retries every 5 s; the viewer simply
hides its API features (static hosting keeps working).

## Build, typecheck, test

```sh
npm run build      # tsc --noEmit (app + node configs) && vite build -> dist/
npm run typecheck  # type-check only
npm test           # unit tests (node --test): routing, job-progress reducer, region helpers,
                   # building styles, graphics levels + frame-rate governor, solar position
```

`dist/` contains both pages (`index.html`, `generate.html`) and is built with `base: './'`,
so it works from any path. The intended setup is the viewer at `/` with the compiled world at
`/world/` and the API at `/api/` (for example `worldgen serve`). The pages call the API
relative to themselves (`api/...`), so they must be served next to it.

## URL parameters

Viewer (`index.html`):

| Parameter | Meaning |
|---|---|
| `world` | Base URL of the compiled world directory. It can be absolute or relative to the page. Default: `world/` relative to the page, which is `/world/` when the viewer is served at `/`. A trailing `/` is added if missing. Example: `?world=https://example.org/worlds/taipei/` |
| `render` | `photo` opens in the photo-realistic mode, `standard` in the standard one (default: the last choice, else standard) |
| `graphics` | Photo-realistic graphics level: `auto` (default), `low`, `medium`, `high`, `ultra` |
| `tod` | Local solar time in hours for the sun, 4.5 … 20.5 (default 10.5) |
| `ground` | Photo-realistic ground colour: `satellite` (default when the world has `tiles/<tile>.ortho.jpg` imagery) or `procedural` (landcover colours). Overrides the **Ground** choice |
| `cam` | Initial view `x,y,z[,distance,yaw,pitch]`: look at local ENU `(x, y, z)` from `distance` m (default 150), heading `yaw`° (0 = from the south, 90 = from the east), `pitch`° above the horizon (default 35). For reproducible screenshots and bug reports |

Generate page (`generate.html`):

| Parameter | Meaning |
|---|---|
| `tiles` | Leaflet tile URL template for the base map instead of OpenStreetMap's standard tiles (`https://tile.openstreetmap.org/{z}/{x}/{y}.png`, max zoom 19). Example: `?tiles=http://localhost:8080/{z}/{x}/{y}.png` |
| `tiles_attribution` | Attribution text for a custom `tiles` layer (default: "© OpenStreetMap contributors") |

## Generate page

`generate.html` (linked as **＋ New world** from the viewer header):

- **Region.** *Circle*: click the map to set the centre, drag the centre handle to move it and
  the edge handle (or the slider / number field, 100 m ... `limits.max_radius_m`) to resize.
  *Rectangle*: drag on the map to draw a bounding box (the draw tool is armed when you switch
  to rectangle mode or press **Draw rectangle**; Esc cancels), then drag the corner handles to
  resize or the centre handle to move it. Lat/lon (or south/west/north/east) fields are
  synced both ways. Presets jump to a few well-known places; **Fit to selection** zooms to the
  region. The last region is remembered in `localStorage`.
- **Plan preview.** Every change asks `GET /api/plan` (debounced) and draws the footprint and
  the tile grid, with the tile count, area and tile size. More than 200 tiles shows a warning;
  a region the server rejects (HTTP 400) disables **Generate** with the server's reason.
- **Build.** The name is suggested from the preset / world name or the centre
  (`n25_033_e121_565`), editable, and validated against `[a-z0-9_-]{1,48}`; reusing a world
  name rebuilds it incrementally. Profile (from `/api/profiles`), *offline* and *force*.
- **Progress.** Overall bar with percentage, elapsed time, the current stage and an estimate
  of the time left; a stage timeline (pending / running / done / failed / skipped) with
  durations, source chips under *Download sources* (ok / partial / missing / failed /
  disabled; the detail is the tooltip) and a tile bar under *Export tiles*. On the map, tiles
  are coloured live: pending (grey outline), building (amber, pulsing), built (green),
  unchanged (teal); click a tile for its id and state. The log console keeps the last 300
  lines, highlights warnings and errors, can show warnings only and follows new lines unless
  you scroll up. **Cancel build** asks the server to cancel. When the build succeeds, a summary
  card shows roads, buildings, trees, bridges, tunnels, validation counts, tiles built /
  unchanged and the duration, with **Open in 3D viewer** (`index.html?world=<world_url>`).
- **Reconnects.** Events come from `EventSource(/api/jobs/<id>/events?after=<seq>)`. A dropped
  stream is reopened after the last seen `seq` (with backoff; from the second failure on, the
  job record is re-read first). The active job id is kept in `localStorage`, so a reload
  re-attaches to it; closing the panel of a running build only detaches (it keeps running).
- **Jobs and worlds.** Recent jobs (click one to follow it; failed or cancelled ones have
  **Retry**) and existing worlds, which are also drawn on the map as dashed outlines with a
  popup (**Open in viewer**, **Rebuild** fills the form with that world's region and name).

In the viewer, when the API answers `/api/health`, a **World** selector lists the worlds of
`/api/worlds` and switches with `?world=<url>`. If the viewer is opened without `?world=` and
`world/` cannot be loaded, it offers those worlds (and **New world**) instead of only the error.

## Controls

| Input | Action |
|---|---|
| **Orbit/map mode** (default) | Left-drag pans, right-drag rotates, the wheel zooms towards the cursor |
| **Fly mode** | W/A/S/D (or the arrow keys) move, Q/E move down/up, Shift moves 5× faster, drag looks around, the wheel changes speed |
| `F` | Switches between orbit and fly mode (the button in **View** does the same) |
| Click | Inspects the clicked feature, graph edge or node, or validation marker |
| `Esc` | Stops route picking, or clears the selection |

## Photo-realistic mode

**View → Render: Photo-realistic** switches to a physically based look. **Graphics** picks a level;
**Time** moves the sun. **Ground** picks what the ground colour comes from. The choices are
remembered in `localStorage`.

- **Satellite ground.** Where a world has orthophotos (`sources.imagery`, see
  `../docs/output-format.md`), they colour the ground by default. Up close, where one satellite
  pixel would cover many screen pixels, the imagery fades to a low-frequency tint and the
  procedural detail carries the surface. Very dark imagery (baked shadows) is lifted towards the
  landcover brightness. Worlds without imagery use the landcover colours, and the selector is
  disabled.

- **Lighting.** A Preetham sky with clouds lights the scene through image-based lighting, and the
  sun sits at the real solar position for the world's latitude and today's date. The sun-to-sky
  ratio is about 6 : 1, as on a clear day. At night the windows light up.
- **Atmosphere.** Height fog takes its colour from the sky, so valleys are hazier than ridges.
- **Shadows.** One sun shadow map follows what the camera looks at. Its size is quantised and its
  centre snapped to texels, and it re-renders only when it moves.
- **Post-processing.** HDR rendering with MSAA, depth-only ambient occlusion (GTAO), bloom on
  highlights only, and ACES tone mapping.
- **Materials.** The tiles' glTF materials are patched in place: every kind compiles once and is
  shared by all tiles, and the colour modes keep working.
  - Terrain gets slope-dependent triplanar rock and grass, soil or forest-floor detail.
  - Roads get asphalt and gravel; bridges and tunnels get concrete.
  - Water gets animated normals and Fresnel reflection.
  - Trees get per-instance colour variation, translucency and wind.
- **Procedural texture engine.** On every graphics-level change, 22 material layers are baked on
  the GPU into mipmapped texture arrays (no image assets): 10 facade archetypes, 5 roofs and 7
  ground layers.
- **Buildings from geo data.** Each building gets a facade archetype, a ground floor (shopfront
  or not), a roof material, tints and a storey height.
  - They come from its metadata: OSM class, amenity, `building:material`, `building:colour`,
    `roof:colour`, `roof:shape`, levels/height (observed or inferred) and ground elevation.
  - A regional style profile from the world's location sets the defaults. Examples:
    subtropical East Asia → ceramic-tile facades with window grilles, metal roofs and rooftop
    water tanks; Nordic → timber; Mediterranean → plaster and clay.
  - Window rows line up with the real storeys, and results are deterministic per building id.
  - See `src/render/buildingStyle.ts`.
- **Tree level of detail.** Per tile, trees are drawn full near the camera, as canopies without
  trunks further out, and as 4–8-triangle impostors beyond that. Past the draw distance they are
  not drawn. Trees are most of a world's triangles.
- **Frame rate.** Each level is sized to run well above 45 fps. A governor watches the frame time
  (and the GPU time where `EXT_disjoint_timer_query_webgl2` exists) and holds 45 fps:
  - It first lowers the render resolution (down to 0.5×).
  - Then it drops effects in this order: AO, bloom, shadow resolution, tree shadows, tree
    distance, MSAA, shadows.
  - Everything comes back with hysteresis when there is headroom.
  - **Auto** also moves between levels.
  - The status bar shows fps, level and scale. It turns amber while effects are lowered; the
    tooltip lists them.

| Level | Resolution | Shadows | AO / bloom | Trees full / canopy / impostor | Textures |
|---|---|---|---|---|---|
| Low | 0.5× | 1024², no tree shadows | – | 150 / 350 / 1600 m | 128 px, albedo only |
| Medium | 0.65× | 2048² | – | 250 / 550 / 2600 m | 256 px + normals |
| High | 0.8× | 4096² | half-res / on | 350 / 800 / 4200 m | 256 px + glass reflections |
| Ultra | 1× | 4096², terrain casts | full-res / on | 500 / 1200 m / all | 512 px + window parallax |

(Resolution is relative to `min(devicePixelRatio, 2)`.)

## Features

- **Loading.** Tiles load nearest-first with 4 parallel requests, and a progress bar
  shows how far it has got. Each tile's `meta.json` loads alongside its GLB.
  `road_graph.json` and `diagnostics.json` are optional. Missing or broken files show a
  message (a toast, plus a line in **Messages**) and never stop the app. A document whose
  `schema_version` major version is not `1` is rejected.
- **Layers.** You can toggle `terrain`, `roads`, `bridges`, `tunnels`, `buildings`,
  `water`, `vegetation_regions` and `trees`. This shows or hides every layer node with
  that name in every tile, including tiles that load later.
- **Inspection.** Clicking a feature raycasts the visible meshes, reads
  `_feature_id_0` at the hit face, maps it through `feature_ids` to the entity id, and
  shows that entity's record from the tile `meta.json`. The panel lists:
  - kind, id, source and source id (OSM ids link to openstreetmap.org), and confidence
  - observed properties
  - inferred properties, in orange italics, with their method and confidence
  - missing properties
  - an *uncertain* badge when the geometry is an uncertain estimate

  Clicking an instanced tree resolves `instanceId` through `instance_ids`. The selected
  feature's own triangles or segments are overlaid in yellow, with a bounding box. Ids in
  the panel are links: clicking one selects that entity, graph edge or node.
- **Colour modes.**
  - *Materials*: the original materials.
  - *Observed vs inferred*: green when fully observed, orange when the feature has
    inferred properties, red when it is uncertain.
  - *Confidence*: a red-to-green gradient.
  - *Source*: one colour per `source`.

  Features without metadata are grey. Terrain is drawn muted so that features on top of
  it stay readable.
- **Topology overlay.** Graph edges are drawn as screen-space lines:
  - blue: two-way road
  - cyan: one-way, with a chevron showing the direction
  - orange: bridge
  - purple, dashed: tunnel, drawn through the terrain
  - grey: no motor vehicles

  Nodes are coloured by kind. Clicking an edge or node shows its record, its incident
  edges and its turn restrictions.
- **Routing.** In **Route**, pick a start and end point on the map, or type them as
  `lat, lon`. Each point snaps to the nearest graph node that the chosen mode can use.
  Routing is Dijkstra over directed arcs:
  - Edge `direction` is honoured (`both`, `forward` = from→to, `backward` = to→from).
  - Edges with `modes[mode] === false` are skipped.
  - Turn restrictions with `enforced: true` are applied, using edge-based states
    (node, incoming edge). A *prohibitory* restriction forbids `from_edge → to_edge` at
    `via_node`. A *mandatory* one allows only `to_edge` after `from_edge` at `via_node`.
  - Immediate U-turns back along the same edge are allowed only at dead ends.
  - Via-way restrictions (non-empty `via_edges`) are skipped and counted.
  - "Ignore one-way restrictions" is an option, off by default.

  The route is drawn as a thick magenta line, with its length and edge count. When no
  route exists, the panel says so clearly.
- **Validation.** Issues that have a `location` appear as pins coloured by severity:
  red for errors, amber for warnings, blue for info. The issue list can be filtered by
  severity, by text and by "has a location". Clicking an issue flies to it and shows its
  message and entity links.
- **Geography.** The status bar shows the cursor's lat/lon (proj4 inverse of
  `manifest.crs.local.proj`), its local x/y and the surface elevation. **Go to** accepts
  `lat, lon`. The **World** panel lists the region, CRS, sources and licences.
  Attribution is always shown at the bottom right of the status bar.

## Source layout

| File | Purpose |
|---|---|
| `src/main.ts` | App wiring: scene, lights, sidebar, events, selection, routing UI |
| `src/types.ts` | Types that mirror `docs/output-format.md` |
| `src/loader.ts` | `?world=` resolution, JSON fetching with 404 handling, schema checks, progressive tile loader |
| `src/world.ts` | Loaded tiles, layer nodes, pickable objects, entity registry and occurrences |
| `src/picking.ts` | Feature-id and instance-id resolution, raycasting helper, selection highlight |
| `src/colorModes.ts` | Vertex-colour and instance-colour recolouring per mode; restores original materials |
| `src/topology.ts` | Road-graph overlay (fat lines, chevrons, node points) and graph picking |
| `src/routing.ts` | Pure Dijkstra router with one-way handling and turn restrictions. No three.js; unit-tested |
| `src/routeView.ts` | Route line and start/end markers |
| `src/validation.ts` | Validation pins and the filterable issue list |
| `src/controls.ts` | Orbit (MapControls) and fly camera, framing, fly-to, dynamic near/far |
| `src/geo.ts` | proj4 WGS84 ↔ local ENU; ENU ↔ three axes `(x, z, -y)` |
| `src/render/photoreal.ts` | Photo-realistic mode: sky + IBL, sun, fitted shadows, fog, post chain, governor wiring, satellite ground |
| `src/render/quality.ts` | Graphics levels and the frame-rate governor. Pure; unit-tested |
| `src/render/materials.ts` | In-place shader patching of the tile materials (one program per kind × flags) |
| `src/render/shaders/` | GLSL: shared noise, the texture bake, the surface patches (terrain, roads, water, trees, buildings, fog) |
| `src/render/textureEngine.ts` | GPU bake of the procedural material layers into mipmapped texture arrays |
| `src/render/buildingStyle.ts` | Geo data → building style; regional profiles; texture layer layout. Pure; unit-tested |
| `src/render/treeLod.ts` | Per-tile tree LOD with impostors sharing the instance matrices |
| `src/render/sun.ts` | Solar position. Pure; unit-tested |
| `src/render/gpuTimer.ts` | GPU frame time via `EXT_disjoint_timer_query_webgl2` |
| `src/ui.ts` | DOM helpers, info panel, status bar, toasts, progress bar |
| `src/dom.ts` | Shared DOM helpers (`h`, sections, checkboxes) without three.js |
| `src/api.ts`, `src/apiTypes.ts` | `worldgen serve` API client and types (docs/studio-api.md) |
| `src/studio.ts` | Viewer integration: API probe, world selector, world picker |
| `src/generate/main.ts` | Generate page wiring: map, form, plan preview, jobs and worlds lists |
| `src/generate/progress.ts` | Pure job-progress reducer (events -> UI state, idempotent by `seq`). No DOM; unit-tested |
| `src/generate/session.ts` | Follows one job: replay, SSE, reconnect after the last `seq`, batching into the reducer |
| `src/generate/progressView.ts` | Progress panel: overall bar, stage timeline, source chips, result card, log console |
| `src/generate/regionEditor.ts` | Leaflet circle / rectangle editing with handles and rectangle drawing |
| `src/generate/mapLayers.ts` | Base map (`?tiles=`), plan preview grid, live tile grid + legend, world outlines |
| `src/generate/region.ts` | Pure region helpers: API query/body, name suggestion and validation, presets. Unit-tested |
| `src/generate/generate.css` | Generate page styles (on top of `src/style.css`) |
| `test/routing.test.ts` | Routing tests on a hand-made mini graph (one-way edges, `no_left_turn`, mandatory turns, modes, U-turns) |
| `test/progress.test.ts` | Reducer tests: full build, failure, cancel, replay idempotency, log trimming |
| `test/region.test.ts` | Region helper tests: plan/job parameters, names, conversions |
| `test/buildingStyle.test.ts` | Region profiles, class/material/levels mapping, colour parsing, determinism, packing |
| `test/quality.test.ts` | Level monotonicity, feature shedding, governor scale-down/shed/recover/hysteresis |
| `test/sun.test.ts` | Solar position: equinox, solstice, morning/afternoon, night |
| `vite.config.ts` | Vite config: both pages, the `/world/*` middleware and the `/api` + `/worlds` proxy |

## How three.js' GLTFLoader exposes the contract

- `_FEATURE_ID_0` becomes `geometry.attributes._feature_id_0`, because attribute names
  are lower-cased.
- Primitive `extras` go to `geometry.userData`, and mesh `extras` to `mesh.userData`.
  Node `extras` go to the node object's `userData`. When a node holds a single
  primitive, the node *is* the `Mesh`, so the node and mesh extras are merged.
  The viewer looks up `feature_ids` on the geometry first, then the mesh, then its
  ancestors.
- Layer nodes are found by `userData.layer` (`extras.layer`), falling back to the
  original glTF node name (`userData.name`). `object.name` can be sanitised or given a
  suffix.
- `EXT_mesh_gpu_instancing` becomes an `InstancedMesh`. The node's `instance_ids` are
  on the `InstancedMesh` itself, or on its parent `Group` when the mesh has several
  primitives. The viewer walks up the ancestors to find them.
