# WorldGen viewer

Interactive Three.js viewer for worlds compiled by WorldGen. It reads the output
contract in [`../docs/output-format.md`](../docs/output-format.md): `manifest.json`,
`tiles/index.json`, the tile `.glb` + `.meta.json` files,
`graph/road_graph.json` and `diagnostics.json`.

Stack: TypeScript, Three.js, Vite, proj4. There is no UI framework, only plain DOM and CSS.

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
with a hint, and the viewer shows a "could not load the world manifest" message.
`npm run preview` (which serves the production build) uses the same middleware.

## Build, typecheck, test

```sh
npm run build      # tsc --noEmit (app + node configs) && vite build -> dist/
npm run typecheck  # type-check only
npm test           # routing unit tests (node --test)
```

`dist/` is built with `base: './'`, so it works from any path. The intended setup is the
viewer at `/` with the compiled world at `/world/` (for example `worldgen serve`).

## URL parameters

| Parameter | Meaning |
|---|---|
| `world` | Base URL of the compiled world directory. It can be absolute or relative to the page. Default: `world/` relative to the page, which is `/world/` when the viewer is served at `/`. A trailing `/` is added if missing. Example: `?world=https://example.org/worlds/taipei/` |

## Controls

| Input | Action |
|---|---|
| **Orbit/map mode** (default) | Left-drag pans, right-drag rotates, the wheel zooms towards the cursor |
| **Fly mode** | W/A/S/D (or the arrow keys) move, Q/E move down/up, Shift moves 5× faster, drag looks around, the wheel changes speed |
| `F` | Switches between orbit and fly mode (the button in **View** does the same) |
| Click | Inspects the clicked feature, graph edge or node, or validation marker |
| `Esc` | Stops route picking, or clears the selection |

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
| `src/ui.ts` | DOM helpers, info panel, status bar, toasts, progress bar |
| `test/routing.test.ts` | Routing tests on a hand-made mini graph (one-way edges, `no_left_turn`, mandatory turns, modes, U-turns) |
| `vite.config.ts` | Vite config and the `/world/*` dev/preview middleware |

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
