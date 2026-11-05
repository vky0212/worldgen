"""Tile asset generation: per-tile GLB, collision GLB and semantic metadata."""

from __future__ import annotations

import dataclasses
from typing import Any

import numpy as np
import shapely
from shapely.strtree import STRtree

from worldgen import SCHEMA_VERSION, __version__
from worldgen.buildings.buildings import build_building_collision, build_building_pieces, building_extent, is_rendered
from worldgen.export.gltf import GltfBuilder
from worldgen.export.ortho import imagery_hash
from worldgen.geometry.mesh import LayerBuilder
from worldgen.hashing import round_floats, sha256_bytes, sha256_json
from worldgen.landcover.landcover import terrain_colors
from worldgen.landcover.water import build_water_pieces
from worldgen.tiling.grid import Piece, TileGrid, group_pieces, parse_tile_id, tile_id
from worldgen.transportation.meshing import (
    build_bridge_pieces,
    build_collision_pieces,
    build_intersections,
    build_road_pieces,
    build_tunnel_pieces,
)
from worldgen.vegetation.regions import ARCHETYPE_ORDER, archetype_meshes, instance_transforms, region_outline_pieces
from worldgen.world.world import CanonicalWorld

LAYER_ORDER = ["terrain", "roads", "bridges", "tunnels", "buildings", "water", "vegetation_regions", "trees"]

NO_TREES = np.zeros(0, dtype=np.int64)


def world_tiles(world: CanonicalWorld) -> list[str]:
    return [tile_id(i, j) for i, j in world.region.tiles()]


def _owner_cell(v: np.ndarray, size: float) -> np.ndarray:
    """Tile index k with k*size <= v < (k+1)*size, exactly as the tile bounds test decides it."""
    k = np.floor(v / size).astype(np.int64)
    # v / size can round across a tile line; settle it with the same products TileGrid.bounds uses.
    k -= (v < k * size).astype(np.int64)
    k += (v >= (k + 1) * size).astype(np.int64)
    return k


def trees_by_tile(world: CanonicalWorld) -> dict[str, np.ndarray]:
    """Indices (ascending) of the rendered trees owned by each tile.

    Built once per export: testing every tree against every tile's bounds is
    O(tiles x trees). Ownership equals ``x0 <= x < x1 and y0 <= y < y1`` on the tile bounds.
    """
    tr = world.trees
    if not len(tr):
        return {}
    size = float(world.config.tiling.tile_size)
    idx = np.nonzero(tr.rendered_mask())[0]
    i = _owner_cell(tr.x[idx], size)
    j = _owner_cell(tr.y[idx], size)
    order = np.lexsort((idx, j, i))
    idx, i, j = idx[order], i[order], j[order]
    starts = np.flatnonzero(np.r_[True, (i[1:] != i[:-1]) | (j[1:] != j[:-1])])
    ends = np.r_[starts[1:], len(idx)]
    return {tile_id(int(i[a]), int(j[a])): idx[a:b] for a, b in zip(starts, ends)}


# Configuration keys that never change a tile's files, kept out of the tile input hashes.
NOT_TILE_INPUTS: dict[str, tuple[str, ...]] = {"export": ("combined_glb_max_tiles", "chunk_tiles")}
# Keys added after format 1.0, hashed only when set away from their default, so worlds built
# with the defaults keep their tile hashes (and incremental rebuilds) across the upgrade.
LATER_KEY_DEFAULTS: dict[str, dict[str, Any]] = {"export": {"mode": "baked"}, "terrain": {"height_quantum_m": 0.0}}


def buildings_by_tile(world: CanonicalWorld) -> dict[str, list[Any]]:
    """Rendered buildings per owning tile (that of the footprint's representative point), in
    world order: the buildings ``build_building_pieces`` meshes for each tile."""
    grid = TileGrid(world.config.tiling.tile_size)
    out: dict[str, list[Any]] = {}
    for b in world.buildings:
        if is_rendered(b):
            rp = b.geometry.representative_point()
            out.setdefault(grid.tile_of(rp.x, rp.y), []).append(b)
    return out


def tile_config_hash(cfg: Any) -> str:
    """Hash of the configuration sections that can change a tile's files."""
    exclude = {s: list(keys) for s, keys in NOT_TILE_INPUTS.items()}
    data = cfg.to_dict()
    for section, defaults in LATER_KEY_DEFAULTS.items():
        exclude.setdefault(section, []).extend(k for k, v in defaults.items() if data[section][k] == v)
    return cfg.section_hash("tiling", "terrain", "roads", "bridges", "tunnels", "buildings", "water",
                            "vegetation", "smoothing", "export", exclude={k: tuple(v) for k, v in exclude.items()})


def tile_input_hash(world: CanonicalWorld, tid: str, entity_tree: tuple[STRtree, list[Any]] | None = None,
                    tree_index: dict[str, np.ndarray] | None = None, config_hash: str | None = None) -> str:
    """Hash of everything that can influence a tile's assets.

    ``entity_tree`` (``entity_strtree``), ``tree_index`` (``trees_by_tile``) and ``config_hash``
    (``tile_config_hash``) are world-level; pass them when hashing many tiles.
    """
    cfg = world.config
    grid = TileGrid(cfg.tiling.tile_size)
    x0, y0, x1, y1 = grid.bounds(tid)
    hf = world.terrain
    parts: dict[str, Any] = {
        "compiler": __version__,
        "config": config_hash if config_hash is not None else tile_config_hash(cfg),
        "tile": tid,
    }
    if cfg.export.mode == "data":
        # Data-first tiles change with their format version too (baked tiles keep their hashes).
        from worldgen.export.tileset import DATA_SCHEMA_VERSION
        parts["format"] = DATA_SCHEMA_VERSION
    if hf is not None:
        r = hf.resolution
        c0 = max(int(round((x0 - hf.xmin) / r)) - 1, 0)
        c1 = min(int(round((x1 - hf.xmin) / r)) + 1, hf.cols - 1)
        r0 = max(int(round((y0 - hf.ymin) / r)) - 1, 0)
        r1 = min(int(round((y1 - hf.ymin) / r)) + 1, hf.rows - 1)
        win = np.round(hf.z[r0:r1 + 1, c0:c1 + 1] * 1000).astype(np.int64)
        parts["terrain"] = sha256_bytes(win.tobytes())
        if world.terrain_codes is not None:
            parts["landcover"] = sha256_bytes(world.terrain_codes[r0:r1 + 1, c0:c1 + 1].tobytes())
        if world.terrain_rgba is not None:
            # Blended colours also depend on classes just outside the window.
            rgb = np.round(world.terrain_rgba[r0:r1 + 1, c0:c1 + 1] * 1000).astype(np.int32)
            parts["terrain_colors"] = sha256_bytes(rgb.tobytes())
    tree, ents = entity_tree if entity_tree is not None else entity_strtree(world)
    hits = sorted(tree.query(shapely.box(x0 - 2, y0 - 2, x1 + 2, y1 + 2)).tolist())
    ent_parts = []
    for k in hits:
        e = ents[k]
        rec = e.record()
        g = sha256_bytes(shapely.to_wkb(shapely.set_precision(e.geometry, 0.001)))
        item = [e.id, g, rec]
        if e.kind == "road" and getattr(e, "elevation", None) is not None:
            item.append(np.round(e.elevation, 2).tolist())
        if e.kind == "building":
            item.append(round(float(getattr(e, "base_elevation", 0.0) or 0.0), 2))
        ent_parts.append(item)
    parts["entities"] = sha256_json(round_floats(ent_parts, 3))
    # The tile's trees (an empty list for none; the export may hold only one chunk's trees).
    tr = world.trees
    if not len(tr):
        idx = NO_TREES
    elif tree_index is not None:
        idx = tree_index.get(tid, NO_TREES)
    else:
        idx = np.nonzero((tr.x >= x0) & (tr.x < x1) & (tr.y >= y0) & (tr.y < y1) & tr.rendered_mask())[0]
    parts["trees"] = sha256_json([[tr.ids[i], round(float(tr.x[i]), 2), round(float(tr.y[i]), 2),
                                   round(float(tr.height[i]), 2), tr.archetype[i]] for i in idx])
    # Orthophoto: imagery content + bake settings. World-level, because the colour levels are.
    im_hash = imagery_hash(world)
    if im_hash is not None:
        parts["imagery"] = im_hash
    return sha256_json(parts)


def entity_strtree(world: CanonicalWorld) -> tuple[STRtree, list[Any]]:
    """Spatial index over all entities (road elevations cover bridge and tunnel decks)."""
    ents = [e for e in world.entities() if e.geometry is not None and not e.geometry.is_empty]
    return STRtree([e.geometry for e in ents]), ents


def generate_pieces(world: CanonicalWorld, wanted: set[str], selector: PieceSelector | None = None,
                    building_meshes: bool = True) -> dict[str, list[Piece]]:
    """Mesh pieces of the ``wanted`` tiles, grouped by tile.

    With a ``selector`` the builders only see the entities that can own a piece in those tiles
    (export calls this per block of tiles); every tile gets the same pieces, in the same order.
    Without ``building_meshes`` (data-first tiles: buildings ship as footprint tables) there are
    no building pieces; building collision is still generated.
    """
    cfg = world.config
    grid = TileGrid(cfg.tiling.tile_size)
    graph: Any = world.graph
    layout = world.junctions
    bridges, tunnels, buildings = world.bridges, world.tunnels, world.buildings
    water, regions = world.water, world.vegetation_regions
    if selector is not None:
        sel = selector.select(wanted)
        graph = _GraphView(world.graph, sel["edges"])
        layout = dataclasses.replace(world.junctions, junctions=sel["junctions"])
        bridges, tunnels, buildings = sel["bridges"], sel["tunnels"], sel["buildings"]
        water, regions = sel["water"], sel["regions"]
    pieces: list[Piece] = []
    drape = world.terrain if cfg.roads.surface_mode == "drape" else None
    _, inter_pieces = build_intersections(graph, layout, grid, drape, wanted)
    pieces += inter_pieces
    pieces += build_road_pieces(graph, world.profiles, world.junctions, cfg, grid, wanted, drape)
    pieces += build_bridge_pieces(bridges, graph, world.profiles, world.terrain.sample, cfg, grid, wanted)
    pieces += build_tunnel_pieces(tunnels, graph, world.profiles, cfg, grid, wanted)
    if building_meshes:
        pieces += build_building_pieces(buildings, cfg.buildings, grid, wanted)
    pieces += build_water_pieces(water, world.water_levels, world.waterway_profiles, world.covered_waterways,
                                 cfg, grid, wanted)
    pieces += region_outline_pieces(regions, world.terrain, grid, wanted)
    if cfg.export.collision:
        pieces += build_collision_pieces(graph, world.profiles, cfg, grid, wanted)
        pieces += build_building_collision(buildings, cfg.buildings, grid, wanted)
    return group_pieces(pieces)


class _EdgeSubset(dict):
    """The edges a builder iterates; looking up any other edge falls back to the whole graph."""

    def __init__(self, items: Any, full: dict[str, Any]) -> None:
        super().__init__(items)
        self._full = full

    def __missing__(self, key: str) -> Any:
        return self._full[key]


class _GraphView:
    """A road graph whose ``edges`` iterate a subset (nodes and everything else: the whole graph)."""

    def __init__(self, graph: Any, edge_ids: list[str]) -> None:
        self._graph = graph
        self.nodes = graph.nodes
        self.edges = _EdgeSubset(((e, graph.edges[e]) for e in edge_ids), graph.edges)

    def __getattr__(self, name: str) -> Any:
        return getattr(self._graph, name)


class PieceSelector:
    """The entities that can own pieces in a set of tiles, from bounds computed once per export.

    Every piece builder assigns a piece to the tile of its anchor (a centreline segment, a
    node, a footprint's representative point), which lies inside the entity's bounding box; so
    an entity whose box misses the tiles has no piece there. Edges get a wider margin because
    bridge piers test the ground roads within ``pier_width + 8`` m of each pier. Selections keep
    the world's order, so each tile receives exactly the pieces, in the same order, as from one
    call over the whole world, while the work per block follows the block, not the world.
    """

    MARGIN = 2.0

    def __init__(self, world: CanonicalWorld) -> None:
        cfg = world.config
        self.grid = TileGrid(cfg.tiling.tile_size)
        self.edge_margin = cfg.bridges.pier_width + 8.0 + self.MARGIN
        g = world.graph
        self.edge_ids = sorted(g.edges)
        self.edge_box = _boxes([g.edges[e].line for e in self.edge_ids])
        self.junctions = world.junctions.junctions
        self.junction_tile = {nid: self.grid.tile_of(g.nodes[nid].x, g.nodes[nid].y) for nid in self.junctions}
        self.bridges, self.tunnels = world.bridges, world.tunnels
        self.bridge_box = _boxes([shapely.union_all([g.edges[e].line for e in b.edge_ids]) for b in self.bridges])
        self.tunnel_box = _boxes([shapely.union_all([g.edges[e].line for e in t.edge_ids]) for t in self.tunnels])
        self.buildings = world.buildings
        self.building_tile = [self.grid.tile_of(p.x, p.y) for p in
                              (b.geometry.representative_point() for b in self.buildings)]
        self.water = world.water
        self.water_box = _boxes([w.geometry for w in self.water])
        self.regions = world.vegetation_regions
        self.region_box = _boxes([r.geometry for r in self.regions])

    def select(self, tiles: set[str]) -> dict[str, Any]:
        bounds = np.array([self.grid.bounds(t) for t in tiles])
        b = (bounds[:, 0].min(), bounds[:, 1].min(), bounds[:, 2].max(), bounds[:, 3].max())

        def pick(items: list[Any], boxes: np.ndarray, margin: float = self.MARGIN) -> list[Any]:
            return [items[k] for k in np.nonzero(_overlaps(boxes, b, margin))[0]]

        return {
            "edges": pick(self.edge_ids, self.edge_box, self.edge_margin),
            "junctions": {nid: j for nid, j in self.junctions.items() if self.junction_tile[nid] in tiles},
            "bridges": pick(self.bridges, self.bridge_box),
            "tunnels": pick(self.tunnels, self.tunnel_box),
            "buildings": [bd for bd, t in zip(self.buildings, self.building_tile) if t in tiles],
            "water": pick(self.water, self.water_box),
            "regions": pick(self.regions, self.region_box),
        }


def _boxes(geoms: list[Any]) -> np.ndarray:
    if not geoms:
        return np.zeros((0, 4))
    return shapely.bounds(np.array(geoms, dtype=object))


def _overlaps(boxes: np.ndarray, b: tuple[float, float, float, float], margin: float) -> np.ndarray:
    return ((boxes[:, 0] <= b[2] + margin) & (boxes[:, 2] >= b[0] - margin)
            & (boxes[:, 1] <= b[3] + margin) & (boxes[:, 3] >= b[1] - margin))


def tile_chunks(tiles: list[str], chunk: int, tile_size: float
                ) -> list[tuple[tuple[float, float, float, float] | None, list[str]]]:
    """Tiles grouped into ``chunk`` x ``chunk`` squares of the tile lattice, with each square's
    bounds, squares in (j, i) order and tiles in their given order (``chunk`` <= 0: one chunk
    covering everything, bounds None)."""
    if chunk <= 0:
        return [(None, list(tiles))]
    span = chunk * tile_size
    groups: dict[tuple[int, int], list[str]] = {}
    for t in tiles:
        i, j = parse_tile_id(t)
        groups.setdefault((i // chunk, j // chunk), []).append(t)
    return [((ci * span, cj * span, (ci + 1) * span, (cj + 1) * span), groups[(ci, cj)])
            for ci, cj in sorted(groups, key=lambda ij: (ij[1], ij[0]))]


def tile_blocks(tiles: list[str], block: int) -> list[list[str]]:
    """``tiles`` grouped into square blocks of ``block`` x ``block`` tiles (order kept within a block)."""
    groups: dict[tuple[int, int], list[str]] = {}
    for t in tiles:
        i, j = parse_tile_id(t)
        groups.setdefault((i // block, j // block), []).append(t)
    return [groups[k] for k in sorted(groups, key=lambda ij: (ij[1], ij[0]))]


def build_tile_assets(world: CanonicalWorld, tid: str, pieces: list[Piece], terrain_normals: np.ndarray | None,
                      terrain_rgba: np.ndarray, arche: dict[str, dict[str, Any]], *,
                      trees_in_tile: np.ndarray | None = None,
                      entity_index: dict[str, Any] | None = None, terrain_mesh: bool = True,
                      trees_as_data: bool = False, buildings_as_data: list[Any] | None = None) -> dict[str, Any]:
    """Return {"glb": bytes, "collision": bytes|None, "meta": dict, "counts": dict, "zrange": [..]}.

    ``terrain_normals`` may be None: the terrain meshes then compute normals for the tile's own
    window (same values). ``trees_in_tile`` (this tile's entry of ``trees_by_tile``) and
    ``entity_index`` (``world.entity_index()``) are computed once per export by the caller;
    without them they are derived here, which costs O(trees + entities) per tile. Without
    ``terrain_mesh`` (data-first tiles, whose terrain ships as a heightmap) the GLB has no
    terrain layer; the terrain record and height range are still produced. With
    ``trees_as_data`` the GLB has no tree instances and meta.json no tree records: the trees
    ship as a ``TRE `` section built from ``tree_idx`` (rendered trees, in TRE order).
    ``buildings_as_data`` (the tile's ``buildings_by_tile`` entry) are buildings shipped as a
    ``BLD `` section instead of meshes (``pieces`` then hold none): their records stay in
    meta.json, their counts and height range in the tile's index entry.
    """
    cfg = world.config
    grid = TileGrid(cfg.tiling.tile_size)
    x0, y0, x1, y1 = grid.bounds(tid)
    origin = (x0, y0, 0.0)
    hf = world.terrain
    layers: dict[str, LayerBuilder] = {name: LayerBuilder(name) for name in LAYER_ORDER}
    collision = LayerBuilder("collision")
    # Terrain.
    terrain_id = f"terrain:{tid}"
    zmin, zmax = np.inf, -np.inf
    if terrain_mesh:
        tmesh = hf.tile_mesh(x0, y0, x1, y1, colors=terrain_rgba, normals=terrain_normals)
        layers["terrain"].add(tmesh, "terrain", terrain_id)
    else:
        win = hf.z[_node_window(hf, (x0, y0, x1, y1))]
        zmin, zmax = float(win.min()), float(win.max())
    if cfg.export.collision:
        collision.add(hf.tile_mesh(x0, y0, x1, y1, stride=2, normals=terrain_normals), "collision", terrain_id)
    if buildings_as_data:
        for b in buildings_as_data:
            layers["buildings"].feature_index(b.id)
            bottom, top = building_extent(b, cfg.buildings)
            zmin, zmax = min(zmin, bottom), max(zmax, top)
    for p in pieces:
        if p.layer == "collision":
            collision.add(p.mesh, p.material, p.feature_id)
        else:
            layers[p.layer].add(p.mesh, p.material, p.feature_id)

    gb = GltfBuilder()
    children = []
    for name in LAYER_ORDER:
        lb = layers[name]
        if name == "trees" or lb.is_empty():
            continue
        prims = lb.build()
        for m in prims.values():
            zmin = min(zmin, float(m.positions[:, 2].min()))
            zmax = max(zmax, float(m.positions[:, 2].max()))
        mi = gb.add_mesh(name, prims, origin, extras={"layer": name, "feature_ids": lb.features})
        if mi is not None:
            children.append(gb.add_node(name, mesh=mi, extras={"layer": name}))
    # Trees: instanced archetypes.
    tr = world.trees
    tree_idx: list[np.ndarray] = []
    n_trees = 0
    if len(tr):
        if trees_in_tile is None:
            trees_in_tile = np.nonzero((tr.x >= x0) & (tr.x < x1) & (tr.y >= y0) & (tr.y < y1)
                                       & tr.rendered_mask())[0]
        tile_arch = np.array([tr.archetype[k] for k in trees_in_tile], dtype=object)
        tree_children = []
        for arch in ARCHETYPE_ORDER:
            idx = trees_in_tile[tile_arch == arch]
            if len(idx) == 0:
                continue
            tree_idx.append(idx)
            n_trees += len(idx)
            if not trees_as_data:
                ids = [tr.ids[i] for i in idx]
                mi = gb.add_mesh(f"tree_{arch}", arche[arch], (0.0, 0.0, 0.0), extras={"archetype": arch})
                ext = gb.instancing(instance_transforms(tr, idx, (x0, y0)))
                tree_children.append(gb.add_node(arch, mesh=mi, extras={"layer": "trees", "archetype": arch,
                                                                         "instance_ids": ids}, extensions=ext))
            top = tr.ground_z[idx] + tr.height[idx]
            zmax = max(zmax, float(top.max()))
        if tree_children:
            children.append(gb.add_node("trees", children=tree_children, extras={"layer": "trees"}))
    gb.add_node(f"tile_{tid}", children=children, translation=(x0, 0.0, -y0),
                extras={"tile": tid, "bounds": [x0, y0, x1, y1], "schema_version": SCHEMA_VERSION}, root=True)
    glb = gb.to_glb()

    col_bytes = None
    if cfg.export.collision and not collision.is_empty():
        cb = GltfBuilder()
        mi = cb.add_mesh("collision", collision.build(), origin, extras={"layer": "collision",
                                                                         "feature_ids": collision.features})
        node = cb.add_node("collision", mesh=mi, extras={"layer": "collision"})
        cb.add_node(f"tile_{tid}_collision", children=[node], translation=(x0, 0.0, -y0), root=True)
        col_bytes = cb.to_glb()

    all_trees = np.concatenate(tree_idx) if tree_idx else NO_TREES
    meta = tile_metadata(world, tid, layers, NO_TREES if trees_as_data else all_trees, entity_index)
    counts = {name: len(layers[name].features) for name in LAYER_ORDER if name != "trees"}
    counts["trees"] = n_trees
    if not terrain_mesh:
        # The runtime terrain mesh carries this feature id; its record stays in the tile's meta.
        meta["entities"] = dict(sorted({**meta["entities"], terrain_id: terrain_record(world, tid)}.items()))
        counts["terrain"] = 1
    return {"glb": glb, "collision": col_bytes, "meta": meta, "counts": counts, "tree_idx": all_trees,
            "zrange": [round(zmin, 3), round(zmax, 3)] if np.isfinite(zmin) else None}


def _node_window(hf: Any, bounds: tuple[float, float, float, float]) -> tuple[slice, slice]:
    """Grid nodes of a tile (shared edges included), as ``Heightfield.tile_mesh`` picks them."""
    x0, y0, x1, y1 = bounds
    r = hf.resolution
    c0, c1 = max(int(round((x0 - hf.xmin) / r)), 0), min(int(round((x1 - hf.xmin) / r)), hf.cols - 1)
    r0, r1 = max(int(round((y0 - hf.ymin) / r)), 0), min(int(round((y1 - hf.ymin) / r)), hf.rows - 1)
    return slice(r0, r1 + 1), slice(c0, c1 + 1)


def tile_metadata(world: CanonicalWorld, tid: str, layers: dict[str, LayerBuilder], tree_idx: np.ndarray,
                  entity_index: dict[str, Any] | None = None) -> dict[str, Any]:
    """Semantic records of a tile's features; ``tree_idx`` are row indices into ``world.trees``."""
    index = entity_index if entity_index is not None else world.entity_index()
    ents: dict[str, Any] = {}
    hf = world.terrain
    for name, lb in layers.items():
        for fid in lb.features:
            if fid in ents:
                continue
            if fid.startswith("terrain:"):
                ents[fid] = terrain_record(world, tid)
                continue
            e = index.get(fid)
            if e is None:
                continue
            rec = e.record()
            rec["layer"] = name
            c = e.geometry.representative_point() if e.geometry.geom_type.endswith("Polygon") else e.geometry.centroid
            z = float(hf.sample(c.x, c.y)) if hf is not None else 0.0
            rec["centroid"] = [round(c.x, 2), round(c.y, 2), round(z, 2)]
            ents[fid] = rec
    tr = world.trees
    for k in tree_idx.tolist():
        rec = tr.record(k)
        rec["centroid"] = [round(float(tr.x[k]), 2), round(float(tr.y[k]), 2), round(float(tr.ground_z[k]), 2)]
        ents[tr.ids[k]] = rec
    return {"schema": "worldgen.tile_meta", "schema_version": SCHEMA_VERSION, "tile": tid,
            "entities": dict(sorted(ents.items()))}


def terrain_record(world: CanonicalWorld, tid: str) -> dict[str, Any]:
    hf = world.terrain
    grid = TileGrid(world.config.tiling.tile_size)
    x0, y0, x1, y1 = grid.bounds(tid)
    dem = next((s for s in world.sources if s.name == "dem"), None)
    r = hf.resolution
    c0, c1 = int(round((x0 - hf.xmin) / r)), int(round((x1 - hf.xmin) / r))
    r0, r1 = int(round((y0 - hf.ymin) / r)), int(round((y1 - hf.ymin) / r))
    win = hf.z[r0:r1 + 1, c0:c1 + 1]
    valid = hf.valid[r0:r1 + 1, c0:c1 + 1] if hf.valid is not None else None
    props = {
        "resolution_m": r, "vertical_datum": hf.vertical_datum,
        "elevation_min": round(float(win.min()), 2), "elevation_max": round(float(win.max()), 2),
        "source_coverage": round(float(valid.mean()), 3) if valid is not None else None,
    }
    inferred = {}
    if hf.meta.get("dsm_correction"):
        inferred["bare_earth"] = {"value": True, "method": "dsm_morphological_opening", "confidence": 0.5}
    return {
        "kind": "terrain", "layer": "terrain", "source": dem.provider if dem else "none",
        "source_id": dem.dataset_version or "" if dem else "", "confidence": 0.7 if dem and dem.status == "ok" else 0.3,
        "properties": props, "inferred": inferred, "missing": [] if valid is None or valid.all() else ["elevation (partial)"],
        "uncertain": bool(valid is not None and not valid.all()),
        "centroid": [round((x0 + x1) / 2, 2), round((y0 + y1) / 2, 2), round(float(win.mean()), 2)],
    }


def terrain_vertex_colors(world: CanonicalWorld) -> np.ndarray:
    """RGBA per terrain node (smoothed colours, else the land-cover class colours).

    Normals are not precomputed for the whole grid: each tile's terrain mesh derives them from
    its own window (``Heightfield.normals_window``), which gives the same values.
    """
    if world.terrain_rgba is not None:
        return world.terrain_rgba
    hf = world.terrain
    codes = world.terrain_codes if world.terrain_codes is not None else np.zeros(hf.z.shape, dtype=np.uint8)
    return terrain_colors(codes)


def archetypes_for(world: CanonicalWorld) -> dict[str, dict[str, Any]]:
    return archetype_meshes(world.config.vegetation.quality)
