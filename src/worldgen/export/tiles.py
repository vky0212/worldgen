"""Tile asset generation: per-tile GLB, collision GLB and semantic metadata."""

from __future__ import annotations

from typing import Any

import numpy as np
import shapely
from shapely.strtree import STRtree

from worldgen import SCHEMA_VERSION, __version__
from worldgen.buildings.buildings import build_building_collision, build_building_pieces
from worldgen.export.gltf import GltfBuilder
from worldgen.geometry.mesh import LayerBuilder
from worldgen.hashing import round_floats, sha256_bytes, sha256_json
from worldgen.landcover.landcover import terrain_colors
from worldgen.landcover.water import build_water_pieces
from worldgen.tiling.grid import Piece, TileGrid, group_pieces, tile_id
from worldgen.transportation.meshing import (
    build_bridge_pieces,
    build_collision_pieces,
    build_intersections,
    build_road_pieces,
    build_tunnel_pieces,
    intersection_nodes,
)
from worldgen.vegetation.regions import ARCHETYPE_ORDER, archetype_meshes, instance_transforms, region_outline_pieces
from worldgen.world.world import CanonicalWorld

LAYER_ORDER = ["terrain", "roads", "bridges", "tunnels", "buildings", "water", "vegetation_regions", "trees"]


def world_tiles(world: CanonicalWorld) -> list[str]:
    return [tile_id(i, j) for i, j in world.region.tiles()]


def tile_input_hash(world: CanonicalWorld, tid: str, entity_tree: tuple[STRtree, list[Any]] | None = None) -> str:
    """Hash of everything that can influence a tile's assets."""
    cfg = world.config
    grid = TileGrid(cfg.tiling.tile_size)
    x0, y0, x1, y1 = grid.bounds(tid)
    hf = world.terrain
    parts: dict[str, Any] = {
        "compiler": __version__,
        "config": cfg.section_hash("tiling", "terrain", "roads", "bridges", "tunnels", "buildings", "water",
                                   "vegetation", "export"),
        "tile": tid,
    }
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
    tr = world.trees
    if len(tr):
        m = (tr.x >= x0) & (tr.x < x1) & (tr.y >= y0) & (tr.y < y1)
        idx = np.nonzero(m)[0]
        parts["trees"] = sha256_json([[tr.ids[i], round(float(tr.x[i]), 2), round(float(tr.y[i]), 2),
                                       round(float(tr.height[i]), 2), tr.archetype[i]] for i in idx])
    return sha256_json(parts)


def entity_strtree(world: CanonicalWorld) -> tuple[STRtree, list[Any]]:
    """Spatial index over all entities (road elevations cover bridge and tunnel decks)."""
    ents = [e for e in world.entities() if e.geometry is not None and not e.geometry.is_empty]
    return STRtree([e.geometry for e in ents]), ents


def generate_pieces(world: CanonicalWorld, wanted: set[str]) -> dict[str, list[Piece]]:
    cfg = world.config
    grid = TileGrid(cfg.tiling.tile_size)
    graph = world.graph
    pieces: list[Piece] = []
    patches = intersection_nodes(graph, world.profiles) if cfg.roads.intersections else set()
    _, inter_pieces = build_intersections(graph, patches, cfg, grid)
    pieces += [p for p in inter_pieces if p.tile in wanted]
    pieces += build_road_pieces(graph, world.profiles, patches, cfg, grid, wanted)
    pieces += build_bridge_pieces(world.bridges, graph, world.profiles, world.terrain.sample, cfg, grid, wanted)
    pieces += build_tunnel_pieces(world.tunnels, graph, world.profiles, cfg, grid, wanted)
    pieces += build_building_pieces(world.buildings, cfg.buildings, grid, wanted)
    pieces += build_water_pieces(world.water, world.water_levels, world.waterway_profiles, world.covered_waterways,
                                 cfg, grid, wanted)
    pieces += region_outline_pieces(world.vegetation_regions, world.terrain, grid, wanted)
    if cfg.export.collision:
        pieces += build_collision_pieces(graph, world.profiles, cfg, grid, wanted)
        pieces += build_building_collision(world.buildings, cfg.buildings, grid, wanted)
    return group_pieces(pieces)


def build_tile_assets(world: CanonicalWorld, tid: str, pieces: list[Piece], terrain_normals: np.ndarray,
                      terrain_rgba: np.ndarray, arche: dict[str, dict[str, Any]]) -> dict[str, Any]:
    """Return {"glb": bytes, "collision": bytes|None, "meta": dict, "counts": dict, "zrange": [..]}."""
    cfg = world.config
    grid = TileGrid(cfg.tiling.tile_size)
    x0, y0, x1, y1 = grid.bounds(tid)
    origin = (x0, y0, 0.0)
    hf = world.terrain
    layers: dict[str, LayerBuilder] = {name: LayerBuilder(name) for name in LAYER_ORDER}
    collision = LayerBuilder("collision")
    # Terrain.
    tmesh = hf.tile_mesh(x0, y0, x1, y1, colors=terrain_rgba, normals=terrain_normals)
    terrain_id = f"terrain:{tid}"
    layers["terrain"].add(tmesh, "terrain", terrain_id)
    if cfg.export.collision:
        collision.add(hf.tile_mesh(x0, y0, x1, y1, stride=2, normals=terrain_normals), "collision", terrain_id)
    for p in pieces:
        if p.layer == "collision":
            collision.add(p.mesh, p.material, p.feature_id)
        else:
            layers[p.layer].add(p.mesh, p.material, p.feature_id)

    gb = GltfBuilder()
    children = []
    zmin, zmax = np.inf, -np.inf
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
    tree_ids: list[str] = []
    n_trees = 0
    if len(tr):
        inside = (tr.x >= x0) & (tr.x < x1) & (tr.y >= y0) & (tr.y < y1)
        tree_children = []
        for arch in ARCHETYPE_ORDER:
            idx = np.nonzero(inside & (np.array(tr.archetype, dtype=object) == arch))[0]
            if len(idx) == 0:
                continue
            ids = [tr.ids[i] for i in idx]
            tree_ids += ids
            n_trees += len(idx)
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

    meta = tile_metadata(world, tid, layers, tree_ids)
    counts = {name: len(layers[name].features) for name in LAYER_ORDER if name != "trees"}
    counts["trees"] = n_trees
    return {"glb": glb, "collision": col_bytes, "meta": meta, "counts": counts,
            "zrange": [round(zmin, 3), round(zmax, 3)] if np.isfinite(zmin) else None}


def tile_metadata(world: CanonicalWorld, tid: str, layers: dict[str, LayerBuilder], tree_ids: list[str]
                  ) -> dict[str, Any]:
    index = world.entity_index()
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
    if tree_ids:
        pos = {tid_: k for k, tid_ in enumerate(tr.ids)}
        for t_id in tree_ids:
            k = pos[t_id]
            rec = tr.record(k)
            rec["centroid"] = [round(float(tr.x[k]), 2), round(float(tr.y[k]), 2), round(float(tr.ground_z[k]), 2)]
            ents[t_id] = rec
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


def terrain_render_inputs(world: CanonicalWorld) -> tuple[np.ndarray, np.ndarray]:
    hf = world.terrain
    normals = hf.vertex_normals()
    codes = world.terrain_codes if world.terrain_codes is not None else np.zeros(hf.z.shape, dtype=np.uint8)
    return normals, terrain_colors(codes)


def archetypes_for(world: CanonicalWorld) -> dict[str, dict[str, Any]]:
    return archetype_meshes(world.config.vegetation.quality)
