"""Conformance vectors shared by the compiler tests and the viewer's runtime generators.

``python -m worldgen.conformance [DIR]`` (default ``tests/conformance``) writes:

* ``hash64.json``: ``hashing.hash_u64`` / ``hash_unit`` cases (words as decimal strings,
  results as hex strings: JSON numbers cannot hold 64-bit integers).
* ``terrain_tiles.json``: data-first tile containers (base64) of a small synthetic world,
  one interior and one world-edge tile and the level-1 tile over all four tiles, with the exact
  vertex arrays the baked GLB terrain of that tile would have (glTF axes, relative to the tile
  corner, as float32): positions, normals, texture coordinates, colours (u8) and triangle
  indices. The level-1 tile's mesh is that of the every-other-node heightfield.
* ``tree_tiles.json``: a tile with tree records (TRE, TOB) and the ids, instance transforms and
  seeds a runtime derives from them; and the level-1 tile over it (``coarse``: thinned trees
  and flat building prisms without ids) with its instance transforms and prism heights.
* ``building_tiles.json``: a tile with building footprint records (BLD, IDS) and the volume,
  roof / floor areas and height range of each building's baked mesh.
* ``graph_core.json``: a small road graph's ``core.wgt`` (base64) and ``strings.json`` with the
  ``road_graph.json`` document they decode to.
* ``locate.json``: FNV-1a 32 hashes and buckets of entity ids (the locator), bucket counts.

Python (tests/unit/test_data_tiles.py) checks the files are current; the viewer
(viewer/test/conformance.test.ts) checks its decoders and meshers against them.
"""

from __future__ import annotations

import base64
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np

from worldgen.export.container import write_container
from worldgen.export.gltf import enu_to_gltf
from worldgen.export.heightmap import level_colors, terrain_sections
from worldgen.hashing import hash_u64, hash_unit
from worldgen.terrain.heightfield import Heightfield

HASH_CASES: list[list[int]] = [
    [0], [42], [1337, 1, 0, 0, 0], [1337, 2, -3, 5, 7, 6], [2**63 + 5, -1], [-1], [2**64 - 1, 2**32, -(2**40)],
    [1337, 5, 123456, -654321, 99, 4], [7, 7, 7, 7, 7, 7, 7, 7],
]

# Synthetic world for the terrain vectors: 2 x 2 tiles of 32 m at 4 m spacing.
TILE = 32.0
SPACING = 4.0
QUANTUM = 0.01


def hash_vectors() -> dict[str, Any]:
    cases = []
    for words in HASH_CASES:
        cases.append({"words": [str(w) for w in words], "u64": f"0x{int(hash_u64(*words)[0]):016x}",
                      "unit": float(hash_unit(*words)[0])})
    return {"schema": "worldgen.conformance.hash64", "cases": cases}


def _synthetic() -> tuple[Heightfield, np.ndarray, np.ndarray]:
    n = int(2 * TILE / SPACING) + 1
    yy, xx = np.mgrid[0:n, 0:n].astype(float)
    z = 37.0 + 4.0 * np.sin(xx * 0.7) * np.cos(yy * 0.45) + 0.37 * xx - 0.21 * yy + 0.003 * xx * yy * yy
    z = np.round(z / QUANTUM) * QUANTUM
    hf = Heightfield(0.0, 0.0, SPACING, z)
    rgba = np.zeros(z.shape + (4,), dtype=np.float32)
    rgba[..., 0] = (0.2 + 0.6 * (xx / n)).astype(np.float32)
    rgba[..., 1] = (0.3 + 0.5 * (yy / n)).astype(np.float32)
    rgba[..., 2] = np.float32(0.25)
    rgba[..., 3] = np.float32(1.0)
    codes = ((xx + 2 * yy) % 8).astype(np.uint8)
    return hf, rgba, codes


def _expected_mesh(hf: Heightfield, rgba: np.ndarray, bounds: tuple[float, float, float, float]) -> dict[str, Any]:
    """The terrain mesh attributes a baked tile GLB would contain (GltfBuilder.add_mesh)."""
    x0, y0, x1, y1 = bounds
    m = hf.tile_mesh(x0, y0, x1, y1, colors=rgba)
    pos = enu_to_gltf(m.positions - np.array([x0, y0, 0.0])).astype(np.float32)
    nrm = enu_to_gltf(m.normals).astype(np.float32)
    uv = np.column_stack([m.uvs[:, 0], 1.0 - m.uvs[:, 1]]).astype(np.float32)
    col = np.clip(np.round(m.colors * 255.0), 0, 255).astype(np.uint8)
    return {"positions": pos.ravel().tolist(), "normals": nrm.ravel().tolist(), "uvs": uv.ravel().tolist(),
            "colors": col.ravel().tolist(), "indices": m.indices.astype(np.uint32).ravel().tolist()}


def terrain_vectors() -> dict[str, Any]:
    hf, rgba, codes = _synthetic()
    tiles = []
    for i, j in ((0, 0), (1, 1)):
        bounds = (i * TILE, j * TILE, (i + 1) * TILE, (j + 1) * TILE)
        wgt = write_container(0, i, j, terrain_sections(hf, bounds, QUANTUM, rgba, codes))
        tiles.append({"level": 0, "i": i, "j": j, "size": TILE, "bounds": list(bounds),
                      "wgt_base64": base64.b64encode(wgt).decode("ascii"), "expected": _expected_mesh(hf, rgba, bounds)})
    # Level 1: every other node of the same grid, the colours of export.heightmap.level_colors.
    bounds = (0.0, 0.0, 2 * TILE, 2 * TILE)
    wgt = write_container(1, 0, 0, terrain_sections(hf, bounds, QUANTUM, rgba, codes, level=1))
    coarse = Heightfield(0.0, 0.0, 2 * SPACING, hf.z[::2, ::2])
    col = level_colors(rgba, (0, hf.rows - 1, 0, hf.cols - 1), 1, codes)
    crgba = np.concatenate([col, np.ones(col.shape[:2] + (1,))], axis=2)
    tiles.append({"level": 1, "i": 0, "j": 0, "size": 2 * TILE, "bounds": list(bounds),
                  "wgt_base64": base64.b64encode(wgt).decode("ascii"), "expected": _expected_mesh(coarse, crgba, bounds)})
    return {"schema": "worldgen.conformance.terrain_tiles", "tile_size": TILE, "spacing": SPACING,
            "quantum": QUANTUM, "note": "normals: compare within 1e-6 (central differences, float32)",
            "tiles": tiles}


TREE_SEED = 1337
TREE_CELL = 16.0


def tree_vectors() -> dict[str, Any]:
    """A tile with generated trees, ground cover and one mapped tree: its container (HGT + TRE +
    TOB) and what a runtime derives from it — ids, instance transforms, per-tree seeds."""
    from worldgen.export.container import decode_heights, read_container
    from worldgen.export.instances import FLAG_GROUND, FLAG_TOB, decode_trees, instances_from_records, tree_ids, tree_sections
    from worldgen.world.model import TreeTable

    hf, rgba, codes = _synthetic()
    i, j = 1, 1
    x0, y0 = i * TILE, j * TILE
    rng = np.random.default_rng(5)
    n = 9
    xs = x0 + rng.uniform(0.0, TILE, n)
    ys = y0 + rng.uniform(0.0, TILE, n)
    ci, cj = np.floor(xs / TREE_CELL).astype(int), np.floor(ys / TREE_CELL).astype(int)
    ids = [f"tree:gen:{a}_{b}_{k}" for k, (a, b) in enumerate(zip(ci, cj))]
    ids[3] = f"tree:gen:{ci[3]}_{cj[3]}_g7"
    ids[5] = "tree:osm:node/42"
    arch = ["broadleaf", "conifer", "palm", "grass", "broadleaf", "broadleaf", "shrub", "conifer", "broadleaf"]
    trees = TreeTable(
        ids=ids, x=xs, y=ys, ground_z=hf.sample(xs, ys), height=rng.uniform(0.4, 25.0, n),
        crown_radius=rng.uniform(0.2, 6.0, n), trunk_radius=np.where(np.isin(arch, ["grass", "shrub"]), 0.0,
                                                                     rng.uniform(0.05, 0.6, n)),
        yaw=rng.uniform(0.0, 2 * np.pi, n), archetype=arch, species=[None] * 5 + ["Ficus"] + [None] * 3,
        category=["dense_forest"] * 3 + ["grassland"] + ["urban_trees"] * 2 + ["shrubland"] + ["dense_forest"] * 2,
        confidence=rng.uniform(0.05, 0.9, n), observed=np.array([k == 5 for k in range(n)]),
        seed=np.arange(n, dtype=np.uint64), source=["procedural"] * 5 + ["osm"] + ["procedural"] * 3,
        source_id=[f"seed:{TREE_SEED}"] * 5 + ["node/42"] + [f"seed:{TREE_SEED}"] * 3,
        inferred=[["height", "crown_radius", "trunk_radius"]] * 5 + [["crown_radius"]] + [["height"]] * 3)
    idx = np.array([k for a in ("broadleaf", "conifer", "palm", "shrub", "grass") for k in range(n) if arch[k] == a])
    bounds = (x0, y0, x0 + TILE, y0 + TILE)
    sections = terrain_sections(hf, bounds, QUANTUM, rgba, codes) + tree_sections(trees, idx, bounds, TREE_CELL)
    wgt = write_container(0, i, j, sections)
    c = read_container(wgt)
    rec = decode_trees(c.sections["TRE"])
    import json as _json
    tob = _json.loads(c.sections["TOB"])
    inst = instances_from_records(rec, decode_heights(c.sections["HGT"]), x0, y0, TILE)
    seeds = []
    for r in rec:
        if r["flags"] & FLAG_TOB:
            seeds.append(None)
            continue
        salt = 6 if r["flags"] & FLAG_GROUND else 3
        cell_i, cell_j = int(np.floor(x0 / TREE_CELL)) + int(r["dx"]), int(np.floor(y0 / TREE_CELL)) + int(r["dy"])
        seeds.append(str(int(hash_u64(TREE_SEED, salt, cell_i, cell_j, int(r["k"]))[0]) & 0x7FFFFFFFFFFFFFFF))
    return {"schema": "worldgen.conformance.tree_tiles", "tile_size": TILE, "cell_size": TREE_CELL,
            "seed": TREE_SEED, "i": i, "j": j, "wgt_base64": base64.b64encode(wgt).decode("ascii"),
            "expected": {"ids": tree_ids(rec, tob, bounds, TREE_CELL),
                         "archetype_codes": [int(v) & 15 for v in rec["arch_cat"]],
                         "translation": inst["TRANSLATION"].ravel().tolist(),
                         "rotation": inst["ROTATION"].ravel().tolist(),
                         "scale": inst["SCALE"].ravel().tolist(), "seeds": seeds},
            "coarse": _coarse_vector(hf, rgba, codes, c.sections, x0, y0),
            "note": "translation: compare within 1e-4 (float32 of a float64 height lookup)"}


def _coarse_vector(hf: Heightfield, rgba: np.ndarray, codes: np.ndarray, tree_tile: dict[str, bytes],
                   x0: float, y0: float) -> dict[str, Any]:
    """The level-1 tile (0, 0) over the tree tile and two buildings of tile (0, 0)."""
    from shapely.geometry import box

    from worldgen.export.coarse import LeafContent, coarse_buildings, coarse_trees
    from worldgen.export.container import decode_heights, read_container
    from worldgen.export.footprints import COLUMNS, decode_buildings, encode_buildings
    from worldgen.export.instances import decode_trees, instances_from_records
    from worldgen.export.pyramid import coarse_heightfield

    cols: dict[str, list[Any]] = {name: [] for name, _, _ in COLUMNS}
    xy = []
    # A flat block (80 m2) and a small, tall gabled building (36 m2, 14 m), both kept at level 1.
    for poly, height, roof in ((box(6, 5, 16, 13), 9.0, 0.0), (box(20, 20, 26, 26), 14.0, 3.0)):
        base = float(hf.sample(*np.asarray(poly.exterior.coords)[:-1].T).min())
        cols["bottom"].append(base - 1.0)
        cols["top"].append(base + height)
        cols["polygons"].append(1)
        cols["material"].append(len(cols["material"]))
        cols["shape"].append(1 if roof else 0)
        cols["rings"].append(1)
        cols["eave"].append(base + height - roof)
        for name, v in (("cx", poly.centroid.x), ("cy", poly.centroid.y), ("ux", 1.0), ("uy", 0.0), ("half", 3.0)):
            cols[name].append(v)
        v = np.asarray(poly.exterior.coords)[:-1]
        cols["vertices"].append(len(v))
        xy.append(np.rint(v * 1000.0))
    leaves = LeafContent()
    leaves.add({"BLD": encode_buildings(cols, xy)}, 0.0, 0.0, TILE)
    leaves.add(tree_tile, x0, y0, TILE)
    bounds = (0.0, 0.0, 2 * TILE, 2 * TILE)
    bld = coarse_buildings(leaves, hf, 1, bounds)[0]
    tre = coarse_trees(leaves, coarse_heightfield(hf, bounds, 1), 1, bounds)[0]
    wgt = write_container(1, 0, 0, terrain_sections(hf, bounds, QUANTUM, rgba, codes, level=1) + bld + tre)
    c = read_container(wgt)
    rec = decode_trees(c.sections["TRE"])
    inst = instances_from_records(rec, decode_heights(c.sections["HGT"]), 0.0, 0.0, 2 * TILE)
    b = decode_buildings(c.sections["BLD"])
    return {"level": 1, "i": 0, "j": 0, "size": 2 * TILE, "wgt_base64": base64.b64encode(wgt).decode("ascii"),
            "expected": {"archetype_codes": [int(v) & 15 for v in rec["arch_cat"]],
                         "translation": inst["TRANSLATION"].ravel().tolist(),
                         "scale": inst["SCALE"].ravel().tolist(),
                         "bottom": b["bottom"].astype(float).tolist(), "top": b["top"].astype(float).tolist()}}


BUILDING_TILE = (3, -2, 256.0)


def _building_cases() -> list[Any]:
    """Footprints covering every roof the compiler resolves, its fallbacks, holes, multipolygons,
    raised parts and a footprint reaching past the tile."""
    from shapely import affinity
    from shapely.geometry import MultiPolygon, Polygon, box

    from worldgen.world.model import Building

    i, j, size = BUILDING_TILE
    x0, y0 = i * size, j * size

    def at(g: Any, dx: float, dy: float, rot: float = 0.0) -> Any:
        g = affinity.rotate(g, rot, origin=(0, 0)) if rot else g
        return shapely_round(affinity.translate(g, x0 + dx, y0 + dy))

    def shapely_round(g: Any) -> Any:
        import shapely
        return shapely.set_precision(g, 0.001)

    ell = Polygon([(0, 0), (20, 0), (20, 8), (8, 8), (8, 18), (0, 18)])
    cases = [
        ("flat_rotated", at(box(-6, -4, 6, 4), 30, 30, 20), {"height": 9.0, "class": "apartments"}, 31.27),
        ("flat_ell", at(ell, 60, 20), {"height": 14.5, "class": "retail"}, 30.5),
        ("flat_courtyard", at(Polygon(box(0, 0, 30, 24).exterior.coords, [box(8, 7, 22, 17).exterior.coords]), 100, 10),
         {"height": 18.0, "class": "school"}, 29.9),
        ("multipolygon", at(MultiPolygon([box(0, 0, 10, 6), box(14, 0, 22, 9)]), 150, 60), {"height": 7.0, "class": "warehouse"},
         33.0),
        ("gabled_house", at(box(-5, -3.5, 5, 3.5), 40, 90, 30), {"height": 8.0, "roof_shape": "gabled", "class": "house"},
         32.1),
        ("gabled_ell", at(ell, 70, 90, -15), {"height": 10.0, "roof_shape": "gabled", "class": "house"}, 32.4),
        ("gabled_given_height", at(Polygon([(0, 0), (12, -1), (14, 7), (6, 12), (-1, 8)]), 120, 110),
         {"height": 11.0, "roof_shape": "gabled", "roof_height": 3.5}, 31.0),
        ("skillion_shed", at(box(-4, -2.5, 4, 2.5), 160, 120, 75), {"height": 4.5, "roof_shape": "skillion", "class": "shed"},
         30.2),
        ("pyramidal_square", at(box(-6, -6, 6, 6), 200, 30, 10), {"height": 21.0, "roof_shape": "pyramidal"}, 35.5),
        ("hipped_rect", at(box(-7, -4, 7, 4), 210, 90), {"height": 9.0, "roof_shape": "hipped", "class": "detached"}, 34.0),
        ("pyramidal_nonconvex_flat", at(ell, 20, 150), {"height": 12.0, "roof_shape": "pyramidal"}, 30.0),
        ("gabled_hole_flat", at(Polygon(box(0, 0, 16, 12).exterior.coords, [box(5, 4, 11, 8).exterior.coords]), 60, 160),
         {"height": 9.0, "roof_shape": "gabled"}, 30.0),
        ("raised_part", at(box(0, 0, 12, 9), 120, 190), {"height": 20.0, "min_height": 6.0, "class": "office"}, 31.0),
        ("past_tile_edge", at(box(-14, -6, 14, 6), 250, 230, 35), {"height": 6.0, "roof_shape": "gabled"}, 30.6),
    ]
    out = []
    for name, geom, props, base in cases:
        b = Building(id=f"building:test:{name}", geometry=geom, properties=dict(props), source="test", source_id=name)
        b.base_elevation = base
        out.append(b)
    return out


def _mesh_invariants(m: Any, x0: float, y0: float) -> dict[str, float]:
    """Volume (closed, outward triangles), up- and down-facing projected areas, z range."""
    p = m.positions - (x0, y0, 0.0)
    a, b, c = p[m.indices[:, 0]], p[m.indices[:, 1]], p[m.indices[:, 2]]
    vol = float(np.einsum("ij,ij->i", a, np.cross(b, c)).sum() / 6.0)
    cz = (b[:, 0] - a[:, 0]) * (c[:, 1] - a[:, 1]) - (b[:, 1] - a[:, 1]) * (c[:, 0] - a[:, 0])
    return {"volume": vol, "up_area": float(cz[cz > 0].sum() / 2), "down_area": float(-cz[cz < 0].sum() / 2),
            "zmin": float(p[:, 2].min()), "zmax": float(p[:, 2].max())}


def building_vectors() -> dict[str, Any]:
    """A tile's ``BLD `` / ``IDS `` sections over synthetic buildings and the invariants of the
    baked meshes a runtime mesher must reproduce: volume, roof and floor areas, height range."""
    from worldgen.buildings.buildings import building_material, building_mesh, building_plan
    from worldgen.config import BuildingsConfig
    from worldgen.export.footprints import BUILDING_MATERIALS, ROOF_SHAPES, building_sections, tileset_entry

    bc = BuildingsConfig()
    i, j, size = BUILDING_TILE
    bounds = (i * size, j * size, (i + 1) * size, (j + 1) * size)
    buildings = _building_cases()
    plans = [building_plan(b, bc) for b in buildings]
    wgt = write_container(0, i, j, building_sections(buildings, plans, bounds))
    expected = []
    for b, plan in zip(buildings, plans):
        # Rounded: these sums over mesh triangles may differ in the last bits between platforms.
        inv = {k: round(v, 6) for k, v in _mesh_invariants(building_mesh(b, bc), bounds[0], bounds[1]).items()}
        expected.append({"id": b.id, "material": BUILDING_MATERIALS.index(building_material(b)),
                         "roofs": [r.shape for r in plan.roofs], "footprint_area": round(float(b.geometry.area), 6),
                         **inv})
    return {"schema": "worldgen.conformance.building_tiles", "tile_size": size, "i": i, "j": j,
            "materials": tileset_entry()["materials"], "roof_shapes": ROOF_SHAPES,
            "wgt_base64": base64.b64encode(wgt).decode("ascii"), "expected": expected,
            "note": "volume and areas: compare within 1e-4 relative; z range within 1e-3 m"}


def _graph_doc() -> dict[str, Any]:
    """A small road graph exercising every stored field: one-way, footway, bridge deck with a
    profile, unnamed and non-ASCII names, a fractional speed limit, both restriction forms."""
    def edge(eid: str, u: str, v: str, geom: list[list[float]], **kw: Any) -> dict[str, Any]:
        e = {"id": eid, "road_id": "road:" + eid.split(":", 1)[1].rsplit(":", 1)[0], "from": u, "to": v,
             "direction": "both", "class": "residential", "name": None, "bridge": False, "tunnel": False,
             "structure_implied": False, "layer": 0, "structure_id": None,
             "modes": {"motor_vehicle": True, "bicycle": True, "foot": True}, "maxspeed_kmh": None,
             "width_m": 6.5, "geometry": geom, "tiles": ["0_0"]}
        e.update(kw)
        return e

    nodes = [{"id": "n:osm:101", "x": 0.0, "y": 0.0, "z": 12.345}, {"id": "n:osm:102", "x": 120.5, "y": -3.25, "z": 13.0},
             {"id": "n:osm:103", "x": 240.125, "y": 60.0, "z": 14.5}, {"id": "n:bnd:104-105", "x": 256.0, "y": 61.75, "z": 14.6},
             {"id": "n:osm:106", "x": 120.5, "y": 90.0, "z": 9.875}]
    edges = [
        edge("e:osm:way/1:0", "n:osm:101", "n:osm:102", [[0, 0, 12.345], [60.25, -1.5, 12.7], [120.5, -3.25, 13.0]],
             name="中山路", maxspeed_kmh=48.28032),
        edge("e:osm:way/1:1", "n:osm:102", "n:osm:103", [[120.5, -3.25, 13.0], [240.125, 60.0, 14.5]],
             name="中山路", maxspeed_kmh=48.28032, direction="forward", tiles=["0_0", "1_0"]),
        edge("e:osm:way/2:0", "n:osm:103", "n:bnd:104-105", [[240.125, 60.0, 14.5], [256.0, 61.75, 14.6]],
             **{"class": "primary"}, maxspeed_kmh=60, width_m=10.25, tiles=["1_0"]),
        edge("e:osm:way/3:0", "n:osm:102", "n:osm:106",
             [[120.5, -3.25, 13.0], [120.5, 20.0, 15.25], [120.5, 70.0, 15.25], [120.5, 90.0, 9.875]],
             bridge=True, layer=1, structure_id="bridge:osm:way/3", **{"class": "tertiary"}),
        edge("e:osm:way/4:0", "n:osm:106", "n:osm:101", [[120.5, 90.0, 9.875], [0, 0, 12.345]],
             **{"class": "footway"}, modes={"motor_vehicle": False, "bicycle": False, "foot": True}, layer=-1,
             tunnel=True, structure_implied=True, width_m=2.0),
    ]
    restrictions = [
        {"id": "r:osm:relation/7", "restriction": "no_left_turn", "from_edge": "e:osm:way/1:0",
         "via_node": "n:osm:102", "to_edge": "e:osm:way/3:0", "via_edges": [], "from_way": "way/1",
         "to_way": "way/3", "source_id": "relation/7", "enforced": True},
        {"id": "r:osm:relation/8", "restriction": "only_straight_on", "from_edge": None, "via_node": None,
         "to_edge": "e:osm:way/2:0", "via_edges": ["e:osm:way/1:1"], "from_way": "way/9", "to_way": "way/2",
         "source_id": "relation/8", "enforced": False},
        # References the graph does not contain (a via node off the graph, a via way outside it).
        {"id": "r:osm:relation/9", "restriction": "no_right_turn", "from_edge": "e:osm:way/2:0",
         "via_node": "n:osm:618956032", "to_edge": None, "via_edges": [], "from_way": "way/2",
         "to_way": "way/77", "source_id": "relation/9", "enforced": False},
        {"id": "r:osm:relation/10", "restriction": "no_u_turn", "from_edge": None, "via_node": None,
         "to_edge": "e:osm:way/1:0", "via_edges": ["e:osm:way/1:1", "e:osm:way/88:0"], "from_way": "way/5",
         "to_way": "way/1", "source_id": "relation/10", "enforced": False},
    ]
    return {"schema": "worldgen.road_graph", "schema_version": "1.1.0", "nodes": nodes, "edges": edges,
            "restrictions": restrictions}


def graph_vectors() -> dict[str, Any]:
    """``graph/core.wgt`` + ``strings.json`` of a small graph and the road graph document they decode to."""
    from worldgen.export.graphcore import decode_graph, graph_files
    from worldgen.export.tileset import DATA_SCHEMA_VERSION
    from worldgen.topology.graph import RoadGraph

    g = RoadGraph.from_dict(_graph_doc())
    origin = {"lat": 25.0, "lon": 121.5, "height": 0.0}
    core, strings = graph_files(g, origin, "+proj=tmerc +lat_0=25 +lon_0=121.5", DATA_SCHEMA_VERSION)
    expected = decode_graph(core, strings)
    ref = g.to_dict(None, origin)
    ref["crs"] = expected["crs"]
    if json.loads(json.dumps(expected)) != json.loads(json.dumps(ref)):
        raise AssertionError("graph core does not round-trip the road graph document")
    return {"schema": "worldgen.conformance.graph_core", "core_base64": base64.b64encode(core).decode("ascii"),
            "strings": strings, "expected": expected}


LOCATE_IDS = ["", "a", "building:osm:way/1", "road:osm:way/1005161805", "bridge:osm:way/107", "water:osm:relation/9",
              "tree:osm:node/42", "建物:äöü"]


def locate_vectors() -> dict[str, Any]:
    from worldgen.export.locate import bucket_count, fnv1a32

    return {"schema": "worldgen.conformance.locate", "hash": "fnv1a32 over UTF-8 bytes",
            "cases": [{"id": i, "fnv1a32": f"0x{fnv1a32(i):08x}", "bucket_64": fnv1a32(i) & 63} for i in LOCATE_IDS],
            "bucket_counts": [[n, bucket_count(n)] for n in (0, 1, 4096, 4097, 100_000, 10**9)]}


def vectors() -> dict[str, dict[str, Any]]:
    return {"hash64.json": hash_vectors(), "terrain_tiles.json": terrain_vectors(), "tree_tiles.json": tree_vectors(),
            "building_tiles.json": building_vectors(), "graph_core.json": graph_vectors(),
            "locate.json": locate_vectors()}


def write(out_dir: Path) -> list[Path]:
    out_dir.mkdir(parents=True, exist_ok=True)
    written = []
    for name, doc in vectors().items():
        p = out_dir / name
        p.write_text(json.dumps(doc, indent=1) + "\n", encoding="utf-8")
        written.append(p)
    return written


if __name__ == "__main__":
    target = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("tests/conformance")
    for p in write(target):
        print(p)
