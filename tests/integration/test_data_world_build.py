"""Builds with export.mode=data: heightmap terrain equals the compiled surface, layout, validation."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from worldgen.build import BuildOptions, build, fixture_region
from worldgen.config import load_config
from worldgen.export.container import decode_heights, read_container
from worldgen.export.gltf import read_glb_json
from worldgen.export.heightmap import terrain_mesh_from_sections
from worldgen.validation.assets import validate_outputs

pytestmark = pytest.mark.slow


def _build(out: Path, mode: str = "data", **kw):  # noqa: ANN003, ANN202
    cfg = load_config("basic", overrides=[f"export.mode={mode}"], region=fixture_region("demo_town"))
    return build(cfg, BuildOptions(out, fixture="demo_town", **kw))


@pytest.fixture(scope="module")
def data_world(tmp_path_factory):  # noqa: ANN001, ANN201
    out = tmp_path_factory.mktemp("data_world")
    return out, _build(out)


def _tiles(out: Path) -> list[tuple[Path, Path]]:
    ts = json.loads((out / "tiles" / "tileset.json").read_text(encoding="utf-8"))
    res = []
    for st in ts["subtrees"]:
        sub = json.loads((out / ts["uri"]["subtree"].format(**st)).read_text(encoding="utf-8"))
        for i, j in zip(sub["levels"][0]["i"], sub["levels"][0]["j"]):
            keys = {"level": 0, "si": st["si"], "sj": st["sj"], "i": i, "j": j}
            res.append((out / ts["uri"]["content"].format(**keys), out / ts["uri"]["meta"].format(**keys)))
    return res


def test_layout_and_manifest(data_world):
    out, res = data_world
    m = res.manifest
    assert m["schema_version"].startswith("2.") and m["tiling"]["tileset"] == "tiles/tileset.json"
    assert not (out / "tiles" / "index.json").exists() and not (out / "world.glb").exists()
    tiles = _tiles(out)
    assert len(tiles) == m["tiling"]["tile_count"] and all(c.is_file() and mt.is_file() for c, mt in tiles)
    v = validate_outputs(out)
    assert v["errors"] == 0, v["lines"]


def test_heightmaps_equal_the_compiled_terrain(data_world):
    out, res = data_world
    hf = res.world.terrain
    cfg = res.world.config
    assert np.array_equal(hf.z, np.round(hf.z / 0.01) * 0.01)            # shaped terrain is on the 1 cm grid
    for content, _ in _tiles(out):
        c = read_container(content.read_bytes())
        x0, y0 = c.i * cfg.tiling.tile_size, c.j * cfg.tiling.tile_size
        mesh = terrain_mesh_from_sections(c.sections, cfg.tiling.tile_size)
        ref = hf.tile_mesh(x0, y0, x0 + cfg.tiling.tile_size, y0 + cfg.tiling.tile_size)
        assert np.array_equal(mesh.positions, ref.positions) and np.array_equal(mesh.indices, ref.indices)
        assert np.allclose(mesh.normals, ref.normals, atol=1e-12)


def test_draped_roads_sit_on_the_runtime_terrain(data_world):
    out, res = data_world
    cfg = res.world.config
    checked = 0
    for content, _ in _tiles(out):
        c = read_container(content.read_bytes())
        doc = read_glb_json(c.sections["GLB"])
        assert all(n.get("name") != "terrain" for n in doc["nodes"])         # terrain ships as data only
        g = decode_heights(c.sections["HGT"])
        z = g.z
        import struct
        blob_off = 20 + struct.unpack_from("<I", c.sections["GLB"], 12)[0] + 8
        x0, y0 = c.i * cfg.tiling.tile_size, c.j * cfg.tiling.tile_size
        for node in doc["nodes"]:
            if node.get("name") != "roads":
                continue
            for prim in doc["meshes"][node["mesh"]]["primitives"]:
                acc = doc["accessors"][prim["attributes"]["POSITION"]]
                view = doc["bufferViews"][acc["bufferView"]]
                p = np.frombuffer(c.sections["GLB"], dtype="<f4", count=3 * acc["count"],
                                  offset=blob_off + view["byteOffset"]).reshape(-1, 3).astype(float)
                ex, ey, ez = p[:, 0] + x0, -p[:, 2] + y0, p[:, 1]
                # Terrain height on the runtime triangles (same diagonal as the compiler).
                fx, fy = (ex - g.x0) / g.spacing, (ey - g.y0) / g.spacing
                cc = np.clip(np.floor(fx).astype(int), 1, z.shape[1] - 3)
                rr = np.clip(np.floor(fy).astype(int), 1, z.shape[0] - 3)
                u, v = np.clip(fx - cc, 0, 1), np.clip(fy - rr, 0, 1)
                z00, z10, z01, z11 = z[rr, cc], z[rr, cc + 1], z[rr + 1, cc], z[rr + 1, cc + 1]
                t = np.where(u >= v, z00 + u * (z10 - z00) + v * (z11 - z10), z00 + v * (z01 - z00) + u * (z11 - z01))
                lift = ez - t
                assert lift.min() > -1e-3                                         # never below the terrain
                assert np.abs(lift - np.round(lift, 2)).max() < 2e-3              # a constant surface offset
                checked += len(p)
    assert checked > 100


def test_tree_records_replace_instances_and_meta(data_world):
    from worldgen.export.instances import decode_trees, instances_from_records, tree_ids
    from worldgen.vegetation.regions import instance_transforms

    out, res = data_world
    world = res.world
    T = world.config.tiling.tile_size
    cs = world.config.vegetation.cell_size
    tr = world.trees
    pos = {t: k for k, t in enumerate(tr.ids)}
    assert len(read_glb_json((out / "tiles" / "archetypes.glb").read_bytes())["scenes"][0]["nodes"]) == 5
    total = 0
    for content, meta_path in _tiles(out):
        c = read_container(content.read_bytes())
        doc = read_glb_json(c.sections["GLB"])
        assert all(n.get("name") != "trees" for n in doc["nodes"])
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert not any(v["kind"] == "tree" for v in meta["entities"].values())
        rec = decode_trees(c.sections["TRE"])
        tob = json.loads(c.sections["TOB"]) if "TOB" in c.sections else []
        x0, y0 = c.i * T, c.j * T
        ids = tree_ids(rec, tob, (x0, y0, x0 + T, y0 + T), cs)
        k = np.array([pos[t] for t in ids], dtype=np.int64)
        inside = (tr.x >= x0) & (tr.x < x0 + T) & (tr.y >= y0) & (tr.y < y0 + T) & tr.rendered_mask()
        assert sorted(k.tolist()) == np.nonzero(inside)[0].tolist()        # exactly the tile's rendered trees
        if not len(k):
            continue
        ref = instance_transforms(tr, k, (x0, y0))
        got = instances_from_records(rec, decode_heights(c.sections["HGT"]), x0, y0, T)
        assert np.abs(got["TRANSLATION"] - ref["TRANSLATION"]).max() < 0.06    # 4 mm position, 5 mm trunk
        assert np.abs(got["SCALE"] - ref["SCALE"]).max() < 0.06               # 0.1 m crown, 1 cm height
        assert np.abs(np.abs(got["ROTATION"]) - np.abs(ref["ROTATION"])).max() < 0.02
        total += len(k)
    assert total > 1000


def test_incremental_and_mode_switch(tmp_path):
    out = tmp_path / "w"
    first = _build(out)
    assert _build(out).tiles_written == [] and first.tiles_written
    baked = _build(out, mode="baked")
    assert len(baked.tiles_written) == len(first.tiles_written)
    assert (out / "tiles" / "index.json").is_file() and not (out / "tiles" / "tileset.json").exists()
    assert not (out / "tiles" / "0").exists() and not (out / "tiles" / "archetypes.glb").exists()
    again = _build(out)
    assert len(again.tiles_written) == len(first.tiles_written)
    assert not (out / "tiles" / "index.json").exists()
    assert not [p for p in (out / "tiles").glob("*_*.glb")]                  # no baked tile files left
    assert validate_outputs(out)["errors"] == 0


def test_building_records_replace_meshes(data_world):
    from shapely.geometry.polygon import orient

    from worldgen.buildings.buildings import building_extent, building_plan, is_rendered
    from worldgen.export.footprints import ROOF_SHAPES, decode_buildings

    out, res = data_world
    world = res.world
    bc = world.config.buildings
    T = world.config.tiling.tile_size
    by_id = {b.id: b for b in world.buildings}
    ts = json.loads((out / "tiles" / "tileset.json").read_text(encoding="utf-8"))
    assert ts["buildings"]["roof_shapes"] == ROOF_SHAPES and len(ts["buildings"]["materials"]) == 5
    seen: list[str] = []
    for content, meta_path in _tiles(out):
        c = read_container(content.read_bytes())
        doc = read_glb_json(c.sections["GLB"])
        assert all(n.get("name") != "buildings" for n in doc["nodes"])
        if "BLD" not in c.sections:
            continue
        rec = decode_buildings(c.sections["BLD"])
        ids = json.loads(c.sections["IDS"])["BLD"]
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        assert all(meta["entities"][i]["kind"] == "building" for i in ids)       # records stay in meta.json
        x0, y0 = c.i * T, c.j * T
        for k, bid in enumerate(ids):
            b = by_id[bid]
            bottom, top = building_extent(b, bc)
            assert rec["bottom"][k] == np.float32(bottom) and rec["top"][k] == np.float32(top)
            plan = building_plan(b, bc)
            polys = rec.polygons(k, x0, y0)
            assert len(polys) == len(plan.roofs)
            for p, r, code in zip(polys, plan.roofs, rec["shape"][rec.poly_start[k]:rec.poly_start[k + 1]]):
                assert ROOF_SHAPES[code] == r.shape
                assert p.symmetric_difference(orient(r.poly, 1.0)).area < 1e-6    # exact footprint (1 mm grid)
        seen += ids
    assert sorted(seen) == sorted(b.id for b in world.buildings if is_rendered(b)) and seen
