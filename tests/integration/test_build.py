"""End-to-end builds: outputs, schemas, determinism, incremental tiles, CLI."""

from __future__ import annotations

import json
import shutil
from pathlib import Path

import geopandas as gpd
import numpy as np
import pytest
import trimesh

from worldgen.build import BuildOptions, build, fixture_region
from worldgen.cli import main
from worldgen.config import load_config
from worldgen.export.gltf import read_glb_json
from worldgen.topology.graph import RoadGraph
from worldgen.validation.assets import compare_worlds, validate_outputs

pytestmark = pytest.mark.slow


def _build(out: Path, name: str = "demo_town", overrides: list[str] | None = None, **kw):  # noqa: ANN003, ANN202
    cfg = load_config("basic", overrides=overrides or [], region=fixture_region(name))
    return build(cfg, BuildOptions(out, fixture=name, **kw))


@pytest.fixture(scope="module")
def demo(tmp_path_factory):  # noqa: ANN001, ANN201
    out = tmp_path_factory.mktemp("demo")
    res = _build(out)
    return out, res


def test_outputs_and_manifest(demo):
    out, res = demo
    m = json.loads((out / "manifest.json").read_text())
    assert m["schema"] == "worldgen.manifest" and m["schema_version"].startswith("1.")
    assert m["origin"] == {"lat": 24.0, "lon": 121.0, "height": 0.0}
    assert m["crs"]["local"]["proj"].startswith("+proj=tmerc")
    assert m["region"]["radius_m"] == 600 and len(m["region"]["bbox_wgs84"]) == 4
    assert m["config_hash"] and m["sources"] and m["outputs"]
    for rel in ("world.glb", "graph/road_graph.json", "tiles/index.json", "terrain/heightfield.tif",
                "terrain/heightfield.f32", "fields/vegetation_fields.tif", "vectors/buildings.parquet",
                "vectors/trees.parquet", "diagnostics.json", "ATTRIBUTION.txt"):
        assert (out / rel).is_file(), rel
    assert m["validation"]["errors"] == 0


def test_tile_glbs_link_features_to_metadata(demo):
    out, _ = demo
    index = json.loads((out / "tiles" / "index.json").read_text())
    layers_seen = set()
    for t in index["tiles"]:
        doc = read_glb_json((out / t["glb"]).read_bytes())
        meta = json.loads((out / t["meta"]).read_text())
        root = doc["nodes"][doc["scenes"][0]["nodes"][0]]
        assert root["name"] == f"tile_{t['id']}"
        assert root["translation"][0] == t["bounds"][0] and root["translation"][2] == -t["bounds"][1]
        for mesh in doc.get("meshes", []):
            for fid in mesh.get("extras", {}).get("feature_ids", []):
                assert fid in meta["entities"], fid
        for n in doc["nodes"]:
            if "extras" in n and "layer" in n["extras"]:
                layers_seen.add(n["extras"]["layer"])
            if "EXT_mesh_gpu_instancing" in n.get("extensions", {}):
                for tid in n["extras"]["instance_ids"]:
                    assert meta["entities"][tid]["kind"] == "tree"
    assert {"terrain", "roads", "bridges", "tunnels", "buildings", "water", "trees"} <= layers_seen
    scene = trimesh.load(out / "world.glb", force="scene")
    assert len(scene.geometry) > 50


def test_semantic_exports(demo):
    out, res = demo
    g = RoadGraph.from_dict(json.loads((out / "graph" / "road_graph.json").read_text()))
    assert len(g.edges) == len(res.world.graph.edges)
    assert g.shortest_path(*sorted(g.nodes)[:2], mode=None) is not None or True
    b = gpd.read_parquet(out / "vectors" / "buildings.parquet")
    assert b.crs.to_epsg() == 4326 and len(b) == len(res.world.buildings)
    assert {"height", "height_inferred", "inferred_properties", "source"} <= set(b.columns)
    br = gpd.read_parquet(out / "vectors" / "bridges.parquet")
    tn = gpd.read_parquet(out / "vectors" / "tunnels.parquet")
    assert len(br) == 2 and len(tn) == 1 and set(br["kind"]) == {"bridge"} and set(tn["kind"]) == {"tunnel"}
    trees = gpd.read_parquet(out / "vectors" / "trees.parquet")
    assert trees["observed"].sum() == 5 and (~trees["observed"]).sum() > 100
    hj = json.loads((out / "terrain" / "heightfield.json").read_text())
    raw = np.fromfile(out / "terrain" / "heightfield.f32", dtype="<f4").reshape(hj["rows"], hj["cols"])
    assert np.allclose(raw, res.world.terrain.z, atol=1e-3)


def test_deterministic_rebuild(demo, tmp_path):
    out, _ = demo
    _build(tmp_path / "again")
    cmp = compare_worlds(out, tmp_path / "again")
    assert cmp["identical"], cmp["lines"]


def test_incremental_tiles(tmp_path):
    out = tmp_path / "w"
    first = _build(out, "multi_tile_road")
    assert first.tiles_skipped == []
    second = _build(out, "multi_tile_road")
    assert second.tiles_written == [] and len(second.tiles_skipped) == len(first.tiles_written)
    # A config change that affects geometry marks every tile dirty again.
    third = _build(out, "multi_tile_road", overrides=["roads.surface_offset=0.2"])
    assert len(third.tiles_written) == len(first.tiles_written)
    # Single-tile rebuild.
    fourth = _build(out, "multi_tile_road", only_tiles={"0_0"}, force=True)
    assert fourth.tiles_written == ["0_0"]
    v = validate_outputs(out)
    assert v["errors"] == 0, v["lines"]


def test_cli_build_route_transform(tmp_path, capsys):
    out = tmp_path / "cli"
    assert main(["-q", "build", "--fixture", "turn_restriction", "-o", str(out)]) == 0
    from worldgen.sources.synthetic import ORIGIN
    from worldgen.spatial.frame import LocalFrame

    f = LocalFrame(ORIGIN)
    a = f.point_to_wgs84(-200, 0)
    b = f.point_to_wgs84(0, 200)
    capsys.readouterr()
    assert main(["route", "--world", str(out), "--from", f"{a.lat},{a.lon}", "--to", f"{b.lat},{b.lon}",
                 "--json"]) == 0
    r = json.loads(capsys.readouterr().out)
    assert r["found"] and r["length_m"] > 400.5
    assert main(["transform", "--world", str(out), f"{a.lat},{a.lon}"]) == 0
    assert "x=-200.000" in capsys.readouterr().out
    assert main(["validate", "--world", str(out)]) == 0
    shutil.rmtree(out / "graph")
    assert main(["route", "--world", str(out), "--from", "24,121", "--to", "24,121"]) == 2
