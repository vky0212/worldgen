"""Validation of compiled world *files* (no source data needed) and world comparison."""

from __future__ import annotations

import io
import json
from pathlib import Path
from typing import Any

import numpy as np
import trimesh

from worldgen.export.container import decode_heights, read_container
from worldgen.export.gltf import read_glb_json
from worldgen.hashing import sha256_file
from worldgen.topology.graph import RoadGraph


def validate_outputs(world: Path) -> dict[str, Any]:
    lines: list[str] = []
    errors = 0
    warnings = 0
    manifest_path = world / "manifest.json"
    if not manifest_path.is_file():
        return {"errors": 1, "lines": [f"ERROR no manifest.json in {world}"]}
    m = json.loads(manifest_path.read_text(encoding="utf-8"))
    # 1. Output integrity (detects modified or missing files).
    changed, missing = [], []
    for rel, info in m.get("outputs", {}).items():
        p = world / rel
        if not p.is_file():
            missing.append(rel)
        elif sha256_file(p) != info["sha256"]:
            changed.append(rel)
    if missing or changed:
        errors += 1
        lines.append(f"ERROR outputs differ from manifest: {len(changed)} changed, {len(missing)} missing "
                     f"(e.g. {(changed + missing)[:3]})")
    else:
        lines.append(f"ok    {len(m.get('outputs', {}))} output files match manifest hashes")
    # 2. Graph schema and connectivity.
    g = RoadGraph.from_dict(json.loads((world / "graph" / "road_graph.json").read_text(encoding="utf-8")))
    comps = g.components("motor_vehicle")
    lines.append(f"ok    road graph: {len(g.nodes)} nodes, {len(g.edges)} edges, {len(comps)} drivable components")
    dangling = [e.id for e in g.edges.values() if e.u not in g.nodes or e.v not in g.nodes]
    if dangling:
        errors += 1
        lines.append(f"ERROR {len(dangling)} edges reference missing nodes")
    for e in g.edges.values():
        if e.z is not None and len(e.z) and not np.isfinite(e.z).all():
            errors += 1
            lines.append(f"ERROR non-finite elevations on {e.id}")
            break
    # 3. Tile meshes (baked GLBs, or the GLB section and heightmaps of data-first tiles).
    tiles = _tile_files(world, m)
    degenerate = bad_normals = 0
    feature_errors = 0
    heights: dict[tuple[int, int], np.ndarray] = {}
    for t in tiles:
        data = (world / t["content"]).read_bytes()
        if t["data"]:
            c = read_container(data)
            data = c.sections.get("GLB", b"")
            heights[(c.i, c.j)] = decode_heights(c.sections["HGT"]).z[1:-1, 1:-1]
            if not np.isfinite(heights[(c.i, c.j)]).all():
                errors += 1
                lines.append(f"ERROR non-finite terrain heights in {t['content']}")
        meta = json.loads((world / t["meta"]).read_text(encoding="utf-8"))
        if not data:
            continue
        doc = read_glb_json(data)
        for mesh in doc.get("meshes", []):
            for fid in mesh.get("extras", {}).get("feature_ids", []):
                if fid not in meta["entities"]:
                    feature_errors += 1
        scene = trimesh.load(io.BytesIO(data), file_type="glb", force="scene", process=False)
        for name, geom in scene.geometry.items():
            if not isinstance(geom, trimesh.Trimesh) or len(geom.faces) == 0:
                continue
            areas = geom.area_faces
            degenerate += int((areas < 1e-9).sum())
            vn = geom.vertex_normals
            bad_normals += int((~np.isfinite(vn)).any(axis=1).sum())
    if feature_errors:
        errors += 1
        lines.append(f"ERROR {feature_errors} feature ids without metadata")
    else:
        lines.append(f"ok    {len(tiles)} tiles: every feature id has metadata")
    if heights:
        seams = sum(1 for (i, j), z in heights.items() for (di, dj) in ((1, 0), (0, 1))
                    if (i + di, j + dj) in heights and not np.array_equal(
                        z[:, -1] if di else z[-1, :], heights[(i + di, j + dj)][:, 0] if di
                        else heights[(i + di, j + dj)][0, :]))
        if seams:
            errors += 1
            lines.append(f"ERROR {seams} heightmap tile seams with different edge heights")
        else:
            lines.append(f"ok    heightmaps: {len(heights)} tiles, shared edges identical")
    if bad_normals:
        errors += 1
        lines.append(f"ERROR {bad_normals} vertices with invalid normals")
    if degenerate:
        warnings += 1
        lines.append(f"warn  {degenerate} degenerate triangles across tiles")
    else:
        lines.append("ok    no degenerate triangles")
    # 4. Build-time diagnostics summary.
    d = json.loads((world / "diagnostics.json").read_text(encoding="utf-8"))
    s = d["summary"]
    lines.append(f"info  build diagnostics: {s['errors']} errors, {s['warnings']} warnings, {s['info']} info")
    for name, c in sorted(s["checks"].items()):
        lines.append(f"      {c['status']:<5} {name} ({c['issues']})")
    return {"errors": errors, "warnings": warnings, "lines": lines}


def _tile_files(world: Path, manifest: dict[str, Any]) -> list[dict[str, Any]]:
    """Every tile's content and meta paths, for either export mode."""
    tiling = manifest.get("tiling", {})
    if tiling.get("tileset"):
        tileset = json.loads((world / tiling["tileset"]).read_text(encoding="utf-8"))
        uri = tileset["uri"]
        out = []
        for st in tileset["subtrees"]:
            sub = json.loads((world / uri["subtree"].format(si=st["si"], sj=st["sj"])).read_text(encoding="utf-8"))
            for lvl in sub["levels"]:
                for i, j in zip(lvl["i"], lvl["j"]):
                    keys = {"level": lvl["level"], "si": st["si"], "sj": st["sj"], "i": i, "j": j}
                    out.append({"content": uri["content"].format(**keys), "meta": uri["meta"].format(**keys),
                                "data": True})
        return out
    index = json.loads((world / tiling.get("index", "tiles/index.json")).read_text(encoding="utf-8"))
    return [{"content": t["glb"], "meta": t["meta"], "data": False} for t in index["tiles"]]


def compare_worlds(a: Path, b: Path) -> dict[str, Any]:
    ma = json.loads((a / "manifest.json").read_text(encoding="utf-8"))
    mb = json.loads((b / "manifest.json").read_text(encoding="utf-8"))
    lines = []
    same = True
    if ma["config_hash"] != mb["config_hash"]:
        same = False
        lines.append("config: DIFFERENT")
        lines += [f"  {k}" for k in _dict_diff(ma["config"], mb["config"])[:20]]
    else:
        lines.append("config: identical")
    sa = {s["name"]: s for s in ma["sources"]}
    sb = {s["name"]: s for s in mb["sources"]}
    for name in sorted(set(sa) | set(sb)):
        ha, hb = sa.get(name, {}).get("snapshot_hash"), sb.get(name, {}).get("snapshot_hash")
        va, vb = sa.get(name, {}).get("dataset_version"), sb.get(name, {}).get("dataset_version")
        if ha != hb or va != vb:
            same = False
            lines.append(f"source {name}: CHANGED (version {va} -> {vb})")
        else:
            lines.append(f"source {name}: identical")
    oa, ob = ma["outputs"], mb["outputs"]
    diff = sorted(k for k in set(oa) | set(ob) if oa.get(k, {}).get("sha256") != ob.get(k, {}).get("sha256"))
    if diff:
        same = False
        lines.append(f"outputs: {len(diff)} of {len(set(oa) | set(ob))} files differ")
        lines += [f"  {k}" for k in diff[:30]]
    else:
        lines.append(f"outputs: all {len(oa)} files identical")
    return {"identical": same, "lines": lines, "differing_outputs": diff}


def _dict_diff(a: Any, b: Any, prefix: str = "") -> list[str]:
    if isinstance(a, dict) and isinstance(b, dict):
        out = []
        for k in sorted(set(a) | set(b)):
            out += _dict_diff(a.get(k), b.get(k), f"{prefix}.{k}" if prefix else k)
        return out
    return [] if a == b else [f"{prefix}: {a!r} -> {b!r}"]
