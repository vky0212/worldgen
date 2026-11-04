"""Merge several GLB files into one scene (used for the combined ``world.glb``).

Works on the binary level so unchanged tiles never need to be regenerated.
"""

from __future__ import annotations

import json
import struct
from typing import Any

from worldgen.export.gltf import read_glb_json

# GLB stores the file and chunk lengths as uint32.
GLB_MAX_BYTES = 2**32 - 1


def _split(data: bytes) -> tuple[dict[str, Any], bytes]:
    doc = read_glb_json(data)
    jlen = struct.unpack_from("<I", data, 12)[0]
    off = 20 + jlen
    bin_chunk = b""
    if off < len(data):
        blen, btype = struct.unpack_from("<II", data, off)
        if btype == 0x004E4942:
            bin_chunk = data[off + 8: off + 8 + blen]
    return doc, bin_chunk


def merge_glbs(files: list[bytes], root_name: str = "world", extras: dict[str, Any] | None = None) -> bytes:
    out: dict[str, Any] = {"asset": {"version": "2.0", "generator": "worldgen"}, "scene": 0,
                           "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
                           "accessors": [], "bufferViews": []}
    blob = bytearray()
    used: set[str] = set()
    material_index: dict[str, int] = {}
    roots: list[int] = []
    for data in files:
        doc, bin_chunk = _split(data)
        while len(blob) % 4:
            blob.append(0)
        base = len(blob)
        blob.extend(bin_chunk)
        bv0 = len(out["bufferViews"])
        for v in doc.get("bufferViews", []):
            nv = dict(v)
            nv["byteOffset"] = v.get("byteOffset", 0) + base
            out["bufferViews"].append(nv)
        acc0 = len(out["accessors"])
        for a in doc.get("accessors", []):
            na = dict(a)
            na["bufferView"] = a["bufferView"] + bv0
            out["accessors"].append(na)
        mat_map = {}
        for i, m in enumerate(doc.get("materials", [])):
            key = json.dumps(m, sort_keys=True)
            if key not in material_index:
                material_index[key] = len(out["materials"])
                out["materials"].append(m)
            mat_map[i] = material_index[key]
        mesh0 = len(out["meshes"])
        for m in doc.get("meshes", []):
            nm = dict(m)
            prims = []
            for p in m["primitives"]:
                np_ = dict(p)
                np_["attributes"] = {k: v + acc0 for k, v in p["attributes"].items()}
                if "indices" in p:
                    np_["indices"] = p["indices"] + acc0
                if "material" in p:
                    np_["material"] = mat_map[p["material"]]
                prims.append(np_)
            nm["primitives"] = prims
            out["meshes"].append(nm)
        node0 = len(out["nodes"])
        for n in doc.get("nodes", []):
            nn = dict(n)
            if "mesh" in n:
                nn["mesh"] = n["mesh"] + mesh0
            if "children" in n:
                nn["children"] = [c + node0 for c in n["children"]]
            ext = n.get("extensions", {}).get("EXT_mesh_gpu_instancing")
            if ext:
                nn["extensions"] = {"EXT_mesh_gpu_instancing": {
                    "attributes": {k: v + acc0 for k, v in ext["attributes"].items()}}}
            out["nodes"].append(nn)
        for r in doc["scenes"][doc.get("scene", 0)]["nodes"]:
            roots.append(r + node0)
        used |= set(doc.get("extensionsUsed", []))
    root = {"name": root_name, "children": roots}
    if extras:
        root["extras"] = extras
    out["nodes"].append(root)
    out["scenes"][0]["nodes"] = [len(out["nodes"]) - 1]
    while len(blob) % 4:
        blob.append(0)
    out["buffers"] = [{"byteLength": len(blob)}]
    if used:
        out["extensionsUsed"] = sorted(used)
    for k in ("meshes", "materials", "accessors", "bufferViews"):
        if not out[k]:
            out.pop(k)
    js = json.dumps(out, separators=(",", ":")).encode("utf-8")
    while len(js) % 4:
        js += b" "
    total = 12 + 8 + len(js) + 8 + len(blob)
    if total > GLB_MAX_BYTES:
        raise ValueError(f"combined GLB would be {total / 2**30:.2f} GiB; GLB lengths are 32-bit (max 4 GiB). "
                         "Lower export.combined_glb_max_tiles or set export.combined_glb=false")
    chunks = struct.pack("<II", len(js), 0x4E4F534A) + js + struct.pack("<II", len(blob), 0x004E4942) + bytes(blob)
    return struct.pack("<III", 0x46546C67, 2, 12 + len(chunks)) + chunks
