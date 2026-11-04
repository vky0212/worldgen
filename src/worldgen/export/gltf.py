"""Minimal, deterministic glTF 2.0 binary (GLB) writer.

Written in-house (instead of going through a scene library) to control exactly
what ends up in the file: node names and extras for semantic ids, the custom
``_FEATURE_ID_0`` vertex attribute, ``EXT_mesh_gpu_instancing`` for trees and
byte-for-byte reproducible output.

Input meshes are in local ENU; they are converted to glTF axes (+Y up) here:
``gltf = (x, z, -y)``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass, field
from typing import Any

import numpy as np

from worldgen.geometry.mesh import Mesh

FLOAT = 5126
UINT32 = 5125
UNSIGNED_BYTE = 5121
ARRAY_BUFFER = 34962
ELEMENT_ARRAY_BUFFER = 34963


def enu_to_gltf(p: np.ndarray) -> np.ndarray:
    return np.column_stack([p[:, 0], p[:, 2], -p[:, 1]])


@dataclass
class Material:
    name: str
    color: tuple[float, float, float, float]
    roughness: float = 0.9
    metallic: float = 0.0
    double_sided: bool = False
    alpha_blend: bool = False
    vertex_colors: bool = False


MATERIALS: dict[str, Material] = {m.name: m for m in [
    Material("terrain", (1.0, 1.0, 1.0, 1.0), 0.95, vertex_colors=True),
    Material("road_major", (0.22, 0.22, 0.24, 1.0), 0.85),
    Material("road_minor", (0.30, 0.30, 0.31, 1.0), 0.88),
    Material("road_path", (0.62, 0.58, 0.52, 1.0), 0.9),
    Material("bridge_deck", (0.62, 0.60, 0.57, 1.0), 0.8),
    Material("bridge_railing", (0.72, 0.72, 0.70, 1.0), 0.7),
    Material("bridge_pier", (0.55, 0.54, 0.52, 1.0), 0.85),
    Material("tunnel_wall", (0.45, 0.44, 0.42, 1.0), 0.9, double_sided=True),
    Material("tunnel_portal", (0.52, 0.50, 0.47, 1.0), 0.85),
    Material("building_residential", (0.84, 0.79, 0.70, 1.0), 0.85),
    Material("building_commercial", (0.66, 0.72, 0.78, 1.0), 0.6),
    Material("building_industrial", (0.64, 0.62, 0.60, 1.0), 0.8),
    Material("building_civic", (0.80, 0.70, 0.62, 1.0), 0.8),
    Material("building_default", (0.78, 0.76, 0.72, 1.0), 0.85),
    Material("water", (0.16, 0.36, 0.52, 0.85), 0.1, alpha_blend=True),
    Material("tree_trunk", (0.36, 0.26, 0.18, 1.0), 0.95),
    Material("foliage_broadleaf", (0.22, 0.42, 0.18, 1.0), 0.95),
    Material("foliage_conifer", (0.14, 0.32, 0.18, 1.0), 0.95),
    Material("foliage_palm", (0.30, 0.50, 0.20, 1.0), 0.9, double_sided=True),
    Material("foliage_shrub", (0.30, 0.42, 0.20, 1.0), 0.95),
    Material("foliage_grass", (0.45, 0.58, 0.26, 1.0), 0.95, double_sided=True),
    Material("collision", (1.0, 0.0, 1.0, 1.0), 1.0),
]}
VEGETATION_LINE_COLORS = {
    "dense_forest": (0.10, 0.45, 0.12), "sparse_woodland": (0.35, 0.60, 0.20), "shrubland": (0.55, 0.60, 0.25),
    "grassland": (0.70, 0.80, 0.30), "agricultural": (0.85, 0.75, 0.30), "urban_trees": (0.20, 0.75, 0.45),
    "wetland": (0.30, 0.60, 0.60), "mangrove": (0.10, 0.50, 0.40),
}
for _k, _c in VEGETATION_LINE_COLORS.items():
    MATERIALS[f"vegetation_{_k}"] = Material(f"vegetation_{_k}", (*_c, 1.0), 1.0)


@dataclass
class GltfBuilder:
    json: dict[str, Any] = field(default_factory=lambda: {
        "asset": {"version": "2.0", "generator": "worldgen"},
        "scene": 0, "scenes": [{"nodes": []}], "nodes": [], "meshes": [], "materials": [],
        "accessors": [], "bufferViews": [], "buffers": [],
    })
    blob: bytearray = field(default_factory=bytearray)
    _materials: dict[str, int] = field(default_factory=dict)
    extensions_used: set[str] = field(default_factory=set)

    # -- low level -----------------------------------------------------------------------
    def _view(self, data: bytes, target: int | None = None, stride: int | None = None) -> int:
        while len(self.blob) % 4:
            self.blob.append(0)
        view: dict[str, Any] = {"buffer": 0, "byteOffset": len(self.blob), "byteLength": len(data)}
        if target is not None:
            view["target"] = target
        if stride is not None:
            view["byteStride"] = stride
        self.blob.extend(data)
        self.json["bufferViews"].append(view)
        return len(self.json["bufferViews"]) - 1

    def _accessor(self, arr: np.ndarray, kind: str, comp: int, target: int | None, minmax: bool = False,
                  normalized: bool = False) -> int:
        view = self._view(arr.tobytes(), target)
        acc: dict[str, Any] = {"bufferView": view, "componentType": comp, "count": int(arr.shape[0]), "type": kind}
        if normalized:
            acc["normalized"] = True
        if minmax:
            acc["min"] = [float(v) for v in arr.min(axis=0)]
            acc["max"] = [float(v) for v in arr.max(axis=0)]
        self.json["accessors"].append(acc)
        return len(self.json["accessors"]) - 1

    def material(self, name: str) -> int:
        if name in self._materials:
            return self._materials[name]
        m = MATERIALS.get(name, Material(name, (0.8, 0.8, 0.8, 1.0)))
        d: dict[str, Any] = {
            "name": m.name,
            "pbrMetallicRoughness": {"baseColorFactor": list(m.color), "metallicFactor": m.metallic,
                                     "roughnessFactor": m.roughness},
        }
        if m.double_sided:
            d["doubleSided"] = True
        if m.alpha_blend:
            d["alphaMode"] = "BLEND"
        self.json["materials"].append(d)
        self._materials[name] = len(self.json["materials"]) - 1
        return self._materials[name]

    # -- meshes and nodes ----------------------------------------------------------------
    def add_mesh(self, name: str, prims: dict[str, Mesh], origin: tuple[float, float, float] = (0.0, 0.0, 0.0),
                 extras: dict[str, Any] | None = None) -> int | None:
        primitives = []
        for mat_name, m in prims.items():
            if m.is_empty():
                continue
            pos = enu_to_gltf(m.positions - np.asarray(origin)).astype(np.float32)
            attrs: dict[str, int] = {"POSITION": self._accessor(pos, "VEC3", FLOAT, ARRAY_BUFFER, minmax=True)}
            if m.mode == "triangles":
                m.ensure_normals()
                nrm = enu_to_gltf(m.normals).astype(np.float32)  # type: ignore[arg-type]
                attrs["NORMAL"] = self._accessor(nrm, "VEC3", FLOAT, ARRAY_BUFFER)
            if m.uvs is not None:
                uv = np.column_stack([m.uvs[:, 0], 1.0 - m.uvs[:, 1]]).astype(np.float32)
                attrs["TEXCOORD_0"] = self._accessor(uv, "VEC2", FLOAT, ARRAY_BUFFER)
            if m.colors is not None:
                col = np.clip(np.round(m.colors * 255.0), 0, 255).astype(np.uint8)
                attrs["COLOR_0"] = self._accessor(col, "VEC4", UNSIGNED_BYTE, ARRAY_BUFFER, normalized=True)
            if m.feature is not None:
                attrs["_FEATURE_ID_0"] = self._accessor(m.feature.astype(np.float32), "SCALAR", FLOAT, ARRAY_BUFFER)
            idx = m.indices.astype(np.uint32).ravel()
            prim: dict[str, Any] = {
                "attributes": attrs,
                "indices": self._accessor(idx, "SCALAR", UINT32, ELEMENT_ARRAY_BUFFER),
                "material": self.material(mat_name),
                "mode": 4 if m.mode == "triangles" else 1,
            }
            if extras:
                prim["extras"] = extras
            primitives.append(prim)
        if not primitives:
            return None
        mesh: dict[str, Any] = {"name": name, "primitives": primitives}
        if extras:
            mesh["extras"] = extras
        self.json["meshes"].append(mesh)
        return len(self.json["meshes"]) - 1

    def add_node(self, name: str, mesh: int | None = None, children: list[int] | None = None,
                 translation: tuple[float, float, float] | None = None, extras: dict[str, Any] | None = None,
                 extensions: dict[str, Any] | None = None, root: bool = False) -> int:
        node: dict[str, Any] = {"name": name}
        if mesh is not None:
            node["mesh"] = mesh
        if children:
            node["children"] = children
        if translation is not None:
            node["translation"] = [float(v) + 0.0 for v in translation]
        if extras:
            node["extras"] = extras
        if extensions:
            node["extensions"] = extensions
        self.json["nodes"].append(node)
        idx = len(self.json["nodes"]) - 1
        if root:
            self.json["scenes"][0]["nodes"].append(idx)
        return idx

    def instancing(self, transforms: dict[str, np.ndarray]) -> dict[str, Any]:
        self.extensions_used.add("EXT_mesh_gpu_instancing")
        attrs = {k: self._accessor(np.ascontiguousarray(v, dtype=np.float32), {3: "VEC3", 4: "VEC4"}[v.shape[1]],
                                   FLOAT, None)
                 for k, v in transforms.items()}
        return {"EXT_mesh_gpu_instancing": {"attributes": attrs}}

    def to_glb(self) -> bytes:
        doc = dict(self.json)
        while len(self.blob) % 4:
            self.blob.append(0)
        doc["buffers"] = [{"byteLength": len(self.blob)}] if self.blob else []
        if not self.blob:
            doc.pop("bufferViews", None)
            doc.pop("accessors", None)
        for k in ("meshes", "materials", "accessors", "bufferViews", "buffers"):
            if k in doc and not doc[k]:
                doc.pop(k)
        if self.extensions_used:
            doc["extensionsUsed"] = sorted(self.extensions_used)
        js = json.dumps(doc, separators=(",", ":"), sort_keys=False).encode("utf-8")
        while len(js) % 4:
            js += b" "
        chunks = struct.pack("<II", len(js), 0x4E4F534A) + js
        if self.blob:
            chunks += struct.pack("<II", len(self.blob), 0x004E4942) + bytes(self.blob)
        header = struct.pack("<III", 0x46546C67, 2, 12 + len(chunks))
        return header + chunks


def read_glb_json(data: bytes) -> dict[str, Any]:
    magic, version, _ = struct.unpack_from("<III", data, 0)
    if magic != 0x46546C67 or version != 2:
        raise ValueError("not a glTF 2.0 binary")
    length, ctype = struct.unpack_from("<II", data, 12)
    if ctype != 0x4E4F534A:
        raise ValueError("first GLB chunk is not JSON")
    return json.loads(data[20:20 + length].decode("utf-8"))
