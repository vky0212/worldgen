"""Tree instances of data-first tiles: the ``TRE `` and ``TOB `` sections and the archetype GLB.

A data-first tile lists its rendered trees as 16-byte records (``TRE ``) instead of GLB
instances, per-tree ids and per-tree JSON records: consumers place their own tree assets
(``tiles/archetypes.glb`` holds the compiler's unit archetype meshes) and derive ids,
attributes and the inspect record from the record itself. Trees whose id is not
derivable (mapped trees) carry their full record in ``TOB ``.

``TRE `` layout (little-endian): ``u32 count``, ``u32 record size (16)``, then per tree::

    u16 x, u16 y        position in the tile, (x - x0) / tile_size * 65535 (rounded)
    u16 height          cm
    u16 k               candidate index of a generated tree, or the TOB index
    u8  crown           dm (crown radius)
    u8  trunk           cm (trunk radius)
    u8  yaw             yaw / 2pi * 256 (rounded, mod 256)
    u8  arch_cat        archetype code (ARCHETYPES index) | category code (CATEGORIES index) << 4
    u8  confidence      * 255
    u8  flags           bit 0 observed, bit 1 record in TOB, bit 2 ground-cover id (``_g{k}``)
    u8  dx, u8 dy       sampling cell relative to (floor(x0 / cell_size), floor(y0 / cell_size))

Records are grouped by archetype in ARCHETYPE_ORDER, ascending tree index within a group.
The tree's ground height is not stored: it is the terrain height at its position (the
``HGT `` section), as for the compiler.
"""

from __future__ import annotations

import json
import math
import re
import struct
from typing import Any

import numpy as np

from worldgen.export.gltf import GltfBuilder
from worldgen.vegetation.classify import CATEGORIES
from worldgen.vegetation.placement import ARCHETYPES
from worldgen.vegetation.regions import ARCHETYPE_ORDER, archetype_meshes

TRE_DTYPE = np.dtype([("x", "<u2"), ("y", "<u2"), ("h", "<u2"), ("k", "<u2"), ("crown", "u1"), ("trunk", "u1"),
                      ("yaw", "u1"), ("arch_cat", "u1"), ("conf", "u1"), ("flags", "u1"), ("dx", "u1"),
                      ("dy", "u1")])
assert TRE_DTYPE.itemsize == 16
FLAG_OBSERVED, FLAG_TOB, FLAG_GROUND = 1, 2, 4
_GEN_ID = re.compile(r"^tree:gen:(-?\d+)_(-?\d+)_(g?)(\d+)$")


def tree_sections(trees: Any, idx: np.ndarray, bounds: tuple[float, float, float, float], cell_size: float
                  ) -> list[tuple[str, bytes, bool]]:
    """``TRE `` (and ``TOB `` when needed) for the tile's rendered trees ``idx`` (in TRE order)."""
    x0, y0, x1, _ = bounds
    size = x1 - x0
    n = len(idx)
    rec = np.zeros(n, dtype=TRE_DTYPE)
    if n:
        rec["x"] = np.clip(np.rint((trees.x[idx] - x0) / size * 65535.0), 0, 65535)
        rec["y"] = np.clip(np.rint((trees.y[idx] - y0) / size * 65535.0), 0, 65535)
        rec["h"] = np.clip(np.rint(trees.height[idx] * 100.0), 0, 65535)
        rec["crown"] = np.clip(np.rint(trees.crown_radius[idx] * 10.0), 0, 255)
        rec["trunk"] = np.clip(np.rint(trees.trunk_radius[idx] * 100.0), 0, 255)
        rec["yaw"] = np.mod(np.rint(trees.yaw[idx] / (2 * math.pi) * 256.0), 256)
        rec["conf"] = np.clip(np.rint(trees.confidence[idx] * 255.0), 0, 255)
    cx0, cy0 = math.floor(x0 / cell_size), math.floor(y0 / cell_size)
    arch_code = {a: c for c, a in enumerate(ARCHETYPES)}
    cat_code = {c: k for k, c in enumerate(CATEGORIES)}
    tob: list[dict[str, Any]] = []
    for row, k in enumerate(idx.tolist()):
        tid = trees.ids[k]
        rec["arch_cat"][row] = arch_code.get(trees.archetype[k], 0) | (cat_code.get(trees.category[k], 0) << 4)
        flags = FLAG_OBSERVED if trees.observed[k] else 0
        m = _GEN_ID.match(tid)
        dx = dy = -1
        if m and not trees.observed[k]:
            dx, dy = int(m.group(1)) - cx0, int(m.group(2)) - cy0
        if m and not trees.observed[k] and 0 <= dx <= 255 and 0 <= dy <= 255 and int(m.group(4)) <= 65535:
            rec["k"][row] = int(m.group(4))
            rec["dx"][row], rec["dy"][row] = dx, dy
            flags |= FLAG_GROUND if m.group(3) else 0
        else:
            record = trees.record(k)
            record["centroid"] = [round(float(trees.x[k]), 2), round(float(trees.y[k]), 2),
                                  round(float(trees.ground_z[k]), 2)]
            rec["k"][row] = len(tob)
            tob.append({"id": tid, "record": record})
            flags |= FLAG_TOB
        rec["flags"][row] = flags
    sections = [("TRE", struct.pack("<II", n, TRE_DTYPE.itemsize) + rec.tobytes(), True)]
    if tob:
        sections.append(("TOB", json.dumps(tob, separators=(",", ":"), ensure_ascii=False).encode("utf-8"), True))
    return sections


def decode_trees(data: bytes) -> np.ndarray:
    """``TRE `` payload -> structured array (TRE_DTYPE)."""
    n, size = struct.unpack_from("<II", data, 0)
    if size != TRE_DTYPE.itemsize:
        raise ValueError(f"unsupported TRE record size {size}")
    return np.frombuffer(data, dtype=TRE_DTYPE, count=n, offset=8)


def tree_ids(rec: np.ndarray, tob: list[dict[str, Any]], bounds: tuple[float, float, float, float],
             cell_size: float) -> list[str]:
    """Entity ids of the records, as the viewer derives them."""
    cx0, cy0 = math.floor(bounds[0] / cell_size), math.floor(bounds[1] / cell_size)
    out = []
    for r in rec:
        if r["flags"] & FLAG_TOB:
            out.append(tob[int(r["k"])]["id"])
        else:
            g = "g" if r["flags"] & FLAG_GROUND else ""
            out.append(f"tree:gen:{cx0 + int(r['dx'])}_{cy0 + int(r['dy'])}_{g}{int(r['k'])}")
    return out


def height_at(g: Any, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    """Terrain height of a decoded ``HGT `` grid at local (x, y), on the mesh triangles
    (cells split along the (c, r)-(c+1, r+1) diagonal), like ``Heightfield.sample``."""
    z = g.z
    fx = (np.asarray(x, dtype=float) - g.x0) / g.spacing
    fy = (np.asarray(y, dtype=float) - g.y0) / g.spacing
    c = np.clip(np.floor(fx).astype(np.int64), 0, z.shape[1] - 2)
    r = np.clip(np.floor(fy).astype(np.int64), 0, z.shape[0] - 2)
    u = np.clip(fx - c, 0.0, 1.0)
    v = np.clip(fy - r, 0.0, 1.0)
    z00, z10, z01, z11 = z[r, c], z[r, c + 1], z[r + 1, c], z[r + 1, c + 1]
    return np.where(u >= v, z00 + u * (z10 - z00) + v * (z11 - z10), z00 + v * (z01 - z00) + u * (z11 - z01))


def instances_from_records(rec: np.ndarray, g: Any, x0: float, y0: float, tile_size: float
                           ) -> dict[str, np.ndarray]:
    """Instance transforms (glTF axes, relative to the tile corner, float32) of ``TRE `` records:
    the reference for runtimes placing their own archetypes (see ``archetypes_glb``)."""
    x = x0 + rec["x"].astype(float) / 65535.0 * tile_size
    y = y0 + rec["y"].astype(float) / 65535.0 * tile_size
    trunk = rec["trunk"].astype(float) / 100.0
    sink = 0.15 + trunk
    ground = height_at(g, x, y)
    yaw = rec["yaw"].astype(float) / 256.0 * 2.0 * math.pi
    crown = rec["crown"].astype(float) / 10.0
    height = rec["h"].astype(float) / 100.0
    t = np.column_stack([x - x0, ground - sink, -(y - y0)]).astype(np.float32)
    r = np.column_stack([np.zeros_like(yaw), np.sin(yaw / 2), np.zeros_like(yaw), np.cos(yaw / 2)]).astype(np.float32)
    s = np.column_stack([crown, height + sink, crown]).astype(np.float32)
    return {"TRANSLATION": t, "ROTATION": r, "SCALE": s}


def archetypes_glb(quality: str) -> bytes:
    """The unit archetype meshes, one root node per archetype (``tiles/archetypes.glb``).

    Instances use them like ``EXT_mesh_gpu_instancing`` does in baked tiles: translation at the
    trunk base minus ``0.15 + trunk radius``, rotation ``yaw`` about the vertical axis, scale
    ``(crown radius, height + 0.15 + trunk radius, crown radius)`` in glTF axes.
    """
    meshes = archetype_meshes(quality)
    gb = GltfBuilder()
    for arch in ARCHETYPE_ORDER:
        mi = gb.add_mesh(f"tree_{arch}", meshes[arch], (0.0, 0.0, 0.0), extras={"archetype": arch})
        gb.add_node(arch, mesh=mi, extras={"archetype": arch}, root=True)
    return gb.to_glb()
