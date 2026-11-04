"""Conformance vectors shared by the compiler tests and the viewer's runtime generators.

``python -m worldgen.conformance [DIR]`` (default ``tests/conformance``) writes:

* ``hash64.json``: ``hashing.hash_u64`` / ``hash_unit`` cases (words as decimal strings,
  results as hex strings: JSON numbers cannot hold 64-bit integers).
* ``terrain_tiles.json``: data-first tile containers (base64) of a small synthetic world,
  one interior and one world-edge tile, with the exact vertex arrays the baked GLB terrain of
  that tile would have (glTF axes, relative to the tile corner, as float32): positions,
  normals, texture coordinates, colours (u8) and triangle indices.

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
from worldgen.export.heightmap import terrain_sections
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
        tiles.append({"i": i, "j": j, "bounds": list(bounds), "wgt_base64": base64.b64encode(wgt).decode("ascii"),
                      "expected": _expected_mesh(hf, rgba, bounds)})
    return {"schema": "worldgen.conformance.terrain_tiles", "tile_size": TILE, "spacing": SPACING,
            "quantum": QUANTUM, "note": "normals: compare within 1e-6 (central differences, float32)",
            "tiles": tiles}


def vectors() -> dict[str, dict[str, Any]]:
    return {"hash64.json": hash_vectors(), "terrain_tiles.json": terrain_vectors()}


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
