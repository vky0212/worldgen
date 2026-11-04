"""Engine-independent triangle mesh containers.

Positions are in the local ENU frame (x east, y north, z up, metres). Triangles
are counter-clockwise when seen from the side their normal points to. Conversion
to glTF axes happens only in the exporter.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class Mesh:
    positions: np.ndarray                      # (n, 3) float64
    indices: np.ndarray                        # (m, 3) int64
    normals: np.ndarray | None = None          # (n, 3)
    uvs: np.ndarray | None = None              # (n, 2)
    colors: np.ndarray | None = None           # (n, 4) float 0..1
    feature: np.ndarray | None = None          # (n,) int feature index
    mode: str = "triangles"                    # "triangles" or "lines" (indices (m, 2))

    @property
    def vertex_count(self) -> int:
        return int(self.positions.shape[0])

    @property
    def triangle_count(self) -> int:
        return int(self.indices.shape[0]) if self.mode == "triangles" else 0

    def is_empty(self) -> bool:
        return self.positions.shape[0] == 0 or self.indices.shape[0] == 0

    def ensure_normals(self) -> None:
        if self.normals is None and self.mode == "triangles":
            self.normals = vertex_normals(self.positions, self.indices)

    def translated(self, dx: float, dy: float, dz: float = 0.0) -> Mesh:
        return Mesh(self.positions + np.array([dx, dy, dz]), self.indices, self.normals, self.uvs,
                    self.colors, self.feature, self.mode)

    def bounds(self) -> tuple[np.ndarray, np.ndarray]:
        return self.positions.min(axis=0), self.positions.max(axis=0)


def empty_mesh(mode: str = "triangles") -> Mesh:
    return Mesh(np.zeros((0, 3)), np.zeros((0, 3 if mode == "triangles" else 2), dtype=np.int64), mode=mode)


def triangle_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    a = positions[indices[:, 0]]
    b = positions[indices[:, 1]]
    c = positions[indices[:, 2]]
    return np.cross(b - a, c - a)


def vertex_normals(positions: np.ndarray, indices: np.ndarray) -> np.ndarray:
    n = np.zeros_like(positions, dtype=np.float64)
    if len(indices):
        fn = triangle_normals(positions, indices)  # area weighted
        for k in range(3):
            np.add.at(n, indices[:, k], fn)
    lens = np.linalg.norm(n, axis=1)
    bad = lens < 1e-12
    n[bad] = (0.0, 0.0, 1.0)
    lens[bad] = 1.0
    return n / lens[:, None]


def merge_meshes(meshes: list[Mesh]) -> Mesh:
    meshes = [m for m in meshes if not m.is_empty()]
    if not meshes:
        return empty_mesh()
    mode = meshes[0].mode
    for m in meshes:
        if m.mode == "triangles":
            m.ensure_normals()
    pos = np.concatenate([m.positions for m in meshes])
    offs = np.cumsum([0] + [m.vertex_count for m in meshes[:-1]])
    idx = np.concatenate([m.indices + o for m, o in zip(meshes, offs)])

    def cat(attr: str, width: int, fill: float) -> np.ndarray | None:
        if all(getattr(m, attr) is None for m in meshes):
            return None
        parts = []
        for m in meshes:
            v = getattr(m, attr)
            if v is None:
                shape = (m.vertex_count, width) if width else (m.vertex_count,)
                v = np.full(shape, fill)
            parts.append(v)
        return np.concatenate(parts)

    return Mesh(
        pos, idx,
        normals=cat("normals", 3, 0.0) if mode == "triangles" else None,
        uvs=cat("uvs", 2, 0.0),
        colors=cat("colors", 4, 1.0),
        feature=cat("feature", 0, 0).astype(np.int64) if any(m.feature is not None for m in meshes) else None,
        mode=mode,
    )


@dataclass
class LayerBuilder:
    """Collects meshes of one layer, grouped by material, with a shared feature table."""

    layer: str
    features: list[str] = field(default_factory=list)
    _feature_index: dict[str, int] = field(default_factory=dict)
    parts: dict[str, list[Mesh]] = field(default_factory=dict)

    def feature_index(self, feature_id: str) -> int:
        idx = self._feature_index.get(feature_id)
        if idx is None:
            idx = len(self.features)
            self.features.append(feature_id)
            self._feature_index[feature_id] = idx
        return idx

    def add(self, mesh: Mesh, material: str, feature_id: str) -> None:
        if mesh.is_empty():
            return
        fi = self.feature_index(feature_id)
        mesh.feature = np.full(mesh.vertex_count, fi, dtype=np.int64)
        self.parts.setdefault(material, []).append(mesh)

    def build(self) -> dict[str, Mesh]:
        return {mat: merge_meshes(ms) for mat, ms in sorted(self.parts.items()) if ms}

    def is_empty(self) -> bool:
        return not any(self.parts.values())
