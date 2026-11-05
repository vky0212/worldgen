"""Buildings of data-first tiles as footprint tables: the ``BLD `` and ``IDS `` sections.

A data-first tile ships its buildings as footprints plus the compiler's resolved roof decision
per footprint polygon (``buildings.building_plan``) instead of meshes; consumers build walls,
roofs and floors themselves (the viewer's ``gen/buildingMesh.ts``). Geometry the runtime builds
encloses the same solid as the baked mesh: same footprint, floor, eaves, ridge and top.

``BLD `` layout (little-endian): ``u32 buildings B``, ``u32 polygons P``, ``u32 rings R``,
``u32 vertices V``, then these arrays in order, each starting at a 4-byte aligned offset (zero
padding between them)::

    f32 bottom[B]     floor z: base - foundation depth, or base + min_height for raised parts
    f32 top[B]        roof top z (base + height)
    u16 polygons[B]   footprint polygons of the building
    u8  material[B]   index into tileset.json ``buildings.materials``
    u8  shape[P]      roof built on the polygon: 0 flat, 1 gabled, 2 skillion, 3 pyramidal
    u16 rings[P]      rings of the polygon (exterior first, then holes; pitched roofs have 1)
    f32 eave[P]       eave z (top for flat roofs)
    f32 cx[P], cy[P]  ridge centre (gabled, skillion) or apex (pyramidal), metres from the tile
                      corner (x0, y0)
    f32 ux[P], uy[P]  ridge direction, unit (gabled, skillion)
    f32 half[P]       half width across the ridge (gabled, skillion)
    u32 vertices[R]   vertices of the ring
    i32 xy[2V]        vertex (x, y) in millimetres from the tile corner; exteriors run
                      counter-clockwise, holes clockwise, the closing vertex is not repeated

Roof surfaces (``d = (p - c) . (-u_y, u_x)``, ``h = top - eave``): gabled
``z = eave + h * clip(1 - |d| / half, 0, 1)``, skillion ``z = eave + h * clip((d + half) / (2 half), 0, 1)``,
pyramidal: one triangle per exterior edge from the eaves to ``(cx, cy, top)``. Gabled walls follow
the roof, so their rings get a vertex where they cross the ridge line. Buildings are in world
order (by id); a building belongs to the tile holding its footprint's representative point, so
footprints may extend past the tile.

``IDS `` is UTF-8 JSON ``{"BLD": [entity id per building]}``.
"""

from __future__ import annotations

import json
import struct
from dataclasses import dataclass
from typing import Any

import numpy as np
from shapely.geometry import Polygon
from shapely.geometry.polygon import orient

from worldgen.buildings.buildings import BuildingPlan, building_material
from worldgen.export.gltf import MATERIALS
from worldgen.world.model import Building

BUILDING_MATERIALS = ["building_residential", "building_commercial", "building_industrial", "building_civic",
                      "building_default"]
ROOF_SHAPES = ["flat", "gabled", "skillion", "pyramidal"]
_MATERIAL_CODE = {m: k for k, m in enumerate(BUILDING_MATERIALS)}
_SHAPE_CODE = {s: k for k, s in enumerate(ROOF_SHAPES)}
_HEADER = struct.Struct("<4I")
# (column, dtype, length: buildings / polygons / rings / 2 x vertices)
COLUMNS: list[tuple[str, str, str]] = [
    ("bottom", "<f4", "B"), ("top", "<f4", "B"), ("polygons", "<u2", "B"), ("material", "u1", "B"),
    ("shape", "u1", "P"), ("rings", "<u2", "P"), ("eave", "<f4", "P"), ("cx", "<f4", "P"), ("cy", "<f4", "P"),
    ("ux", "<f4", "P"), ("uy", "<f4", "P"), ("half", "<f4", "P"),
    ("vertices", "<u4", "R"), ("xy", "<i4", "V2"),
]


def building_sections(buildings: list[Building], plans: list[BuildingPlan],
                      bounds: tuple[float, float, float, float]) -> list[tuple[str, bytes, bool]]:
    """``BLD `` and ``IDS `` of a tile's rendered buildings (with their ``building_plan``)."""
    x0, y0 = bounds[0], bounds[1]
    cols: dict[str, list[Any]] = {name: [] for name, _, _ in COLUMNS}
    xy: list[np.ndarray] = []
    for b, plan in zip(buildings, plans):
        if len(plan.roofs) > 0xFFFF:
            raise ValueError(f"{b.id}: more than 65535 footprint polygons")
        cols["bottom"].append(plan.bottom)
        cols["top"].append(plan.top)
        cols["polygons"].append(len(plan.roofs))
        cols["material"].append(_MATERIAL_CODE.get(building_material(b), _MATERIAL_CODE["building_default"]))
        for r in plan.roofs:
            poly = orient(r.poly, 1.0)
            rings = [poly.exterior, *poly.interiors]
            if len(rings) > 0xFFFF:
                raise ValueError(f"{b.id}: more than 65535 rings in one polygon")
            cols["shape"].append(_SHAPE_CODE[r.shape])
            cols["rings"].append(len(rings))
            cols["eave"].append(r.eave)
            pitched = r.shape != "flat"
            cols["cx"].append(float(r.centre[0]) - x0 if pitched else 0.0)
            cols["cy"].append(float(r.centre[1]) - y0 if pitched else 0.0)
            cols["ux"].append(float(r.u[0]))
            cols["uy"].append(float(r.u[1]))
            cols["half"].append(float(r.half))
            for ring in rings:
                c = np.asarray(ring.coords)[:-1, :2]
                cols["vertices"].append(len(c))
                xy.append(np.rint((c - (x0, y0)) * 1000.0))
    v = np.concatenate(xy) if xy else np.zeros((0, 2))
    cols["xy"] = v.ravel().tolist()
    counts = (len(cols["bottom"]), len(cols["shape"]), len(cols["vertices"]), len(v))
    out = bytearray(_HEADER.pack(*counts))
    for name, dtype, _ in COLUMNS:
        out += np.asarray(cols[name], dtype=dtype).tobytes()
        out += b"\0" * (-len(out) % 4)
    ids = json.dumps({"BLD": [b.id for b in buildings]}, separators=(",", ":"), ensure_ascii=False).encode("utf-8")
    return [("BLD", bytes(out), True), ("IDS", ids, True)]


@dataclass
class BuildingRecords:
    """Decoded ``BLD `` columns, plus the first polygon / ring / vertex of each row."""

    columns: dict[str, np.ndarray]
    poly_start: np.ndarray      # (B + 1,) first polygon of each building
    ring_start: np.ndarray      # (P + 1,) first ring of each polygon
    vert_start: np.ndarray      # (R + 1,) first vertex of each ring

    def __len__(self) -> int:
        return len(self.columns["bottom"])

    def __getitem__(self, name: str) -> np.ndarray:
        return self.columns[name]

    def polygons(self, k: int, x0: float, y0: float) -> list[Polygon]:
        """Footprint polygons of building ``k`` in world coordinates."""
        xy = self.columns["xy"].reshape(-1, 2).astype(float) / 1000.0 + (x0, y0)
        out = []
        for p in range(int(self.poly_start[k]), int(self.poly_start[k + 1])):
            rings = [xy[self.vert_start[r]:self.vert_start[r + 1]]
                     for r in range(int(self.ring_start[p]), int(self.ring_start[p + 1]))]
            out.append(Polygon(rings[0], rings[1:]))
        return out


def decode_buildings(data: bytes) -> BuildingRecords:
    """``BLD `` payload -> columns."""
    sizes = dict(zip("BPRV", _HEADER.unpack_from(data, 0)))
    sizes["V2"] = 2 * sizes["V"]
    off = _HEADER.size
    cols: dict[str, np.ndarray] = {}
    for name, dtype, length in COLUMNS:
        n = sizes[length]
        cols[name] = np.frombuffer(data, dtype=dtype, count=n, offset=off)
        off += n * np.dtype(dtype).itemsize
        off += -off % 4
    if off > len(data):
        raise ValueError("truncated BLD section")

    def starts(counts: np.ndarray) -> np.ndarray:
        return np.concatenate([[0], np.cumsum(counts.astype(np.int64))])

    return BuildingRecords(cols, starts(cols["polygons"]), starts(cols["rings"]), starts(cols["vertices"]))


def tileset_entry() -> dict[str, Any]:
    """``tileset.json`` ``buildings``: sections, material palette (glTF PBR factors), roof codes."""
    mats = []
    for name in BUILDING_MATERIALS:
        m = MATERIALS[name]
        mats.append({"name": name, "color": list(m.color), "roughness": m.roughness, "metallic": m.metallic})
    return {"sections": ["BLD", "IDS"], "materials": mats, "roof_shapes": ROOF_SHAPES, "uv_scale": 4.0}
