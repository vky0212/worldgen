"""Data-first tile container (``.wgt``) and its terrain sections (format 2.x).

Layout, all little-endian (the full contract is in docs/output-format.md)::

    header (24 B)   char[4] "WGT1" | u16 version (1) | u16 section count | u8 level | 3 x pad
                    | i32 i | i32 j | u32 reserved
    sections        count x 16 B: char[4] tag | u32 offset | u32 stored length | u16 flags | u16 pad
                    flags bit 0: the stored bytes are raw DEFLATE (RFC 1951) of the section
    data            each section at an 8-byte aligned offset

Sections written by the compiler: ``HGT `` heights, ``RGB `` terrain colours, ``LCC ``
land-cover codes and ``GLB `` the tile's other layers as a GLB. Readers ignore unknown tags.
"""

from __future__ import annotations

import struct
import zlib
from dataclasses import dataclass

import numpy as np

MAGIC = b"WGT1"
VERSION = 1
FLAG_DEFLATE = 1
_HEADER = struct.Struct("<4sHHB3xiiI")
_SECTION = struct.Struct("<4sIIHH")
# HGT header (48 B): u16 nx, u16 ny, u8 dtype (1 = u16, 2 = u32), u8 apron, u16 reserved,
# f64 spacing, f64 quantum, i64 offset, f64 x0, f64 y0. Heights: z = (offset + value) * quantum.
_HGT = struct.Struct("<HHBBHddqdd")
HGT_U16, HGT_U32 = 1, 2
_GRID = struct.Struct("<HH")


def _tag(name: str) -> bytes:
    t = name.encode("ascii")
    if not 1 <= len(t) <= 4:
        raise ValueError(f"section tag {name!r} must be 1-4 ASCII characters")
    return t.ljust(4, b" ")


def _deflate(data: bytes) -> bytes:
    c = zlib.compressobj(level=6, method=zlib.DEFLATED, wbits=-15)
    return c.compress(data) + c.flush()


def write_container(level: int, i: int, j: int, sections: list[tuple[str, bytes, bool]]) -> bytes:
    """``sections``: (tag, payload, deflate) in the order they are stored."""
    stored = [(_tag(t), _deflate(p) if z else p, FLAG_DEFLATE if z else 0) for t, p, z in sections]
    offset = _HEADER.size + _SECTION.size * len(stored)
    table, body = [], bytearray()
    for tag, data, flags in stored:
        pad = (-(offset + len(body))) % 8
        body.extend(b"\0" * pad)
        table.append(_SECTION.pack(tag, offset + len(body), len(data), flags, 0))
        body.extend(data)
    return _HEADER.pack(MAGIC, VERSION, len(stored), level, i, j, 0) + b"".join(table) + bytes(body)


@dataclass
class Container:
    level: int
    i: int
    j: int
    sections: dict[str, bytes]      # tag (trailing spaces stripped) -> decoded payload


def read_container(data: bytes) -> Container:
    magic, version, count, level, i, j, _ = _HEADER.unpack_from(data, 0)
    if magic != MAGIC:
        raise ValueError("not a WorldGen tile container (.wgt)")
    if version != VERSION:
        raise ValueError(f"unsupported .wgt version {version}")
    sections = {}
    for k in range(count):
        tag, off, length, flags, _ = _SECTION.unpack_from(data, _HEADER.size + k * _SECTION.size)
        raw = data[off:off + length]
        sections[tag.decode("ascii").rstrip()] = zlib.decompress(raw, wbits=-15) if flags & FLAG_DEFLATE else raw
    return Container(level, i, j, sections)


# --------------------------------------------------------------------------------------
# terrain sections


@dataclass
class HeightGrid:
    """Decoded ``HGT``: heights (ny, nx) in metres, row 0 south, with a one-node apron."""

    z: np.ndarray
    spacing: float
    quantum: float
    x0: float                 # local ENU of node (0, 0), i.e. the tile corner minus the apron
    y0: float
    apron: int


def encode_heights(n: np.ndarray, spacing: float, quantum: float, x0: float, y0: float, apron: int = 1) -> bytes:
    """``n``: integer heights in quanta (ny, nx), apron included; z = n * quantum."""
    n = np.asarray(n, dtype=np.int64)
    offset = int(n.min())
    span = int(n.max()) - offset
    dtype, code = (np.dtype("<u2"), HGT_U16) if span <= 0xFFFF else (np.dtype("<u4"), HGT_U32)
    if span > 0xFFFFFFFF:
        raise ValueError("height range too large for a heightmap tile")
    ny, nx = n.shape
    head = _HGT.pack(nx, ny, code, apron, 0, spacing, quantum, offset, x0, y0)
    return head + (n - offset).astype(dtype).tobytes()


def decode_heights(data: bytes) -> HeightGrid:
    nx, ny, code, apron, _, spacing, quantum, offset, x0, y0 = _HGT.unpack_from(data, 0)
    dtype = {HGT_U16: np.dtype("<u2"), HGT_U32: np.dtype("<u4")}[code]
    q = np.frombuffer(data, dtype=dtype, count=nx * ny, offset=_HGT.size).astype(np.int64).reshape(ny, nx)
    # Same float64 arithmetic as the compiler's rounding (integer quanta * quantum): bit-identical.
    return HeightGrid((q + offset).astype(np.float64) * quantum, spacing, quantum, x0, y0, apron)


def encode_grid(values: np.ndarray) -> bytes:
    """``RGB `` (ny, nx, 3) or ``LCC `` (ny, nx) uint8 grids."""
    v = np.ascontiguousarray(values, dtype=np.uint8)
    return _GRID.pack(v.shape[1], v.shape[0]) + v.tobytes()


def decode_grid(data: bytes, channels: int) -> np.ndarray:
    nx, ny = _GRID.unpack_from(data, 0)
    arr = np.frombuffer(data, dtype=np.uint8, count=nx * ny * channels, offset=_GRID.size)
    return arr.reshape((ny, nx, channels) if channels > 1 else (ny, nx))
