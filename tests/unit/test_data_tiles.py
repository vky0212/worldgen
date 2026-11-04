"""Data-first tile container, heightmap sections and conformance vectors."""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pytest

from worldgen.conformance import vectors
from worldgen.export.container import (
    decode_grid,
    decode_heights,
    encode_grid,
    encode_heights,
    read_container,
    write_container,
)

CONFORMANCE = Path(__file__).resolve().parents[1] / "conformance"


def test_container_round_trip():
    payloads = [("HGT", b"\x01\x02\x03", True), ("GLB", b"glTF" + bytes(range(50)), False), ("X", b"", True)]
    data = write_container(2, -5, 17, payloads)
    c = read_container(data)
    assert (c.level, c.i, c.j) == (2, -5, 17)
    assert c.sections == {"HGT": b"\x01\x02\x03", "GLB": b"glTF" + bytes(range(50)), "X": b""}
    # Section data is 8-byte aligned (readers may view it as typed arrays).
    import struct
    for k in range(3):
        _, off, _, _, _ = struct.unpack_from("<4sIIHH", data, 24 + 16 * k)
        assert off % 8 == 0
    with pytest.raises(ValueError):
        read_container(b"XXXX" + data[4:])


@pytest.mark.parametrize("span", [500, 70000])
def test_heights_round_trip_exact(span):
    rng = np.random.default_rng(span)
    n = rng.integers(-1000, -1000 + span, size=(67, 67))
    g = decode_heights(encode_heights(n, 4.0, 0.01, -260.0, 124.0))
    assert np.array_equal(g.z, n.astype(np.float64) * 0.01)       # the compiler's rounding, bit for bit
    assert (g.spacing, g.quantum, g.x0, g.y0, g.apron) == (4.0, 0.01, -260.0, 124.0, 1)


def test_grids_round_trip():
    rgb = np.arange(5 * 4 * 3, dtype=np.uint8).reshape(5, 4, 3)
    codes = np.arange(20, dtype=np.uint8).reshape(5, 4)
    assert np.array_equal(decode_grid(encode_grid(rgb), 3), rgb)
    assert np.array_equal(decode_grid(encode_grid(codes), 1), codes)


def test_conformance_vectors_are_current():
    """tests/conformance is what `python -m worldgen.conformance` writes (the viewer tests read it)."""
    for name, doc in vectors().items():
        on_disk = json.loads((CONFORMANCE / name).read_text(encoding="utf-8"))
        assert on_disk == json.loads(json.dumps(doc)), f"run: uv run python -m worldgen.conformance ({name})"
