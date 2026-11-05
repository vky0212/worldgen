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


# Vectors whose numbers come from floating-point reductions over meshes (summation order and
# BLAS kernels differ between platforms): their numbers are compared within a tolerance.
TOLERANT = {"building_tiles.json": 2e-6}


def _differences(a, b, path: str, tol: float, out: list[str]) -> None:  # noqa: ANN001
    number = (int, float)
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b)):
            if k not in a or k not in b:
                out.append(f"{path}/{k}: only on {'disk' if k in a else 'regeneration'}")
            else:
                _differences(a[k], b[k], f"{path}/{k}", tol, out)
    elif isinstance(a, list) and isinstance(b, list):
        if len(a) != len(b):
            out.append(f"{path}: {len(a)} items on disk, {len(b)} regenerated")
        for k, (x, y) in enumerate(zip(a, b)):
            _differences(x, y, f"{path}[{k}]", tol, out)
    elif tol and isinstance(a, number) and isinstance(b, number) and not isinstance(a, bool):
        if abs(a - b) > tol + 1e-9 * abs(b):
            out.append(f"{path}: {a!r} on disk, {b!r} regenerated")
    elif a != b:
        out.append(f"{path}: {str(a)[:60]!r} on disk, {str(b)[:60]!r} regenerated")


def test_conformance_vectors_are_current():
    """tests/conformance is what `python -m worldgen.conformance` writes (the viewer tests read it)."""
    for name, doc in vectors().items():
        on_disk = json.loads((CONFORMANCE / name).read_text(encoding="utf-8"))
        diffs: list[str] = []
        _differences(on_disk, json.loads(json.dumps(doc)), "", TOLERANT.get(name, 0.0), diffs)
        assert not diffs, f"run: uv run python -m worldgen.conformance ({name}): " + "; ".join(diffs[:8])


def test_conformance_comparison_reports_differences():
    diffs: list[str] = []
    _differences({"a": [1.0, 2.0], "s": "x"}, {"a": [1.0, 2.000001], "s": "x"}, "", 2e-6, diffs)
    assert diffs == []
    _differences({"a": [1.0, 2.0], "s": "x"}, {"a": [1.0, 2.1], "s": "y", "n": 1}, "", 2e-6, diffs)
    assert diffs == ["/a[1]: 2.0 on disk, 2.1 regenerated", "/n: only on regeneration", "/s: 'x' on disk, 'y' regenerated"]
    exact: list[str] = []
    _differences({"a": 1.0}, {"a": 1.0000001}, "", 0.0, exact)
    assert exact == ["/a: '1.0' on disk, '1.0000001' regenerated"]
