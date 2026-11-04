"""Stable hashing helpers used for determinism, caching and provenance."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Any

import numpy as np


def _default(o: Any) -> Any:
    if isinstance(o, np.integer):
        return int(o)
    if isinstance(o, np.floating):
        return float(o)
    if isinstance(o, np.ndarray):
        return o.tolist()
    if isinstance(o, Path):
        return o.as_posix()
    if isinstance(o, (set, frozenset)):
        return sorted(o)
    raise TypeError(f"not JSON serializable: {type(o).__name__}")


def canonical_json(data: Any) -> str:
    return json.dumps(data, sort_keys=True, separators=(",", ":"), default=_default, allow_nan=True)


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def sha256_json(data: Any) -> str:
    return sha256_bytes(canonical_json(data).encode("utf-8"))


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        while True:
            b = fh.read(chunk)
            if not b:
                break
            h.update(b)
    return h.hexdigest()


def stable_u64(*parts: Any) -> int:
    """Deterministic 64-bit integer from arbitrary JSON-able parts (seed derivation)."""
    digest = hashlib.blake2b(canonical_json(parts).encode("utf-8"), digest_size=8).digest()
    return int.from_bytes(digest, "little")


def rng_for(*parts: Any) -> np.random.Generator:
    """A numpy Generator whose stream depends only on ``parts``."""
    return np.random.Generator(np.random.PCG64(stable_u64(*parts)))


def unit_hash(*parts: Any) -> float:
    """Deterministic float in [0, 1)."""
    return stable_u64(*parts) / 2.0**64


# --------------------------------------------------------------------------------------
# portable counter-based hash (vectorised; specified in docs/vegetation.md)

_U64_MASK = 0xFFFFFFFFFFFFFFFF
_GOLDEN = np.uint64(0x9E3779B97F4A7C15)
_MIX1 = np.uint64(0xBF58476D1CE4E5B9)
_MIX2 = np.uint64(0x94D049BB133111EB)


def _as_u64(word: Any) -> np.ndarray:
    """An integer word (scalar or array) as uint64, two's complement for negative values."""
    if isinstance(word, (int, np.integer)) and not isinstance(word, np.ndarray):
        return np.array([int(word) & _U64_MASK], dtype=np.uint64)
    a = np.asarray(word)
    if a.dtype == np.uint64:
        return a
    if a.dtype.kind in "iub":
        return a.astype(np.int64).astype(np.uint64)
    raise TypeError(f"hash words must be integers, got {a.dtype}")


def hash_u64(*words: Any) -> np.ndarray:
    """Portable 64-bit hash of integer words (scalars or arrays, broadcast together).

    ``h = 0; for w in words: h = mix64((h XOR w) + 0x9E3779B97F4A7C15)``, all modulo 2**64,
    with ``mix64`` the splitmix64 finaliser (xor-shift 30, * 0xBF58476D1CE4E5B9, xor-shift 27,
    * 0x94D049BB133111EB, xor-shift 31) and words taken as two's-complement 64-bit integers.
    Unlike ``stable_u64`` it is cheap per element and simple to reproduce in other languages
    (the viewer's runtime generators use the same function). Returns a uint64 array (shape
    (1,) when every word is a scalar).
    """
    arrays = [_as_u64(w) for w in words]
    h = np.zeros(np.broadcast_shapes(*(a.shape for a in arrays)), dtype=np.uint64)
    for a in arrays:
        h = (h ^ a) + _GOLDEN
        h ^= h >> np.uint64(30)
        h *= _MIX1
        h ^= h >> np.uint64(27)
        h *= _MIX2
        h ^= h >> np.uint64(31)
    return h


def hash_unit(*words: Any) -> np.ndarray:
    """``hash_u64`` mapped to [0, 1): the top 53 bits divided by 2**53."""
    return (hash_u64(*words) >> np.uint64(11)).astype(np.float64) * (1.0 / float(1 << 53))


def round_floats(obj: Any, ndigits: int = 6) -> Any:
    """Round floats recursively so tiny numeric noise does not change hashes."""
    if isinstance(obj, float):
        if math.isnan(obj) or math.isinf(obj):
            return obj
        return round(obj, ndigits)
    if isinstance(obj, dict):
        return {k: round_floats(v, ndigits) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [round_floats(v, ndigits) for v in obj]
    return obj
