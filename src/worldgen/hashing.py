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
