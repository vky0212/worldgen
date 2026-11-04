"""Smooth world-space value noise (a hash of lattice coordinates and a seed).

Values depend only on the world position and the seed, never on the grid or region they
are evaluated on, so any window of the world gets the same noise as the whole world.
"""

from __future__ import annotations

import numpy as np

_K1 = np.uint64(0x9E3779B97F4A7C15)
_K2 = np.uint64(0xBF58476D1CE4E5B9)
_K3 = np.uint64(0x94D049BB133111EB)


def _lattice(ix: np.ndarray, iy: np.ndarray, seed: int) -> np.ndarray:
    """Uniform value in [-1, 1] per integer lattice point (splitmix64 finaliser)."""
    salt = np.uint64(((seed & 0xFFFFFFFF) * 0x94D049BB133111EB) & 0xFFFFFFFFFFFFFFFF)
    h = ix.astype(np.int64).astype(np.uint64) * _K1
    h ^= iy.astype(np.int64).astype(np.uint64) * _K2 + salt
    h ^= h >> np.uint64(30)
    h *= _K2
    h ^= h >> np.uint64(27)
    h *= _K3
    h ^= h >> np.uint64(31)
    return (h >> np.uint64(11)).astype(np.float64) * (2.0 / float(1 << 53)) - 1.0


def value_noise(x: np.ndarray, y: np.ndarray, wavelength: float, seed: int) -> np.ndarray:
    """Smooth value noise in [-1, 1] at world coordinates (C2 quintic interpolation)."""
    fx = np.asarray(x, dtype=float) / wavelength
    fy = np.asarray(y, dtype=float) / wavelength
    x0 = np.floor(fx)
    y0 = np.floor(fy)
    tx = fx - x0
    ty = fy - y0
    tx = tx * tx * tx * (tx * (tx * 6.0 - 15.0) + 10.0)
    ty = ty * ty * ty * (ty * (ty * 6.0 - 15.0) + 10.0)
    ix = x0.astype(np.int64)
    iy = y0.astype(np.int64)
    a = _lattice(ix, iy, seed)
    b = _lattice(ix + 1, iy, seed)
    c = _lattice(ix, iy + 1, seed)
    d = _lattice(ix + 1, iy + 1, seed)
    return a + (b - a) * tx + (c - a) * ty + (a - b - c + d) * tx * ty


def warp_offsets(x: np.ndarray, y: np.ndarray, amplitude: float, wavelength: float, seed: int
                 ) -> tuple[np.ndarray, np.ndarray]:
    """Smooth displacement (dx, dy) in metres at world coordinates (two octaves)."""
    if amplitude <= 0:
        z = np.zeros(np.shape(x))
        return z, z.copy()

    def fbm(s: int) -> np.ndarray:
        return (0.7 * value_noise(x, y, wavelength, s)
                + 0.3 * value_noise(np.asarray(x) + 17.3, np.asarray(y) - 9.1, wavelength / 2.7, s + 1))

    return amplitude * fbm(4 * seed + 11), amplitude * fbm(4 * seed + 23)
