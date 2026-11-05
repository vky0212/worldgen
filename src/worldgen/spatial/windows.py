"""Window-by-window processing of large grids.

A grid stage that computes each node from a bounded neighbourhood can run over windows of
the grid (``grid_windows``: a core plus the halo the stage needs) and write only the core of
each, so its temporaries follow the window instead of the world. For the result not to
depend on the window, every operation must give a node bit-identical values whatever the
window: exact stencils (``ndimage.correlate1d`` / ``gaussian_filter``, min / max filters)
instead of running sums (``uniform_filter``), sampling weights that do not depend on the
array origin, and searches with explicit tie rules.
"""

from __future__ import annotations

from collections.abc import Iterator

import numpy as np
from scipy import ndimage

Slices = tuple[slice, slice]


def grid_windows(shape: tuple[int, int], size: int, halo: int, align: int = 1) -> Iterator[tuple[Slices, Slices, Slices]]:
    """(window, core within the window, core) slices covering ``shape`` with square cores of
    ``size`` nodes and ``halo`` nodes around them (clipped at the grid); window edges are
    multiples of ``align``."""
    rows, cols = shape
    size = max(align, size // align * align)
    halo = -(-halo // align) * align
    for r0 in range(0, rows, size):
        for c0 in range(0, cols, size):
            r1, c1 = min(r0 + size, rows), min(c0 + size, cols)
            R0, C0 = max(r0 - halo, 0), max(c0 - halo, 0)
            R1, C1 = min(r1 + halo, rows), min(c1 + halo, cols)
            yield ((slice(R0, R1), slice(C0, C1)), (slice(r0 - R0, r1 - R0), slice(c0 - C0, c1 - C0)),
                   (slice(r0, r1), slice(c0, c1)))


def gauss_radius(sigma: float) -> int:
    """Support radius (cells) of ``ndimage.gaussian_filter`` (truncate 4); 0 below sigma 0.3."""
    return int(4.0 * sigma + 0.5) if sigma >= 0.3 else 0


def box_filter(a: np.ndarray, size: int, mode: str = "reflect") -> np.ndarray:
    """Mean over a ``size`` x ``size`` box, like ``ndimage.uniform_filter`` but as an explicit
    stencil: each output sums its own inputs in a fixed order, so a node gets the same value
    in any window (a running sum carries rounding from the start of each line)."""
    k = np.full(size, 1.0 / size)
    out = ndimage.correlate1d(np.asarray(a, dtype=float), k, axis=0, mode=mode)
    return ndimage.correlate1d(out, k, axis=1, mode=mode)
