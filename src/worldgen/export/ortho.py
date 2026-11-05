"""Per-tile orthophoto bake: source imagery -> ``tiles/<id>.ortho.jpg`` in the local frame.

Orientation (the contract with consumers; also in docs/output-format.md):

* The image covers exactly the tile square ``[xmin, xmax] x [ymin, ymax]`` of
  ``TileGrid.bounds(id)`` in local ENU metres, pixel-is-area: ``N x N`` pixels of ``T / N``
  metres (``T`` = tile size), no border or overlap.
* North up: **row 0 is the north edge** (``y = ymax``), **column 0 is the west edge**
  (``x = xmin``). Pixel (column ``c``, row ``r``) is centred at
  ``x = xmin + (c + 0.5) * T / N`` and ``y = ymax - (r + 0.5) * T / N``.
* Texture coordinates for a local point: ``u = (x - xmin) / T``, ``v = (y - ymin) / T``, so
  ``v = 1`` is the north edge, i.e. image row 0. That is how WebGL samples an image uploaded
  with ``UNPACK_FLIP_Y`` (three.js ``Texture.flipY = true``, the ``TextureLoader`` default).
  With ``flipY = false`` (glTF convention: image row 0 at ``v = 0``) use ``1 - v``.

Pixel size: ``N = min(tile_px_max, T / m_per_px)`` rounded up to a power of two (at least 32),
where ``m_per_px`` is the source ground resolution, but never finer than ``max_m_per_px``.

Colour: Sentinel-2 TCI is dark and hazy, so one set of levels is computed per world over the
region extent and applied to every tile: per-channel 1st/99th percentile levels (the black
points remove haze; the white points' spread between channels is damped so hues survive), a
gray-world white balance (weak: vegetation-dominated regions are legitimately green, and a full
gray-world turns forest grey and grassland pink), a mild gamma lift and a soft highlight
shoulder. Display values up to the knee (0.8) are unchanged; above it they are compressed
logarithmically (``knee + a ln(1 + (v - knee) / a)``, slope 1 at the knee) instead of clipped,
with ``a`` fitted so that the brightest 0.5 % of every tile stays at or below 0.86 (219/255).
Bright roofs, roads and bare ground stay light and keep their brightness order instead of
burning out to white.

Seams: every tile samples one world-level image at its own pixel centres with bilinear
interpolation. The outermost pixels interpolate imagery from up to half a pixel beyond the tile
edge instead of clamping at it, so adjacent tiles are samples of one continuous image and meet
without a seam. Imagery is box-averaged first when an output pixel spans two or more source
pixels, so fine sources do not alias.
"""

from __future__ import annotations

import math
import warnings
from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
import rasterio
from affine import Affine
from pyproj import Transformer
from rasterio.errors import NotGeoreferencedWarning
from rasterio.io import MemoryFile
from scipy.ndimage import map_coordinates

from worldgen.config import ImagerySourceConfig
from worldgen.hashing import sha256_bytes, sha256_json
from worldgen.sources.imagery import ground_m_per_px, rgb_valid_mask
from worldgen.spatial.frame import LocalFrame
from worldgen.spatial.raster import GeoRaster
from worldgen.world.world import CanonicalWorld

ORTHO_VERSION = 2                # part of the tile input hash; bump when the bake changes
MIN_PX = 32
LEVEL_PERCENTILES = (1.0, 99.0)
# White point = black + span; each channel keeps this share of its own 1-99% span, the rest is
# the mean span of all channels (1 = fully independent channels, which shifts hues).
LEVEL_CHANNEL_INDEPENDENCE = 0.25
GRAY_WORLD_STRENGTH = 0.1        # 0 = off, 1 = full gray-world
GRAY_WORLD_GAIN_LIMITS = (0.95, 1.05)
GAMMA = 1.3                      # out = in ** (1 / GAMMA)
HIGHLIGHT_KNEE = 0.8             # display values above this are compressed instead of clipped
HIGHLIGHT_TARGET = 0.86          # ... so that this percentile of every tile stays at or below 0.86:
HIGHLIGHT_TILE_PERCENTILE = 99.5
HIGHLIGHT_GENTLE = 0.3           # weakest shoulder (the old white point lands at ~0.95)
STATS_MAX_SIDE = 2048            # colour statistics sample at most this many points per axis
LATTICE = 17                     # exact reprojection on a 17x17 lattice per tile, linear in between


@dataclass(frozen=True)
class ColourLevels:
    """World-level colour normalisation, applied identically to every tile."""

    black: tuple[float, float, float]
    white: tuple[float, float, float]
    gain: tuple[float, float, float]
    gamma: float = GAMMA
    knee: float = 1.0          # display value where the highlight shoulder starts
    shoulder: float = 0.0      # shoulder scale a (smaller = stronger); 0 = hard clip at 1

    def display(self, rgb: np.ndarray) -> np.ndarray:
        """Source values ``(..., 3)`` -> display values (levels, gain, gamma), not yet clipped."""
        b = np.asarray(self.black, dtype=np.float32)
        w = np.asarray(self.white, dtype=np.float32)
        g = np.asarray(self.gain, dtype=np.float32)
        x = (rgb.astype(np.float32) - b) / np.maximum(w - b, 1.0)
        return np.maximum(x * g, 0.0) ** np.float32(1.0 / self.gamma)

    def tone(self, v: np.ndarray) -> np.ndarray:
        """Highlight shoulder: identity up to the knee, then ``knee + a ln(1 + (v - knee) / a)``."""
        if self.shoulder > 0.0 and self.knee < 1.0:
            k, a = np.float32(self.knee), np.float32(self.shoulder)
            v = np.where(v > k, k + a * np.log1p(np.maximum(v - k, 0.0) / a), v)
        return np.clip(v, 0.0, 1.0)

    def apply(self, rgb: np.ndarray) -> np.ndarray:
        """Source values ``(..., 3)`` -> normalised uint8 ``(..., 3)``."""
        return np.rint(self.tone(self.display(rgb)) * 255.0).astype(np.uint8)

    def to_dict(self) -> dict[str, Any]:
        return {k: [round(float(v), 3) for v in val] if isinstance(val, tuple) else val
                for k, val in asdict(self).items()}


IDENTITY_LEVELS = ColourLevels((0.0, 0.0, 0.0), (255.0, 255.0, 255.0), (1.0, 1.0, 1.0), 1.0)


def highlight_shoulder(ref: float, knee: float = HIGHLIGHT_KNEE, target: float = HIGHLIGHT_TARGET) -> float:
    """Shoulder scale ``a`` (slope 1 at the knee) that maps display value ``ref`` to at most ``target``.

    The gentlest shoulder is ``HIGHLIGHT_GENTLE``; brighter references get a stronger one.
    """
    gentle = HIGHLIGHT_GENTLE

    def f(a: float) -> float:                 # shoulder output above the knee; increasing in a
        return a * math.log1p((ref - knee) / a)

    if ref <= target or f(gentle) <= target - knee:
        return gentle
    lo, hi = 1e-6, gentle
    for _ in range(60):
        mid = 0.5 * (lo + hi)
        if f(mid) > target - knee:
            hi = mid
        else:
            lo = mid
    return lo


def _highlight_reference(whiteness: np.ndarray, groups: np.ndarray | None) -> float:
    """Brightest per-tile ``HIGHLIGHT_TILE_PERCENTILE`` of the whiteness (min over channels)."""
    if groups is None or len(groups) != len(whiteness):
        return float(np.percentile(whiteness, HIGHLIGHT_TILE_PERCENTILE))
    order = np.argsort(groups, kind="stable")
    cuts = np.flatnonzero(np.diff(groups[order])) + 1
    return max(float(np.percentile(chunk, HIGHLIGHT_TILE_PERCENTILE)) for chunk in np.split(whiteness[order], cuts))


def compute_levels(pixels: np.ndarray, groups: np.ndarray | None = None) -> ColourLevels:
    """Levels from valid source pixels ``(n, 3)``: percentile stretch, weak gray-world, gamma and a
    highlight shoulder fitted to the brightest tile (``groups``: tile of each pixel)."""
    if pixels.size == 0:
        return IDENTITY_LEVELS
    px = pixels.astype(np.float32)
    lo = np.percentile(px, LEVEL_PERCENTILES[0], axis=0)
    span = np.maximum(np.percentile(px, LEVEL_PERCENTILES[1], axis=0) - lo, 1.0)
    a = LEVEL_CHANNEL_INDEPENDENCE
    hi = lo + a * span + (1.0 - a) * span.mean()
    x = np.clip((px - lo) / (hi - lo), 0.0, 1.0)
    means = np.maximum(x.mean(axis=0), 1e-6)
    gain = np.clip((means.mean() / means) ** GRAY_WORLD_STRENGTH, *GRAY_WORLD_GAIN_LIMITS)
    def t(arr: np.ndarray) -> tuple[float, float, float]:
        return (float(arr[0]), float(arr[1]), float(arr[2]))

    base = ColourLevels(t(lo), t(hi), t(gain), GAMMA)
    ref = _highlight_reference(base.display(px).min(axis=1), groups)
    return ColourLevels(base.black, base.white, base.gain, GAMMA, HIGHLIGHT_KNEE, highlight_shoulder(ref))


def tile_pixels(tile_size: float, m_per_px: float, tile_px_max: int) -> int:
    """Tile image edge: ``min(tile_px_max, T / m_per_px)`` rounded up to a power of two, >= 32."""
    want = min(float(tile_px_max), tile_size / m_per_px)
    n = 1 << max(0, math.ceil(math.log2(max(want, 1.0)) - 1e-9))
    return int(min(max(n, MIN_PX), max(tile_px_max, MIN_PX)))


def _box_average(data: np.ndarray, valid: np.ndarray, transform: Affine, k: int
                 ) -> tuple[np.ndarray, np.ndarray, Affine]:
    """Average k x k blocks of valid pixels (anti-aliasing before minification)."""
    rows, cols = valid.shape[0] // k, valid.shape[1] // k
    v = valid[: rows * k, : cols * k].reshape(rows, k, cols, k)
    n = v.sum(axis=(1, 3))
    out = np.zeros((3, rows, cols), dtype=np.uint8)
    for ch in range(3):
        d = data[ch, : rows * k, : cols * k].reshape(rows, k, cols, k).astype(np.float32)
        s = (d * v).sum(axis=(1, 3))
        out[ch] = np.clip(np.rint(s / np.maximum(n, 1)), 0, 255).astype(np.uint8)
    return out, n > 0, transform @ Affine.scale(k)


def _interp_matrix(n: int, lattice: np.ndarray) -> np.ndarray:
    """(n, m) matrix W with W @ f(lattice) = linear interpolation of f at 0..n-1."""
    full = np.arange(n, dtype=np.float64)
    eye = np.eye(len(lattice))
    return np.stack([np.interp(full, lattice, eye[j]) for j in range(len(lattice))], axis=1)


def encode_jpeg(rgb: np.ndarray, quality: int) -> bytes:
    """``(rows, cols, 3)`` uint8 -> baseline JPEG bytes (GDAL JPEG driver; no sidecar files)."""
    h, w, _ = rgb.shape
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", NotGeoreferencedWarning)
        with rasterio.Env(GDAL_PAM_ENABLED="NO"), MemoryFile() as mf:
            with mf.open(driver="JPEG", width=w, height=h, count=3, dtype="uint8", QUALITY=int(quality)) as ds:
                ds.write(np.ascontiguousarray(rgb.transpose(2, 0, 1)))
            return bytes(mf.read())


class OrthoBaker:
    """Bakes north-up per-tile orthophotos from one source raster (see module docstring)."""

    def __init__(self, imagery: GeoRaster, frame: LocalFrame, extent: tuple[float, float, float, float],
                 tile_size: float, cfg: ImagerySourceConfig, levels: ColourLevels | None = None) -> None:
        self.cfg = cfg
        self.tile_size = float(tile_size)
        self.source_m_per_px = ground_m_per_px(imagery)
        self.effective_m_per_px = max(self.source_m_per_px, cfg.max_m_per_px)
        self.px = tile_pixels(self.tile_size, self.effective_m_per_px, cfg.tile_px_max)
        self.m_per_px = self.tile_size / self.px
        data = imagery.data
        valid = rgb_valid_mask(data)
        transform = imagery.transform
        ratio = self.m_per_px / self.source_m_per_px
        self.prefilter = int(ratio + 0.5) if ratio >= 1.5 else 1
        if self.prefilter > 1:
            data, valid, transform = _box_average(data, valid, transform, self.prefilter)
        self.transform = transform
        self._inv = ~transform
        self._to_src = Transformer.from_crs(frame.local_crs(), imagery.crs, always_xy=True)
        # Coarser levels of detail reuse the level-0 colour levels, so every level looks alike.
        self.levels = levels if levels is not None else compute_levels(*self._region_pixels(data, valid, extent))
        if not valid.all():
            # Holes (no imagery anywhere) get the mean colour; bilinear sampling softens the edge.
            fill = data[:, valid].mean(axis=1) if valid.any() else np.full(3, 128.0)
            data = data.copy()
            data[:, ~valid] = np.rint(fill).astype(np.uint8)[:, None]
        self.valid_fraction = float(valid.mean())
        self.src = data

    def _src_pixel_coords(self, xs: np.ndarray, ys: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Local (x, y) -> fractional source (row, col) with pixel centres at integers."""
        sx, sy = self._to_src.transform(xs, ys)
        inv = self._inv
        cols = inv.a * sx + inv.b * sy + inv.c - 0.5
        rows = inv.d * sx + inv.e * sy + inv.f - 0.5
        return rows, cols

    def _region_pixels(self, data: np.ndarray, valid: np.ndarray, extent: tuple[float, float, float, float]
                       ) -> tuple[np.ndarray, np.ndarray]:
        """Valid source pixels inside the region extent, (n, 3), nearest-sampled on a regular grid,
        and the tile (an integer key) each one falls in."""
        xmin, ymin, xmax, ymax = extent
        nx = int(min(STATS_MAX_SIDE, max(2, math.ceil((xmax - xmin) / (self.source_m_per_px * self.prefilter)))))
        ny = int(min(STATS_MAX_SIDE, max(2, math.ceil((ymax - ymin) / (self.source_m_per_px * self.prefilter)))))
        gx, gy = np.meshgrid(np.linspace(xmin, xmax, nx), np.linspace(ymin, ymax, ny))
        rows, cols = self._src_pixel_coords(gx.ravel(), gy.ravel())
        r = np.rint(rows).astype(np.int64)
        c = np.rint(cols).astype(np.int64)
        h, w = valid.shape
        inside = (r >= 0) & (r < h) & (c >= 0) & (c < w)
        r, c = r[inside], c[inside]
        ok = valid[r, c]
        t = self.tile_size
        ntx = max(1, int(round((xmax - xmin) / t)))
        nty = max(1, int(round((ymax - ymin) / t)))
        ti = np.clip(np.floor((gx.ravel()[inside][ok] - xmin) / t), 0, ntx - 1).astype(np.int64)
        tj = np.clip(np.floor((gy.ravel()[inside][ok] - ymin) / t), 0, nty - 1).astype(np.int64)
        return data[:, r[ok], c[ok]].T, tj * ntx + ti

    def sample(self, bounds: tuple[float, float, float, float]) -> np.ndarray:
        """Normalised ``(N, N, 3)`` uint8 image of a tile; row 0 = north, column 0 = west."""
        xmin, ymin, xmax, ymax = bounds
        n = self.px
        lat = np.linspace(0.0, n - 1.0, min(n, LATTICE))
        lx = xmin + (lat + 0.5) / n * (xmax - xmin)
        ly = ymax - (lat + 0.5) / n * (ymax - ymin)
        gx, gy = np.meshgrid(lx, ly)                       # [lattice row (north->south), lattice col]
        rows, cols = self._src_pixel_coords(gx, gy)
        wm = _interp_matrix(n, lat)
        rr = wm @ rows @ wm.T
        cc = wm @ cols @ wm.T
        out = np.empty((n, n, 3), dtype=np.float32)
        for ch in range(3):
            out[..., ch] = map_coordinates(self.src[ch], [rr, cc], order=1, mode="nearest", output=np.float32)
        return self.levels.apply(out)

    def bake(self, bounds: tuple[float, float, float, float]) -> bytes:
        return encode_jpeg(self.sample(bounds), self.cfg.jpeg_quality)

    def describe(self) -> dict[str, Any]:
        return {
            "format": "image/jpeg", "pixels": self.px, "m_per_px": round(self.m_per_px, 4),
            "source_m_per_px": round(self.source_m_per_px, 3),
            "effective_source_m_per_px": round(self.effective_m_per_px, 3),
            "source_valid_fraction": round(self.valid_fraction, 4), "prefilter": self.prefilter,
            "jpeg_quality": self.cfg.jpeg_quality, "levels": self.levels.to_dict(),
            "orientation": "north-up: row 0 = north edge (ymax), column 0 = west edge (xmin)",
            "uv": "u = (x - xmin) / tile_size, v = (y - ymin) / tile_size; v = 1 is image row 0 (flipY = true)",
            "version": ORTHO_VERSION,
        }


def imagery_hash(world: CanonicalWorld) -> str | None:
    """Content hash of everything that shapes the orthophotos (memoised on the world)."""
    im = world.imagery
    if im is None:
        return None
    if world.imagery_hash is None:
        c = world.config.sources.imagery
        world.imagery_hash = sha256_json({
            "version": ORTHO_VERSION,
            "data": sha256_bytes(np.ascontiguousarray(im.data).tobytes()),
            "shape": list(im.data.shape), "transform": list(im.transform)[:6], "crs": im.crs.to_wkt(),
            "bake": {"max_m_per_px": c.max_m_per_px, "tile_px_max": c.tile_px_max, "jpeg_quality": c.jpeg_quality},
            "extent": list(world.region.extent),
        })
    return world.imagery_hash


def ortho_baker(world: CanonicalWorld, level: int = 0, levels: ColourLevels | None = None) -> OrthoBaker | None:
    """The orthophoto baker of tiles of ``level`` (``tile_size * 2**level`` metres; see
    ``export.pyramid``), with the colour ``levels`` of level 0 for the coarser levels."""
    if world.imagery is None:
        return None
    return OrthoBaker(world.imagery, world.region.frame, world.region.extent,
                      world.config.tiling.tile_size * (1 << level), world.config.sources.imagery, levels)
