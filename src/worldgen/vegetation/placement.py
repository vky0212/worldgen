"""Deterministic, tile-independent tree placement.

Candidates are drawn per *sampling cell* of a global lattice: the count and
every uniform of a candidate are counter-based hashes of ``(seed, salt, cell
index, candidate index, column)`` (``hashing.hash_unit``), so all cells are
generated at once with array arithmetic and any window gives the same values;
attributes depend only on the global vegetation grid at the candidate
position. A candidate survives if

1. it lies in a vegetated cell, outside every exclusion (buildings, road
   surfaces, water, bridges) and on terrain no steeper than the slope limit;
2. it passes an intensity thinning that converts the local target stem density
   into the candidate intensity of a Matérn type-II hard-core process;
3. no higher-priority candidate (lower random mark, or an observed tree) lies
   within the hard-core distance.

Every rule only looks at candidates within a bounded radius, so any tile can be
generated on its own (with a halo) and produce exactly the trees of a global
run: instances are stable across tile boundaries and never duplicated.
Generated trees are marked as inferred; mapped trees are kept as observed.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import shapely
from scipy.spatial import cKDTree
from shapely.geometry.base import BaseGeometry

from worldgen.config import VegetationConfig
from worldgen.hashing import hash_u64, hash_unit, stable_u64
from worldgen.terrain.heightfield import Heightfield
from worldgen.vegetation.classify import CAT, CATEGORIES, TREE_CATEGORIES, VegetationGrid
from worldgen.world.model import ObservedTree, TreeTable

ARCHETYPES = ["broadleaf", "conifer", "palm", "shrub", "grass"]
PALM_GENERA = {"roystonea", "cocos", "phoenix", "washingtonia", "livistona", "syagrus", "areca",
               "trachycarpus", "sabal", "dypsis", "archontophoenix", "caryota"}


def crown_radius(archetype: str, height: np.ndarray | float) -> np.ndarray:
    h = np.maximum(np.asarray(height, dtype=float), 0.3)
    if archetype == "conifer":
        return np.clip(0.18 * h ** 0.8, 0.6, 5.0)
    if archetype == "palm":
        return np.clip(0.9 + 0.12 * h, 1.2, 4.5)
    if archetype == "shrub":
        return np.clip(0.45 * h + 0.25, 0.4, 2.5)
    if archetype == "grass":
        return np.clip(0.6 * h + 0.2, 0.2, 0.8)
    return np.clip(0.32 * h ** 0.85, 0.8, 9.0)


def trunk_radius(archetype: str, height: np.ndarray | float) -> np.ndarray:
    h = np.asarray(height, dtype=float)
    if archetype in ("shrub", "grass"):
        return np.zeros_like(h)
    if archetype == "palm":
        return np.clip(0.012 * h + 0.1, 0.1, 0.4)
    return np.clip(0.018 * h + 0.03, 0.05, 1.2)


# The largest crown radius any archetype reaches (broadleaf clip): bounds every hard core.
CROWN_RADIUS_MAX = 9.0


def crown_radius_codes(arch: np.ndarray, height: np.ndarray) -> np.ndarray:
    """``crown_radius`` for archetype codes (indices into ARCHETYPES), elementwise."""
    a = np.asarray(arch)
    h = np.maximum(np.asarray(height, dtype=float), 0.3)
    return np.select(
        [a == 1, a == 2, a == 3, a == 4],
        [np.clip(0.18 * h ** 0.8, 0.6, 5.0), np.clip(0.9 + 0.12 * h, 1.2, 4.5),
         np.clip(0.45 * h + 0.25, 0.4, 2.5), np.clip(0.6 * h + 0.2, 0.2, 0.8)],
        np.clip(0.32 * h ** 0.85, 0.8, CROWN_RADIUS_MAX))


def trunk_radius_codes(arch: np.ndarray, height: np.ndarray) -> np.ndarray:
    """``trunk_radius`` for archetype codes, elementwise."""
    a = np.asarray(arch)
    h = np.asarray(height, dtype=float)
    return np.select([a >= 3, a == 2], [np.zeros_like(h), np.clip(0.012 * h + 0.1, 0.1, 0.4)],
                     np.clip(0.018 * h + 0.03, 0.05, 1.2))


# Salts of the counter-based random streams (hashing.hash_unit; see docs/vegetation.md).
SALT_TREE_COUNT = 1     # (seed, salt, i, j): candidates in sampling cell (i, j)
SALT_TREE_U = 2         # (seed, salt, i, j, k, column): uniforms of candidate k
SALT_TREE_SEED = 3      # (seed, salt, i, j, k): published per-tree seed
SALT_GROUND_COUNT = 4
SALT_GROUND_U = 5
SALT_GROUND_SEED = 6

# Sampling-cell rows generated together (bounds the candidate arrays held at once).
CELL_ROWS_PER_BATCH = 64


def poisson_icdf(u: np.ndarray, lam: np.ndarray | float) -> np.ndarray:
    """Smallest n with P(N <= n) >= u for N ~ Poisson(lam), elementwise (lam up to a few hundred)."""
    u = np.asarray(u, dtype=float)
    uf = u.ravel()
    lf = np.broadcast_to(np.asarray(lam, dtype=float), u.shape).ravel()
    pf = np.exp(-lf)
    cf = pf.copy()
    nf = np.zeros(uf.shape, dtype=np.int64)
    lmax = float(lf.max()) if lf.size else 0.0
    # Far beyond any probability mass: stops rounding (cdf never quite reaching u) from looping.
    k_max = int(np.ceil(lmax + 40.0 * np.sqrt(lmax) + 40.0))
    active = np.flatnonzero(uf > cf)
    k = 0
    while active.size and k < k_max:
        k += 1
        pf[active] *= lf[active] / k
        cf[active] += pf[active]
        nf[active] = k
        active = active[uf[active] > cf[active]]
    return nf.reshape(u.shape)


@dataclass
class Exclusions:
    geometry: BaseGeometry | None

    def contains(self, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        if self.geometry is None or self.geometry.is_empty or len(x) == 0:
            return np.zeros(len(x), dtype=bool)
        return shapely.contains_xy(self.geometry, x, y)


def _archetype_tables(cfg: VegetationConfig) -> tuple[np.ndarray, np.ndarray]:
    """Per category: cumulative archetype weights (n_cat, n_arch) and hard-core factors."""
    cum = np.zeros((len(CATEGORIES), len(ARCHETYPES)))
    spacing = np.ones(len(CATEGORIES)) * 1.6
    for name, cc in cfg.categories.items():
        if name not in CAT:
            continue
        w = np.array([cc.archetypes.get(a, 0.0) for a in ARCHETYPES], dtype=float)
        if w.sum() <= 0:
            w = np.array([1.0, 0, 0, 0, 0])
        cum[CAT[name]] = np.cumsum(w / w.sum())
        spacing[CAT[name]] = cc.spacing_factor
    return cum, spacing


def _leaf_adjusted(arch_u: np.ndarray, cat: np.ndarray, leaf: np.ndarray, cum: np.ndarray) -> np.ndarray:
    # Count of cumulative weights <= u, i.e. searchsorted(cum[cat], u, side="right") per row.
    idx = (cum[np.asarray(cat, dtype=np.int64)] <= np.asarray(arch_u)[:, None]).sum(axis=1).astype(np.int64)
    idx = np.clip(idx, 0, len(ARCHETYPES) - 1)
    trees = np.isin(idx, [0, 1])
    idx = np.where(trees & (leaf == 1), 0, idx)
    idx = np.where(trees & (leaf == 2), 1, idx)
    return idx


def place_trees(
    grid: VegetationGrid, terrain: Heightfield, exclusions: Exclusions, observed: list[ObservedTree],
    cfg: VegetationConfig, bounds: tuple[float, float, float, float] | None = None,
) -> TreeTable:
    """Place trees inside ``bounds`` (default: whole grid). Results do not depend on bounds."""
    xmin = grid.xmin
    ymin = grid.ymin
    xmax = grid.xmin + grid.shape[1] * grid.resolution
    ymax = grid.ymin + grid.shape[0] * grid.resolution
    bx0, by0, bx1, by1 = bounds if bounds is not None else (xmin, ymin, xmax, ymax)
    cum, spacing = _archetype_tables(cfg)
    cs = cfg.cell_size
    tree_codes = np.array([CAT[c] for c in TREE_CATEGORIES])

    # Observed trees first: they are always kept and dominate generated candidates.
    obs = _observed_table(observed, grid, terrain, cfg)

    # Halo: the largest hard core any candidate can have, so trees inside ``bounds`` see every
    # candidate that could suppress them (a constant, so it does not depend on the window).
    r_int = float(np.max(spacing)) * CROWN_RADIUS_MAX
    hx0, hy0, hx1, hy1 = bx0 - r_int, by0 - r_int, bx1 + r_int, by1 + r_int

    i0, i1 = math.floor(max(hx0, xmin) / cs), math.floor(min(hx1, xmax - 1e-9) / cs)
    j0, j1 = math.floor(max(hy0, ymin) / cs), math.floor(min(hy1, ymax - 1e-9) / cs)
    parts = [_tree_candidates(grid, cfg, cum, spacing, tree_codes, i0, i1, jb, min(jb + CELL_ROWS_PER_BATCH - 1, j1))
             for jb in range(j0, j1 + 1, CELL_ROWS_PER_BATCH)] if i1 >= i0 else []
    parts = [p for p in parts if p is not None]
    if not parts:
        gen = TreeTable()
    else:
        C = {k: np.concatenate([p[k] for p in parts]) for k in parts[0]}
        # Exclusions and slope before the hard-core rule.
        keep = ~exclusions.contains(C["x"], C["y"])
        inside_t = terrain.contains(C["x"], C["y"])
        keep &= inside_t
        slope = np.where(inside_t, terrain.slope_deg(C["x"], C["y"]), 90.0)
        keep &= slope <= cfg.max_slope_deg
        C = {k: v[keep] for k, v in C.items()}
        keep = _matern_ii(C["x"], C["y"], C["mark"], C["delta"], obs)
        C = {k: v[keep] for k, v in C.items()}
        inb = (C["x"] >= bx0) & (C["x"] < bx1) & (C["y"] >= by0) & (C["y"] < by1)
        C = {k: v[inb] for k, v in C.items()}
        gen = _generated_table(C, terrain, cfg)
    if bounds is not None and len(obs):
        m = (obs.x >= bx0) & (obs.x < bx1) & (obs.y >= by0) & (obs.y < by1)
        obs = obs.subset(m)
    return concat_tables([obs, gen])


def _range_max(a: np.ndarray, lo: np.ndarray, hi: np.ndarray, axis: int) -> np.ndarray:
    """Maximum of ``a`` over the index ranges lo[k]..hi[k] (inclusive) along ``axis``."""
    out = np.take(a, lo, axis=axis)
    for d in range(1, int((hi - lo).max(initial=0)) + 1):
        out = np.maximum(out, np.take(a, np.minimum(lo + d, hi), axis=axis))
    return out


def _tree_candidates(grid: VegetationGrid, cfg: VegetationConfig, cum: np.ndarray, spacing: np.ndarray,
                     tree_codes: np.ndarray, i0: int, i1: int, j0: int, j1: int) -> dict[str, np.ndarray] | None:
    """Candidates of sampling cells i0..i1 x j0..j1 that pass the intensity thinning.

    Every sampling cell draws a Poisson number of candidates at the largest target intensity
    of the grid cells it touches; each candidate's seven uniforms are hashed from
    (seed, cell, candidate index, column), so a cell's candidates never depend on which other
    cells are generated with it.
    """
    cs = cfg.cell_size
    ii = np.arange(i0, i1 + 1, dtype=np.int64)
    jj = np.arange(j0, j1 + 1, dtype=np.int64)
    # Grid block of each sampling cell, resolved like cell_of does for the cell's corners.
    _, c_lo, _ = grid.cell_of(ii * cs, np.full(len(ii), grid.ymin))
    _, c_hi, _ = grid.cell_of((ii + 1) * cs - 1e-6, np.full(len(ii), grid.ymin))
    r_lo, _, _ = grid.cell_of(np.full(len(jj), grid.xmin), jj * cs)
    r_hi, _, _ = grid.cell_of(np.full(len(jj), grid.xmin), (jj + 1) * cs - 1e-6)
    R0, R1, C0, C1 = int(r_lo.min()), int(r_hi.max()), int(c_lo.min()), int(c_hi.max())
    win = (slice(R0, R1 + 1), slice(C0, C1 + 1))
    lam = _lambda0(grid.coverage[win], grid.canopy_height[win], grid.category[win], spacing, tree_codes)
    cap = _range_max(_range_max(lam, c_lo - C0, c_hi - C0, axis=1), r_lo - R0, r_hi - R0, axis=0)
    jg, ig = np.meshgrid(jj, ii, indexing="ij")
    act = cap > 0
    ci, cj, cap = ig[act], jg[act], cap[act]
    if len(ci) == 0:
        return None
    n = poisson_icdf(hash_unit(cfg.seed, SALT_TREE_COUNT, ci, cj), cap * cs * cs)
    if n.sum() == 0:
        return None
    rep = np.repeat(np.arange(len(n)), n)
    k = np.arange(int(n.sum()), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
    ci, cj, cap = ci[rep], cj[rep], cap[rep]
    u = [hash_unit(cfg.seed, SALT_TREE_U, ci, cj, k, col) for col in range(7)]
    x = (ci + u[0]) * cs
    y = (cj + u[1]) * cs
    rr, cc, inside = grid.cell_of(x, y)
    cat = grid.category[rr, cc]
    ok = inside & np.isin(cat, tree_codes)
    lam0 = _lambda0(grid.coverage[rr, cc], grid.canopy_height[rr, cc], cat, spacing, tree_codes)
    ok &= u[2] < lam0 / cap
    sel = np.nonzero(ok)[0]
    if len(sel) == 0:
        return None
    rr, cc, cat = rr[sel], cc[sel], cat[sel]
    h_local = np.maximum(grid.canopy_height[rr, cc], 3.0)
    # Lognormal height jitter around the local canopy height (mean canopy -> individual trees).
    h = h_local * np.exp(0.18 * _norm_ppf(u[3][sel]))
    arch = _leaf_adjusted(u[4][sel], cat, grid.leaf_type[rr, cc], cum)
    arch = np.where(np.isin(arch, [3, 4]), 0, arch)  # ground cover handled separately
    delta = spacing[cat] * crown_radius_codes(arch, h_local)
    return {"x": x[sel], "y": y[sel], "mark": u[5][sel], "cell": _cell_key(ci[sel], cj[sel]), "k": k[sel],
            "h": h, "arch": arch, "yaw": u[6][sel] * 2 * math.pi, "cat": cat, "delta": delta,
            "conf": grid.confidence[rr, cc]}


def _cell_key(i: int | np.ndarray, j: int | np.ndarray) -> int | np.ndarray:
    return (i + (1 << 20)) * (1 << 21) + (j + (1 << 20))


def _lambda0(coverage: np.ndarray, height: np.ndarray, cat: np.ndarray, spacing: np.ndarray,
             tree_codes: np.ndarray) -> np.ndarray:
    """Matérn-II candidate intensity giving the target stem density (stems / m2)."""
    h = np.maximum(np.asarray(height, dtype=float), 3.0)
    rc = crown_radius("broadleaf", h)
    sp = spacing[np.asarray(cat, dtype=np.int64)]
    delta = sp * rc
    target = np.asarray(coverage, dtype=float) / (math.pi * rc ** 2)
    cap = 0.95 / (math.pi * delta ** 2)
    lam = np.minimum(target, cap)
    a = math.pi * delta ** 2
    with np.errstate(divide="ignore", invalid="ignore"):
        lam0 = -np.log1p(-lam * a) / a
    lam0 = np.where(np.isin(cat, tree_codes), lam0, 0.0)
    return np.nan_to_num(lam0, nan=0.0, posinf=0.0)


# Side of the square blocks _matern_ii queries one at a time (metres).
MATERN_BLOCK = 256.0


def _matern_ii(x: np.ndarray, y: np.ndarray, mark: np.ndarray, delta: np.ndarray, obs: TreeTable) -> np.ndarray:
    """Keep a candidate unless a lower-mark candidate or an observed tree is within the hard core.

    Two candidates conflict when their distance is below the larger of their hard-core distances,
    so every conflict is found from the endpoint with the larger one. Candidates are queried block
    by block with that block's own largest hard core: memory follows the block, not one
    world-wide radius around every candidate. A conflict's loser is the higher mark (on equal
    marks, the higher index); the kept set does not depend on the query order.
    """
    n = len(x)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    bi = np.floor(x / MATERN_BLOCK).astype(np.int64)
    bj = np.floor(y / MATERN_BLOCK).astype(np.int64)
    order = np.lexsort((bj, bi))
    starts = np.flatnonzero(np.r_[True, (bi[order][1:] != bi[order][:-1]) | (bj[order][1:] != bj[order][:-1])])
    for s, e in zip(starts, np.r_[starts[1:], n]):
        idx = order[s:e]
        # Slightly beyond the largest hard core: pairs are confirmed below with the exact test.
        r = float(delta[idx].max()) * (1.0 + 1e-9) + 1e-9
        found = cKDTree(pts[idx]).sparse_distance_matrix(tree, r, output_type="ndarray")
        p, q = idx[found["i"]], found["j"]
        a, b = np.minimum(p, q), np.maximum(p, q)
        other = a != b
        a, b = a[other], b[other]
        d = np.hypot(x[a] - x[b], y[a] - y[b])
        close = d < np.maximum(delta[a], delta[b])
        a, b = a[close], b[close]
        keep[np.where(mark[a] > mark[b], a, b)] = False
    if len(obs):
        otree = cKDTree(np.column_stack([obs.x, obs.y]))
        r = np.maximum(delta, 1.5)
        idx = np.nonzero(keep)[0]
        if len(idx):
            near = otree.query_ball_point(pts[idx], r[idx], return_length=True)
            keep[idx[near > 0]] = False
    return keep


def _norm_ppf(u: np.ndarray) -> np.ndarray:
    from scipy.special import ndtri

    return ndtri(np.clip(u, 1e-6, 1 - 1e-6))


def _generated_table(C: dict[str, np.ndarray], terrain: Heightfield, cfg: VegetationConfig) -> TreeTable:
    n = len(C["x"])
    order = np.lexsort((C["k"], C["cell"]))
    C = {k: v[order] for k, v in C.items()}
    arch = [ARCHETYPES[a] for a in C["arch"]]
    h = np.clip(C["h"], 2.5, 60.0)
    crown = crown_radius_codes(C["arch"], h)
    trunk = trunk_radius_codes(C["arch"], h)
    gz = terrain.sample(C["x"], C["y"])
    ci = C["cell"] // (1 << 21) - (1 << 20)
    cj = C["cell"] % (1 << 21) - (1 << 20)
    ids = [f"tree:gen:{i}_{j}_{k}" for i, j, k in zip(ci.tolist(), cj.tolist(), C["k"].tolist())]
    seeds = hash_u64(cfg.seed, SALT_TREE_SEED, ci, cj, C["k"]) & np.uint64(0x7FFFFFFFFFFFFFFF)
    return TreeTable(
        ids=ids, x=C["x"], y=C["y"], ground_z=gz, height=h, crown_radius=crown, trunk_radius=trunk,
        yaw=C["yaw"], archetype=arch, species=[None] * n, category=[CATEGORIES[c] for c in C["cat"]],
        confidence=np.clip(C["conf"] * 0.5, 0.05, 0.5), observed=np.zeros(n, dtype=bool),
        seed=seeds, source=["procedural"] * n,
        source_id=[f"seed:{cfg.seed}"] * n, inferred=[["height", "crown_radius", "trunk_radius"]] * n,
    )


def _observed_table(observed: list[ObservedTree], grid: VegetationGrid, terrain: Heightfield,
                    cfg: VegetationConfig) -> TreeTable:
    rows = [t for t in sorted(observed, key=lambda t: t.id) if terrain.contains(t.x, t.y)]
    if not rows:
        return TreeTable()
    x = np.array([t.x for t in rows])
    y = np.array([t.y for t in rows])
    rr, cc, inside = grid.cell_of(x, y)
    local_h = np.where(inside, grid.canopy_height[rr, cc], 0.0)
    ids, arch, species, cat, h, crown, trunk, inferred, conf, src, sid, yaw, seeds = ([] for _ in range(13))
    for k, t in enumerate(rows):
        p = t.properties
        a = _observed_archetype(p)
        inf: list[str] = []
        hh = p.get("height")
        if hh is None:
            hh = float(local_h[k]) if local_h[k] >= 3.0 else 8.0
            inf.append("height")
        cr = p.get("crown_radius")
        if cr is None:
            cr = float(crown_radius(a, hh))
            inf.append("crown_radius")
        tr = p.get("trunk_radius")
        if tr is None:
            tr = float(trunk_radius(a, hh))
            inf.append("trunk_radius")
        ids.append(t.id)
        arch.append(a)
        species.append(p.get("species") or p.get("genus"))
        cat.append(CATEGORIES[int(grid.category[rr[k], cc[k]])] if inside[k] else "none")
        h.append(float(hh))
        crown.append(float(cr))
        trunk.append(float(tr))
        inferred.append(inf)
        conf.append(0.7 if t.from_row else 0.9)
        src.append(t.source)
        sid.append(t.source_id)
        s = stable_u64(cfg.seed, t.id) & 0x7FFFFFFFFFFFFFFF
        seeds.append(s)
        yaw.append((s % 3600) / 3600.0 * 2 * math.pi)
    table = TreeTable(
        ids=ids, x=x, y=y, ground_z=terrain.sample(x, y), height=np.array(h), crown_radius=np.array(crown),
        trunk_radius=np.array(trunk), yaw=np.array(yaw), archetype=arch, species=species, category=cat,
        confidence=np.array(conf), observed=np.ones(len(ids), dtype=bool), seed=np.array(seeds, dtype=np.uint64),
        source=src, source_id=sid, inferred=inferred,
    )
    # Trees interpolated along mapped tree rows are positioned by the compiler.
    for k, t in enumerate(rows):
        if t.from_row:
            table.observed[k] = False
            table.inferred[k] = sorted(set(table.inferred[k]) | {"position"})
    return table


def _observed_archetype(p: dict) -> str:
    genus = str(p.get("genus") or p.get("species") or "").split(" ")[0].lower()
    if genus in PALM_GENERA or "palm" in str(p.get("species_en", "")).lower():
        return "palm"
    if p.get("leaf_type") == "needleleaved":
        return "conifer"
    return "broadleaf"


SHRUB_CATEGORIES = ("shrubland", "wetland")
GRASS_CATEGORIES = ("grassland", "agricultural")


def ground_cover_scale(grid: VegetationGrid, cfg: VegetationConfig) -> float:
    """Density factor keeping the world's expected ground cover under ``max_instances``.

    A world-level constant (it counts the whole grid's shrub and grass cells), computed once
    and passed to every window's ``place_ground_cover`` so windows agree."""
    gc = cfg.ground_cover
    area_shrub = float(np.isin(grid.category, [CAT[c] for c in SHRUB_CATEGORIES]).sum()) * grid.resolution ** 2
    area_grass = float(np.isin(grid.category, [CAT[c] for c in GRASS_CATEGORIES]).sum()) * grid.resolution ** 2
    expected = area_shrub * gc.shrub_density + area_grass * gc.grass_density
    return min(1.0, gc.max_instances / expected) if expected > 0 else 1.0


def place_ground_cover(grid: VegetationGrid, terrain: Heightfield, exclusions: Exclusions,
                       cfg: VegetationConfig, bounds: tuple[float, float, float, float] | None = None,
                       scale: float | None = None) -> TreeTable:
    """Shrubs and grass clusters (no hard core; deterministic per sampling cell). ``scale``:
    ``ground_cover_scale(grid, cfg)``, computed here when not given."""
    gc = cfg.ground_cover
    if not gc.enabled:
        return TreeTable()
    shrub_cats = np.array([CAT[c] for c in SHRUB_CATEGORIES])
    grass_cats = np.array([CAT[c] for c in GRASS_CATEGORIES])
    if scale is None:
        scale = ground_cover_scale(grid, cfg)
    cs = cfg.cell_size
    xmin, ymin = grid.xmin, grid.ymin
    xmax = xmin + grid.shape[1] * grid.resolution
    ymax = ymin + grid.shape[0] * grid.resolution
    bx0, by0, bx1, by1 = bounds if bounds is not None else (xmin, ymin, xmax, ymax)
    lam = max(gc.shrub_density, gc.grass_density) * scale
    i0, i1 = math.floor(bx0 / cs), math.floor((bx1 - 1e-9) / cs)
    j0, j1 = math.floor(by0 / cs), math.floor((by1 - 1e-9) / cs)
    parts = []
    for jb in range(j0, j1 + 1, CELL_ROWS_PER_BATCH):
        # Cells in (j, i) order and candidates by index, the order of a cell-by-cell loop.
        jg, ig = np.meshgrid(np.arange(jb, min(jb + CELL_ROWS_PER_BATCH - 1, j1) + 1, dtype=np.int64),
                             np.arange(i0, i1 + 1, dtype=np.int64), indexing="ij")
        ci, cj = ig.ravel(), jg.ravel()
        n = poisson_icdf(hash_unit(cfg.seed, SALT_GROUND_COUNT, ci, cj), lam * cs * cs)
        if n.sum() == 0:
            continue
        rep = np.repeat(np.arange(len(n)), n)
        k = np.arange(int(n.sum()), dtype=np.int64) - np.repeat(np.cumsum(n) - n, n)
        ci, cj = ci[rep], cj[rep]
        u = [hash_unit(cfg.seed, SALT_GROUND_U, ci, cj, k, col) for col in range(5)]
        x = (ci + u[0]) * cs
        y = (cj + u[1]) * cs
        rr, cc, inside = grid.cell_of(x, y)
        cat = grid.category[rr, cc]
        is_shrub = np.isin(cat, shrub_cats)
        is_grass = np.isin(cat, grass_cats)
        dens = np.where(is_shrub, gc.shrub_density, np.where(is_grass, gc.grass_density, 0.0)) * scale
        ok = inside & (is_shrub | is_grass) & (u[2] < dens / max(lam, 1e-12))
        ok &= (x >= bx0) & (x < bx1) & (y >= by0) & (y < by1)
        s = np.nonzero(ok)[0]
        if len(s):
            arch = np.where(is_shrub[s], 3, 4)
            parts.append({"x": x[s], "y": y[s], "k": k[s], "i": ci[s], "j": cj[s], "arch": arch, "cat": cat[s],
                          "h": np.where(arch == 3, 1.6, 0.5) * (0.7 + 0.6 * u[3][s]), "yaw": u[4][s] * 2 * math.pi})
    if not parts:
        return TreeTable()
    G = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    keep = ~exclusions.contains(G["x"], G["y"]) & terrain.contains(G["x"], G["y"])
    keep &= terrain.slope_deg(G["x"], G["y"]) <= cfg.max_slope_deg
    G = {key: v[keep] for key, v in G.items()}
    n = len(G["x"])
    seeds = hash_u64(cfg.seed, SALT_GROUND_SEED, G["i"], G["j"], G["k"]) & np.uint64(0x7FFFFFFFFFFFFFFF)
    return TreeTable(
        ids=[f"tree:gen:{i}_{j}_g{k}" for i, j, k in zip(G["i"].tolist(), G["j"].tolist(), G["k"].tolist())],
        x=G["x"], y=G["y"], ground_z=terrain.sample(G["x"], G["y"]), height=G["h"],
        crown_radius=crown_radius_codes(G["arch"], G["h"]), trunk_radius=np.zeros(n), yaw=G["yaw"],
        archetype=[ARCHETYPES[a] for a in G["arch"]], species=[None] * n,
        category=[CATEGORIES[c] for c in G["cat"]], confidence=np.full(n, 0.2), observed=np.zeros(n, dtype=bool),
        seed=seeds, source=["procedural"] * n, source_id=[f"seed:{cfg.seed}"] * n,
        inferred=[["height", "crown_radius", "position"]] * n,
    )


def concat_tables(tables: list[TreeTable]) -> TreeTable:
    tables = [t for t in tables if len(t)]
    if not tables:
        return TreeTable()
    cat = lambda f: np.concatenate([getattr(t, f) for t in tables])  # noqa: E731
    lst = lambda f: [v for t in tables for v in getattr(t, f)]  # noqa: E731
    return TreeTable(
        ids=lst("ids"), x=cat("x"), y=cat("y"), ground_z=cat("ground_z"), height=cat("height"),
        crown_radius=cat("crown_radius"), trunk_radius=cat("trunk_radius"), yaw=cat("yaw"),
        archetype=lst("archetype"), species=lst("species"), category=lst("category"),
        confidence=cat("confidence"), observed=cat("observed"), seed=cat("seed"), source=lst("source"),
        source_id=lst("source_id"), inferred=lst("inferred"),
        rendered=None if all(t.rendered is None for t in tables) else
        np.concatenate([t.rendered_mask() for t in tables]),
    )
