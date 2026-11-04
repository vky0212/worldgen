"""Deterministic, tile-independent tree placement.

Candidates are drawn per *sampling cell* of a global lattice from a random
stream seeded only by ``(seed, cell index)``; their attributes depend only on
the global vegetation grid at the candidate position. A candidate survives if

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
from worldgen.hashing import rng_for, stable_u64
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
    idx = np.array([np.searchsorted(cum[c], u, side="right") for c, u in zip(cat, arch_u)], dtype=np.int64)
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

    # Interaction radius bound for the halo.
    max_h = float(np.nanmax(grid.canopy_height)) if grid.canopy_height.size else 0.0
    r_int = float(np.max(spacing)) * float(crown_radius("broadleaf", max(max_h * 1.5, 2.0)))
    hx0, hy0, hx1, hy1 = bx0 - r_int, by0 - r_int, bx1 + r_int, by1 + r_int

    cand = {k: [] for k in ("x", "y", "mark", "cell", "k", "h", "arch", "yaw", "cat", "delta", "conf")}
    i0, i1 = math.floor(max(hx0, xmin) / cs), math.floor(min(hx1, xmax - 1e-9) / cs)
    j0, j1 = math.floor(max(hy0, ymin) / cs), math.floor(min(hy1, ymax - 1e-9) / cs)
    res = grid.resolution
    for j in range(j0, j1 + 1):
        for i in range(i0, i1 + 1):
            # Upper bound of the target intensity over grid cells touching this sampling cell.
            r0, c0, _ = grid.cell_of(np.array([i * cs]), np.array([j * cs]))
            r1, c1, _ = grid.cell_of(np.array([(i + 1) * cs - 1e-6]), np.array([(j + 1) * cs - 1e-6]))
            sl = (slice(int(r0[0]), int(r1[0]) + 1), slice(int(c0[0]), int(c1[0]) + 1))
            cats = grid.category[sl]
            if not np.isin(cats, tree_codes).any():
                continue
            lam0_cap = _lambda0(grid.coverage[sl], grid.canopy_height[sl], cats, spacing, tree_codes).max()
            if not np.isfinite(lam0_cap) or lam0_cap <= 0:
                continue
            rng = rng_for(cfg.seed, "trees", i, j)
            n = int(rng.poisson(lam0_cap * cs * cs))
            if n == 0:
                continue
            u = rng.random((n, 7))
            x = (i + u[:, 0]) * cs
            y = (j + u[:, 1]) * cs
            rr, cc, inside = grid.cell_of(x, y)
            cat = grid.category[rr, cc]
            ok = inside & np.isin(cat, tree_codes)
            lam0 = _lambda0(grid.coverage[rr, cc], grid.canopy_height[rr, cc], cat, spacing, tree_codes)
            ok &= u[:, 2] < lam0 / lam0_cap
            if not ok.any():
                continue
            h_local = np.maximum(grid.canopy_height[rr, cc], 3.0)
            # Lognormal height jitter around the local canopy height (mean canopy -> individual trees).
            h = h_local * np.exp(0.18 * _norm_ppf(u[:, 3]))
            arch = _leaf_adjusted(u[:, 4], cat, grid.leaf_type[rr, cc], cum)
            arch = np.where(np.isin(arch, [3, 4]), 0, arch)  # ground cover handled separately
            delta = np.array([spacing[c] for c in cat]) * np.array(
                [float(crown_radius(ARCHETYPES[a], hh)) for a, hh in zip(arch, h_local)])
            sel = np.nonzero(ok)[0]
            cand["x"].append(x[sel]); cand["y"].append(y[sel]); cand["mark"].append(u[sel, 5])
            cand["cell"].append(np.full(len(sel), _cell_key(i, j), dtype=np.int64))
            cand["k"].append(sel); cand["h"].append(h[sel]); cand["arch"].append(arch[sel])
            cand["yaw"].append(u[sel, 6] * 2 * math.pi); cand["cat"].append(cat[sel])
            cand["delta"].append(delta[sel]); cand["conf"].append(grid.confidence[rr, cc][sel])
    if not cand["x"]:
        gen = TreeTable()
    else:
        C = {k: np.concatenate(v) for k, v in cand.items()}
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


def _cell_key(i: int, j: int) -> int:
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


def _matern_ii(x: np.ndarray, y: np.ndarray, mark: np.ndarray, delta: np.ndarray, obs: TreeTable) -> np.ndarray:
    """Keep a candidate unless a lower-mark candidate or an observed tree is within the hard core."""
    n = len(x)
    keep = np.ones(n, dtype=bool)
    if n == 0:
        return keep
    pts = np.column_stack([x, y])
    tree = cKDTree(pts)
    rmax = float(delta.max())
    pairs = tree.query_pairs(rmax, output_type="ndarray")
    if len(pairs):
        a, b = pairs[:, 0], pairs[:, 1]
        d = np.hypot(x[a] - x[b], y[a] - y[b])
        close = d < np.maximum(delta[a], delta[b])
        a, b = a[close], b[close]
        loser = np.where(mark[a] > mark[b], a, b)
        keep[loser] = False
    if len(obs):
        otree = cKDTree(np.column_stack([obs.x, obs.y]))
        r = np.maximum(delta, 1.5)
        for k in np.nonzero(keep)[0]:
            if otree.query_ball_point(pts[k], r[k]):
                keep[k] = False
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
    crown = np.array([float(crown_radius(a, hh)) for a, hh in zip(arch, h)])
    trunk = np.array([float(trunk_radius(a, hh)) for a, hh in zip(arch, h)])
    gz = terrain.sample(C["x"], C["y"])
    ids = []
    seeds = []
    for cell, k in zip(C["cell"], C["k"]):
        i, j = int(cell // (1 << 21)) - (1 << 20), int(cell % (1 << 21)) - (1 << 20)
        ids.append(f"tree:gen:{i}_{j}_{int(k)}")
        seeds.append(stable_u64(cfg.seed, "tree", i, j, int(k)) & 0x7FFFFFFFFFFFFFFF)
    return TreeTable(
        ids=ids, x=C["x"], y=C["y"], ground_z=gz, height=h, crown_radius=crown, trunk_radius=trunk,
        yaw=C["yaw"], archetype=arch, species=[None] * n, category=[CATEGORIES[c] for c in C["cat"]],
        confidence=np.clip(C["conf"] * 0.5, 0.05, 0.5), observed=np.zeros(n, dtype=bool),
        seed=np.array(seeds, dtype=np.uint64), source=["procedural"] * n,
        source_id=[f"seed:{cfg.seed}" for _ in range(n)], inferred=[["height", "crown_radius", "trunk_radius"]] * n,
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


def place_ground_cover(grid: VegetationGrid, terrain: Heightfield, exclusions: Exclusions,
                       cfg: VegetationConfig, bounds: tuple[float, float, float, float] | None = None) -> TreeTable:
    """Shrubs and grass clusters (no hard core; deterministic per sampling cell)."""
    gc = cfg.ground_cover
    if not gc.enabled:
        return TreeTable()
    shrub_cats = np.array([CAT["shrubland"], CAT["wetland"]])
    grass_cats = np.array([CAT["grassland"], CAT["agricultural"]])
    area_shrub = float(np.isin(grid.category, shrub_cats).sum()) * grid.resolution ** 2
    area_grass = float(np.isin(grid.category, grass_cats).sum()) * grid.resolution ** 2
    expected = area_shrub * gc.shrub_density + area_grass * gc.grass_density
    scale = min(1.0, gc.max_instances / expected) if expected > 0 else 1.0
    cs = cfg.cell_size
    xmin, ymin = grid.xmin, grid.ymin
    xmax = xmin + grid.shape[1] * grid.resolution
    ymax = ymin + grid.shape[0] * grid.resolution
    bx0, by0, bx1, by1 = bounds if bounds is not None else (xmin, ymin, xmax, ymax)
    xs, ys, ks, cells, arch, h, yaw = [], [], [], [], [], [], []
    for j in range(math.floor(by0 / cs), math.floor((by1 - 1e-9) / cs) + 1):
        for i in range(math.floor(bx0 / cs), math.floor((bx1 - 1e-9) / cs) + 1):
            rng = rng_for(cfg.seed, "ground", i, j)
            lam = max(gc.shrub_density, gc.grass_density) * scale
            n = int(rng.poisson(lam * cs * cs))
            if n == 0:
                continue
            u = rng.random((n, 5))
            x = (i + u[:, 0]) * cs
            y = (j + u[:, 1]) * cs
            rr, cc, inside = grid.cell_of(x, y)
            cat = grid.category[rr, cc]
            is_shrub = np.isin(cat, shrub_cats)
            is_grass = np.isin(cat, grass_cats)
            dens = np.where(is_shrub, gc.shrub_density, np.where(is_grass, gc.grass_density, 0.0)) * scale
            ok = inside & (is_shrub | is_grass) & (u[:, 2] < dens / max(lam, 1e-12))
            ok &= (x >= bx0) & (x < bx1) & (y >= by0) & (y < by1)
            for k in np.nonzero(ok)[0]:
                xs.append(x[k]); ys.append(y[k]); ks.append(int(k)); cells.append((i, j))
                a = "shrub" if is_shrub[k] else "grass"
                arch.append(a)
                base = 1.6 if a == "shrub" else 0.5
                h.append(base * (0.7 + 0.6 * u[k, 3]))
                yaw.append(u[k, 4] * 2 * math.pi)
    if not xs:
        return TreeTable()
    X, Y = np.array(xs), np.array(ys)
    keep = ~exclusions.contains(X, Y) & terrain.contains(X, Y)
    keep &= terrain.slope_deg(X, Y) <= cfg.max_slope_deg
    idx = np.nonzero(keep)[0]
    n = len(idx)
    H = np.array(h)[idx]
    A = [arch[k] for k in idx]
    return TreeTable(
        ids=[f"tree:gen:{cells[k][0]}_{cells[k][1]}_g{ks[k]}" for k in idx], x=X[idx], y=Y[idx],
        ground_z=terrain.sample(X[idx], Y[idx]), height=H,
        crown_radius=np.array([float(crown_radius(a, hh)) for a, hh in zip(A, H)]),
        trunk_radius=np.zeros(n), yaw=np.array(yaw)[idx], archetype=A, species=[None] * n,
        category=[CATEGORIES[int(grid.category[r, c])] for r, c in zip(*grid.cell_of(X[idx], Y[idx])[:2])],
        confidence=np.full(n, 0.2), observed=np.zeros(n, dtype=bool),
        seed=np.array([stable_u64(cfg.seed, "ground", cells[k][0], cells[k][1], ks[k]) & 0x7FFFFFFFFFFFFFFF
                       for k in idx], dtype=np.uint64),
        source=["procedural"] * n, source_id=[f"seed:{cfg.seed}"] * n,
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
