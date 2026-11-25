"""Stricter versions of the tiled blend that only remove moves (research/growpatch-safe-blend-protocol.md).

The locked tiled blend (growbench_tiled_blend.py) is reused unchanged. This module adds:
- per-ray ridge features computed in each tile, from the same surface prediction the graph solver uses, and
- a transfer step that can require agreement among the overlapping tiles that see a vertex.
"""
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
from scipy.ndimage import gaussian_filter1d, map_coordinates

import growbench_tiled_blend as T
from growbench_geometry import RADIUS, STEP, faces, frame, project
from growbench_recover import solve

WINDOW = 24   # voxels either side of GrowPatch's depth searched for ridges


def ridge_features(volume, site, stock_center, landing, rays):
    """For the rays in `rays` (bool grid): distance from the landing ridge to the nearest other ridge, and whether
    the ridge nearest to GrowPatch's depth is the landing ridge. Ridges as in growbench_recover.evidence: local
    maxima of the prediction (normalised to its maximum, smoothed 0.85 voxels along the ray) above 0.15."""
    clear = np.full(rays.shape, np.nan)
    nearest = np.zeros(rays.shape, bool)
    iv, iu = np.nonzero(rays)
    if not len(iv):
        return clear, nearest
    F = frame(site)
    ax = np.arange(-RADIUS, RADIUS + 1, STEP, dtype=np.float64)
    off = np.arange(-WINDOW, WINDOW + 1, dtype=np.float64)
    s = stock_center[iv, iu]
    w = s[:, None] + off[None, :]
    P = (np.asarray(site["seed_xyz"]) + ax[iu][:, None, None] * F[0] + ax[iv][:, None, None] * F[1] + w[..., None] * F[2])
    idx = (P - np.asarray(site["origin_xyz"]))[..., ::-1].reshape(-1, 3).T
    vmax = float(volume.max()) or 1.0
    prof = map_coordinates(volume, idx, order=1, mode="constant", cval=0.0, prefilter=False).reshape(w.shape) / vmax
    sig = gaussian_filter1d(prof, 0.85, axis=1)
    pk = np.zeros(sig.shape, bool)
    pk[:, 1:-1] = (sig[:, 1:-1] >= sig[:, :-2]) & (sig[:, 1:-1] > sig[:, 2:]) & (sig[:, 1:-1] > .15)
    b = np.roll(sig, 1, axis=1)
    a = np.roll(sig, -1, axis=1)
    dz = np.clip((a - b) / (2 * np.maximum(2 * sig - b - a, 1e-6)), -.5, .5)
    Z = np.where(pk, w + dz, np.nan)
    has = pk.any(1)
    land = landing[iv, iu]
    dl = np.where(np.isfinite(Z), np.abs(Z - land[:, None]), np.inf)
    own = np.argmin(dl, axis=1)
    zown = Z[np.arange(len(iv)), own]
    do = np.where(np.isfinite(Z), np.abs(Z - zown[:, None]), np.inf)
    do[np.arange(len(iv)), own] = np.inf
    c = np.min(do, axis=1)
    ds = np.where(np.isfinite(Z), np.abs(Z - s[:, None]), np.inf)
    near_own = np.argmin(ds, axis=1) == own
    clear[iv, iu] = np.where(has, c, np.nan)
    nearest[iv, iu] = has & near_own
    return clear, nearest


def tile_shift_features(task):
    """The locked tile_shift, plus ridge features on the rays where the locked rule would move."""
    site, tri = task["site"], task["triangles"]
    stock = project(tri, site)
    with TemporaryDirectory() as tmp:
        try:
            solve(T._VOLUME, site, Path(tmp))
        except ValueError as exc:
            return dict(index=task["index"], abstained=str(exc))
        with np.load(Path(tmp) / "graph.npz") as g:
            gtri, _ = faces(g["xyz"], g["valid"])
    graph = project(gtri, site)
    unique_s = np.isfinite(stock.low) & (stock.high - stock.low <= 1)
    unique_g = np.isfinite(graph.low) & (graph.high - graph.low <= 1)
    both = unique_s & unique_g
    gap = np.zeros(stock.low.shape)
    gap[both] = graph.center[both] - stock.center[both]
    shared = both & (np.abs(gap) <= T.LIMIT)
    center = np.where(unique_s, stock.center, 0.0)
    clear, nearest = ridge_features(T._VOLUME, site, center, center + gap, shared)
    return dict(index=task["index"], shift=np.where(shared, 0.5 * gap, 0.0).astype(np.float32),
                center=np.where(unique_s, stock.center, np.nan).astype(np.float32), unique=unique_s, shared=shared,
                clear=clear.astype(np.float32), nearest=nearest)


def solve_tiles_features(xyz, valid, volume_path, origin_xyz, spacing_vox=32):
    sites, lattice_nodes, step = T.tile_seeds(xyz, valid, spacing_vox)
    for k, s in enumerate(sites):
        s["id"] = f"tile_{k:04d}"
        s["origin_xyz"] = [float(c) for c in origin_xyz]
    tri_all, _ = faces(xyz, valid)
    centroids = tri_all.mean(1)
    T._init_worker(str(volume_path))
    results = []
    for k, s in enumerate(sites):
        near = np.linalg.norm(centroids - np.asarray(s["seed_xyz"]), axis=1) <= RADIUS * 1.75 + 2 * step
        results.append(tile_shift_features(dict(index=k, site=s, triangles=tri_all[near])))
    return sites, results, dict(lattice_nodes=lattice_nodes, grid_step_vox=step)


def ray_mask(res, rule):
    m = res["shared"].copy()
    clear = rule.get("clearance")
    if clear is not None:
        m &= np.nan_to_num(res["clear"], nan=np.inf) >= clear
    if rule.get("nearest"):
        m &= res["nearest"]
    return m


def transfer_safe(points, results, sites, rule):
    """Shared-only transfer with optional ray masks and tile consensus. With rule = {} it equals the locked transfer."""
    acc = np.zeros_like(points)
    w_sh = np.zeros(len(points))
    w_vis = np.zeros(len(points))
    s1 = np.zeros(len(points))
    s2 = np.zeros(len(points))
    for res in results:
        if "shift" not in res:
            continue
        site = sites[res["index"]]
        F = frame(site)
        seed = np.asarray(site["seed_xyz"], float)
        lo, hi = seed - RADIUS * 1.75, seed + RADIUS * 1.75
        cand = np.flatnonzero(((points >= lo) & (points <= hi)).all(1))
        if not len(cand):
            continue
        loc = (points[cand] - seed) @ F.T
        r = np.hypot(loc[:, 0], loc[:, 1])
        inside = r < RADIUS
        cand, loc, r = cand[inside], loc[inside], r[inside]
        fu, fv = (loc[:, 0] + RADIUS) / STEP, (loc[:, 1] + RADIUS) / STEP
        iu = np.clip(np.rint(fu).astype(int), 0, T.N_RAYS - 1)
        iv = np.clip(np.rint(fv).astype(int), 0, T.N_RAYS - 1)
        center = res["center"][iv, iu]
        visible = res["unique"][iv, iu] & np.isfinite(center) & (np.abs(loc[:, 2] - np.nan_to_num(center)) <= T.VISIBLE_TOL)
        if not visible.any():
            continue
        w = np.cos(np.pi * r / (2 * RADIUS)) ** 2
        w_vis[cand[visible]] += w[visible]
        mask = ray_mask(res, rule)
        sh = visible & mask[iv, iu]
        if not sh.any():
            continue
        c, fu2, fv2, w2 = cand[sh], fu[sh], fv[sh], w[sh]
        s = map_coordinates(res["shift"], [fv2, fu2], order=1, mode="nearest")
        acc[c] += (w2 * s)[:, None] * F[2]
        w_sh[c] += w2
        s1[c] += w2 * s
        s2[c] += w2 * s * s
    disp = np.zeros_like(points)
    has = w_sh > 0
    q = rule.get("consensus")
    if q is not None:
        frac = np.where(w_vis > 0, w_sh / np.maximum(w_vis, 1e-12), 0)
        mean = np.where(has, s1 / np.maximum(w_sh, 1e-12), 0)
        sd = np.sqrt(np.maximum(np.where(has, s2 / np.maximum(w_sh, 1e-12), 0) - mean ** 2, 0))
        has &= (frac >= q - 1e-9) & (sd <= rule.get("spread", 1.0) + 1e-9)
    disp[has] = acc[has] / w_sh[has, None]
    return disp


def safe_blend(xyz, valid, solved, rule):
    sites, results, lattice = solved
    k = max(1, int(round(lattice["grid_step_vox"] / STEP)))
    up, ok = T.upsample(xyz, valid, k)
    out = up.copy()
    out[ok] = up[ok] + transfer_safe(up[ok], results, sites, rule)
    return out, ok
