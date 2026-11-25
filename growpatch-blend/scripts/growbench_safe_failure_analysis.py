"""Exploratory, development regions only (manual_ids): where does the locked tiled blend move surface onto a wrong sheet?

For every scoring ray (seed frame, 2-voxel grid, 12-128 voxels from the seed) that the blend moved, record the label
class before and after, and simple features of the surface-prediction profile along the seed normal.
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
DATA = ROOT / "results/growbench"   # the project data layout (see evidence/tiled/README.md)
PRESETS = ("oriented8_loss", "oriented8_th128_loss", "oriented8_th192_loss")
WIDE = 128


def peaks_along(volume, site, F, u, v, center, half=24):
    from scipy.ndimage import gaussian_filter1d, map_coordinates
    off = np.arange(-half, half + 1, dtype=np.float32)
    w = center[:, None] + off[None, :]
    P = (np.asarray(site["seed_xyz"]) + u[:, None, None] * F[0] + v[:, None, None] * F[1] + w[..., None] * F[2])
    idx = (P - site["origin_xyz"])[..., ::-1].reshape(-1, 3).T
    prof = map_coordinates(volume, idx, order=1, mode="constant", cval=0, prefilter=False).reshape(w.shape) / 255.0
    sig = gaussian_filter1d(prof, 0.85, axis=1)
    pk = np.zeros(sig.shape, bool)
    pk[:, 1:-1] = (sig[:, 1:-1] >= sig[:, :-2]) & (sig[:, 1:-1] > sig[:, 2:]) & (sig[:, 1:-1] > .15)
    out = []
    for i in range(len(center)):
        j = np.flatnonzero(pk[i])
        if len(j):
            b, m, a = sig[i, j - 1], sig[i, j], sig[i, j + 1]
            dz = np.clip((a - b) / (2 * np.maximum(2 * m - b - a, 1e-6)), -.5, .5)
            out.append(w[i, j] + dz)
        else:
            out.append(np.array([]))
    return out


def run(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    from growbench_geometry import faces, frame
    from growbench_regional import region_project
    from growbench_tiled_blend import tiled_blend
    preset, site, target = job
    i = site["id"]
    lab = DATA / "labels/manual_ids"
    ids = np.load(lab / "truth" / i / "instance_ids.npy")
    m7 = np.load(lab / "inputs" / i / "m7.npy")
    with np.load(DATA / "stock_manual" / preset / i / "surface.npz") as d:
        xyz, valid = d["xyz"].astype(np.float64), d["valid"].astype(bool)
    tiled, ok, _, _ = tiled_blend(xyz, valid, str(lab / "inputs" / i / "m7.npy"), site["origin_xyz"], 32, "shared")
    F = frame(site)
    ps_, pt_ = region_project(faces(xyz, valid)[0], site, WIDE), region_project(faces(tiled, ok)[0], site, WIDE)
    ax = np.arange(-WIDE, WIDE + 1, 2)
    u, v = np.meshgrid(ax, ax)
    rr = u * u + v * v
    us = np.isfinite(ps_.low) & (ps_.high - ps_.low <= 1)
    ut = np.isfinite(pt_.low) & (pt_.high - pt_.low <= 1)
    m = us & ut & (rr > 144) & (rr <= WIDE ** 2)
    s, t = ps_.center[m], pt_.center[m]
    moved = np.abs(t - s) > 0.05

    def cls(center, mask):
        xyzp = np.stack((u[mask], v[mask], center), -1) @ F
        pt = np.floor(xyzp + np.asarray(site["seed_xyz"]) - site["origin_xyz"] + .5).astype(int)[:, ::-1]
        inside = ((pt >= 0) & (pt < ids.shape)).all(1)
        val = np.zeros(len(pt), np.int64)
        val[inside] = ids[tuple(pt[inside].T)]
        return np.where(val == target, 1, np.where(val > 0, 2, 0))   # 1 target, 2 other, 0 unlabeled
    cs, ct = cls(s, m), cls(t, m)
    sel = moved
    uu, vv, ss, tt = u[m][sel], v[m][sel], s[sel], t[sel]
    pks = peaks_along(m7, site, F, uu.astype(np.float32), vv.astype(np.float32), ss.astype(np.float32))
    rows = []
    for k in range(len(ss)):
        p = pks[k]
        g = ss[k] + 2 * (tt[k] - ss[k])
        if len(p):
            ps = p[np.argmin(np.abs(p - ss[k]))]
            pg = p[np.argmin(np.abs(p - g))]
            others = p[np.abs(p - pg) > 1.0]
            ogap = float(np.min(np.abs(others - pg))) if len(others) else 99.0
            # nearest other peak to the stock point, excluding the graph's peak
            alt = p[np.abs(p - pg) > 1.0]
            d_alt = float(np.min(np.abs(alt - ss[k]))) if len(alt) else 99.0
            rows.append((int(cs[sel][k]), int(ct[sel][k]), float(tt[k] - ss[k]), float(abs(ps - pg) < 1.0),
                         ogap, float(abs(ss[k] - pg)), d_alt, len(p)))
        else:
            rows.append((int(cs[sel][k]), int(ct[sel][k]), float(tt[k] - ss[k]), 0.0, 99.0, 99.0, 99.0, 0))
    return preset, i, rows


if __name__ == "__main__":
    lab = DATA / "labels/manual_ids"
    seeds = json.loads((lab / "seeds.json").read_text())["sites"]
    targets = {r["id"]: int(r["target_id"]) for r in json.loads((lab / "refs.json").read_text())["sites"]}
    jobs = [(p, s, targets[s["id"]]) for p in PRESETS for s in seeds]
    allrows = []
    with ProcessPoolExecutor(20) as pool:
        for preset, i, rows in pool.map(run, jobs, chunksize=1):
            for r in rows:
                allrows.append((preset, i) + r)
            print(preset, i, len(rows), flush=True)
    out = Path(sys.argv[1])
    json.dump(allrows, open(out, "w"))
    print("rows", len(allrows))
