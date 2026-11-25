"""Ray-by-ray label transitions between official GrowPatch and the locked tiled blend (diagnostic only).

For each labeled region, both meshes are rasterized in the region's seed frame within 128 voxels. Each
ray is classified as target sheet, other named sheet, unassigned (label 0), ambiguous (multi-valued) or
absent. The area moving between classes is pooled per GrowPatch setting.
    python growbench_tiled_transitions.py manual_external
"""
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from growbench_geometry import faces, frame
from growbench_regional import region_project
from growbench_tiled_blend import tiled_blend
from growbench_tiled_eval import COHORTS, DATA, LOCK, OUT, PITCH, PRESETS, WIDE

CLASSES = ("target", "other", "unassigned", "ambiguous", "absent")


def classify(site, ref, p, uv, volume, target):
    u, v = uv
    rr = u * u + v * v
    known = np.isfinite(ref.low) & (ref.high - ref.low <= .5) & (ref.area_factor <= 2) & (rr <= WIDE ** 2) & (rr > 12 ** 2)
    present = np.isfinite(p.low)
    unique = present & (p.high - p.low <= 1)
    xyz = np.stack((u[unique], v[unique], p.center[unique]), -1) @ frame(site)
    pt = np.floor(xyz + np.asarray(site["seed_xyz"]) - site["origin_xyz"] + .5).astype(int)[:, ::-1]
    inside = ((pt >= 0) & (pt < volume.shape)).all(1)
    val = np.zeros(len(pt), np.int32)
    val[inside] = volume[tuple(pt[inside].T)]
    ids = np.zeros(known.shape, np.int32)
    ids[unique] = val
    cls = np.full(known.shape, 4)
    cls[present & ~unique] = 3
    cls[unique & (ids == 0)] = 2
    cls[unique & (ids != 0) & (ids != target)] = 1
    cls[unique & (ids == target)] = 0
    area = np.where(known, ref.area_factor, 0) * (2 * PITCH / 10000) ** 2
    return cls, area, (p.center if True else None), unique


def job(args):
    os.environ["OMP_NUM_THREADS"] = "1"
    cohort, preset, site, target, lock = args
    i = site["id"]
    lab = DATA / "labels" / cohort
    volume = np.load(lab / "truth" / i / "instance_ids.npy", mmap_mode="r")
    ref = region_project(np.load(lab / "truth" / i / "triangles.npy"), site, WIDE)
    axis = np.arange(-WIDE, WIDE + 1, 2)
    uv = np.meshgrid(axis, axis)
    with np.load(DATA / COHORTS[cohort] / preset / i / "surface.npz") as d:
        xyz, valid = d["xyz"].astype(np.float64), d["valid"].astype(bool)
    stock = region_project(faces(xyz, valid)[0], site, WIDE)
    out, ok, _, _ = tiled_blend(xyz, valid, lab / "inputs" / i / "m7.npy", site["origin_xyz"], lock["spacing_vox"], lock["mode"])
    tiled = region_project(faces(out, ok)[0], site, WIDE)
    a, area, ca, ua = classify(site, ref, stock, uv, volume, target)
    b, _, cb, ub = classify(site, ref, tiled, uv, volume, target)
    m = np.zeros((5, 5))
    np.add.at(m, (a.ravel(), b.ravel()), area.ravel())
    both = ua & ub & (area > 0)
    move = np.abs(cb[both] - ca[both])
    t2o = both & (a == 0) & (b == 1)
    return dict(preset=preset, id=i, matrix=m.tolist(),
                target_to_other_depth_change_vox=np.abs(cb[t2o] - ca[t2o]).tolist(),
                median_depth_change_vox=float(np.median(move)) if move.size else 0.0)


def main():
    cohort = sys.argv[1] if len(sys.argv) > 1 else "manual_external"
    lock = json.loads(LOCK.read_text())
    seeds = json.loads((DATA / "labels" / cohort / "seeds.json").read_text())["sites"]
    targets = {r["id"]: int(r["target_id"]) for r in json.loads((DATA / "labels" / cohort / "refs.json").read_text())["sites"]}
    jobs = [(cohort, p, s, targets[s["id"]], lock) for p in PRESETS for s in seeds]
    with ProcessPoolExecutor(20) as pool:
        rows = list(pool.map(job, jobs, chunksize=1))
    report = {}
    for p in PRESETS:
        m = sum(np.array(r["matrix"]) for r in rows if r["preset"] == p) * 100  # cm2 -> mm2
        report[p] = dict(classes=CLASSES, transition_mm2=m.round(5).tolist(),
                         per_region_target_to_other_mm2={r["id"]: round(np.array(r["matrix"])[0][1] * 100, 5) for r in rows if r["preset"] == p},
                         per_region_unassigned_to_other_mm2={r["id"]: round(np.array(r["matrix"])[2][1] * 100, 5) for r in rows if r["preset"] == p})
        print("==", p)
        print("from \\ to".ljust(12) + "".join(c.rjust(12) for c in CLASSES))
        for k, c in enumerate(CLASSES):
            print(c.ljust(12) + "".join(f"{x:12.4f}" for x in m[k]))
    dest = OUT / f"transitions_{cohort}.json"
    dest.write_text(json.dumps(dict(cohort=cohort, lock=lock, zone="12 < r <= 128 voxels", report=report, rows=rows), indent=1), encoding="utf-8")
    print("wrote", dest)


if __name__ == "__main__":
    main()
