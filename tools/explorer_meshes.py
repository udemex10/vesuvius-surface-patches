"""Regenerate the locked tiled-blend meshes for the 16 held-out regions and check them against the published scores.

    python tools/explorer_meshes.py --data <project>/results/growbench --out <work>/explorer-meshes

For each region and GrowPatch cutoff this reruns `tiled_blend` with the locked setting
(`growpatch-blend/scripts/tiled_blend_lock.json`), rescored with the evaluator's own `score_all`, and compares
stock and tiled rows with `evidence/tiled/heldout_manual_external.json`. It stops if any value differs.
The meshes feed `tools/build_explorer.py`.
"""
import argparse
import json
import os
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "growpatch-blend" / "scripts"))

PRESETS = ("oriented8_loss", "oriented8_th128_loss", "oriented8_th192_loss")
COHORT = "manual_external"


def run_case(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    from growbench_geometry import faces, project
    from growbench_regional import region_project
    from growbench_tiled_blend import tiled_blend
    from growbench_tiled_eval import WIDE, score_all

    data, out, preset, site, target, spacing, mode = job
    i = site["id"]
    lab = Path(data) / "labels" / COHORT
    volume = np.load(lab / "truth" / i / "instance_ids.npy", mmap_mode="r")
    ref_tri = np.load(lab / "truth" / i / "triangles.npy")
    ref96, ref_w = project(ref_tri, site), region_project(ref_tri, site, WIDE)
    axis = np.arange(-WIDE, WIDE + 1, 2)
    uv = np.meshgrid(axis, axis)
    with np.load(Path(data) / "stock_manual_external" / preset / i / "surface.npz") as d:
        xyz, valid = d["xyz"].astype(np.float64), d["valid"].astype(bool)
    tiled, ok, _, rep = tiled_blend(xyz, valid, str(lab / "inputs" / i / "m7.npy"), site["origin_xyz"], spacing, mode)
    rows = {"stock": score_all(faces(xyz, valid)[0], site, ref96, ref_w, uv, volume, target),
            "tiled": score_all(faces(tiled, ok)[0], site, ref96, ref_w, uv, volume, target)}
    dest = Path(out) / preset
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f"{i}.npz", stock_xyz=xyz, stock_valid=valid, tiled_xyz=tiled, tiled_valid=ok)
    return preset, i, rows, rep


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True, help="project results/growbench folder (labels, stock meshes)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    lock = json.loads((REPO / "growpatch-blend" / "scripts" / "tiled_blend_lock.json").read_text(encoding="utf-8"))
    spacing, mode = lock["spacing_vox"], lock["mode"]
    published = json.loads((REPO / "evidence" / "tiled" / "heldout_manual_external.json").read_text(encoding="utf-8"))
    pub = {(c["preset"], c["id"]): c["rows"] for c in published["cases"]}
    lab = Path(a.data) / "labels" / COHORT
    seeds = json.loads((lab / "seeds.json").read_text(encoding="utf-8"))["sites"]
    targets = {r["id"]: int(r["target_id"]) for r in json.loads((lab / "refs.json").read_text(encoding="utf-8"))["sites"]}
    jobs = [(a.data, a.out, p, s, targets[s["id"]], spacing, mode) for p in PRESETS for s in seeds]
    worst = 0.0
    with ProcessPoolExecutor(a.workers) as pool:
        for preset, i, rows, rep in pool.map(run_case, jobs, chunksize=1):
            ref = pub[(preset, i)]
            for mine, theirs in (("stock", "stock"), ("tiled", f"tiled_{spacing}_{mode}")):
                for zone, vals in rows[mine].items():
                    for k, v in vals.items():
                        worst = max(worst, abs(v - ref[theirs][zone][k]))
            print(f"{preset} {i}: {rep['tiles_solved']}/{rep['tiles']} tiles solved, max |diff| so far {worst:.3g}", flush=True)
    if worst > 1e-12:
        sys.exit(f"Regenerated meshes do not reproduce the published scores (max diff {worst}).")
    print(f"All {len(jobs)} cases reproduce the published stock and tiled scores exactly (max diff {worst}).")


if __name__ == "__main__":
    main()
