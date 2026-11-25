"""Score the stricter tiled-blend candidates (research/growpatch-safe-blend-protocol.md).

    python growbench_safe_eval.py dev        # 16 manual_ids regions, 8 candidates, writes the lock
    python growbench_safe_eval.py heldout    # 16 manual_external regions, locked candidate (third look)
    python growbench_safe_eval.py fresh      # 4 manual_fresh cubes, locked candidate (descriptive)
"""
import argparse
import hashlib
import json
import os
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from growbench_geometry import faces, project
from growbench_regional import region_project
from growbench_safe_blend import safe_blend, solve_tiles_features
from growbench_tiled_eval import COHORTS, PRESETS, WIDE, bootstrap, compare, pooled, score_all

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/growbench"
OUT = DATA / "safe"
PROTOCOL = ROOT / "research/growpatch-safe-blend-protocol.md"
LOCK = Path(__file__).with_name("safe_blend_lock.json")
CANDIDATES = {
    "locked": {},
    "clearance_10": {"clearance": 10.0},
    "clearance_14": {"clearance": 14.0},
    "nearest_ridge": {"nearest": True},
    "consensus_0.5": {"consensus": 0.5, "spread": 1.0},
    "consensus_0.75": {"consensus": 0.75, "spread": 1.0},
    "consensus_1.0": {"consensus": 1.0, "spread": 1.0},
    "consensus_0.75_nearest": {"consensus": 0.75, "spread": 1.0, "nearest": True},
}


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def run_case(job):
    os.environ["OMP_NUM_THREADS"] = "1"
    cohort, preset, site, target, rules = job
    i = site["id"]
    lab = DATA / "labels" / cohort
    volume_path = lab / "inputs" / i / "m7.npy"
    volume = np.load(lab / "truth" / i / "instance_ids.npy", mmap_mode="r")
    ref_tri = np.load(lab / "truth" / i / "triangles.npy")
    ref96, ref_w = project(ref_tri, site), region_project(ref_tri, site, WIDE)
    axis = np.arange(-WIDE, WIDE + 1, 2)
    uv = np.meshgrid(axis, axis)
    with np.load(DATA / COHORTS[cohort] / preset / i / "surface.npz") as d:
        xyz, valid = d["xyz"].astype(np.float64), d["valid"].astype(bool)
    rows = {"stock": score_all(faces(xyz, valid)[0], site, ref96, ref_w, uv, volume, target)}
    solved = solve_tiles_features(xyz, valid, volume_path, site["origin_xyz"], 32)
    meshes = {}
    for name, rule in rules.items():
        out, ok = safe_blend(xyz, valid, solved, rule)
        rows[name] = score_all(faces(out, ok)[0], site, ref96, ref_w, uv, volume, target)
        meshes[name] = (out.astype(np.float32), ok)
    dest = OUT / "meshes" / cohort / preset
    dest.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(dest / f"{i}.npz", **{f"{k}_xyz": v[0] for k, v in meshes.items()}, **{f"{k}_valid": v[1] for k, v in meshes.items()})
    return dict(cohort=cohort, preset=preset, id=i, rows=rows,
                tiles=len(solved[0]), tiles_solved=sum("shift" in r for r in solved[1]))


def run(cohort, rules, workers):
    lab = DATA / "labels" / cohort
    seeds = json.loads((lab / "seeds.json").read_text())["sites"]
    targets = {r["id"]: int(r["target_id"]) for r in json.loads((lab / "refs.json").read_text())["sites"]}
    jobs = [(cohort, p, s, targets[s["id"]], rules) for p in PRESETS for s in seeds]
    started = time.time()
    results = []
    with ProcessPoolExecutor(workers) as pool:
        for r in pool.map(run_case, jobs, chunksize=1):
            results.append(r)
            print(f"{r['preset']} {r['id']} done ({len(results)}/{len(jobs)}, {time.time()-started:.0f}s)", flush=True)
    by = {p: {r["id"]: r["rows"] for r in results if r["preset"] == p} for p in PRESETS}
    summary = {p: {k: {z: compare(by[p], "stock", k, z) for z in ("r96", "r128", "annulus")} for k in rules} for p in PRESETS}
    return results, summary, round(time.time() - started, 1)


def gate_ok(summary, key, zone="r128"):
    return all(summary[p][key][zone]["gate_other_share_named"] for p in PRESETS)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("dev", "heldout", "fresh"))
    ap.add_argument("--workers", type=int, default=20)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = dict(protocol_sha256=file_hash(PROTOCOL), eval_sha256=file_hash(Path(__file__)),
                      blend_sha256=file_hash(Path(__file__).with_name("growbench_safe_blend.py")),
                      tiled_sha256=file_hash(Path(__file__).with_name("growbench_tiled_blend.py")))
    if a.stage == "dev":
        if LOCK.exists():
            sys.exit("Lock exists; the development choice has already been made.")
        results, summary, secs = run("manual_ids", CANDIDATES, a.workers)
        ranking = []
        for k in CANDIDATES:
            gain = float(np.mean([summary[p][k]["r128"]["connected_gain_percent"] for p in PRESETS]))
            ranking.append(dict(candidate=k, gate_r128_all_cutoffs=gate_ok(summary, k), mean_r128_gain_percent=gain,
                                other_share=[(summary[p][k]["r128"]["base"]["other_share_named"], summary[p][k]["r128"]["candidate"]["other_share_named"]) for p in PRESETS]))
        passing = [r for r in ranking if r["gate_r128_all_cutoffs"] and r["candidate"] != "locked"]
        chosen = max(passing, key=lambda r: (r["mean_r128_gain_percent"], -list(CANDIDATES).index(r["candidate"]))) if passing else None
        (OUT / "dev_manual_ids.json").write_text(json.dumps(dict(summary=summary, cases=results, ranking=ranking, seconds=secs, **provenance), indent=1))
        lock = dict(chosen=chosen["candidate"] if chosen else None, rule=CANDIDATES[chosen["candidate"]] if chosen else None,
                    ranking=ranking, locked_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **provenance)
        LOCK.write_text(json.dumps(lock, indent=1))
        for r in ranking:
            print(f"{r['candidate']:24s} gate={'pass' if r['gate_r128_all_cutoffs'] else 'FAIL'}  gain={r['mean_r128_gain_percent']:6.2f}%  "
                  + "  ".join(f"{100*b:.2f}->{100*c:.2f}" for b, c in r["other_share"]))
        print("chosen:", lock["chosen"])
        return
    if not LOCK.exists():
        sys.exit("No lock; run the development stage first.")
    lock = json.loads(LOCK.read_text())
    if lock["chosen"] is None:
        sys.exit("No candidate passed at development; the protocol stops here.")
    cohort = "manual_external" if a.stage == "heldout" else "manual_fresh"
    target = OUT / f"{a.stage}_{cohort}.json"
    if target.exists():
        sys.exit(f"{target.name} exists; this run is done once.")
    rules = {"locked": {}, lock["chosen"]: lock["rule"]}
    results, summary, secs = run(cohort, rules, a.workers)
    target.write_text(json.dumps(dict(summary=summary, cases=results, seconds=secs, lock=lock, **provenance), indent=1))
    k = lock["chosen"]
    for p in PRESETS:
        for z in ("r96", "r128", "annulus"):
            c = summary[p][k][z]
            print(f"{p:22s} {z:7s} {100*c['base']['connected']:.2f} -> {100*c['candidate']['connected']:.2f} mm2 "
                  f"({c['connected_gain_percent']:+.1f}%, {c['better']}/{c['tie']}/{c['worse']}, CI {100*c['interval_95_cm2'][0]:+.2f}..{100*c['interval_95_cm2'][1]:+.2f}) "
                  f"other share {100*c['base']['other_share_named']:.2f}->{100*c['candidate']['other_share_named']:.2f} gate={'pass' if c['gate_other_share_named'] else 'FAIL'}")


if __name__ == "__main__":
    main()
