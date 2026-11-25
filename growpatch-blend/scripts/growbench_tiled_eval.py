"""Score the whole-mesh tiled blend on manually labeled PHerc. Paris. 4 regions.

Protocol: research/growpatch-tiled-blend-protocol.md (pre-registered before any tiled scoring).
    python growbench_tiled_eval.py dev        # 16 manual_ids regions, 4 candidates, writes the lock
    python growbench_tiled_eval.py heldout    # 16 manual_external regions, locked candidate only
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
from scipy.ndimage import label

from growbench_geometry import faces, frame, project
from growbench_identity import identities
from growbench_regional import displaced, region_project
from growbench_tiled_blend import RADIUS, solve_tiles, tile_shift, tiled_blend, transfer, upsample, _init_worker

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT / "results/growbench"
OUT = DATA / "tiled"
PROTOCOL = ROOT / "research/growpatch-tiled-blend-protocol.md"
LOCK = Path(__file__).with_name("tiled_blend_lock.json")
PRESETS = ("oriented8_loss", "oriented8_th128_loss", "oriented8_th192_loss")
CUTOFF = {"oriented8_loss": 1, "oriented8_th128_loss": 128, "oriented8_th192_loss": 192}
COHORTS = {"manual_ids": "stock_manual", "manual_external": "stock_manual_external", "manual_fresh": "stock_manual_fresh"}
CANDIDATES = [(32, "visible"), (32, "shared"), (64, "visible"), (64, "shared")]
WIDE = 128
PITCH = 7.91
KEYS = ("reference", "accepted", "correct", "incorrect", "connected", "known_other", "unassigned", "ambiguous")


def file_hash(path):
    with open(path, "rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def wide_measure(site, ref, p, uv, volume, target):
    """Regional instance-ID scorer on a 128-voxel disc, split into the full disc and the 96-128 annulus."""
    u, v = uv
    rr = u * u + v * v
    known = np.isfinite(ref.low) & (ref.high - ref.low <= .5) & (ref.area_factor <= 2) & (rr <= WIDE ** 2)
    a = np.where(known, ref.area_factor, 0) * (2 * PITCH / 10000) ** 2
    use = known & np.isfinite(p.low)
    unique = use & (p.high - p.low <= 1)
    xyz = np.stack((u[unique], v[unique], p.center[unique]), -1) @ frame(site)
    pt = np.floor(xyz + np.asarray(site["seed_xyz"]) - site["origin_xyz"] + .5).astype(int)[:, ::-1]
    inside = ((pt >= 0) & (pt < volume.shape)).all(1)
    val = np.zeros(len(pt), np.int32)
    val[inside] = volume[tuple(pt[inside].T)]
    ids = np.zeros(known.shape, np.int32)
    ids[unique] = val
    correct = unique & (ids == target)
    other = unique & (ids != 0) & (ids != target)
    labs, _ = label(correct)
    roots = np.unique(labs[correct & (rr <= 4 ** 2)])
    connected = correct & np.isin(labs, roots[roots > 0])
    a[rr <= 12 ** 2] = 0
    out = {}
    for name, m in (("r128", rr <= WIDE ** 2), ("annulus", (rr > RADIUS ** 2) & (rr <= WIDE ** 2))):
        out[name] = dict(reference=float(a[m].sum()), accepted=float(a[use & m].sum()), correct=float(a[correct & m].sum()),
                         incorrect=float(a[use & ~correct & m].sum()), connected=float(a[connected & m].sum()),
                         known_other=float(a[other & m].sum()), unassigned=float(a[unique & ~correct & ~other & m].sum()),
                         ambiguous=float(a[use & ~unique & m].sum()))
    return out


def score_all(tri, site, ref96, ref_w, uv, volume, target, shift96=None, shift_w=None, stock_proj=None):
    if stock_proj is not None:
        p96 = displaced(stock_proj[0], shift96)
        pw = displaced(stock_proj[1], shift_w)
    else:
        p96 = project(tri, site)
        pw = region_project(tri, site, WIDE)
    row = {"r96": {k: identities(ref96, p96, 0, site, volume, target)[k] for k in KEYS}}
    row.update(wide_measure(site, ref_w, pw, uv, volume, target))
    return row


def run_case(job):
    """Score stock, single-seed blend, single-tile check and the tiled candidates for one region and preset."""
    os.environ["OMP_NUM_THREADS"] = "1"
    cohort, preset, site, target, spacings, modes = job
    i = site["id"]
    lab = DATA / "labels" / cohort
    volume_path = lab / "inputs" / i / "m7.npy"
    volume = np.load(lab / "truth" / i / "instance_ids.npy", mmap_mode="r")
    ref_tri = np.load(lab / "truth" / i / "triangles.npy")
    ref96, ref_w = project(ref_tri, site), region_project(ref_tri, site, WIDE)
    axis = np.arange(-WIDE, WIDE + 1, 2)
    uv = np.meshgrid(axis, axis)
    stock_file = DATA / COHORTS[cohort] / preset / i / "surface.npz"
    with np.load(stock_file) as d:
        xyz, valid = d["xyz"].astype(np.float64), d["valid"].astype(bool)
    tri, _ = faces(xyz, valid)
    rows = {"stock": score_all(tri, site, ref96, ref_w, uv, volume, target)}
    # Single-seed blend from the frozen graph at the region seed (exactly as the published comparison).
    with np.load(DATA / "recover" / cohort / i / "graph.npz") as g:
        gtri, _ = faces(g["xyz"], g["valid"])
    s96, sw = project(tri, site), region_project(tri, site, WIDE)
    g96, gw = project(gtri, site), region_project(gtri, site, WIDE)
    shifts = []
    for st, gr in ((s96, g96), (sw, gw)):
        both = np.isfinite(st.low) & np.isfinite(gr.low) & (st.high - st.low <= 1) & (gr.high - gr.low <= 1)
        gap = np.zeros(st.low.shape)
        gap[both] = gr.center[both] - st.center[both]
        shifts.append(np.where(both & (np.abs(gap) <= 8), .5 * gap, 0))
    rows["single_seed_blend"] = score_all(None, site, ref96, ref_w, uv, volume, target, shifts[0], shifts[1], (s96, sw))
    # Implementation check: one tile at the region seed, transferred to resampled vertices.
    _init_worker(str(volume_path))
    one = tile_shift(dict(index=0, site=site, triangles=tri))
    up, ok = upsample(xyz, valid, 4)
    disp, _ = transfer(up[ok], [one], [site], "visible")
    moved = up.copy()
    moved[ok] = up[ok] + disp
    rows["single_tile_check"] = score_all(faces(moved, ok)[0], site, ref96, ref_w, uv, volume, target)
    reports = {}
    for spacing in spacings:
        solved = solve_tiles(xyz, valid, volume_path, site["origin_xyz"], spacing, workers=1)
        for mode in modes:
            out, ok2, _, rep = tiled_blend(xyz, valid, volume_path, site["origin_xyz"], spacing, mode, solved=solved)
            key = f"tiled_{spacing}_{mode}"
            rows[key] = score_all(faces(out, ok2)[0], site, ref96, ref_w, uv, volume, target)
            reports[key] = rep
    return dict(cohort=cohort, preset=preset, id=i, rows=rows, reports=reports,
                stock_sha256=file_hash(stock_file), foreground_sha256=file_hash(volume_path))


def bootstrap(before, after, metric="connected", draws=200_000, seed=20261124):
    ids = sorted(before)
    delta = np.array([after[i][metric] - before[i][metric] for i in ids])
    rng = np.random.default_rng(seed)
    idx = rng.integers(0, len(ids), size=(draws, len(ids)))
    sums = delta[idx].sum(1)
    return [float(x) for x in np.percentile(sums, [2.5, 97.5])]


def pooled(table):
    s = {k: sum(r[k] for r in table.values()) for k in KEYS}
    s["strict_error"] = s["incorrect"] / s["accepted"] if s["accepted"] else None
    named = s["known_other"] + s["correct"]
    s["other_share_named"] = s["known_other"] / named if named else None
    s["other_share_accepted"] = s["known_other"] / s["accepted"] if s["accepted"] else None
    return s


def compare(rows_by_case, base, cand, zone):
    b = {i: r[base][zone] for i, r in rows_by_case.items()}
    c = {i: r[cand][zone] for i, r in rows_by_case.items()}
    pb, pc = pooled(b), pooled(c)
    d = np.array([c[i]["connected"] - b[i]["connected"] for i in sorted(b)])
    return dict(base=pb, candidate=pc, connected_delta_cm2=float(d.sum()),
                connected_gain_percent=100 * float(d.sum()) / pb["connected"] if pb["connected"] else None,
                better=int((d > 1e-10).sum()), tie=int((np.abs(d) <= 1e-10).sum()), worse=int((d < -1e-10).sum()),
                interval_95_cm2=bootstrap(b, c),
                gate_other_share_named=bool(pc["other_share_named"] is not None and pb["other_share_named"] is not None
                                            and pc["other_share_named"] <= pb["other_share_named"] + 1e-12))


def run(cohort, candidates, workers):
    seeds = json.loads((DATA / "labels" / cohort / "seeds.json").read_text())["sites"]
    targets = {r["id"]: int(r["target_id"]) for r in json.loads((DATA / "labels" / cohort / "refs.json").read_text())["sites"]}
    spacings = sorted({c[0] for c in candidates})
    modes = sorted({c[1] for c in candidates})
    jobs = [(cohort, p, s, targets[s["id"]], spacings, modes) for p in PRESETS for s in seeds]
    started = time.time()
    with ProcessPoolExecutor(workers) as pool:
        results = []
        for r in pool.map(run_case, jobs, chunksize=1):
            results.append(r)
            print(f"{r['preset']} {r['id']} done ({len(results)}/{len(jobs)}, {time.time()-started:.0f}s)", flush=True)
    by = {p: {r["id"]: r["rows"] for r in results if r["preset"] == p} for p in PRESETS}
    cand_keys = [f"tiled_{s}_{m}" for s, m in candidates]
    summary = {}
    for p in PRESETS:
        summary[p] = {}
        for key in ["single_seed_blend", "single_tile_check"] + cand_keys:
            summary[p][key] = {z: compare(by[p], "stock", key, z) for z in ("r96", "r128", "annulus")}
        for key in cand_keys:
            summary[p][key + "_vs_single_seed"] = {z: compare(by[p], "single_seed_blend", key, z) for z in ("r96", "r128", "annulus")}
    return results, summary, round(time.time() - started, 1)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("stage", choices=("dev", "heldout", "fresh"))
    ap.add_argument("--workers", type=int, default=22)
    a = ap.parse_args()
    OUT.mkdir(parents=True, exist_ok=True)
    provenance = dict(protocol_sha256=file_hash(PROTOCOL), eval_sha256=file_hash(Path(__file__)),
                      tiled_sha256=file_hash(Path(__file__).with_name("growbench_tiled_blend.py")))
    if a.stage == "dev":
        if LOCK.exists():
            sys.exit("Lock exists; development selection already made.")
        results, summary, secs = run("manual_ids", CANDIDATES, a.workers)
        def score(c):
            key = f"tiled_{c[0]}_{c[1]}"
            gains = [summary[p][key]["r128"]["connected_gain_percent"] or 0 for p in PRESETS]
            errs = [summary[p][key]["r128"]["candidate"]["strict_error"] or 1 for p in PRESETS]
            return (float(np.mean(gains)), -float(np.mean(errs)))
        ranked = sorted(CANDIDATES, key=score, reverse=True)
        chosen = ranked[0]
        lock = dict(spacing_vox=chosen[0], mode=chosen[1], selected_on="manual_ids (16 regions, 3 presets), radius 128",
                    ranking=[dict(spacing_vox=c[0], mode=c[1], mean_gain_percent=score(c)[0], mean_strict_error=-score(c)[1]) for c in ranked],
                    locked_at=time.strftime("%Y-%m-%dT%H:%M:%S"), **provenance)
        LOCK.write_text(json.dumps(lock, indent=1), encoding="utf-8")
        (OUT / "dev_manual_ids.json").write_text(json.dumps(dict(summary=summary, cases=results, seconds=secs, lock=lock, **provenance), indent=1), encoding="utf-8")
        print(json.dumps(lock, indent=1))
    else:
        lock = json.loads(LOCK.read_text())
        cohort = "manual_external" if a.stage == "heldout" else "manual_fresh"
        dest = OUT / f"{a.stage}_{cohort}.json"
        if dest.exists():
            sys.exit(f"{dest} exists; the held-out stage runs once.")
        results, summary, secs = run(cohort, [(lock["spacing_vox"], lock["mode"])], a.workers)
        dest.write_text(json.dumps(dict(summary=summary, cases=results, seconds=secs, lock=lock, **provenance), indent=1), encoding="utf-8")
        key = f"tiled_{lock['spacing_vox']}_{lock['mode']}"
        for p in PRESETS:
            for z in ("r96", "r128", "annulus"):
                c = summary[p][key][z]
                print(p, z, f"{c['connected_gain_percent']:+.2f}%", f"{c['better']}/{c['tie']}/{c['worse']}",
                      [round(x * 100, 3) for x in c["interval_95_cm2"]], "gate", c["gate_other_share_named"])


if __name__ == "__main__":
    main()
