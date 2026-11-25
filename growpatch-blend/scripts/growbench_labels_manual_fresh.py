"""Freeze a third, untouched source-cube and target-ID cohort for the tiled-blend confirmation.

Same eligibility rule as growbench_labels_manual_external.py with a new salt. Excludes every cube name
and target ID used by the original (manual_ids) and external (manual_external) cohorts, and requires a
Chebyshev cube-origin separation of at least 256 voxels from all of them (no shared raw voxels).
Selection reads label masks only; no CT, foreground or method output enters it.
"""
import argparse
import json
import re

import numpy as np

from growbench_labels_manual import OUT as ORIGINAL, ROOT, choose, digest, fetch, mesh, original, prepare, write

EXTERNAL = ORIGINAL.parent / "manual_external"
OUT = ORIGINAL.parent / "manual_fresh"
SALT = "growbench-manual-fresh-v1:"
RULE = dict(salt=SALT, planned_sites=16, source="Original legacy manual and semantic paired cube catalog",
            selection="Salted hash order of eligible cube names; one target nearest the cube center with the main cohort's "
                      "mask-only eligibility. Exclude all manual_ids and manual_external cube names and target IDs. Require "
                      "another labelled sheet within 32 Euclidean voxels of the seed. Require Chebyshev distance >=256 between "
                      "cube origins and all previously used origins. Sealed as one confirmation set for the locked tiled blend.")


def select():
    if (OUT / "seeds.json").exists():
        raise SystemExit("Already locked")
    OUT.mkdir(exist_ok=True, parents=True)
    write(OUT / "sampling_rule.json", RULE)
    prior = [json.loads((d / "refs.json").read_text())["sites"] for d in (ORIGINAL, EXTERNAL)]
    prior = [r for rows in prior for r in rows]
    used_names = {r["source_cube_zyx"] for r in prior}
    exclude = {r["target_id"] for r in prior}
    origins = [np.array(list(map(int, n.split("_"))))[::-1] for n in sorted(used_names)]
    catalogs = [set(re.findall(r'href="(\d{5}_\d{5}_\d{5})/"', fetch(ROOT + c + "/").decode()))
                for c in ("instance-labels-harmonized", "semantic-labels")]
    nlist = sorted(set.intersection(*catalogs) - used_names, key=lambda n: digest((SALT + n).encode()))
    sites, refs, trial = [], [], []
    for name in nlist:
        origin = np.array(list(map(int, name.split("_"))))[::-1]
        if min(np.max(abs(origin - q)) for q in origins) < 256:
            trial.append(dict(source=name, accepted=False, reason="origin within 256 of a used cube"))
            continue
        labels, linfo = original(name, "instance.nrrd", cache_root=OUT)
        excluded = set(exclude)
        while True:
            choice = choose(labels, excluded)
            if not choice:
                break
            tid, point, n, u, v, yy = choice
            cx, cy, cz = np.rint(point).astype(int)
            rad = 32
            near = labels[cz - rad:cz + rad + 1, cy - rad:cy + rad + 1, cx - rad:cx + rad + 1]
            pts = np.argwhere((near != 0) & (near != tid))
            gap = float(np.linalg.norm(pts - rad, axis=1).min()) if len(pts) else float("inf")
            if gap <= rad:
                break
            excluded.add(tid)
        trial.append(dict(source=name, accepted=bool(choice)))
        if not choice:
            continue
        sid = f"fresh_{len(sites) + 1:02d}"
        s = dict(id=sid, split="test", seed_xyz=(point + origin).tolist(), origin_xyz=origin.tolist(), normal_xyz=n.tolist(),
                 tangent_u_xyz=u.tolist(), tangent_v_xyz=v.tolist(), cube_shape_zyx=[256] * 3,
                 algorithm_input="semantic_foreground_centers_uint8")
        truth = OUT / "truth" / sid
        truth.mkdir(exist_ok=True, parents=True)
        xyz, triangles = mesh(yy, origin)
        np.save(truth / "instance_ids.npy", labels)
        np.save(truth / "ref_xyz.npy", xyz)
        np.save(truth / "triangles.npy", triangles)
        refs.append(dict(id=sid, source_cube_zyx=name, target_id=tid, source=linfo, label_ids=np.unique(labels).tolist(),
                         other_id_distance_to_rounded_seed_vox=gap, unknown_value=0))
        sites.append(s)
        used_names.add(name)
        exclude.add(tid)
        origins.append(origin)
        print("fresh selected", sid, name, tid, flush=True)
        if len(sites) == 16:
            break
    write(OUT / "refs.json", dict(source=ROOT + "README.txt", rule=RULE, prior_cube_names=sorted(
        {r["source_cube_zyx"] for r in prior}), paired_remaining=len(nlist), selection_history=trial, sites=refs))
    write(OUT / "seeds.json", dict(benchmark="Paris4-legacy-manual-fresh-v1", frame="global XYZ 7.91um historical Scroll 1",
                                   voxel_size_um=7.91, area_radius_voxels=96, excluded_seed_radius_voxels=12, sites=sites))
    print("MANUAL FRESH LOCK", len(sites), "sites", digest((OUT / "seeds.json").read_bytes()), flush=True)


if __name__ == "__main__":
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs", action="store_true", help="Only (re)build inputs for the frozen sites")
    a = ap.parse_args()
    if not a.inputs:
        select()
    prepare(OUT)
