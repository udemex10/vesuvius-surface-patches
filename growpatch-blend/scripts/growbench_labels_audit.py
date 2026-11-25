"""Validate frozen inputs, coverage and close neighboring sheet IDs before adjudication."""
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.spatial import cKDTree
from growbench_labels import OUT, digest, write
from growbench_labels_replay import COHORTS, file_hash


def cohort(path,manual=False):
    seeds=json.loads((path/"seeds.json").read_text())
    refs={r["id"]:r for r in json.loads((path/"refs.json").read_text())["sites"]}
    rows=[]
    for s in seeds["sites"]:
        sid=s["id"];src=path/"inputs"/sid;target=path/"truth"/sid;ref=refs[sid]
        origin=np.array(s["origin_xyz"]);seed=np.array(s["seed_xyz"]);m7=np.load(src/"m7.npy",mmap_mode="r");ct=np.load(src/"ct.npy",mmap_mode="r")
        assert m7.shape==ct.shape==tuple(s["cube_shape_zyx"]) and m7.dtype==np.uint8
        assert m7.max()>0 and ct.max()>ct.min()
        xyz=np.load(target/"ref_xyz.npy");valid=np.isfinite(xyz).all(-1)
        delta=xyz-seed;u=np.asarray(s["tangent_u_xyz"]);v=np.asarray(s["tangent_v_xyz"])
        radius=seeds["area_radius_voxels"];collar=seeds["excluded_seed_radius_voxels"]
        rr=((delta@u)**2+(delta@v)**2);ok=valid&(rr<=radius**2)&(rr>=collar**2)
        row=dict(id=sid,split=s["split"],annulus_finite_reference_pixels=int(ok.sum()),pitch_um=7.91 if manual else 9.6,
                 nominal_xz_area_mm2=float(ok.sum()*(.00791 if manual else .0096)**2),m7_file_sha256=digest((src/"m7.npy").read_bytes()),
                 ct_file_sha256=digest((src/"ct.npy").read_bytes()))
        if manual:
            qa=json.loads((target/"source_qa.json").read_text())
            vals=sorted([(qscan,p,c) for qscan,items in qa["registration"].items() for p,c in items["correlations"].items()],key=lambda t:-t[-1])
            assert vals[0][0:2]==("A","(0, 1, 2)") and vals[0][2]>.9
            ids=np.load(target/"instance_ids.npy",mmap_mode="r");target_id=ref["target_id"]
            competitors=(ids!=0)&(ids!=target_id)
            nearest=distance_transform_edt(~competitors)
            points=np.rint((xyz[ok]-origin)[:,::-1]).astype(int);zz,yy,xx=points.T
            assert np.all(ids[zz,yy,xx]==target_id)
            distances=nearest[zz,yy,xx]
            row.update(target_id=target_id,number_other_ids=int(len(np.unique(ids[competitors]))),manual_semantic_foreground_iou=qa["semantic_manual_voxel_iou"],
                       best_registration_pearson=vals[0][-1],best_alternative_orientation_or_scan_pearson=vals[1][-1],
                       target_to_other_id_rounded_center_vox_p10_p50_p90=np.percentile(distances,[10,50,90]).tolist(),
                       fraction_targets_with_other_id_within={str(k):float((distances<=k).mean()) for k in (4,8,12,20)})
        rows.append(row)
    pairs=[]
    for i,a in enumerate(seeds["sites"]):
        for b in seeds["sites"][i+1:]:
            gap_xyz=np.abs(np.subtract(a["origin_xyz"],b["origin_xyz"]))-(np.asarray(a["cube_shape_zyx"])[::-1]-1)
            pairs.append(float(np.linalg.norm(np.maximum(gap_xyz,0))))
    assert min(pairs)>0.
    return dict(seeds_sha256=digest((path/"seeds.json").read_bytes()),ref_manifest_sha256=digest((path/"refs.json").read_bytes()),
                min_pair_bounds_clearance_vox=min(pairs),sites=rows)


def scored_points(path):
    result={}
    for s in json.loads((path/"seeds.json").read_text())["sites"]:
        grid=np.load(path/"truth"/s["id"]/"ref_xyz.npy")
        pts=grid[np.isfinite(grid).all(-1)]
        d=pts-s["seed_xyz"];r=(d@s["tangent_u_xyz"])**2+(d@s["tangent_v_xyz"])**2
        result[s["id"]]=pts[(r>=12**2)&(r<=96**2)]
    return result


def gap_between_clouds(a,b,intra=False):
    pairs=[]
    for k,v in a.items():
        for n,w in b.items():
            if intra and k>=n:continue
            lower=float(np.linalg.norm(np.maximum(np.maximum(v.min(0)-w.max(0),w.min(0)-v.max(0)),0)))
            pairs.append((lower,k,n))
    best=(float("inf"),None,None)
    for lower,k,n in sorted(pairs):
        if lower>=best[0]:break
        dist=float(cKDTree(a[k]).query(b[n],workers=2)[0].min())
        if dist<best[0]:best=(dist,k,n)
    return dict(min_native_voxel_distance=best[0],first=best[1],second=best[2],sample="All reference midpoint integer X/Z samples in the tangent annulus, bounds.pruned exact nearest pair in this point set.")


def save_replay_hashes():
    records = {}
    for name, subdir in COHORTS.items():
        path = OUT / subdir
        manifest = path / "seeds.json"
        if not manifest.is_file():
            continue
        records[name] = {}
        for case in json.loads(manifest.read_text())["sites"]:
            sid = case["id"]
            targets = [path / "inputs" / sid / filename for filename in ("ct.npy", "m7.npy")]
            targets += [path / "truth" / sid / filename for filename in ("ref_xyz.npy", "triangles.npy")]
            if name in ("manual", "manual_external"):
                targets.append(path / "truth" / sid / "instance_ids.npy")
            records[name][sid] = {f.name: file_hash(f) for f in targets}
    write(OUT / "replay_hashes.json", records)


if __name__=="__main__":
    summary=dict(primary=cohort(OUT),manual=cohort(OUT/"manual_ids",manual=True))
    for name,is_manual in (("pressure",False),("interrupted",False),("manual_external",True)):
        path=OUT/name
        if (path/"seeds.json").is_file():
            sites=json.loads((path/"seeds.json").read_text())["sites"]
            if all((path/"inputs"/s["id"]/"ct.npy").is_file() and ((not is_manual) or (path/"truth"/s["id"]/"source_qa.json").is_file()) for s in sites):
                summary[name]=cohort(path,manual=is_manual)
    if "manual_external" in summary:
        original=scored_points(OUT/"manual_ids");external=scored_points(OUT/"manual_external")
        summary["manual_external"]["between_scoring_domains_and_original"]=gap_between_clouds(original,external)
        summary["manual_external"]["between_own_scoring_domains"]=gap_between_clouds(external,external,True)
    write(OUT/"cohort_qa.json",summary)
    save_replay_hashes()
    m=summary["manual"]["sites"]
    for r in m:
        print(r["id"],r["target_id"],round(r["target_to_other_id_rounded_center_vox_p10_p50_p90"][1],1),r["fraction_targets_with_other_id_within"]["8"])
