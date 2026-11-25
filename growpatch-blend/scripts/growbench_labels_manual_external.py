"""Freeze an untouched second source-cube and target-ID holdout for the historical cohort."""
import argparse
import re
import json
import numpy as np
from growbench_labels_manual import OUT as ORIGINAL, ROOT, original, choose, mesh, prepare, digest, fetch, write

OUT=ORIGINAL.parent/"manual_external"
SALT="growbench-manual-external-v1:"
RULE=dict(salt=SALT,planned_sites=16,source="Original legacy manual and semantic paired cube catalog",
    selection="Salted hash order of eligible cube names, one target nearest cube center with the same mask-only eligibility as main. Exclude all original cube names and target IDs and all newly used targets. Require another labelled sheet within 32 Euclidean voxels of the seed. First pass requires Chebyshev distance >=384 across old/new cube origins. Second pass requires >=256, which ensures zero shared raw voxels. No raw CT, common semantic or contender results enter site selection. Eligible graph-shaped sheets only. All sites seal as one external assessment set without tuning.")


def select():
    if (OUT/"seeds.json").exists():raise SystemExit("Already locked")
    OUT.mkdir(exist_ok=True,parents=True);write(OUT/"sampling_rule.json",RULE)
    orig=json.loads((ORIGINAL/"refs.json").read_text())["sites"]
    oldnames={r["source_cube_zyx"] for r in orig}
    exclude={r["target_id"] for r in orig}
    names_used=set(oldnames)
    origins=[np.array(list(map(int,n.split("_"))))[::-1] for n in sorted(oldnames)]
    catalogs=[set(re.findall(r'href="(\d{5}_\d{5}_\d{5})/"',fetch(ROOT+c+"/").decode())) for c in ("instance-labels-harmonized","semantic-labels")]
    nlist=sorted(set.intersection(*catalogs)-oldnames,key=lambda n:digest((SALT+n).encode()))
    sites=[];refs=[];trial=[];first_count=0
    for margin in (384,256):
        for name in nlist:
            if name in names_used:continue
            origin=np.array(list(map(int,name.split("_"))))[::-1]
            if min(np.max(abs(origin-q)) for q in origins)<margin:continue
            labels,linfo=original(name,"instance.nrrd");excluded=set(exclude)
            while True:
                choice=choose(labels,excluded)
                if not choice:break
                tid,point,n,u,v,yy=choice
                cx,cy,cz=np.rint(point).astype(int)
                rad=32;near=labels[cz-rad:cz+rad+1,cy-rad:cy+rad+1,cx-rad:cx+rad+1]
                pts=np.argwhere((near!=0)&(near!=tid))
                gap=float(np.linalg.norm(pts-rad,axis=1).min()) if len(pts) else float("inf")
                if gap<=rad:break
                excluded.add(tid)
            trial.append(dict(source=name,margin=margin,accepted=bool(choice)))
            if not choice:continue
            sid=f"external_{len(sites)+1:02d}"
            s=dict(id=sid,split="test",seed_xyz=(point+origin).tolist(),origin_xyz=origin.tolist(),normal_xyz=n.tolist(),
                tangent_u_xyz=u.tolist(),tangent_v_xyz=v.tolist(),cube_shape_zyx=[256]*3,algorithm_input="semantic_foreground_centers_uint8")
            truth=OUT/"truth"/sid;truth.mkdir(exist_ok=True,parents=True)
            xyz,triangles=mesh(yy,origin)
            np.save(truth/"instance_ids.npy",labels);np.save(truth/"ref_xyz.npy",xyz);np.save(truth/"triangles.npy",triangles)
            r=dict(id=sid,source_cube_zyx=name,target_id=tid,source=linfo,label_ids=np.unique(labels).tolist(),
                initial_source_margin_vox=margin,other_id_distance_to_rounded_seed_vox=gap,unknown_value=0,
                midpoint_definition="Same midpoint of unique closed target-ID column interval as representative manual cohort.")
            sites.append(s);refs.append(r);names_used.add(name);exclude.add(tid);origins.append(origin)
            print("external selected",sid,name,tid,margin,flush=True)
            if len(sites)==16:break
        if margin==384:first_count=len(sites)
        if len(sites)==16:break
    if len(sites)<8:raise ValueError(f"Only {len(sites)} qualifying sites; original rule preserved.")
    write(OUT/"refs.json",dict(source=ROOT+"README.txt",original_refs_sha256=digest((ORIGINAL/"refs.json").read_bytes()),rule=RULE,
          original_ids=sorted(r["target_id"] for r in orig),paired_remaining=len(nlist),n_384_margin=first_count,
          selection_history=trial,sites=refs))
    write(OUT/"seeds.json",dict(benchmark="Paris4-legacy-manual-external-v1",frame="global XYZ 7.91um historical Scroll 1",
        voxel_size_um=7.91,area_radius_voxels=96,excluded_seed_radius_voxels=12,sites=sites))
    print("MANUAL EXTERNAL LOCK",digest((OUT/"seeds.json").read_bytes()),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__);ap.add_argument("--inputs",action="store_true");args=ap.parse_args()
    if not args.inputs:select()
    prepare(OUT)
