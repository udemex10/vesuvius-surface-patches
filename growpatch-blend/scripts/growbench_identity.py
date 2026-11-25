"""Direct held-out sheet-ID assessment on the manually harmonized historical Scroll 1 cubes."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from growbench_compare import DATA, OUTPUT, file_hash, pair_summary, save, total
from growbench_geometry import R2, STEP, U, V, frame, load_grid, project, score
from growbench_phase_compare import offset_stock

SOURCE=DATA/"labels/manual_ids"
PITCH=7.91


def identities(ref,trial,threshold,site,volume,target):
    known=np.isfinite(ref.low)&(ref.high-ref.low<=.5)&(ref.area_factor<=2)&(R2<=96**2)
    area=np.where(known,ref.area_factor,0)*(STEP*PITCH/10000)**2
    predicted=np.isfinite(trial.low)&(trial.confidence+1e-9>=threshold)
    accepted=known&predicted
    unique=accepted&(trial.high-trial.low<=1)
    points=np.stack((U[unique],V[unique],trial.center[unique]),axis=-1) @ frame(site)
    offset=np.asarray(site["seed_xyz"])-site["origin_xyz"]
    local=np.floor(points+offset+.5).astype(int)[:,::-1]
    inside=((local>=0)&(local<volume.shape)).all(1)
    ident=np.zeros(known.shape,dtype=np.int32)
    ids=np.zeros(len(local),np.int32)
    ids[inside]=volume[tuple(local[inside].T)]
    ident[unique]=ids
    correct=unique&(ident==target)
    wrong=unique&(ident!=target)&(ident!=0)
    outside=unique&(ident==0)
    labs,_=label(correct)
    near=np.unique(labs[correct&(R2<=4**2)])
    area[R2<=12**2]=0
    sizes=np.bincount(labs.ravel(),weights=area.ravel())
    return dict(reference=float(area.sum()),accepted=float(area[accepted].sum()),correct=float(area[correct].sum()),
                incorrect=float(area[accepted&~correct].sum()),known_other=float(area[wrong].sum()),
                unassigned=float(area[outside].sum()),ambiguous=float(area[accepted&~unique].sum()),
                connected=float(area[np.isin(labs,near)&correct].sum()),largest=float(sizes[1:].max(initial=0)),
                out_of_reference_rays=int((predicted&~known&(R2<=96**2)).sum()))


def summarize(before,after):
    summary=pair_summary(before,after)
    def other(row):
        n=row["known_other"]+row["correct"]
        return row["known_other"]/n if n>0 else None
    a,b=(other(summary[key]) for key in ("stock","candidate"))
    summary.update(stock_known_other_rate=a,candidate_known_other_rate=b,
                   identity_gate=bool(summary["positive_point_estimate"] and a is not None and b is not None and b<=a+1e-12))
    return summary


def run(split,stock_folder):
    sites=json.loads((SOURCE/"seeds.json").read_text())["sites"]
    refs={r["id"]:r for r in json.loads((SOURCE/"refs.json").read_text())["sites"]}
    original=json.loads((OUTPUT/"selection.json").read_text())
    phase=json.loads((OUTPUT/"phase_selection.json").read_text())
    methods={k:v["threshold"] for k,v in original["arms"].items() if k in ("graph","two_route")}
    methods[phase["chosen"]["method"]]=phase["chosen"]["threshold"]
    all_rows={}
    geometric={}
    hashes={}
    for s in sites:
        if s["split"]!=split: continue
        i=s["id"]
        target=int(refs[i]["target_id"])
        directory=SOURCE/"truth"/i
        ref=project(np.load(directory/"triangles.npy"),s)
        volume=np.load(directory/"instance_ids.npy",mmap_mode="r")
        stock_file=DATA/"stock_manual"/stock_folder/i/"surface.npz"
        stock=load_grid(stock_file,s)
        # Native diagnostic loss may be present on a full-geometry replay. This
        # comparison tests the complete stock mesh, independently of loss gating.
        stock.confidence=np.ones(stock.low.shape)
        cases={"stock":(stock,0),"stock_offset":(offset_stock(stock)[0],0)}
        hashes[i]={"stock":file_hash(stock_file),"instance_labels":file_hash(directory/"instance_ids.npy")}
        for name,threshold in methods.items():
            path=DATA/"recover/manual_ids"/i/(name+".npz")
            cases[name]=(load_grid(path,s),threshold)
            hashes[i][name]=file_hash(path)
        for name,(p,th) in cases.items():
            all_rows.setdefault(name,{})[i]=identities(ref,p,th,s,volume,target)
            geom=score(ref,p,th)
            for k in ("reference","accepted","correct","incorrect","connected"):
                geom[k]*=(PITCH/9.6)**2
            geometric.setdefault(name,{})[i]=geom
    results={key:summarize(all_rows["stock"],v) for key,v in all_rows.items() if key!="stock"}
    path=OUTPUT/"manual_ids"/stock_folder/(split+".json")
    save(path,dict(split=split,stock=stock_folder,field_digests=hashes,rows=all_rows,summary=results,
                   midpoint_geometry={k:total(v) for k,v in geometric.items()}))
    print(split,stock_folder,[(k,round(r["connected_gain_percent"] or 0,2),round(r["risk_difference_points"] or 0,3),
                       round(100*r["stock_known_other_rate"],2) if r["stock_known_other_rate"] is not None else None,
                       round(100*r["candidate_known_other_rate"],2) if r["candidate_known_other_rate"] is not None else None,
                       r["identity_gate"]) for k,r in results.items()])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("split",choices=("dev","test","confirm"))
    parser.add_argument("--stock-folder",default="oriented8")
    args=parser.parse_args()
    run(args.split,args.stock_folder)
