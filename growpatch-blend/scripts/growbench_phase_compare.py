"""Test the development-informed seed-phase repair and give stock a common-offset control."""
import argparse
import json

import numpy as np
from scipy.ndimage import label

from growbench_compare import DATA, OUTPUT, LEVELS, best_option, file_hash, get_cases, pair_summary, save, total
from growbench_geometry import R2, STEP, PITCH_UM, Projection, load_grid, score


def offset_stock(p):
    valid=np.isfinite(p.low)&(p.high-p.low<=1)&(R2<=6**2)
    delta=float(np.clip(np.median(p.center[valid]),-3,3)) if valid.sum()>=5 else 0.
    return Projection(p.low-delta,p.high-delta,p.confidence,p.area_factor),delta


def sensitivity(ref,baseline,candidate,threshold):
    known=np.isfinite(ref.low)&(ref.high-ref.low<=.5)&(ref.area_factor<=2)&(R2<=96**2)
    weight=np.where(known,ref.area_factor,0)*(STEP*PITCH_UM/10000)**2
    weight[R2<=12**2]=0
    outputs=[]
    for p,q in ((baseline,0),(candidate,threshold)):
        has=known&np.isfinite(p.low)&(p.confidence+1e-9>=q)
        correct=np.zeros(has.shape,bool)
        correct[has]=((np.abs(p.center[has]-ref.center[has])/ref.area_factor[has]<=2+1e-8)&
                      (p.high[has]-p.low[has]<=1+1e-8))
        labs,_=label(correct)
        size=np.bincount(labs.ravel(),weights=weight.ravel())
        collars=np.unique(labs[correct&(R2<=12**2)])
        outputs.append((has,correct,dict(largest_correct_cm2=float(size[1:].max(initial=0)),
                    from_collar_cm2=float(weight[np.isin(labs,collars)&correct].sum()),
                    all_correct_cm2=float(weight[correct].sum()))))
    common=outputs[0][0]&outputs[1][0]
    for _,correct,out in outputs:
        out.update(common_cm2=float(weight[common].sum()),common_bad_cm2=float(weight[common&~correct].sum()))
    return dict(stock=outputs[0][2],candidate=outputs[1][2])


def tables(cases,arm,q,tol):
    return {key:score(ref,fields[arm],q,tol) for key,(ref,fields,_) in cases.items()}


def run(split):
    original=json.loads((OUTPUT/"selection.json").read_text())
    preset=original["preset"]
    cases=get_cases(split)
    manifest=json.loads((DATA/"labels/seeds.json").read_text())
    offsets={}
    for site in manifest["sites"]:
        name=site["id"]
        if name not in cases: continue
        fields=cases[name][1]
        fields["stock_offset"],offsets[name]=offset_stock(fields[preset])
        for m in ("two_route_phase","graph_phase"):
            path=DATA/"recover"/name/(m+".npz")
            fields[m]=load_grid(path,site)
            cases[name][2][m]=file_hash(path)
    lock=OUTPUT/"phase_selection.json"
    if split=="dev":
        budget=original["dev_risk_budget"]
        curves=[dict(method=m,threshold=q,result=total(tables(cases,m,q,2)))
                for m in ("two_route_phase","graph_phase") for q in LEVELS]
        sel=dict(chosen=best_option(curves,budget),stock_preset=preset,dev_budget=budget,
                 source_hashes={p:file_hash(DATA.parents[1]/p) for p in ["scripts/growbench_phase_compare.py",
                    "research/growpatch-seed-offset-protocol.md","results/growbench/comparison/selection.json"]},
                 prediction_lock=file_hash(DATA/"recover/phase_lock.json"))
        if lock.exists() and json.loads(lock.read_text())!=sel: raise ValueError("Existing supplemental lock differs")
        save(OUTPUT/"phase_dev_curves.json",curves)
        save(lock,sel)
    else: sel=json.loads(lock.read_text())
    for p,h in sel["source_hashes"].items():
        if file_hash(DATA.parents[1]/p)!=h: raise ValueError(f"Supplemental lock changed: {p}")
    chosen=sel["chosen"]
    results={}
    for tol in (1,2,4):
        before=tables(cases,preset,0,tol)
        control=tables(cases,"stock_offset",0,tol)
        after=tables(cases,chosen["method"],chosen["threshold"],tol)
        results[str(tol)]=dict(stock=before,adjusted_stock=control,candidate=after,
              candidate_vs_stock=pair_summary(before,after),candidate_vs_adjusted=pair_summary(control,after),
              adjusted_vs_stock=pair_summary(before,control))
    sens={name:sensitivity(ref,fields[preset],fields[chosen["method"]],chosen["threshold"])
          for name,(ref,fields,_) in cases.items()}
    save(OUTPUT/("phase_"+split+".json"),dict(split=split,method=chosen,offsets=offsets,sensitivity=sens,
        case_digests={key:v[2] for key,v in cases.items()},by_tolerance=results))
    print(split,chosen)
    for name in ("candidate_vs_stock","adjusted_vs_stock","candidate_vs_adjusted"):
        result=results["2"][name]
        print(name,round(result["connected_gain_percent"],3),round(result["risk_difference_points"],3),result["improved"],result["same"],result["worse"])


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("split",choices=("dev","test","confirm"))
    run(p.parse_args().split)
