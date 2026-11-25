"""Compare stock's native snap loss and cross-method disagreement at dev-calibrated risk."""
import argparse
import json

import numpy as np

from growbench_compare import DATA,OUTPUT,file_hash,get_cases,pair_summary,save,total
from growbench_geometry import R2,Projection,load_grid,score


def confidence_fields(split):
    cases=get_cases(split)
    selected=json.loads((OUTPUT/"selection.json").read_text())["preset"]
    seed_manifest=json.loads((DATA/"labels/seeds.json").read_text())
    info={}
    for s in seed_manifest["sites"]:
        i=s["id"]
        if i not in cases: continue
        ref,fields,_=cases[i]
        stock=fields[selected]
        diagnostic=load_grid(DATA/"stock"/(selected+"_loss")/i/"surface.npz",s)
        if not(np.array_equal(stock.low,diagnostic.low) and np.array_equal(stock.high,diagnostic.high)):
            raise ValueError(f"Diagnostic output changed stock geometry on {i}")
        alt=fields["two_route"]
        joint=np.isfinite(stock.low)&np.isfinite(alt.low)&(alt.high-alt.low<=1)
        difference=np.zeros(joint.shape)
        difference[joint]=-np.abs(stock.center[joint]-alt.center[joint])
        restricted=Projection(np.where(joint,stock.low,np.inf),np.where(joint,stock.high,-np.inf),
                              difference,stock.area_factor)
        info[i]=(ref,{"snap":diagnostic,"agreement":restricted},stock)
    return info


def rows(info,arm,threshold):
    return {i:score(ref,fields[arm],threshold) for i,(ref,fields,_) in info.items()}


def main(split):
    info=confidence_fields(split)
    lock=OUTPUT/"confidence_selection.json"
    if split=="dev":
        choices={}
        target=json.loads((OUTPUT/"selection.json").read_text())["dev_risk_budget"]
        for arm in ("snap","agreement"):
            pools=[]
            for _,fields,_ in info.values():
                p=fields[arm]
                ok=np.isfinite(p.low)&np.isfinite(p.confidence)&(R2<=96**2)&(R2>12**2)
                pools.append(p.confidence[ok])
            vals=np.unique(np.r_[-1e250,np.percentile(np.concatenate(pools),np.arange(0,101,5))])
            curve=[(float(t),total(rows(info,arm,t))) for t in vals]
            choices[arm]={}
            for key,cap in [("1pct",.01),("2pct",.02),("5pct",.05),("10pct",.10),("stock_dev",target)]:
                ok=[(t,r) for t,r in curve if r["accepted"]>0 and r["risk"]<=cap+1e-12]
                choices[arm][key]=max(ok,key=lambda x:(x[1]["connected"],-x[1]["risk"]))[0] if ok else None
        new=dict(choices=choices,source=file_hash(__file_path__),preset_record=file_hash(OUTPUT/"selection.json"),
                 protocol=file_hash(DATA.parents[1]/"research/growpatch-confidence-protocol.md"))
        if lock.exists() and json.loads(lock.read_text())!=new: raise ValueError("Confidence selection changed")
        save(lock,new)
    spec=json.loads(lock.read_text())
    assert file_hash(__file_path__)==spec["source"]
    outputs={}
    for key in spec["choices"]["snap"]:
        by_arm={a:rows(info,a,ts[key]) for a,ts in spec["choices"].items() if ts[key] is not None}
        pair=pair_summary(by_arm["snap"],by_arm["agreement"]) if len(by_arm)==2 else None
        outputs[key]=dict(arms={a:total(tb) for a,tb in by_arm.items()},paired=pair)
    save(OUTPUT/("confidence_"+split+".json"),outputs)
    print(split,"5% budget",outputs["5pct"]["arms"])


if __name__=="__main__":
    from pathlib import Path
    __file_path__=Path(__file__)
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("split",choices=("dev","test","confirm"))
    main(ap.parse_args().split)
