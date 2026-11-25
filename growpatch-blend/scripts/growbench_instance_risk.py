"""Use held-out sheet identities with confidence cutoffs fixed on development."""
import argparse
import json

import numpy as np

from growbench_compare import DATA,OUTPUT,file_hash,save,total
from growbench_geometry import load_grid,project
from growbench_identity import identities,summarize
from growbench_regional import displaced


def risk_ratio(row):
    classified=row["correct"]+row["known_other"]
    return row["known_other"]/classified if classified else None


def run(cohort,preset):
    old=cohort=="dev"
    group="manual_ids" if old else "manual_external"
    local=DATA/"labels"/group
    seeds=json.loads((local/"seeds.json").read_text())["sites"]
    ref_ids={r["id"]:r["target_id"] for r in json.loads((local/"refs.json").read_text())["sites"]}
    root=DATA/("stock_manual" if old else "stock_manual_external")
    stage=OUTPUT/"named_risk"
    lock=stage/(preset.removesuffix("_loss")+"_selection.json")
    if old:
        variants={key:list(np.arange(41)/40) for key in ("two_route","graph","blend_8_gated")}
    else:
        selection=json.loads(lock.read_text())
        if selection["source_sha256"]!=file_hash(__file_path__):raise ValueError("Named-risk method changed")
        variants={k:[v] for k,v in selection["threshold"].items() if v is not None}
    baselines={}
    tables={m:{q:{} for q in vv} for m,vv in variants.items()}
    for s in seeds:
        i=s["id"]
        if old and s["split"]!="dev":continue
        ref=project(np.load(local/"truth"/i/"triangles.npy"),s)
        ids=np.load(local/"truth"/i/"instance_ids.npy",mmap_mode="r")
        stock=load_grid(root/preset/i/"surface.npz",s)
        stock.confidence=np.ones(stock.low.shape)
        def measure(p,q):
            return identities(ref,p,q,s,ids,ref_ids[i])
        baselines[i]=measure(stock,0)
        fs={k:load_grid(DATA/"recover"/group/i/(k+".npz"),s) for k in ("graph","two_route")}
        g=fs["graph"];both=np.isfinite(g.low)&np.isfinite(stock.low)&(g.high-g.low<=1)&(stock.high-stock.low<=1)
        diff=np.zeros(stock.low.shape);diff[both]=g.center[both]-stock.center[both]
        for method,vv in variants.items():
            for q in vv:
                if method=="blend_8_gated":
                    step=np.where(both&(np.abs(diff)<=8)&(g.confidence+1e-9>=q),.5*diff,0)
                    row=measure(displaced(stock,step),0)
                else:row=measure(fs[method],q)
                tables[method][q][i]=row
    base=total(baselines);risk=risk_ratio(base)
    if old:
        threshold={}
        curves={}
        for name,vv in tables.items():
            c=[(q,total(rows)) for q,rows in vv.items()]
            curves[name]=c
            possible=[(q,t) for q,t in c if t["accepted"] and risk_ratio(t) is not None and
                      t["risk"]<=base["risk"]+1e-12 and risk_ratio(t)<=risk+1e-12]
            threshold[name]=float(max(possible,key=lambda x:(x[1]["connected"],-x[1]["risk"]))[0]) if possible else None
        doc=dict(threshold=threshold,dev_strict=base["risk"],dev_known_other=risk,source_sha256=file_hash(__file_path__),
                 protocol=file_hash(DATA.parents[1]/"research/growpatch-instance-risk-protocol.md"))
        if lock.exists() and json.loads(lock.read_text())!=doc:raise ValueError("Different named-risk lock")
        save(lock,doc)
        save(stage/(preset+"_dev_curves.json"),curves)
    else:threshold=selection["threshold"]
    res={m:summarize(baselines,tables[m][q]) for m,q in threshold.items() if q is not None}
    save(stage/(preset+"_"+cohort+".json"),dict(summary=res,baselines=baselines,threshold=threshold,
          rows={m:tables[m][q] for m,q in threshold.items() if q is not None}))
    print(preset,cohort,threshold,[(k,round(v["connected_gain_percent"] or 0,2),
          round(100*v["stock_known_other_rate"],2),round(100*v["candidate_known_other_rate"],2),v["identity_gate"]) for k,v in res.items()])


if __name__=="__main__":
    from pathlib import Path
    __file_path__=Path(__file__)
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("cohort",choices=("dev","external"));p.add_argument("preset")
    a=p.parse_args();run(a.cohort,a.preset)
