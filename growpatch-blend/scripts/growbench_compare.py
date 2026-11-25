"""Lock a development operating point and evaluate it on untouched spatial regions."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from growbench_extend import extend
from growbench_geometry import load_grid, project, score

ROOT=Path(__file__).resolve().parents[1]
DATA=ROOT/"results/growbench"
OUTPUT=DATA/"comparison"
PRESETS=("default20","oriented8")
LOCAL=("two_route","graph")
EXTENDED=("stock_plus_two_route","stock_plus_graph")
LEVELS=[i/20 for i in range(21)]


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f,"sha256").hexdigest()


def save(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    path.write_text(json.dumps(obj,indent=2,allow_nan=False)+"\n",encoding="utf-8")


def get_cases(split):
    manifest=json.loads((DATA/"labels/seeds.json").read_text())
    cases={}
    for site in manifest["sites"]:
        if site["split"]!=split:
            continue
        name=site["id"]
        reference_path=DATA/"labels/truth"/name/"triangles.npy"
        paths={k:DATA/"stock"/k/name/"surface.npz" for k in PRESETS}
        paths.update({k:DATA/"recover"/name/(k+".npz") for k in LOCAL})
        fields={k:load_grid(p,site) for k,p in paths.items()}
        ref=project(np.load(reference_path),site)
        cases[name]=(ref,fields,{"reference":file_hash(reference_path),**{k:file_hash(p) for k,p in paths.items()}})
    return cases


def evaluate(cases, method, threshold=0, preset=None, tolerance=2):
    out={}
    for case,(ref,fields,_) in cases.items():
        if method in EXTENDED:
            source=method.removeprefix("stock_plus_")
            projection=extend(fields[preset],fields[source],threshold)
            row=score(ref,projection,0,tolerance)
        else:
            row=score(ref,fields[method],threshold,tolerance)
        out[case]=row
    return out


def total(rows):
    vals=list(rows.values())
    summed={k:sum(row[k] for row in vals) for k in vals[0]}
    return dict(**summed,risk=summed["incorrect"]/summed["accepted"] if summed["accepted"] else None)


def pair_summary(a,b):
    """b minus a, with one paired bootstrap draw per physical source site."""
    ids=sorted(a)
    draw=np.random.default_rng(481505).integers(0,len(ids),size=(12000,len(ids)))
    def vec(table,key): return np.array([table[i][key] for i in ids])
    def ci(array):
        array=array[np.isfinite(array)]
        return np.percentile(array,[2.5,97.5]).tolist() if len(array) else [None,None]
    def risk(table):
        n=vec(table,"incorrect")[draw].sum(1)
        d=vec(table,"accepted")[draw].sum(1)
        return np.divide(n,d,out=np.full_like(n,np.nan),where=d>0)
    delta=vec(b,"connected")-vec(a,"connected")
    ta,tb=total(a),total(b)
    return dict(stock=ta,candidate=tb,connected_delta_cm2=float(delta.sum()),
                connected_gain_percent=100*float(delta.sum())/ta["connected"] if ta["connected"] else None,
                risk_difference_points=100*(tb["risk"]-ta["risk"]) if None not in (tb["risk"],ta["risk"]) else None,
                improved=int(sum(delta>1e-10)),same=int(sum(abs(delta)<=1e-10)),worse=int(sum(delta< -1e-10)),
                positive_point_estimate=bool(delta.sum()>1e-10 and None not in (ta["risk"],tb["risk"])
                                         and tb["risk"]<=ta["risk"]+1e-12),
                paired_area_delta_95_cm2=ci(delta[draw].sum(1)),paired_risk_difference_95_points=ci(100*(risk(b)-risk(a))))


def best_option(items,budget):
    feasible=[v for v in items if v["result"]["accepted"]>0 and v["result"]["risk"]<=budget+1e-12]
    if not feasible:
        # A diagnostic operating point. Its failed development gate stays part of the record.
        available=[v for v in items if v["result"]["accepted"]>0]
        if not available: return None
        winner=min(available,key=lambda v:(v["result"]["risk"],-v["result"]["connected"]))
    else:
        winner=max(feasible,key=lambda v:(v["result"]["connected"],-v["result"]["risk"]))
    return dict(method=winner["method"],threshold=winner["threshold"],dev_gate=bool(feasible))


def lock(cases):
    stock={p:total(evaluate(cases,p)) for p in PRESETS}
    preset=max(PRESETS,key=lambda p:(stock[p]["connected"],-(stock[p]["risk"] if stock[p]["risk"] is not None else 1)))
    budget=stock[preset]["risk"]
    if budget is None: raise ValueError("Stock produced no supported development area")
    curves=[dict(method=m,threshold=q,result=total(evaluate(cases,m,q,preset))) for m in LOCAL+EXTENDED for q in LEVELS]
    selection=dict(preset=preset,dev_risk_budget=budget,stock_presets=stock,
                   primary=best_option(curves,budget),standalone=best_option([c for c in curves if c["method"] in LOCAL],budget),
                   arms={m:best_option([c for c in curves if c["method"]==m],budget) for m in LOCAL+EXTENDED},
                   dev_files={k:row[2] for k,row in cases.items()},
                   source_hashes={s:file_hash(ROOT/s) for s in ["research/growpatch-heldout-protocol.md","scripts/growbench_geometry.py",
                       "scripts/growbench_extend.py","scripts/growbench_compare.py","results/growbench/labels/seeds.json"]})
    dest=OUTPUT/"selection.json"
    if dest.exists() and json.loads(dest.read_text())!=selection:
        raise RuntimeError("A different operating point is already locked. Use a separately named experiment.")
    save(OUTPUT/"dev_curves.json",curves)
    save(dest,selection)
    return selection


def run(split):
    cases=get_cases(split)
    sel=lock(cases) if split=="dev" else json.loads((OUTPUT/"selection.json").read_text())
    for name,digest in sel["source_hashes"].items():
        if file_hash(ROOT/name)!=digest:
            raise ValueError(f"Method lock changed: {name}")
    preset=sel["preset"]
    options={"primary":sel["primary"],**sel["arms"]}
    groups={}
    for tol in (1,2,4):
        baseline=evaluate(cases,preset,tolerance=tol)
        arms={}
        for name,opt in options.items():
            if opt is None: continue
            rows=evaluate(cases,opt["method"],opt["threshold"],preset,tolerance=tol)
            arms[name]=dict(rows=rows,summary=pair_summary(baseline,rows))
        groups[str(tol)]=dict(stock=baseline,arms=arms)
    doc=dict(split=split,preset=preset,selection=sel["primary"],case_digests={k:v[2] for k,v in cases.items()},
             selection_sha256=file_hash(OUTPUT/"selection.json"),by_tolerance=groups)
    save(OUTPUT/(split+".json"),doc)
    print(split,preset,sel["primary"],json.dumps(groups["2"]["arms"]["primary"]["summary"]),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("split",choices=["dev","test","confirm"])
    run(p.parse_args().split)
