"""Prospective larger-field and fresh instance-cube comparison of frozen surface refinements."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label

from growbench_compare import DATA,OUTPUT,file_hash,pair_summary,save,total
from growbench_geometry import Projection,faces,frame,project
from growbench_identity import summarize
from growbench_local_refine import refine_grid


def region_project(tri,site,radius):
    if radius==96: return project(tri,site)
    # Assemble the exact two-voxel lattice using local panels. Moving the
    # panel origin in its tangent plane preserves the signed depth coordinate.
    axes=frame(site);base=np.asarray(site["seed_xyz"])
    coord=np.arange(-radius,radius+1,2)
    arrays=[np.empty((len(coord),len(coord))) for _ in range(4)]
    for high_u in (False,True):
        cols=np.flatnonzero((coord>0)==high_u)
        for high_v in (False,True):
            rows=np.flatnonzero((coord>0)==high_v)
            du=(radius-96)*(1 if high_u else -1)
            dv=(radius-96)*(1 if high_v else -1)
            moved={**site,"seed_xyz":(base+du*axes[0]+dv*axes[1]).tolist()}
            part=project(tri,moved)
            src=np.ix_((coord[rows]-dv+96)//2,(coord[cols]-du+96)//2)
            for dest,values in zip(arrays,(part.low,part.high,part.confidence,part.area_factor)):
                dest[np.ix_(rows,cols)]=values[src]
    return Projection(*arrays)


def load_geometry(path):
    with np.load(path) as ds:
        return ds["xyz"],ds["valid"].astype(bool)


def displaced(stock,shift):
    return Projection(stock.low+shift,stock.high+shift,np.ones(stock.low.shape),stock.area_factor)


def measure(site,ref,p,uv,pitch,volume=None,target=None):
    u,v=uv
    rr=u*u+v*v
    known=np.isfinite(ref.low)&(ref.high-ref.low<=.5)&(ref.area_factor<=2)&(rr<=u.max()**2)
    a=np.where(known,ref.area_factor,0)*(2*pitch/10000)**2
    use=known&np.isfinite(p.low)
    unique=use&(p.high-p.low<=1)
    if volume is None:
        correct=np.zeros(known.shape,bool)
        correct[unique]=np.abs(p.center[unique]-ref.center[unique])/ref.area_factor[unique]<=2+1e-8
        other=np.zeros(known.shape,bool)
    else:
        xyz=np.stack((u[unique],v[unique],p.center[unique]),-1)@frame(site)
        pt=np.floor(xyz+np.asarray(site["seed_xyz"])-site["origin_xyz"]+.5).astype(int)[:,::-1]
        inside=((pt>=0)&(pt<volume.shape)).all(1)
        val=np.zeros(len(pt),np.int32)
        val[inside]=volume[tuple(pt[inside].T)]
        ids=np.zeros(known.shape,np.int32)
        ids[unique]=val
        correct=unique&(ids==target)
        other=unique&(ids!=0)&(ids!=target)
    labs,_=label(correct)
    roots=np.unique(labs[correct&(rr<=4**2)])
    a[rr<=12**2]=0
    out=dict(reference=float(a.sum()),accepted=float(a[use].sum()),correct=float(a[correct].sum()),
             incorrect=float(a[use&~correct].sum()),connected=float(a[correct&np.isin(labs,roots)].sum()),
             largest=float(np.bincount(labs.ravel(),weights=a.ravel())[1:].max(initial=0)),
             other_footprint_rays=int((np.isfinite(p.low)&~known&(rr<=u.max()**2)).sum()))
    if volume is not None:
        out.update(known_other=float(a[other].sum()),unassigned=float(a[unique&~correct&~other].sum()),
                   ambiguous=float(a[use&~unique].sum()))
    return out


REFINEMENTS=("stock_offset","tapered_offset","ridge_25","ridge_50","ridge_50_half",
             "blend_2","blend_4","blend_8")


def spec(cohort):
    return {
        "paris_dev":("","","stock",False),"manual_dev":("manual_ids","manual_ids","stock_manual",True),
        "pressure":("pressure","pressure","stock_pressure",False),"gaps":("interrupted","interrupted","stock_interrupted",False),
        "manual_external":("manual_external","manual_external","stock_manual_external",True),
    }[cohort]


def case_results(cohort,stock_preset):
    labels,recovery,stockroot,manual=spec(cohort)
    src=DATA/"labels"/labels
    meta=json.loads((src/"seeds.json").read_text())
    radius=meta["area_radius_voxels"]
    axis=np.arange(-radius,radius+1,2)
    uv=np.meshgrid(axis,axis);rr=uv[0]**2+uv[1]**2
    pitch=7.91 if manual else 9.6
    targets={r["id"]:r["target_id"] for r in json.loads((src/"refs.json").read_text())["sites"]} if manual else {}
    results={}; hashes={}
    for s in meta["sites"]:
        i=s["id"]
        if cohort.endswith("_dev") and s["split"]!="dev":continue
        tpath=src/"truth"/i/"triangles.npy"
        ref=region_project(np.load(tpath),s,radius)
        id_path=src/"truth"/i/"instance_ids.npy"
        ids=np.load(id_path,mmap_mode="r") if manual else None
        stock_path=DATA/stockroot/stock_preset/i/"surface.npz"
        xyz,valid=load_geometry(stock_path)
        stock=region_project(faces(xyz,valid)[0],s,radius)
        candidates={"stock":stock}
        init=np.isfinite(stock.low)&(stock.high-stock.low<=1)&(rr<=6**2)
        offset=float(np.clip(np.median(stock.center[init]),-3,3)) if init.sum()>=5 else 0.
        candidates["stock_offset"]=displaced(stock,-offset)
        candidates["tapered_offset"]=displaced(stock,-.5*offset*np.exp(-rr/(2*64**2)))
        hashes[i]={"stock":file_hash(stock_path),"reference":file_hash(tpath)}
        for key in ("two_route","graph","two_route_phase","graph_phase"):
            path=DATA/"recover"/recovery/i/(key+".npz")
            g,m=load_geometry(path)
            candidates[key]=region_project(faces(g,m)[0],s,radius)
            hashes[i][key]=file_hash(path)
        graph=candidates["graph"]
        both=np.isfinite(stock.low)&np.isfinite(graph.low)&(stock.high-stock.low<=1)&(graph.high-graph.low<=1)
        gap=np.zeros(stock.low.shape)
        gap[both]=graph.center[both]-stock.center[both]
        for lim in (2,4,8):
            # Distance is along the seed normal; retain stock on disagreement.
            shift=np.where(both&(np.abs(gap)<=lim),.5*gap,0)
            candidates["blend_"+str(lim)]=displaced(stock,shift)
        prob_path=src/"inputs"/i/"m7.npy"
        prob=np.load(prob_path,mmap_mode="r")
        hashes[i]["foreground"]=file_hash(prob_path)
        for key,cut,gain in (("ridge_25",.25,1),("ridge_50",.5,1),("ridge_50_half",.5,.5)):
            g,m,_=refine_grid(xyz,valid,s,prob,cut,gain)
            candidates[key]=region_project(faces(g,m)[0],s,radius)
        for k,p in candidates.items():
            results.setdefault(k,{})[i]=measure(s,ref,p,uv,pitch,ids,targets.get(i))
    return results,hashes,manual


def run(cohort,stock_preset):
    outputs,hashes,manual=case_results(cohort,stock_preset)
    before=outputs["stock"]
    pairs={k:(summarize if manual else pair_summary)(before,r) for k,r in outputs.items() if k!="stock"}
    directory=OUTPUT/"regional"
    anchor="manual_dev" if manual else "paris_dev"
    # The harder 9.6-µm and gap runs used the diagnostic replay of the exact oriented preset.
    normalized=stock_preset.removesuffix("_loss")
    lock=directory/(anchor+"_"+normalized+"_choice.json")
    if cohort==anchor:
        def gate(k):
            b=pairs[k]
            return b["identity_gate"] if manual else b["positive_point_estimate"]
        valid=[k for k in REFINEMENTS if gate(k)]
        choose=max(valid,key=lambda k:pairs[k]["candidate"]["connected"]) if valid else "stock"
        body=dict(chosen=choose,passing_dev=valid,reference=cohort,stock=stock_preset,
                  source=file_hash(Path(__file__)),refinement_source=file_hash(Path(__file__).with_name("growbench_local_refine.py")),
                  protocol=file_hash(DATA.parents[1]/"research/growpatch-local-refinement-protocol.md"))
        if lock.exists() and json.loads(lock.read_text())!=body:raise RuntimeError("Candidate lock mismatch")
        save(lock,body)
    selected=json.loads(lock.read_text())
    assert file_hash(Path(__file__))==selected["source"]
    save(directory/(cohort+"_"+stock_preset+".json"),dict(cohort=cohort,stock=stock_preset,selection=selected,
          digests=hashes,rows=outputs,comparisons=pairs))
    print(cohort,stock_preset,selected["chosen"],[(k,round(v["connected_gain_percent"] or 0,2),
        round(v["risk_difference_points"] or 0,2),v.get("identity_gate",v["positive_point_estimate"])) for k,v in pairs.items()])


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("cohort",choices=("paris_dev","manual_dev","pressure","gaps","manual_external"))
    parser.add_argument("stock_preset")
    opts=parser.parse_args()
    run(opts.cohort,opts.stock_preset)
