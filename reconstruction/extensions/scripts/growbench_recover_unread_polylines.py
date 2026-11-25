"""Check new geometry against individual organizer polylines selected inside the old area."""
import hashlib
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from first_letters_0125 import F,GRID,getgrid
from growbench_recover_unread import DEST,OLD


def crossings(paths,axis,value,boundary,sign,old_y,old_good):
    target=np.arange(-128,1,16)
    branches=[]
    if old_good.sum()<5 or np.ptp(target[old_good])<80:return []
    for i,path in enumerate(paths):
        lateral=(path[:,0 if axis=="xy" else 1]-boundary)*sign
        vertical=path[:,1 if axis=="xy" else 0]
        # One branch must enter the inner edge and reach the boundary itself.
        if lateral.min()>-128 or lateral.max()<0:continue
        for forward in (True,False):
            u=lateral if forward else lateral[::-1];y=vertical if forward else vertical[::-1]
            entries=np.flatnonzero((u[:-1]<=-128)&(u[1:]>-128))
            for start in entries:
                if abs(float(y[start])-float(old_y[0]))>25:continue
                pos=u[start:];heights=y[start:]
                peak=np.maximum.accumulate(pos)
                # Organizer polylines sample ordinary connected segments about
                # every 20 native voxels. Allow that native representation and
                # stop at gaps larger than 32 voxels.
                broken=np.flatnonzero((peak-pos>8)|np.r_[False,np.hypot(np.diff(pos),np.diff(heights))>32])
                stop=int(broken[0])if len(broken)else len(pos)
                pos=pos[:stop];heights=heights[:stop];peak=peak[:stop]
                if len(pos)<2 or peak[-1]<0:continue
                advances=np.r_[True,np.diff(peak)>0]
                pos,heights=pos[advances],heights[advances]
                estimate=np.interp(target,pos,heights)
                delta=abs(estimate-old_y)[old_good]
                median=float(np.median(delta));worst=float(np.max(delta))
                if median<=2.5 and worst<=5:
                    branches.append((median,i,forward,int(start),pos,heights))
    return sorted(branches,key=lambda branch:branch[0])


def audit(direction):
    directory=DEST.parent/("pherc125_"+("extension" if direction=="east" else direction))
    # A fixed sweep of original 16-voxel mesh lines, independent of new-path evidence.
    axis="xy" if direction in ("east","west") else "yz"
    transverse_center=10467 if axis=="xy" else 3400
    boundary={"east":4040,"west":2760,"north":11107,"south":9827}[direction]
    sign=1 if direction in ("east","north")else -1
    with np.load(OLD) as old,np.load(directory/"consensus.npz")as new:
        g="x" if axis=="xy"else "z";trans="z" if axis=="xy"else "x"
        old_axis=old[g];old_cross=old[trans]
        old_y=old["fit"] if axis=="xy" else old["fit"].T
        old_valid=old["qualified"] if axis=="xy" else old["qualified"].T
        new_axis=new[g];new_cross=new[trans]
        new_y=new["y"] if axis=="xy" else new["y"].T
        new_ok=new["new"] if axis=="xy" else new["new"].T
    records=[];deltas=[]
    old_ix=np.searchsorted(old_axis,boundary+sign*np.arange(-128,1,16))
    for val in transverse_center+np.arange(-480,481,32):
        val=int(val);out=dict(plane=val,axis=axis);records.append(out)
        oi=int(np.searchsorted(old_cross,val));ni=int(np.searchsorted(new_cross,val))
        x=(new_axis-boundary)*sign
        use=new_ok[ni]&(x>=16)&(x%8==0)
        out["new_consensus_nodes"]=int(use.sum())
        if not use.any():continue
        point_y=old_y[oi,old_ix];good=old_valid[oi,old_ix]
        if good.sum()<5:continue
        paths=getgrid(axis,val)
        source=F/axis/f"{val:06d}.grid"
        out.update(url=GRID+"/"+axis+"/"+source.name,sha256=hashlib.sha256(source.read_bytes()).hexdigest())
        ranked=crossings(paths,axis,val,boundary,sign,point_y,good)
        out["old_anchor_matches"]=len(ranked)
        if len(ranked)!=1:continue
        median,idx,forward,start,cu,cy=ranked[0]
        inside=use&(x<=cu[-1])
        out.update(polyline=idx,forward=forward,entry_vertex=start,old_anchor_median_y_error=median,
                   outward_reach_vox=int(cu[-1]),new_tested=int(inside.sum()))
        if not inside.any():continue
        delta=new_y[ni,inside]-np.interp(x[inside],cu,cy);deltas.extend(delta)
        out.update(within_3_vox=int((abs(delta)<=3).sum()),within_6_vox=int((abs(delta)<=6).sum()),
                   signed_differences=delta.tolist())
    arr=np.array(deltas);n=len(arr)
    report=dict(direction=direction,planes_checked=len(records),planes_with_new_consensus=sum(r["new_consensus_nodes"]>0 for r in records),
                planes_with_unique_old_anchor=sum(r.get("old_anchor_matches")==1 for r in records),
                planes_reaching_new=sum(r.get("new_tested",0)>0 for r in records),new_consensus_nodes=sum(r["new_consensus_nodes"] for r in records),
                new_nodes_on_single_original_polyline=n,
                within_three_voxels=int((abs(arr)<=3).sum()),within_six_voxels=int((abs(arr)<=6).sum()),
                median_abs_y_error_vox=float(np.median(abs(arr)))if n else None,
                control_within_three_by_offset={str(o):int((abs(arr+o)<=3).sum()) for o in (-9,-5,0,5,9)},
                method="One direction-consistent continuous organizer curve must uniquely match the old qualified surface across an interior 128-voxel band. No new coordinates enter curve choice. Reversals of more than 8 voxels or segment gaps greater than 32 voxels terminate continuation. Source polyline segments have ordinary lengths around 20 voxels in the old band. Agreement is measured on newly fitted grid points 8 voxels apart along fixed planes, separated by 32 voxels.",
                readout="Both representations originate in the organizer M7. A single crossing contour can be absent or switch layers. The denominator only includes the subset joined by the same stored polyline.",records=records)
    (directory/"organizer_continuity.json").write_text(json.dumps(report,indent=2))
    print(direction,{k:report[k]for k in ("new_consensus_nodes","new_nodes_on_single_original_polyline","within_three_voxels","planes_reaching_new")},flush=True)


if __name__=="__main__":
    with ThreadPoolExecutor(2) as pool:list(pool.map(audit,("east","north","west","south")))
