"""Verify the portable PHerc. 125 geometry addendum, optionally from original CT."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import io
import itertools
import json
from pathlib import Path

import numpy as np
import requests
from scipy.ndimage import label
import tifffile


def sha(path):
    with path.open("rb") as stream:return hashlib.file_digest(stream,"sha256").hexdigest()


def array_hash(array):
    stream=io.BytesIO();np.save(stream,array)
    return hashlib.sha256(stream.getbuffer()).hexdigest()


def independent_ct(mesh,mask,stack,url):
    rows,cols=np.where(mask);positions=[]
    for index in np.linspace(0,len(rows)-1,11,dtype=int):
        r,c=rows[index],cols[index]
        iy,ix=r//8,c//8;v,u=(r%8)/8,(c%8)/8
        a,b=mesh[iy,ix],mesh[iy,ix+1];d,e=mesh[iy+1,ix],mesh[iy+1,ix+1]
        point=(1-v)*((1-u)*a+u*b)+v*((1-u)*d+u*e)
        du=(b-a)*(1-v)+(e-d)*v;dv=(d-a)*(1-u)+(e-b)*u
        n=np.cross(du,dv);n/=np.linalg.norm(n)
        for k in (0,4,8,12,16):positions.append(((point+(k-8)*n)[::-1],int(stack[k,r,c])))
    corners=np.array(list(itertools.product((0,1),repeat=3)))
    keys={tuple(zyx//128)for pos,_ in positions for zyx in np.floor(pos).astype(int)+corners}
    def read(key):
        uri=url+"/"+"/".join(map(str,key))
        for attempt in range(4):
            try:
                resp=requests.get(uri,timeout=40);resp.raise_for_status()
                return key,np.frombuffer(resp.content,np.uint8).reshape(128,128,128)
            except requests.RequestException:
                if attempt==3:raise
    with ThreadPoolExecutor(5) as pool:source=dict(pool.map(read,keys))
    errors=[]
    for pos,expected in positions:
        lo=np.floor(pos).astype(int);frac=pos-lo
        measured=0.
        for offset in corners:
            target=lo+offset
            val=source[tuple(target//128)][tuple(target%128)]
            measured+=val*np.where(offset,frac,1-frac).prod()
        errors.append(abs(int(np.floor(measured+.5))-expected))
    assert max(errors)<=1,(max(errors),errors)
    return dict(sampled_depth_pixels=len(errors),full_native_chunks=len(keys),exact=sum(x==0 for x in errors),
                maximum_uint8_difference=max(errors),mean_uint8_difference=float(np.mean(errors)))


def verify_case(case,online=False):
    with np.load(case/"native_ct.npz") as d:ct,mask=d["ct17"],d["mask"]
    meta=json.loads((case/"meta.json").read_text());direction=case.name.split("_")[0]
    folder=next(case.glob("*.tifxyz"))
    mesh=np.stack([tifffile.imread(folder/(axis+".tif")).astype(float) for axis in "xyz"],axis=-1)
    valid=(mesh>=0).all(-1);nodes=mesh[valid];a,b,c,d=mesh[:-1,:-1],mesh[:-1,1:],mesh[1:,1:],mesh[1:,:-1]
    good=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    expected=good.repeat(8,axis=0).repeat(8,axis=1)
    assert ct.shape==(17,*expected.shape) and mask.shape==expected.shape and np.all(mask<=expected)
    assert ct.dtype==np.uint8 and mask.dtype==bool
    assert meta["support"]==mask.sum() and array_hash(ct)==meta["ct17_sha256"] and array_hash(mask)==meta["mask_sha256"]
    assert meta["missing_objects"]==0 and np.array_equal(mask,expected)
    ranges=([float(nodes[:,i].min())for i in range(3)],[float(nodes[:,i].max())for i in range(3)])
    assert np.all((nodes[:,0]<=2744) if direction=="west" else (nodes[:,0]>=4056) if direction=="east" else
                  (nodes[:,2]>=11123) if direction=="north" else (nodes[:,2]<=9811))
    if direction in ("west","east"):assert np.all((nodes[:,2]>=9827)&(nodes[:,2]<=11107))
    else:assert np.all((nodes[:,0]>=2760)&(nodes[:,0]<=4040))
    pitch_cm=meta["pitch_um"]/10000
    area=(np.linalg.norm(np.cross(b-a,c-a),axis=-1)+np.linalg.norm(np.cross(c-a,d-a),axis=-1))*.5*pitch_cm**2
    comps,n=label(good);by_comp=np.bincount(comps[good],weights=area[good])[1:]
    distortion=json.loads((case/"distortion.json").read_text())["regular_export"]
    assert distortion["flips"]==0 and np.all(np.isfinite(nodes)) and np.all(area[good]>0)
    result=dict(source_supported_pixels=int(mask.sum()),nominal_cm2=float(mask.sum()*pitch_cm**2),
                triangulated_cm2=float(area[good].sum()),largest_connected_triangulated_cm2=float(by_comp.max()),
                components=int(n),triangles=2*int(good.sum()),distortion_p95_stretch=distortion["p50_p90_p95_max_stretch"][2],
                zero_flips=True,native_xyz_bounds=ranges,missing_objects=meta["missing_objects"],
                complete_nonzero_neighborhood_pixels=meta["all_corners_nonzero"])
    if online:result["independent_raw_reread"]=independent_ct(mesh,mask,ct,meta["native_volume_url"])
    return result


def verify(root,online=False):
    cases={f.name:verify_case(f,online) for f in sorted((root/"cases").iterdir()) if f.is_dir()}
    assert {"east","west","north","south"}<=set(cases)
    for a,b in itertools.combinations(cases.values(),2):
        low,high=np.array(a["native_xyz_bounds"]);lo,hi=np.array(b["native_xyz_bounds"])
        assert np.any((high[[0,2]]+18 < lo[[0,2]])|(hi[[0,2]]+18 < low[[0,2]]))
    out=dict(cases=cases,total_triangulated_cm2=sum(v["triangulated_cm2"]for v in cases.values()),
                total_nominal_cm2=sum(v["nominal_cm2"]for v in cases.values()),
                largest_single_connected_export_cm2=max(v["largest_connected_triangulated_cm2"]for v in cases.values()),
                distinct_native_footprints=True)
    previous={}
    for case in sorted((root/"previous_geometry").glob("PHerc*.npz")):
        with np.load(case)as d:mesh,mask,pitch=d["mesh"],d["mask"],float(d["pitch_um"])
        nodes=mesh[(mesh>=0).all(-1)].astype(float)
        low,high=nodes.min(0),nodes.max(0)
        a,b,c,e=mesh[:-1,:-1],mesh[:-1,1:],mesh[1:,1:],mesh[1:,:-1]
        area=.5*(np.linalg.norm(np.cross(b-a,c-a),axis=-1)+np.linalg.norm(np.cross(c-a,e-a),axis=-1))*(pitch/10000)**2
        mh,mw=mask.shape;assert (mh,mw)==((mesh.shape[0]-1)*8,(mesh.shape[1]-1)*8)
        observed=mask.reshape(mh//8,8,mw//8,8).mean(axis=(1,3))
        assert np.all((observed==1)|(observed==0))
        row=dict(nominal_cm2=float(mask.sum()*(pitch/10000)**2),triangulated_cm2=float(area[observed==1].sum()),
                 native_xyz_bounds=[low.tolist(),high.tolist()],overlap_checks={})
        if case.stem.startswith("PHerc0125"):
            for other,entry in cases.items():
                lo,hi=np.array(entry["native_xyz_bounds"])
                gap=np.maximum(lo-high,low-hi)[[0,2]].max()
                assert gap>18,(case.stem,other,gap)
                row["overlap_checks"][other]=float(gap)
        previous[case.stem]=row
    if previous:
        assert set(previous)=={"PHerc0125_y3310","PHerc0125_y3706","PHerc0268_y2991"}
        old_nominal=sum(v["nominal_cm2"]for v in previous.values())
        old_native=sum(v["triangulated_cm2"]for v in previous.values())
        out.update(previous_interior_cases=previous,previous_interior_nominal_cm2=old_nominal,
                   previous_interior_triangulated_cm2=old_native,combined_interior_nominal_cm2=old_nominal+out["total_nominal_cm2"],
                   combined_interior_triangulated_cm2=old_native+out["total_triangulated_cm2"])
        with np.load(root/"previous_geometry/old_shared_grid.npz")as raw:
            delta=(raw["y3706"]-raw["y3310"])[raw["qualified3310"]&raw["qualified3706"]]
            out["previous_pherc125_native_y_separation_vox"]=dict(shared_qualified_nodes=int(len(delta)),
                minimum=float(delta.min()),median=float(np.median(delta)),maximum=float(delta.max()))
    return out


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("folder",nargs="?",type=Path,default=Path(__file__).resolve().parent)
    p.add_argument("--online",action="store_true")
    p.add_argument("--output",type=Path)
    a=p.parse_args()
    lock=a.folder/"SHA256SUMS"
    if lock.exists():
        for entry in lock.read_text().splitlines():
            expected,rel=entry.split("  ",1);f=a.folder/rel
            assert sha(f)==expected,rel
    report=json.dumps(verify(a.folder,a.online),indent=2);print(report)
    if a.output:a.output.write_text(report)
