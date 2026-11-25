"""Select and freeze Paris4 benchmark sites without consulting algorithm outputs.

All coordinates in the primary benchmark are XYZ at 9.6 um. Published M7
array storage and the input crops use ZYX. Only seeds.json and inputs/ may
be read by the competing surface generators. refs.json and truth/ are for
the separate evaluator.
"""
import argparse
import hashlib
import io
import json
import threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
import requests
import tifffile
from numcodecs import get_codec
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "results/growbench/labels"
INDEX = ROOT / "data/round3/verified_patches/index.json"
BASE = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHercParis4/"
SPIRAL = BASE+"representations/spiral/20260411134726-spiral-dataset-20261019084150/"
M7 = BASE+"representations/predictions/surfaces/20260411134726-surface-20260413222639-surface-m7-L2-th0.2.zarr/0"
CT = BASE+"volumes/20260411134726-2.400um-0.2m-78keV-masked.zarr/2"
SEED = "growbench-v1:"
SPLITS = ["dev"]*4 + ["test"]*8 + ["confirm"]*4
THREAD = threading.local()


def digest(data):
    return hashlib.sha256(data).hexdigest()


def write(path,obj):
    path.parent.mkdir(parents=True,exist_ok=True)
    tmp=path.with_suffix(".tmp")
    tmp.write_text(json.dumps(obj,indent=2)+"\n",encoding="utf-8")
    tmp.replace(path)


def fetch(url,missing=False):
    if not hasattr(THREAD,"s"):
        THREAD.s=requests.Session()
    for attempt in range(4):
        try:
            r=THREAD.s.get(url,timeout=(12,55))
            if missing and r.status_code==404:
                return None
            r.raise_for_status()
            return r.content
        except requests.RequestException:
            if attempt==3:
                raise


def patch(name,cache_root=OUT):
    path=cache_root/"sources"/name
    def get(file):
        dest=path/file
        if not dest.exists():
            dest.parent.mkdir(parents=True,exist_ok=True)
            dest.write_bytes(fetch(SPIRAL+"verified_patches/"+name+"/"+file))
        return dest.read_bytes()
    with ThreadPoolExecutor(4) as pool:
        raw=dict(zip(("x.tif","y.tif","z.tif","meta.json"),pool.map(get,("x.tif","y.tif","z.tif","meta.json"))))
    xyz=np.stack([tifffile.imread(io.BytesIO(raw[a+".tif"])) for a in "xyz"],axis=-1).astype(np.float64)
    meta=json.loads(raw["meta.json"])
    return xyz,meta,{k:digest(v) for k,v in raw.items()}


def triangulate(xyz):
    valid=np.isfinite(xyz).all(-1)&(xyz>=0).all(-1)
    ok=valid[1:,1:]&valid[1:,:-1]&valid[:-1,1:]&valid[:-1,:-1]
    a,b,c,d=xyz[:-1,:-1][ok],xyz[:-1,1:][ok],xyz[1:,:-1][ok],xyz[1:,1:][ok]
    tris=np.concatenate([np.stack((a,b,c),1),np.stack((d,c,b),1)])
    edges=np.linalg.norm(tris-np.roll(tris,1,axis=1),axis=-1)
    return tris[(edges.max(1)<80)&(edges.min(1)>.01)]


def raster(tris,origin,side=257):
    grid=np.full((side,side),np.nan,dtype=np.float32)
    collisions=np.zeros(grid.shape,bool)
    for triangle in tris:
        tri=triangle-origin
        xlo=max(0,int(np.ceil(tri[:,0].min())));xhi=min(side-1,int(np.floor(tri[:,0].max())))
        zlo=max(0,int(np.ceil(tri[:,2].min())));zhi=min(side-1,int(np.floor(tri[:,2].max())))
        if xlo>xhi or zlo>zhi:
            continue
        za,xa=np.mgrid[zlo:zhi+1,xlo:xhi+1]
        a,b,c=tri
        u=b-a;v=c-a;det=u[0]*v[2]-u[2]*v[0]
        if abs(det)<.01:
            continue
        alpha=((xa-a[0])*v[2]-(za-a[2])*v[0])/det
        beta=((za-a[2])*u[0]-(xa-a[0])*u[2])/det
        yy=a[1]+alpha*u[1]+beta*v[1]
        valid=(alpha>=-1e-6)&(beta>=-1e-6)&(alpha+beta<=1+1e-6)&(yy>=0)&(yy<=side-1)
        sub=grid[zlo:zhi+1,xlo:xhi+1]
        collisions[zlo:zhi+1,xlo:xhi+1] |= valid & np.isfinite(sub) & (abs(yy+origin[1]-sub)>2)
        sub[valid]=yy[valid]+origin[1]
    grid[collisions]=np.nan
    return grid,collisions


def site(name,sites,used):
    xyz,meta,hashes=patch(name)
    members=set(meta.get("member_ids",[name]))
    if members&used:
        return
    valid=np.isfinite(xyz).all(-1)&(xyz>=0).all(-1)
    normals=np.cross(xyz[1:-1,2:]-xyz[1:-1,:-2],xyz[2:,1:-1]-xyz[:-2,1:-1])
    good=np.zeros_like(valid)
    good[1:-1,1:-1]=(np.abs(normals[...,1])>.72*np.maximum(np.linalg.norm(normals,axis=-1),1e-7))
    inside=distance_transform_edt(np.pad(valid,1))[1:-1,1:-1]
    good &= inside>=4
    if not good.any():return
    i,j=np.unravel_index(np.argmax(np.where(good,inside,-1)),inside.shape)
    seed=xyz[i,j];origin=np.rint(seed).astype(int)-128
    if any(np.max(abs(seed-s["seed_xyz"]))<650 for s in sites):
        return
    triangles=triangulate(xyz)
    triangles=triangles[((triangles.max(1)>=origin)&(triangles.min(1)<=origin+256)).all(1)]
    if len(triangles)<12:return
    yy,collisions=raster(triangles,origin)
    z,x=np.mgrid[:257,:257];xx=x+origin[0];zz=z+origin[2]
    disk=(xx-seed[0])**2+(zz-seed[2])**2<96**2
    if np.isfinite(yy[disk]).mean()<.70 or collisions[disk].mean()>.03:return
    local=np.isfinite(yy)&((xx-seed[0])**2+(zz-seed[2])**2<=8**2)
    if local.sum()<80:return
    design=np.column_stack((xx[local]-seed[0],zz[local]-seed[2],np.ones(local.sum())))
    ax,az,dy=np.linalg.lstsq(design,yy[local]-seed[1],rcond=None)[0]
    normal=np.array([-ax,1.,-az]);normal/=np.linalg.norm(normal)
    if normal[1]<.70:return
    u=np.array([1.,0.,0.])-normal[0]*normal;u/=np.linalg.norm(u)
    v=np.cross(u,normal)
    n=len(sites);split=SPLITS[n];sid=f"{split}_{SPLITS[:n].count(split)+1:02d}"
    case=dict(id=sid,split=split,seed_xyz=seed.tolist(),origin_xyz=origin.tolist(),normal_xyz=normal.tolist(),
              tangent_u_xyz=u.tolist(),tangent_v_xyz=v.tolist(),cube_shape_zyx=[257]*3)
    dest=OUT/"truth"/sid;dest.mkdir(exist_ok=True,parents=True)
    ref=np.stack([xx,yy,zz],-1).astype(np.float32);ref[~np.isfinite(yy)]=np.nan
    np.save(dest/"ref_xyz.npy",ref);np.save(dest/"triangles.npy",triangles.astype("f4"))
    info=dict(id=sid,name=name,source=SPIRAL+"verified_patches/"+name+"/",hashes=hashes,
              members=sorted(members),source_tool=meta.get("source"),mesh_scale=meta.get("scale"),source_index_ij=[int(i),int(j)],
              projected_disk_coverage=float(np.isfinite(yy[disk]).mean()),ambiguous_multivalued_fraction=float(collisions[disk].mean()),
              triangles=len(triangles),triangle_max_edge=80,local_plane_sheet_offset=float(dy),
              ref_sha256=digest((dest/"ref_xyz.npy").read_bytes()),triangle_sha256=digest((dest/"triangles.npy").read_bytes()))
    return case,info,members


def select():
    if (OUT/"seeds.json").exists():
        raise SystemExit("Seeds already locked.")
    idx=json.loads(INDEX.read_text())
    options=[]
    for name,q in idx.items():
        if not q["bbox_l2"] or (q.get("area_cm2") or 0)<=.05:continue
        extent=np.subtract(*q["bbox_l2"][::-1])
        if min(extent[0],extent[2])>210 and max(extent)<2200:
            options.append(name)
    options.sort(key=lambda name:digest((SEED+name).encode()))
    sites=[];refs=[];used=set();inspected=[]
    for name in options:
        result=site(name,sites,used);inspected.append(name)
        if result:
            s,r,m=result;sites.append(s);refs.append(r);used.update(m)
            print("found",s["id"],s["seed_xyz"],name,flush=True)
        if len(sites)==len(SPLITS):break
    if len(sites)!=len(SPLITS):
        raise ValueError("Insufficient eligible reference sites")
    write(OUT/"refs.json",dict(verified_source=SPIRAL+"verified_patches/",README="https://scrollprize.org/data/datasets/spiral-input-PHercParis4-README.md",
          initial_index_sha256=digest(INDEX.read_bytes()),initial_count=len(idx),eligible_metadata=len(options),salt=SEED,
          attempted_before_completion=inspected,rule="Sort eligible public patch IDs by SHA256(salt+name), never consulting CT, M7, detector or fitted results. Anchor the most interior graph-compatible point, first in row-major ties. Require X/Z extents >210, largest extent <2200, area >0.05 nominal cm2, >=70% reference coverage on central 96vx disk, <=3% ambiguous reference rays, abs normal Y >=0.70, no reused member IDs or overlapping cubes (Linf seed separation >=650). See script for full lock.",sites=refs))
    write(OUT/"seeds.json",dict(benchmark="Paris4-gp9p6-v1",frame="global XYZ in 9.6um voxels = level 2 of CT; M7 and cube arrays ZYX",
          ct=CT,m7=M7,area_radius_voxels=96,excluded_seed_radius_voxels=12,sites=sites))
    print("LOCK",digest((OUT/"seeds.json").read_bytes()),flush=True)


def extract(site, dataset, url,output=OUT,cache_root=OUT):
    info=json.loads(fetch(url+"/.zarray"))
    shape=np.array(info["chunks"]);origin=np.asarray(site["origin_xyz"])[::-1];end=origin+site["cube_shape_zyx"]
    out=np.zeros(site["cube_shape_zyx"],dtype=info["dtype"])
    digestrows=[]
    codec=get_codec(info["compressor"]) if info["compressor"] else None
    for j in np.ndindex(tuple((end-1)//shape-origin//shape+1)):
        idx=np.array(j)+origin//shape;k="/".join(map(str,idx))
        cache=cache_root/"cache"/dataset/("_".join(map(str,idx))+".bin")
        if cache.exists():
            b=cache.read_bytes() or None
        else:
            b=fetch(url+"/"+k,True)
            cache.parent.mkdir(parents=True,exist_ok=True);cache.write_bytes(b or b"")
        digestrows.append([k,None if b is None else digest(b)])
        if b is None:continue
        vals=np.frombuffer(codec.decode(b) if codec else b,dtype=info["dtype"]).reshape(shape)
        lo=np.maximum(origin,idx*shape);hi=np.minimum(end,(idx+1)*shape)
        out[tuple(slice(x,y) for x,y in zip(lo-origin,hi-origin))]=vals[tuple(slice(x,y) for x,y in zip(lo-idx*shape,hi-idx*shape))]
    dest=output/"inputs"/site["id"];dest.mkdir(exist_ok=True,parents=True)
    path=dest/(dataset+".npy");temp=dest/(dataset+".partial.npy");np.save(temp,out);temp.replace(path)
    write(dest/(dataset+".json"),dict(source=url,origin_xyz=site["origin_xyz"],compressed_chunk_sha256=digestrows,
          shape_zyx=list(out.shape),dtype=str(out.dtype),array_file_sha256=digest(path.read_bytes()),nonzero=int(np.count_nonzero(out))))
    print("ready",site["id"],dataset,flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("action",choices=["select","m7","ct"]);a=p.parse_args()
    if a.action=="select":select()
    else:
        cases=json.loads((OUT/"seeds.json").read_text())["sites"]
        with ThreadPoolExecutor(12) as ex:
            list(ex.map(lambda s:extract(s,a.action,CT if a.action=="ct" else M7),cases))
