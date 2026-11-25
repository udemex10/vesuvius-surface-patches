"""Independent manual sheet-ID benchmark from paired 2024 Scroll 1 cubes.

No instance ID enters the common input or its preprocessing. Reference ID
and geometry are exclusively for the scorer; both competitors get the same
seed, local initialization plane and separately published semantic input.
"""
import gzip
import hashlib
import itertools
import json
import re
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from numcodecs import get_codec
from scipy.ndimage import gaussian_filter
from growbench_labels import OUT as PRIMARY, SPLITS, digest, fetch, write

OUT=PRIMARY/"manual_ids"
BASE="https://dl.ash2txt.org/full-scrolls/Scroll1/PHercParis4.volpkg/"
ROOT=BASE+"volumetric-instance-labels/"
PERMITTED={"volume.nrrd":("instance-labels-harmonized","volume"),"instance.nrrd":("instance-labels-harmonized","mask"),
           "semantic.nrrd":("semantic-labels","mask")}
SALT="growbench-manual-v1:"


def original(name,file,cache_root=OUT):
    kind,ft=PERMITTED[file]
    url=f"{ROOT}{kind}/{name}/{name}_{ft}.nrrd"
    dest=cache_root/"sources"/name/file
    if not dest.exists():
        dest.parent.mkdir(parents=True,exist_ok=True)
        dest.write_bytes(fetch(url))
    data=dest.read_bytes()
    header,body=data.split(b"\n\n",1)
    fields=dict(re.findall(r"^(type|sizes|encoding|space origin):\s*(.+)$",header.decode(),re.M))
    assert fields["sizes"]=="256 256 256" and fields["encoding"] in ("gzip","raw")
    dt={"unsigned char":"u1","uint8":"u1","uint16":"<u2","unsigned short":"<u2"}[fields["type"]]
    # NRRD lists z,y,x as its declared axes in this particular release (see
    # coordinate verification). The FIRST axis is the fastest file axis.
    values=gzip.decompress(body) if fields["encoding"]=="gzip" else body
    volume=np.frombuffer(values,dtype=dt).reshape((256,)*3,order="F")
    return np.ascontiguousarray(volume),dict(url=url,sha256=digest(data),fields=fields)


def runs(vol):
    """Extract same-valued Y runs separately for every X/Z column."""
    data=np.ascontiguousarray(vol.transpose(0,2,1))
    padded=np.pad(data,((0,0),(0,0),(1,1)),constant_values=0)
    change=padded[:,:,1:]!=padded[:,:,:-1]
    z,x,s=np.nonzero(change & (padded[:,:,1:]!=0))
    zz,xx,t=np.nonzero(change & (padded[:,:,:-1]!=0))
    assert np.array_equal(z,zz) and np.array_equal(x,xx)
    return z,x,s,t,data[z,x,s]


def common_input(sem):
    z,x,s,t,_=runs(sem.astype("u1"))
    midpoint=(s+t-1)/2
    yy=midpoint.astype(int);f=midpoint-yy
    values=np.zeros(sem.shape,np.float32)
    np.add.at(values,(z,yy,x),1-f)
    np.add.at(values,(z,np.minimum(yy+1,255),x),f)
    values=gaussian_filter(values,(.65,1.15,.65),mode="constant")
    return np.rint(np.clip(values*2.89,0,1)*255).astype("u1")


def mesh(y,origin):
    z,x=np.mgrid[:256,:256]
    xyz=np.stack([x+origin[0],y+origin[1],z+origin[2]],-1).astype("f4")
    good=np.isfinite(y);xyz[~good]=np.nan
    # Native dense indexed surface, without triangles spanning missing
    # rays or a depth discontinuity of >4 voxels between columns.
    a,b,c,d=xyz[:-1,:-1],xyz[:-1,1:],xyz[1:,:-1],xyz[1:,1:]
    parts=[]
    for aa,bb,cc in ((a,b,c),(d,c,b)):
        ok=np.isfinite(aa[...,1])&np.isfinite(bb[...,1])&np.isfinite(cc[...,1])
        ok &= np.maximum.reduce([abs(aa[...,1]-bb[...,1]),abs(aa[...,1]-cc[...,1]),abs(bb[...,1]-cc[...,1])])<=4
        parts.append(np.stack((aa[ok],bb[ok],cc[ok]),1))
    return xyz,np.concatenate(parts)


def choose(labels,excluded=()):
    z,x,s,t,ids=runs(labels)
    candidates=[]
    for cz,cx in itertools.product((128,104,152),(128,104,152)):
        ix=np.flatnonzero((z==cz)&(x==cx)&((s+t-1)/2>=99)&((s+t-1)/2<=157))
        for j in ix:
            cy=(s[j]+t[j]-1)*.5
            candidates.append(((cx-128)**2+(cz-128)**2+(cy-128)**2,int(ids[j]),cx,cy,cz))
    zz,xx=np.mgrid[:256,:256];cache={}
    for _,tid,cx,cy,cz in sorted(candidates):
        if tid in excluded:continue
        if tid not in cache:
            ix=np.flatnonzero((ids==tid)&(s>0)&(t<256))
            count=np.zeros((256,256),np.uint16)
            np.add.at(count,(z[ix],x[ix]),1)
            y=np.full(count.shape,np.nan,np.float32)
            y[z[ix],x[ix]]=(s[ix]+t[ix]-1)/2.
            y[count!=1]=np.nan
            cache[tid]=y
        yy=cache[tid]
        good=np.isfinite(yy)
        disk=(xx-cx)**2+(zz-cz)**2
        local=(disk<=8**2)&good
        if local.sum()<100 or np.mean(good[disk<=96**2])<.65 or not good[cz,cx] or yy[cz,cx]!=cy:continue
        k=np.column_stack((xx[local]-cx,zz[local]-cz,np.ones(local.sum())))
        ax,az,off=np.linalg.lstsq(k,yy[local]-cy,rcond=None)[0]
        norm=np.array([-ax,1.,-az]);norm/=np.linalg.norm(norm)
        if norm[1]<.65 or np.median(abs(k@np.array([ax,az,off])-(yy[local]-cy)))>2.:continue
        u=np.array([1.,0,0])-norm[0]*norm;u/=np.linalg.norm(u);v=np.cross(u,norm)
        return tid,np.asarray([cx,cy,cz]),norm,u,v,yy


def registration(name,ct):
    origin=np.array(list(map(int,name.split("_"))));key=(origin+128)//128;lo=key*128-origin
    sz=np.minimum(128,256-lo)
    sub=tuple(slice(x,x+n,4) for x,n in zip(lo,sz))
    rows={}
    for scan in "AB":
        url=BASE+f"volumes_zarr_standardized/54keV_7.91um_Scroll1{scan}.zarr/0"
        metadata=json.loads(fetch(url+"/.zarray"))
        raw=fetch(url+"/"+"/".join(map(str,key)),True) if np.all(key*128<metadata["shape"]) else None
        if raw is None:continue
        ref=np.frombuffer(get_codec(metadata["compressor"]).decode(raw),"u1").reshape((128,)*3)[tuple(slice(0,n,4) for n in sz)]
        scores={}
        for perm in itertools.permutations(range(3)):
            a=ct.transpose(perm)[sub]
            score=float(np.corrcoef(a.ravel(),ref.ravel())[0,1]) if a.std() and ref.std() else 0.
            scores[str(perm)]=score
        rows[scan]=dict(chunk_zyx=key.tolist(),url=url,sha256=digest(raw),correlations=scores)
    return rows


def select():
    if (OUT/"seeds.json").exists():raise SystemExit("Manual seeds already locked; use --inputs")
    sets=[set(re.findall(r'href="(\d{5}_\d{5}_\d{5})/"',fetch(ROOT+c+"/").decode())) for c in ("instance-labels-harmonized","semantic-labels")]
    names=sorted(set.intersection(*sets),key=lambda n:digest((SALT+n).encode()))
    sites=[];refs=[];checked=[];names_used=[];origins=[]
    for name in names:
        origin=np.array(list(map(int,name.split("_"))))[::-1]
        if any(np.max(abs(origin-v))<384 for v in origins):continue
        labels,linfo=original(name,"instance.nrrd")
        choice=choose(labels);checked.append(name)
        if not choice:continue
        target,point,normal,u,v,refy=choice
        n=len(sites);split=SPLITS[n];sid=f"{split}_{SPLITS[:n].count(split)+1:02d}"
        pub=dict(id=sid,split=split,seed_xyz=(point+origin).tolist(),origin_xyz=origin.tolist(),normal_xyz=normal.tolist(),
                 tangent_u_xyz=u.tolist(),tangent_v_xyz=v.tolist(),cube_shape_zyx=[256]*3,algorithm_input="semantic_foreground_centers_uint8")
        truth=OUT/"truth"/sid;truth.mkdir(parents=True,exist_ok=True)
        np.save(truth/"instance_ids.npy",labels)
        xyz,triangles=mesh(refy,origin)
        np.save(truth/"ref_xyz.npy",xyz);np.save(truth/"triangles.npy",triangles)
        refs.append(dict(id=sid,source_cube_zyx=name,target_id=target,source=linfo,label_ids=np.unique(labels).tolist(),
                    midpoint_definition="Per-voxel native Y coordinate halfway between first and last occupied voxels of a unique target-ID run; censor open-ended and multivalued rays.",
                    unknown_value=0,source_id="manually harmonized instance IDs; unknown/background zero untouched"))
        sites.append(pub);origins.append(origin);names_used.append(name)
        print("selected",sid,name,target,point.tolist(),flush=True)
        if len(sites)==len(SPLITS):break
    if len(sites)!=len(SPLITS):raise ValueError(f"Only {len(sites)} qualifying sites")
    write(OUT/"refs.json",dict(source=ROOT+"README.txt",salt=SALT,ordered_source_ids=names,
         geometry_checked=checked,selection="Intersection of paired semantic and harmonized-instance folders, SHA256 ordering, origins >=384 in L-infinity, one eligible label nearest cube center across nine fixed columns, graph inclination, >=65% assigned rays on 96-voxel disk. Outcomes and raw/predicted intensities unseen.",sites=refs))
    write(OUT/"seeds.json",dict(benchmark="Paris4-legacy-manual-IDs-v1",frame="global XYZ historical Scroll 1, 7.91um; source cubes arranged ZYX",
         voxel_size_um=7.91,area_radius_voxels=96,excluded_seed_radius_voxels=12,sites=sites))
    print("MANUAL LOCK",digest((OUT/"seeds.json").read_bytes()),flush=True)


def prepare(output=OUT):
    sites=json.loads((output/"seeds.json").read_text())["sites"]
    names_used=[r["source_cube_zyx"] for r in json.loads((output/"refs.json").read_text())["sites"]]
    def inputs(info):
        s,name=info;sid=s["id"];dest=output/"inputs"/sid;dest.mkdir(exist_ok=True,parents=True)
        if (output/"truth"/sid/"source_qa.json").is_file():
            return
        sem,sinfo=original(name,"semantic.nrrd")
        vals=np.unique(sem)
        if len(vals)>2:raise ValueError(f"Nonbinary semantic input {vals}")
        sem=sem!=0
        prob=common_input(sem);np.save(dest/"m7.partial.npy",prob);(dest/"m7.partial.npy").replace(dest/"m7.npy")
        write(dest/"m7.json",dict(source=sinfo,common_preprocessor="All positive Y-runs of separately published semantic foreground converted to unit-weight linearly deposited midpoints, then isotropic lateral sigma 0.65 and Y sigma 1.15, ×2.89, quantized uint8.",sha256=digest((dest/"m7.npy").read_bytes())))
        ct,cinfo=original(name,"volume.nrrd");np.save(dest/"ct.partial.npy",ct);(dest/"ct.partial.npy").replace(dest/"ct.npy")
        np.save(dest/"semantic.npy",sem)
        write(dest/"ct.json",cinfo)
        lab=np.load(output/"truth"/sid/"instance_ids.npy",mmap_mode="r")>0
        reg=registration(name,ct)
        write(output/"truth"/sid/"source_qa.json",dict(semantic_manual_voxel_iou=np.count_nonzero(lab&sem)/np.count_nonzero(lab|sem),
             fraction_manual_inside_semantic=np.count_nonzero(lab&sem)/np.count_nonzero(lab),registration=reg,
             density=float(np.mean(lab)),semantic_values=vals.tolist()))
        print("inputs",sid,flush=True)
    with ThreadPoolExecutor(5) as pool:list(pool.map(inputs,zip(sites,names_used)))


if __name__=="__main__":
    import argparse
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--inputs",action="store_true",help="Resume input acquisition after freezing sites.")
    args=ap.parse_args()
    if not args.inputs:select()
    prepare()
