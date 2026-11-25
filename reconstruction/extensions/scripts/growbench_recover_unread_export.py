"""Flatten and check the largest PHerc. 125 extension with original native attenuation."""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont
from scipy.ndimage import gaussian_filter, label, map_coordinates, minimum_filter
import tifffile

from first_letters_intrinsic import distort, flatten, raster
from first_letters_native_sample import sample_native
from growbench_recover_unread import DEST, PITCH
from zarr_crop import read_crop


CT=("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0125/"
    "volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr/0")
EXPORT=DEST/"flat"
ROOT=Path(__file__).resolve().parents[1]
DIRECTION="east"
COMPONENT_RANK=1


def selected_component(mask):
    ids,n=label(mask)
    if n<COMPONENT_RANK:raise ValueError("Missing requested component")
    order=np.argsort(np.bincount(ids.ravel())[1:])[::-1]+1
    return ids==order[COMPONENT_RANK-1]


def faces(mask):
    good=mask[:-1,:-1]&mask[1:,:-1]&mask[:-1,1:]&mask[1:,1:]
    iy,ix=np.where(good);m=mask.shape[1];a=iy*m+ix
    return np.vstack([np.column_stack([a,a+1,a+1+m]),np.column_stack([a,a+1+m,a+m])]),good


def mesh():
    with np.load(DEST/"consensus.npz") as d:
        new=selected_component(d["new"])
        quad_source=minimum_filter(new.astype("u1"),size=9,mode="constant")!=0
        z,x=np.meshgrid(d["z"],d["x"],indexing="ij")
        grid=np.stack([x,d["y"],z],-1)[::4,::4]
        sel=quad_source[::4,::4]
        fine_x0=float(d["x"][0]);fine_z0=float(d["z"][0])
    tri,_=faces(sel)
    nodes=np.unique(tri);lookup=np.full(sel.size,-1,int);lookup[nodes]=np.arange(len(nodes));tri=lookup[tri]
    pos=grid.reshape(-1,3)[nodes]
    uv0=pos[:,[0,2]]-pos[:,[0,2]].min(0)
    uv1=flatten(pos,uv0,tri)
    before=distort(pos,uv0,tri);after=distort(pos,uv1,tri)
    if after["flips"]:
        raise ValueError(f"Intrinsic flattening inverted {after['flips']} triangles")
    world,valid,origin,clash=raster(pos,tri,uv1,8)
    # Require every exported quad midpoint to project inside full original support.
    mid=(world[:-1,:-1]+world[1:,1:]+world[:-1,1:]+world[1:,:-1])*.25
    indices=np.stack([(mid[...,2]-fine_z0)/2,(mid[...,0]-fine_x0)/2])
    valid_mid=map_coordinates(quad_source.astype("u1"),indices,order=0,mode="constant")>0
    _,quad=faces(valid);bad=quad&~valid_mid
    for dy,dx in ((0,0),(0,1),(1,0),(1,1)):
        valid[dy:dy+bad.shape[0],dx:dx+bad.shape[1]] &= ~bad
    final_tri,_=faces(valid);hh,ww=np.indices(valid.shape)
    final_dist=distort(world.reshape(-1,3),np.stack([ww*8,hh*8],-1).reshape(-1,2),final_tri)
    suffix="" if COMPONENT_RANK==1 else f"_{COMPONENT_RANK}"
    folder=EXPORT/f"PHerc0125_{DIRECTION}{suffix}.tifxyz";folder.mkdir(parents=True,exist_ok=True)
    for j,axis in enumerate("xyz"):
        tifffile.imwrite(folder/(axis+".tif"),np.where(valid,world[...,j],-1).astype("f4"))
    (folder/"meta.json").write_text(json.dumps(dict(format="tifxyz",type="seg",name=folder.stem,
        uuid=folder.stem,scale=[.125,.125],bbox=[world[valid].min(0).tolist(),world[valid].max(0).tolist()])))
    record=dict(source_geometry=before,intrinsic=after,regular_export=final_dist,
                retained_vertices=int(valid.sum()),source_triangles=len(tri),regular_triangles=len(final_tri),
                overlap_rejections=int(clash.sum()),hole_rejections=int(bad.sum()))
    (EXPORT/"distortion.json").write_text(json.dumps(record,indent=2))
    return folder


def sample(folder):
    g=np.stack([tifffile.imread(folder/(name+".tif")) for name in "xyz"],-1)
    h,w=g.shape[:2];rr,cc=np.mgrid[:(h-1)*8,:(w-1)*8];i,j=rr//8,cc//8
    v=(rr%8)[...,None]/8;u=(cc%8)[...,None]/8
    a,b,c,d=(g[i+dy,j+dx] for dy,dx in ((0,0),(0,1),(1,0),(1,1)))
    valid=(np.minimum.reduce([a,b,c,d])>=0).all(-1)
    p=(1-v)*((1-u)*a+u*b)+v*((1-u)*c+u*d)
    du=(1-v)*(b-a)+v*(d-c);dv=(1-u)*(c-a)+u*(d-b)
    normal=np.cross(du,dv);norm=np.linalg.norm(normal,axis=2);valid &= norm>.5
    normal/=np.maximum(norm[...,None],1e-8)
    sample_native(EXPORT,CT,p,normal,valid,PITCH,dict(specimen="PHerc0125",native_volume_id="20250821151825",
        mesh=folder.name,scan="PHerc0125",original_surface="PHerc0125_y3310",direction=DIRECTION,component_rank=COMPONENT_RANK,
        source_exclusion="Full old XZ footprint plus 16-voxel guard on the outward growth axis"))


def longest(bits):
    edge=np.diff(np.r_[False,bits,False].astype(int))
    starts=np.flatnonzero(edge==1);ends=np.flatnonzero(edge==-1)
    if not len(starts):return 0,0
    i=np.argmax(ends-starts)
    return int(starts[i]),int(ends[i])


def scale8(array,mask=None):
    mask=np.ones_like(array,bool) if mask is None else mask
    lo,hi=np.percentile(array[mask],[1,99])
    return np.uint8(np.where(mask,np.clip((array.astype(float)-lo)*255/max(hi-lo,1e-9),0,255),0))


def evidence():
    with np.load(DEST/"consensus.npz") as dd:
        new,y,xs,zs=selected_component(dd["new"]),dd["y"],dd["x"],dd["z"]
    # Choose the longest continuous transect in each third of the supported
    # native-Z range, before consulting grayscale CT.
    lines=[]
    contributing=np.flatnonzero(new.any(1))
    for group in np.array_split(contributing,3):
        choices=[]
        for r in group:
            a,b=longest(new[r])
            if b>a:choices.append((b-a,r,a,b))
        length,r,a,b=max(choices)
        if length<6:continue
        if b-a>240:
            center=(a+b)//2;a,b=center-120,center+120
        line=y[r,a:b];z=int(zs[r]);xp=xs[a:b]
        y0=int(np.floor(line.min())-32);y1=int(np.ceil(line.max())+33)
        x0=int(xp[0]);x1=int(xp[-1])+1
        raw,manifest=read_crop(CT,[z,y0,x0,z+1,y1,x1],ROOT/"data/cache",1300*1024**2)
        basic=Image.fromarray(scale8(raw[0])).convert("RGB")
        over=basic.copy();d=ImageDraw.Draw(over)
        d.line([(int(x-x0),float(h-y0))for x,h in zip(xp,line)],fill="#ffdc59",width=1)
        lines.append((z,x0,x1,y0,y1,basic,over,manifest))
    masked=np.load(EXPORT/"mask.npy");ct=np.load(EXPORT/"ct17.npy",mmap_mode="r")[8]
    try:
        heading=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",23)
        body=ImageFont.truetype("C:/Windows/Fonts/segoeui.ttf",16)
    except OSError:
        heading=body=ImageFont.load_default()
    out=Image.new("RGB",(1400,765),"#17191c");draw=ImageDraw.Draw(out)
    draw.text((12,5),f"PHerc. 125 | Interior surface {DIRECTION} of the original patch, component {COMPONENT_RANK}",font=heading,fill="white")
    view=Image.fromarray(scale8(ct,masked));view.thumbnail((620,630))
    draw.text((12,42),"Flattened native CT at the recovered surface",font=body,fill="white")
    out.paste(view,(12,75))
    sb=round(1000/PITCH*view.width/ct.shape[1])
    draw.line((15,719,15+sb,719),fill="white",width=4)
    draw.text((24+sb,706),"1 mm on the flattened chart",font=body,fill="white")
    draw.text((658,44),"Original axial CT",font=body,fill="white")
    draw.text((1027,44),"Same slice with fitted surface",font=body,fill="white")
    receipt=[]
    for k,(z,x0,x1,y0,y1,orig,over,m) in enumerate(lines):
        top=78+219*k;draw.text((658,top),f"Native z {z}    x {x0} to {x1-1}",font=body,fill="white")
        for j,img in enumerate((orig,over)):
            img.thumbnail((356,171))
            out.paste(img,(658+369*j,top+30))
        receipt.append(dict(z=z,x=[x0,x1],y=[y0,y1],source=m))
    out.save(EXPORT/"native_evidence.jpg",quality=94)
    (EXPORT/"native_transects.json").write_text(json.dumps(receipt,indent=2))
    result=json.loads((EXPORT/"meta.json").read_text())
    result["supported_nominal_cm2"]=float(masked.sum()*(PITCH/10000)**2)
    groups,n=label(masked)
    result["connected_component_count"]=int(n)
    result["largest_connected_supported_nominal_cm2"]=float(np.bincount(groups.ravel())[1:].max()*(PITCH/10000)**2)
    result["new_native_xyz_range"]=json.loads(next(EXPORT.glob("*.tifxyz/meta.json")).read_text())["bbox"]
    result["independent_ground_truth"]="unavailable on this new original-scroll region"
    (EXPORT/"meta.json").write_text(json.dumps(result,indent=2))
    print(json.dumps({k:result[k] for k in ("supported_nominal_cm2","new_native_xyz_range","missing_objects","shape")}))


if __name__=="__main__":
    folder=mesh()
    if not (EXPORT/"meta.json").is_file():sample(folder)
    evidence()
