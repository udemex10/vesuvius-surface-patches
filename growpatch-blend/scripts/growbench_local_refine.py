"""Refine a local stock tifxyz mesh against the nearest bounded foreground band."""
import argparse
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates
import tifffile


def band_displacements(points,origin,normal,volume,cutoff=.5):
    """Find bounded runs along each ray. Ambiguous, broad or distant bands do not move a vertex."""
    offsets=np.arange(-8,8.01,.5)
    output=np.zeros(len(points))
    for start in range(0,len(points),12288):
        pts=points[start:start+12288]
        pos=pts[:,None,:]-origin+offsets[None,:,None]*normal
        signal=map_coordinates(volume,pos[:,:,::-1].transpose(2,0,1),order=1,output=np.float32,
                               mode="constant",cval=0,prefilter=False)/255
        inside=((pos>=0)&(pos<np.asarray(volume.shape[::-1])-1)).all(axis=(1,2))
        mask=signal>=cutoff
        pad=np.pad(mask,((0,0),(1,1)),constant_values=False)
        row,left=np.nonzero(pad[:,1:]&~pad[:,:-1])
        endrow,right=np.nonzero(pad[:,:-1]&~pad[:,1:])
        if not np.array_equal(row,endrow): raise ValueError("Foreground run indexing mismatch")
        # Ignore clipped intervals. They cannot supply an observed midpoint.
        closed=(left>0)&(right<len(offsets))&inside[row]
        row,left,right=row[closed],left[closed],right[closed]
        if not len(row):continue
        lval,lnext=signal[row,left-1],signal[row,left]
        rprev,rval=signal[row,right-1],signal[row,right]
        low=offsets[left-1]+.5*(cutoff-lval)/(lnext-lval)
        high=offsets[right-1]+.5*(rprev-cutoff)/(rprev-rval)
        distance=np.maximum(0,np.maximum(low,-high))
        order=np.lexsort((distance,row))
        row,low,high,distance=(a[order] for a in (row,low,high,distance))
        first=np.r_[True,row[1:]!=row[:-1]]
        idx=np.flatnonzero(first)
        second=np.minimum(idx+1,len(row)-1)
        ambiguous=(second!=idx)&(row[second]==row[idx])&(distance[second]-distance[idx]<1)
        keep=(~ambiguous)&(distance[idx]<=2.5)&(high[idx]-low[idx]<=8)
        idx=idx[keep]
        output[start+row[idx]]=np.clip((low[idx]+high[idx])/2,-3,3)
    return output


def refine_grid(xyz,valid,site,volume,cutoff=.5,gain=1):
    normal=np.asarray(site["normal_xyz"],float)
    normal/=np.linalg.norm(normal)
    result=np.asarray(xyz,dtype=float).copy()
    valid=np.asarray(valid)&np.isfinite(result).all(-1)&(result>=0).all(-1)
    shifts=band_displacements(result[valid],np.asarray(site["origin_xyz"]),normal,volume,cutoff)
    result[valid]+=gain*shifts[:,None]*normal
    return result,valid,shifts


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh",type=Path,help="Stock tifxyz folder in native XYZ coordinates")
    p.add_argument("volume",type=Path,help="Matching regional uint8 ZYX foreground NPY")
    p.add_argument("site",type=Path,help="JSON with origin_xyz and a seed normal_xyz")
    p.add_argument("output",type=Path,help="New output tifxyz folder")
    p.add_argument("--cutoff",type=float,default=.5)
    p.add_argument("--gain",type=float,default=1.)
    a=p.parse_args()
    source=[a.mesh/(k+".tif") for k in "xyz"]
    xyz=np.stack([tifffile.imread(f) for f in source],-1)
    volume=np.load(a.volume,mmap_mode="r")
    if volume.ndim!=3 or volume.dtype!=np.uint8 or not(0<a.cutoff<1 and 0<a.gain<=1):
        p.error("Expected 3D uint8 probability, cutoff between 0 and 1 and gain in (0,1].")
    grid,_,displacement=refine_grid(xyz,np.ones(xyz.shape[:2],bool),json.loads(a.site.read_text()),volume,a.cutoff,a.gain)
    a.output.mkdir(parents=True,exist_ok=False)
    for i,k in enumerate("xyz"):
        tifffile.imwrite(a.output/(k+".tif"),grid[:,:,i].astype(np.float32))
    meta=json.loads((a.mesh/"meta.json").read_text())
    meta.update(name=a.output.name,uuid=a.output.name)
    # Derived extents and areas belong to the old mesh.
    for k in ("bbox","area_cm2","area_vx2"):meta.pop(k,None)
    (a.output/"meta.json").write_text(json.dumps(meta,indent=2))
    print("Vertices evaluated",len(displacement),"moved",int(np.count_nonzero(displacement)))


if __name__=="__main__":
    main()
