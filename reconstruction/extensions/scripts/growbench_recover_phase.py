"""Preserve a seed's signed offset from the thick M7 prediction through growth.

The only supervised location used here is the shared starter point and its
supplied tangent plane. No later reference surface or winding IDs are read.
"""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from growbench_recover import LABELS, OUTPUT


def orient_normals(xyz,seed_normal):
    """Estimate mesh normals with short centered differences; reject jumps."""
    du=np.gradient(xyz,axis=1)
    dv=np.gradient(xyz,axis=0)
    cross=np.cross(du,dv)
    length=np.linalg.norm(cross,axis=-1)
    n=np.zeros_like(xyz)
    good=(length>1e-5)&np.all(np.isfinite(cross),axis=-1)
    n[good]=cross[good]/length[good,None]
    dot=n@seed_normal
    n[dot<0]*=-1
    stable=good&(np.abs(dot)>=.55)&(np.linalg.norm(du,axis=-1)<5)&(np.linalg.norm(dv,axis=-1)<5)
    n[~stable]=seed_normal
    return n


def align(xyz,valid,site):
    seed=np.asarray(site["seed_xyz"])
    n=np.asarray(site["normal_xyz"]);n=n/np.linalg.norm(n)
    p=xyz-seed
    dist=p@n
    r2=np.sum((p-dist[...,None]*n)**2,axis=-1)
    near=valid&(r2<=6**2)&(np.abs(dist)<=6)
    if not near.any():
        return xyz,0.0,0
    # A 6-voxel region identifies which isosurface within the predicted band
    # passes through the known starter. This is smaller than the excluded
    # 12-voxel radius in the benchmark.
    signed=float(np.median(dist[near]))
    direction=orient_normals(xyz,n)
    aligned=xyz-signed*direction
    return aligned.astype(np.float32),signed,int(near.sum())


def run(site,root):
    folder=root/site["id"]
    info={}
    for name in ("graph","two_route"):
        path=folder/(name+".npz")
        with np.load(path) as data:
            xyz,offset,n=align(data["xyz"],data["valid"],site)
            np.savez_compressed(folder/(name+"_phase.npz"),xyz=xyz,
                                valid=data["valid"],quality=data["quality"])
        info[name]=dict(signed_offset_voxels=offset,seed_plane_samples=n,
                        input_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
    (folder/"phase.json").write_text(json.dumps(info,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--seeds",type=Path,default=LABELS/"seeds.json")
    p.add_argument("--out",type=Path,default=OUTPUT)
    p.add_argument("--split",choices=["dev","test","confirm"],nargs="+")
    a=p.parse_args()
    for site in json.loads(a.seeds.read_text(encoding="utf8"))["sites"]:
        if not a.split or site["split"] in a.split:
            run(site,a.out)
