"""Re-anchor a local GrowPatch mesh to a trusted seed, in the mesh's native voxel coordinates."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import tifffile

from growbench_geometry import R2,faces,frame,project


def load_surface(folder):
    xyz=np.stack([tifffile.imread(folder/(a+".tif")) for a in "xyz"],axis=-1)
    valid=np.isfinite(xyz).all(-1)&(xyz>=0).all(-1)
    return xyz,valid


def align(xyz,valid,site):
    tri,_=faces(xyz,valid)
    raster=project(tri,site)
    central=(R2<=6**2)&np.isfinite(raster.low)&(raster.high-raster.low<=1)
    count=int(central.sum())
    if count<5:
        shift=0.
    else:
        shift=float(np.clip(np.median(raster.center[central]),-3,3))
    moved=np.asarray(xyz,dtype=float).copy()
    moved[valid]-=shift*frame(site)[2]
    return moved.astype(np.float32),dict(seed_rays=count,shift_along_seed_normal_voxels=-shift)


def canonical_site(seed,normal):
    n=np.asarray(normal,float)
    if not np.isfinite(n).all() or np.linalg.norm(n)<1e-9: raise ValueError("Invalid seed normal")
    n/=np.linalg.norm(n)
    axis=np.eye(3)[np.argmin(abs(n))]
    u=axis-n*np.dot(axis,n)
    u/=np.linalg.norm(u)
    return dict(seed_xyz=list(seed),normal_xyz=n.tolist(),tangent_u_xyz=u.tolist(),tangent_v_xyz=np.cross(n,u).tolist())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("mesh",type=Path,help="Local stock tifxyz in global XYZ, in voxels")
    p.add_argument("output",type=Path,help="A new tifxyz directory")
    p.add_argument("--site",type=Path,help="Evaluation seed JSON including tangent axes, or a seeds manifest with --id")
    p.add_argument("--id",help="Case ID in a multi-seed manifest")
    p.add_argument("--seed",nargs=3,type=float,metavar=("X","Y","Z"))
    p.add_argument("--normal",nargs=3,type=float,metavar=("NX","NY","NZ"))
    a=p.parse_args()
    if a.site:
        site=json.loads(a.site.read_text())
        if "sites" in site:
            site=next((s for s in site["sites"] if s["id"]==a.id),None)
            if site is None:p.error("Supply an ID from the chosen manifest")
    elif a.seed and a.normal:
        site=canonical_site(a.seed,a.normal)
    else:p.error("Supply a seed and its normal, or a site JSON")
    xyz,valid=load_surface(a.mesh)
    moved,record=align(xyz,valid,site)
    meta=json.loads((a.mesh/"meta.json").read_text())
    with (a.mesh/"meta.json").open("rb") as f:
        record["input_metadata_sha256"]=hashlib.file_digest(f,"sha256").hexdigest()
    meta.update(name=a.output.name,uuid=a.output.name)
    if valid.any():
        meta["bbox"]=[moved[valid].min(0).tolist(),moved[valid].max(0).tolist()]
    a.output.mkdir(parents=True,exist_ok=False)
    for i,k in enumerate("xyz"):tifffile.imwrite(a.output/(k+".tif"),moved[:,:,i])
    (a.output/"meta.json").write_text(json.dumps(meta,indent=2))
    (a.output/"seed_alignment.json").write_text(json.dumps(dict(**record,site=site),indent=2))
    print(json.dumps(record))


if __name__=="__main__":
    main()
