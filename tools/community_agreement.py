"""Per-quad agreement between this release's PHerc. 125 meshes and the community spiral-fit windings
(pscamillo/vesuvius-eligible-meshes, MIT). A quad agrees when all four corners lie within TOL native
voxels of the densified community surface. Writes a JSON summary and per-case boolean quad masks.
Usage: python scripts/community_agreement.py <community_PHerc0125_dir> <out_dir> [TOL ...]
"""
import sys, os, glob, json
import numpy as np, tifffile
from scipy.spatial import cKDTree
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
PUB=os.path.join(ROOT,"reconstruction")
comm_dir,out=sys.argv[1],sys.argv[2]; tols=[float(t) for t in sys.argv[3:]] or [3,6]
os.makedirs(out,exist_ok=True)
def grid(d):
    xyz=np.stack([tifffile.imread(os.path.join(d,f"{a}.tif")).astype(np.float64) for a in "xyz"],-1)
    return xyz,(xyz>0).all(-1)&np.isfinite(xyz).all(-1)
def dense(xyz,v,k=6):
    q=v[:-1,:-1]&v[1:,:-1]&v[:-1,1:]&v[1:,1:]; I,J=np.nonzero(q); o=[]
    for a in np.linspace(0,1,k,endpoint=False):
        for b in np.linspace(0,1,k,endpoint=False):
            o.append((1-a)*(1-b)*xyz[I,J]+a*(1-b)*xyz[I+1,J]+(1-a)*b*xyz[I,J+1]+a*b*xyz[I+1,J+1])
    return np.concatenate(o)
names,pts=[],[]
for d in sorted(glob.glob(os.path.join(comm_dir,"*",""))):
    if os.path.exists(os.path.join(d,"x.tif")):
        P=dense(*grid(d)); names.append(os.path.basename(os.path.normpath(d))); pts.append(P)
lab=np.concatenate([np.full(len(p),i) for i,p in enumerate(pts)]); tree=cKDTree(np.concatenate(pts))
idx=json.load(open(os.path.join(PUB,"index.json")))["cases"]
pitch_cm=9.362e-4
res={"community_source":"https://github.com/pscamillo/vesuvius-eligible-meshes","community_meshes_compared":len(names),"tolerances_voxels":tols,"cases":{}}
for c in idx:
    if not c["id"].startswith("PHerc0125"): continue
    xyz,v=grid(os.path.join(PUB,c["mesh"]))
    dist=np.full(v.shape,np.inf); near=np.full(v.shape,-1)
    dd,ii=tree.query(xyz[v]); dist[v]=dd; near[v]=lab[ii]
    q=v[:-1,:-1]&v[1:,:-1]&v[:-1,1:]&v[1:,1:]
    A,B,C,D=xyz[:-1,:-1],xyz[1:,:-1],xyz[:-1,1:],xyz[1:,1:]
    area=(0.5*np.linalg.norm(np.cross(B-A,C-A),axis=-1)+0.5*np.linalg.norm(np.cross(B-D,C-D),axis=-1))*pitch_cm**2
    area=np.where(q,area,0.0)
    qmax=np.maximum.reduce([dist[:-1,:-1],dist[1:,:-1],dist[:-1,1:],dist[1:,1:]])
    rec={"triangulated_cm2":round(float(area.sum()),6)}
    faces=None
    if c["family"]=="extension":
        fp=os.path.join(PUB,"curve-faces",c["id"].removeprefix("PHerc0125_")+"_faces.npz")
        with np.load(fp) as f: faces=f["qualified_quads"]
    masks={}
    for t in tols:
        m=q&(qmax<=t); masks[f"within_{int(t)}"]=m
        rec[f"agree_within_{int(t)}vox_cm2"]=round(float(area[m].sum()),6)
        if faces is not None and faces.shape==m.shape:
            rec[f"curve_faces_and_agree_{int(t)}vox_cm2"]=round(float(area[m&faces].sum()),6)
    wins={}
    for n in near[v][dist[v]<=max(tols)]: wins[names[n]]=wins.get(names[n],0)+1
    rec["matched_windings"]=sorted(wins,key=lambda k:-wins[k])[:3]
    rec["median_vertex_distance_vox"]=round(float(np.median(dist[v])),2)
    res["cases"][c["id"]]=rec
    np.savez_compressed(os.path.join(out,f"{c['id']}_community_agreement.npz"),**masks)
    print(c["id"],rec)
tot={k:round(sum(r[k] for r in res["cases"].values() if k in r),6) for k in next(iter(res["cases"].values())) if k.endswith("cm2")}
res["totals_cm2"]=tot; print("TOTALS",tot)
json.dump(res,open(os.path.join(out,"community_agreement.json"),"w"),indent=1)
