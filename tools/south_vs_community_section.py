"""Vertical CT sections (fixed native X) through the PHerc. 125 south extensions, overlaying
our traces and the community spiral-fit winding w070 (pscamillo/vesuvius-eligible-meshes).
Usage: python scripts/south_vs_community_section.py <community_PHerc0125_dir> <out_dir> X [X ...]
"""
import sys, os, glob, json
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt
ROOT=os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(ROOT,"reconstruction/extensions/scripts"))
from zarr_crop import read_crop
URL="https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr/0"
PUB=os.path.join(ROOT,"reconstruction/")
comm_dir,out_dir=sys.argv[1],sys.argv[2]; xs=[int(a) for a in sys.argv[3:]]
os.makedirs(out_dir,exist_ok=True)
def load(d):
    xyz=np.stack([tifffile.imread(os.path.join(d,f"{a}.tif")).astype(float) for a in "xyz"],-1); return xyz[(xyz>0).all(-1)]
def dense(d,k=6):
    xyz=np.stack([tifffile.imread(os.path.join(d,f"{a}.tif")).astype(float) for a in "xyz"],-1); v=(xyz>0).all(-1)
    q=v[:-1,:-1]&v[1:,:-1]&v[:-1,1:]&v[1:,1:]; I,J=np.nonzero(q); out=[]
    for a in np.linspace(0,1,k,endpoint=False):
        for b in np.linspace(0,1,k,endpoint=False):
            out.append((1-a)*(1-b)*xyz[I,J]+a*(1-b)*xyz[I+1,J]+(1-a)*b*xyz[I,J+1]+a*b*xyz[I+1,J+1])
    return np.concatenate(out)
ours={n:load(p) for n,p in {"y3310":PUB+"original/cases/PHerc0125_y3310/PHerc0125_y3310.tifxyz",
    "south":PUB+"extensions/cases/south/PHerc0125_south.tifxyz","south_2":PUB+"extensions/cases/south_2/PHerc0125_south_2.tifxyz"}.items()}
comm={w:np.concatenate([dense(d) for d in glob.glob(os.path.join(comm_dir,f"*_{w}")) if os.path.exists(os.path.join(d,"x.tif"))]) for w in ["w065","w070"]}
Z0,Z1,Y0,Y1=8250,10050,3300,3640
summary={}
for x0 in xs:
    ct=read_crop(URL,[Z0,Y0,x0,Z1,Y1,x0+1],os.path.join(out_dir,"ct_chunk_cache"))
    if isinstance(ct,tuple): ct=ct[0]
    img=ct[:,:,0].astype(float)  # (z, y)
    lo,hi=np.percentile(img[img>0],[1,99.5]); img=np.clip((img-lo)/(hi-lo),0,1)
    fig,ax=plt.subplots(figsize=(24,5.6),facecolor="#111")
    ax.imshow(img.T,cmap="gray",origin="upper",extent=[Z0,Z1,Y1,Y0],aspect="equal",interpolation="nearest")
    rec={}
    for name,col in [("y3310","#ffd24a"),("south","#ff6b3d"),("south_2","#ff6b3d")]:
        P=ours[name]; s=P[(np.abs(P[:,0]-x0)<1.5)&(P[:,2]>=Z0)&(P[:,2]<Z1)&(P[:,1]>=Y0)&(P[:,1]<Y1)]
        if len(s): ax.scatter(s[:,2],s[:,1],s=10,c=col,lw=0,zorder=5,label=f"this release: {name}"); rec[name]=int(len(s))
    for w,col in [("w070","#39c5ff"),("w065","#8a7dff")]:
        P=comm[w]; s=P[(np.abs(P[:,0]-x0)<1.0)&(P[:,2]>=Z0)&(P[:,2]<Z1)&(P[:,1]>=Y0)&(P[:,1]<Y1)]
        if len(s): ax.scatter(s[:,2],s[:,1],s=5,c=col,lw=0,zorder=4,label=f"community spiral fit {w}"); rec[w]=int(len(s))
    ax.set_xlabel("native z (voxels, 9.362 µm)",color="#ddd"); ax.set_ylabel("native y",color="#ddd"); ax.tick_params(colors="#ccc")
    ax.set_title(f"PHerc. 125 vertical CT section at native x = {x0}",color="#eee",loc="left")
    lg=ax.legend(loc="upper right",markerscale=8,fontsize=8,facecolor="#222",labelcolor="#ddd",framealpha=.8)
    fig.tight_layout(); p=os.path.join(out_dir,f"section_x{x0}_zoom.png"); fig.savefig(p,dpi=110,facecolor="#111"); plt.close(fig)
    summary[x0]=rec; print(p,rec)
json.dump(summary,open(os.path.join(out_dir,"sections.json"),"w"),indent=1)
