"""Grow independent overlapping PHerc. 125 tracks out of a previously released sheet.

Each track is seeded only from the old mesh, then uses its own preceding tile.
The final mesh requires agreement between tracks started on different native
Z sections. No comparison with a verified sheet is available outside the old
published area.
"""
import argparse
from concurrent.futures import ThreadPoolExecutor
import hashlib
import json
import threading
import time
from pathlib import Path

import numpy as np
from scipy.ndimage import label, map_coordinates
import tifffile

from growbench_recover import ROOT, solve
from growbench_recover_phase import align, orient_normals
from r3_chunkvol import ChunkVolume


DEST=ROOT/"results/growbench/recover/pherc125_extension"
OLD=ROOT/"results/first_letters_deep/PHerc0125_y3310/geometry_nodes.npz"
M7=("https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0125/"
    "representations/predictions/surfaces/20250821151825-surface-20260413222639-surface-m7-L0-th0.2.zarr/0")
PITCH=9.362
X0=3880
Z0=10467
STRIDE=80
NZ=13
NX=21
MID=48
FETCH_LOCK=threading.Lock()


def cube(volume,origin):
    lo=np.asarray(origin)[::-1];hi=lo+257
    result=np.zeros((257,)*3,dtype=np.uint8)
    keys=[tuple(k+lo//volume.chunk) for k in np.ndindex(tuple((hi-1)//volume.chunk-lo//volume.chunk+1))]
    # Adjacent independent tracks can request the same upstream chunk together.
    # Serialize the shared on-disk cache writes, then solve tiles in parallel.
    with FETCH_LOCK:
        for attempt in range(5):
            try:
                arrays=volume.chunks(keys)
                break
            except PermissionError:
                if attempt==4:raise
                time.sleep(.2*(attempt+1))
    for key in keys:
        start=np.asarray(key)*volume.chunk
        a=np.maximum(start,lo);b=np.minimum(start+volume.chunk,hi)
        result[tuple(map(slice,a-lo,b-lo))]=arrays[key][tuple(map(slice,a-start,b-start))]
    return result,keys


def route(row):
    volume=ChunkVolume(M7,max_bytes=160*1024**2,allow_missing=True,workers=6)
    center_z=Z0+(row-NZ//2)*STRIDE
    directory=DEST/f"route_{row:02d}";directory.mkdir(exist_ok=True,parents=True)
    with np.load(OLD) as old:
        ix=int(np.argmin(abs(old["x"]-X0)));iz=int(np.argmin(abs(old["z"]-center_z)))
        if not old["qualified"][iz,ix]:
            return
        seed_y=float(old["fit"][iz,ix])
        gy,gx=np.gradient(old["fit"],16,16)
        normal=np.array([-gx[iz,ix],1.,-gy[iz,ix]])
        normal/=np.linalg.norm(normal)
    previous=None;log=[];used=set()
    for i in range(NX):
        cx=X0+i*STRIDE
        tile=directory/f"{i:02d}";tile.mkdir(exist_ok=True)
        if previous is not None:
            j=MID+STRIDE//2
            valid=previous["v"]
            if not valid[MID,j]:
                log.append(dict(cx=cx,stop="unqualified entire next starter"));break
            sy=previous["y"]
            if not(valid[MID-1:MID+2,j-1:j+2].all()):
                log.append(dict(cx=cx,stop="insufficient immediate support at next starter"));break
            seed_y=float(sy[MID,j])
            n=np.array([-(sy[MID,j+1]-sy[MID,j-1])/4.,1.,-(sy[MID+1,j]-sy[MID-1,j])/4.])
            normal=n/np.linalg.norm(n)
        if abs(normal[1]) < .65 or np.argmax(abs(normal))!=1:
            log.append(dict(cx=cx,stop="sheet cannot be represented by the local Y(X,Z) chart"));break
        site=dict(seed_xyz=[cx,seed_y,center_z],normal_xyz=normal.tolist(),
                  origin_xyz=[cx-128,round(seed_y)-128,center_z-128])
        saved=tile/"supported.npz"
        if not saved.exists():
            vol,keys=cube(volume,site["origin_xyz"])
            used.update(keys)
            try:
                stats=solve(vol,site,tile)
            except ValueError as e:
                log.append(dict(cx=cx,stop=str(e)));break
            with np.load(tile/"graph.npz") as f:
                xyz=f["xyz"];valid=f["valid"];quality=f["quality"]
            _,phase,_=align(xyz,valid,site)
            local=orient_normals(xyz,normal)
            y=xyz[...,1]-phase/np.where(np.abs(local[...,1])>.40,local[...,1],np.nan)
            # A discovery mesh requires direct measured M7 support, route confidence,
            # and a locally regular Y chart. A dense center cross is not a template.
            valid &= np.isfinite(y)&(quality>=.58)
            top=np.where(valid,y,np.nan)
            for axis in (0,1):
                grad=np.abs(np.gradient(top,axis=axis)) / 2
                valid &= ~(grad>1.8)
            regions,_=label(valid)
            seed_component=regions[MID,MID]
            if not seed_component:
                log.append(dict(cx=cx,stop="new tile did not retain its seed"));break
            valid &= regions==seed_component
            np.savez_compressed(saved,y=y,v=valid,q=quality)
            (tile/"source.json").write_text(json.dumps(dict(site=site,keys_zyx=[list(map(int,k)) for k in keys],
                   cue=M7,phase_vox=phase,stats=stats,
                   cue_array_sha256=hashlib.sha256(vol.tobytes()).hexdigest())))
        with np.load(saved) as f:current={k:f[k] for k in f.files}
        if previous is not None:
            off=STRIDE//2
            common=previous["v"][:,off:]&current["v"][:,:-off]
            dif=abs(previous["y"][:,off:]-current["y"][:,:-off])[common]
            if len(dif)<240 or (dif>3).mean()>.30:
                log.append(dict(cx=cx,stop="overlap agreement failure",common=int(len(dif)),
                                fraction_over_3=float(np.mean(dif>3))if len(dif)else None));break
        previous=current
        log.append(dict(cx=cx,valid_nodes=int(current["v"].sum()),seed_y=seed_y))
    (directory/"track.json").write_text(json.dumps(dict(row=row,z=center_z,steps=log),indent=2))
    print("route",row,len([s for s in log if "valid_nodes"in s]),log[-1] if log else None,flush=True)


def combine(projected=True):
    x=np.arange(X0-96,X0+(NX-1)*STRIDE+97,2)
    z=np.arange(Z0-(NZ//2)*STRIDE-96,Z0+(NZ//2)*STRIDE+97,2)
    shape=(len(z),len(x))
    tracks=np.full((NZ,*shape),np.nan,np.float32)
    for row in range(NZ):
        folder=DEST/f"route_{row:02d}"
        if not (folder/"track.json").exists():continue
        steps=json.loads((folder/"track.json").read_text())["steps"]
        best=np.zeros(shape,np.float32)
        for i,step in enumerate(steps):
            if "valid_nodes" not in step:continue
            dy=row*(STRIDE//2)
            dx=i*(STRIDE//2)
            sl=np.s_[dy:dy+97,dx:dx+97]
            with np.load(folder/f"{i:02d}"/"supported.npz") as data:
                w=1-(np.abs(np.arange(97)-48)[None,:]/120)
                q=data["q"]*w;good=data["v"]&(q>best[sl])
                tracks[row][sl][good]=data["y"][good];best[sl][good]=q[good]
    count=np.isfinite(tracks).sum(0)
    low=np.min(np.where(np.isfinite(tracks),tracks,np.inf),axis=0)
    high=np.max(np.where(np.isfinite(tracks),tracks,-np.inf),axis=0)
    y=np.nansum(tracks,axis=0)/np.maximum(1,count)
    ok=(count>=2)&(high-low<=3)
    # Any source site remains connected to a band within the original geometry.
    regions,_=label(ok);anchors=np.unique(regions[:,(x>=X0-20)&(x<=X0+20)])
    ok &= np.isin(regions,anchors[anchors>0])
    yy,xx=np.meshgrid(z,x,indexing="ij")
    xyz=np.stack([xx,y,yy],axis=-1)
    # The largest possible X coordinate of the old 125 package is 4040;
    # an additional 16-voxel guard excludes even nearby depth samples.
    new=ok&(xx>=4056)
    quads=new[:-1,:-1]&new[1:,:-1]&new[:-1,1:]&new[1:,1:]
    a=xyz[:-1,:-1];b=xyz[:-1,1:];c=xyz[1:,1:];d=xyz[1:,:-1]
    surface=(np.linalg.norm(np.cross(b-a,c-a),axis=2)+
             np.linalg.norm(np.cross(c-a,d-a),axis=2))*.5*(PITCH/10000)**2
    labs,_=label(quads)
    areas=np.bincount(labs.ravel(),weights=np.where(quads,surface,0).ravel())
    largest=int(np.argmax(areas[1:])+1) if len(areas)>1 else 0
    np.savez_compressed(DEST/"consensus.npz",x=x,z=z,y=y,valid=ok,new=new,
                        source_routes=count,disagreement=np.where(count>1,high-low,np.nan))
    # The projected mesh exposes all verified-by-agreement geometry at native XYZ.
    # It is not described as isometric; the follow-up exporter handles flattening.
    step=4;select=np.s_[::step,::step];nodes=xyz[select];mask=new[select]
    if projected:
        folder=DEST/"PHerc0125_east_projected.tifxyz";folder.mkdir(exist_ok=True)
        for j,axis in enumerate("xyz"):
            tifffile.imwrite(folder/(axis+".tif"),np.where(mask,nodes[...,j],-1).astype("f4"))
        if mask.any():
            (folder/"meta.json").write_text(json.dumps(dict(format="tifxyz",type="seg",name=folder.stem,
                 uuid=folder.stem,scale=[.125,.125],bbox=[nodes[mask].min(0).tolist(),nodes[mask].max(0).tolist()])))
    report=dict(original_seed_mesh="PHerc0125_y3310",old_x_max=4040,new_x_min=4056,
        n_independent_anchor_routes=NZ,n_valid_tiles=sum(len([s for s in json.loads(f.read_text())["steps"] if "valid_nodes" in s])
                                                for f in DEST.glob("route_*/track.json")),
        new_native_connected_area_cm2=float(areas[largest])if largest else 0,
        new_native_total_consensus_area_cm2=float(surface[quads].sum()),
        max_x=int(xx[new].max()) if new.any() else None,
        new_valid_quads=int(quads.sum()),threshold_vox=3,grid_spacing_vox=2,
        interpretation="Conditional agreement of independently seeded routes through the same M7 model; not independent human verification of physical winding identity.")
    (DEST/"extension.json").write_text(json.dumps(report,indent=2))
    print(json.dumps(report),flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("--rows",type=int,nargs="+",default=list(range(NZ)))
    p.add_argument("--threads",type=int,default=3)
    p.add_argument("--combine",action="store_true")
    a=p.parse_args();DEST.mkdir(parents=True,exist_ok=True)
    if a.combine:combine()
    else:
        with ThreadPoolExecutor(a.threads) as e:list(e.map(route,a.rows))
