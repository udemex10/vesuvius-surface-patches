"""Prospectively select higher-pressure frozen Paris4 fields using only local input M7 profiles."""
import argparse
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from scipy.ndimage import distance_transform_edt
from scipy.signal import find_peaks
from growbench_labels import OUT as PRIMARY, INDEX, SPIRAL, CT, M7, digest, extract, patch, raster, triangulate, write

OUT=PRIMARY/"pressure"
SIDE=385;RADIUS=160;SALT="growbench-pressure-v1:";POOL=40
RULE=dict(dataset="Same 2.4um acquisition as representative cohort, CT L2, M7 array /0; physical units 9.6um",
    salt=SALT,candidate_pool_size=POOL,target_sites=8,
    candidate_rule="First 40 eligible by salted SHA256 of public reference ID. Bbox extent in X/Z >350, largest extent <3500, area >0.18cm2. Require at least 65% verified projected disk coverage and <=3% multivalued rays in 160vx disk, local |normal_y|>=.70. Highest-inradius valid graph anchor. Cubes and reference member IDs disjoint from original cohort and every other candidate by >=650 voxels Chebyshev seed separation. Source geometry used only for eligibility, shared initialization and safety from broad chart folds.",
    ranking="Use only the probability signal in shared M7 within a seed-local 97x97 XZ area. Evaluate 13x13 Y-profiles centered on the tangent-plane prediction; offsets +/-40. Local peaks at height>=51, prominence>=32, spacing>=3. Seed must have peak within 4 voxels. Metric=0.65*paired_fraction+0.35*min(gap_fraction,0.60); paired = eligible near peak within 12 and second peak 4-20 voxels away with intervening valley <=0.55*weaker peak. Gap = no peak within 12. Both fractions are fractions of all 169 profiles. Fix ranking before any solver outcomes; top eight eligible. Ties use salted patch order.",
    origin="385x385x385 integer XYZ centered on rounded seed in level-2 space, tangent-radius 160, collar 12, no new tuning")


def geometry(name,others,used):
    xyz,meta,hashes=patch(name)
    members=set(meta.get("member_ids",[name]))
    if used & members:return
    valid=np.isfinite(xyz).all(-1)&(xyz>=0).all(-1)
    v=np.cross(xyz[1:-1,2:]-xyz[1:-1,:-2],xyz[2:,1:-1]-xyz[:-2,1:-1])
    interior=distance_transform_edt(np.pad(valid,1))[1:-1,1:-1]
    g=np.zeros(valid.shape,bool);g[1:-1,1:-1]=abs(v[:,:,1])>.72*np.maximum(np.linalg.norm(v,axis=-1),1e-7)
    g&=interior>=6
    if not g.any():return
    i,j=np.unravel_index(np.argmax(np.where(g,interior,-1)),g.shape)
    seed=xyz[i,j];origin=np.rint(seed).astype(int)-SIDE//2
    if any(np.max(abs(seed-s["seed_xyz"]))<650 for s in others):return
    triangles=triangulate(xyz)
    triangles=triangles[((triangles.max(1)>=origin)&(triangles.min(1)<=origin+SIDE-1)).all(1)]
    if len(triangles)<25:return
    yy,c=raster(triangles,origin,SIDE)
    z,x=np.mgrid[:SIDE,:SIDE];xx=x+origin[0];zz=z+origin[2];dist=(xx-seed[0])**2+(zz-seed[2])**2
    disk=dist<=RADIUS**2;support=np.isfinite(yy)
    if support[disk].mean()<.65 or c[disk].mean()>.03:return
    local=(dist<=8**2)&support
    if local.sum()<80:return
    k=np.c_[xx[local]-seed[0],zz[local]-seed[2],np.ones(local.sum())]
    ax,az,_=np.linalg.lstsq(k,yy[local]-seed[1],rcond=None)[0]
    n=np.array([-ax,1.,-az]);n/=np.linalg.norm(n)
    if n[1]<.70:return
    u=np.array([1.,0,0])-n[0]*n;u/=np.linalg.norm(u)
    s=dict(seed_xyz=seed.tolist(),origin_xyz=origin.tolist(),normal_xyz=n.tolist(),tangent_u_xyz=u.tolist(),tangent_v_xyz=np.cross(u,n).tolist(),cube_shape_zyx=[SIDE]*3)
    r=dict(name=name,source=SPIRAL+"verified_patches/"+name+"/",members=sorted(members),hashes=hashes,source_tool=meta.get("source"),
        projected_disk_coverage=float(support[disk].mean()),multivalued_fraction=float(c[disk].mean()),reference_knot_ij=[int(i),int(j)])
    truth=np.stack([xx,yy,zz],-1).astype("f4");truth[~support]=np.nan
    return s,r,truth,triangles.astype("f4")


def pool():
    if (OUT/"pool_public.json").exists():return
    OUT.mkdir(parents=True,exist_ok=True)
    write(OUT/"sampling_rule.json",RULE)
    idx=json.loads(INDEX.read_text())
    old=json.loads((PRIMARY/"seeds.json").read_text())["sites"]
    used={s for rec in json.loads((PRIMARY/"refs.json").read_text())["sites"] for s in rec["members"]}
    qualified=[]
    for n,q in idx.items():
        if q.get("bbox_l2") and (q.get("area_cm2") or 0)>.18:
            extent=np.subtract(*q["bbox_l2"][::-1])
            if min(extent[0],extent[2])>350 and max(extent)<3500:qualified.append(n)
    chosen=[];refs=[];inspected=[]
    for name in sorted(qualified,key=lambda n:digest((SALT+n).encode())):
        result=geometry(name,old+chosen,used);inspected.append(name)
        if not result:continue
        s,r,xyz,tris=result;s["id"]=f"pool_{len(chosen)+1:02d}";r["id"]=s["id"]
        dest=OUT/"pool_truth"/s["id"];dest.mkdir(parents=True,exist_ok=True)
        np.save(dest/"ref_xyz.npy",xyz);np.save(dest/"triangles.npy",tris)
        chosen.append(s);refs.append(r);used.update(r["members"])
        print("pressure pool",len(chosen),flush=True)
        if len(chosen)==POOL:break
    if len(chosen)!=POOL:raise ValueError("Too few regions for locked rule")
    write(OUT/"pool_refs.json",dict(inspected=inspected,sites=refs));write(OUT/"pool_public.json",dict(sites=chosen))


def rank(s):
    # Only a 193^3 crop needed to cover all short score profiles; the method
    # runners subsequently get separate, untouched 385^3 source cubes.
    small=dict(s,origin_xyz=(np.rint(s["seed_xyz"]).astype(int)-96).tolist(),cube_shape_zyx=[193]*3)
    temp=OUT/"pool_data"
    extract(small,"m7",M7,temp)
    a=np.load(temp/"inputs"/s["id"]/"m7.npy",mmap_mode="r")
    seed=np.array(s["seed_xyz"])-small["origin_xyz"];nx,ny,nz=s["normal_xyz"]
    pair=0;missing=0;near_center=False
    for dx in range(-48,49,8):
        for dz in range(-48,49,8):
            x,z=np.rint([seed[0]+dx,seed[2]+dz]).astype(int)
            y=round(seed[1]-(dx*nx+dz*nz)/ny)
            from_y=max(0,y-40);to_y=min(len(a),y+41)
            profile=a[z,from_y:to_y,x].astype(float)
            pk,_=find_peaks(profile,height=51,prominence=32,distance=3)
            if not len(pk) or min(abs(pk+from_y-y))>12:
                missing+=1;continue
            nearest=pk[np.argmin(abs(pk+from_y-y))]
            if dx==dz==0:near_center=abs(nearest+from_y-y)<=4
            for i in pk:
                if 4<=abs(i-nearest)<=20 and profile[min(i,nearest):max(i,nearest)+1].min()<=.55*min(profile[i],profile[nearest]):
                    pair+=1;break
    return dict(id=s["id"],eligible_seed=bool(near_center),paired=pair,gaps=missing,
                profiles=169,score=.65*pair/169+.35*min(missing/169,.60))


def select():
    if (OUT/"seeds.json").exists():return
    sites=json.loads((OUT/"pool_public.json").read_text())["sites"]
    refs=json.loads((OUT/"pool_refs.json").read_text())["sites"]
    with ThreadPoolExecutor(6) as ex:metrics=list(ex.map(rank,sites))
    ordered=sorted(range(len(sites)),key=lambda j:-metrics[j]["score"])
    kept=[j for j in ordered if metrics[j]["eligible_seed"]][:8]
    if len(kept)<8:raise ValueError("Insufficient seeds in the published M7; preserved the prior rule.")
    selected=[];rout=[]
    import shutil
    for i,j in enumerate(kept,1):
        sid=f"pressure_{i:02d}";s=sites[j];folder=OUT/"truth"/sid
        folder.mkdir(exist_ok=True,parents=True)
        for n in ("ref_xyz.npy","triangles.npy"):
            shutil.copyfile(OUT/"pool_truth"/s["id"]/n,folder/n)
        selected.append(dict(s,id=sid,split="test"))
        rout.append(dict(refs[j],id=sid,score=metrics[j],pool_index=j))
    write(OUT/"input_scores.json",dict(sites=metrics))
    write(OUT/"refs.json",dict(pool_lock=digest((OUT/"pool_public.json").read_bytes()),sites=rout))
    write(OUT/"seeds.json",dict(benchmark="Paris4-M7-pressure-v1",frame="global XYZ 9.6um = CT level 2",ct=CT,m7=M7,
          area_radius_voxels=RADIUS,excluded_seed_radius_voxels=12,sites=selected))
    print("PRESSURE LOCK",digest((OUT/"seeds.json").read_bytes()),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["select","m7","ct"]);args=ap.parse_args()
    if args.action=="select":pool();select()
    else:
        sites=json.loads((OUT/"seeds.json").read_text())["sites"]
        with ThreadPoolExecutor(5) as ex:list(ex.map(lambda s:extract(s,args.action,M7 if args.action=="m7" else CT,OUT),sites))
