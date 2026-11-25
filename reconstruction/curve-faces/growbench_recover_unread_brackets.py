"""Measure delivered mesh faces bracketed on every side by fresh source-curve checks."""
import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
from scipy.ndimage import label
import tifffile


def reference(directory,direction):
    report=json.loads((directory/"organizer_continuity.json").read_text())
    boundary={"east":4040,"west":2760,"north":11107,"south":9827}[direction]
    swap=direction in ("north","south")
    sign=1 if direction in ("east","north") else -1
    with np.load(directory/"route_agreement.npz")as data:
        cross,lat=(data["x"],data["z"])if swap else (data["z"],data["x"])
        new=data["new"].T if swap else data["new"]
    planes=sorted(r["plane"] for r in report["records"])
    vlo=planes[0];ulo=boundary+8*int(np.floor((lat.min()-boundary)/8))
    samples=np.zeros((len(planes),int((lat.max()-ulo)//8)+1),bool)
    for row in report["records"]:
        if not row.get("new_tested"):continue
        idx=np.searchsorted(cross,row["plane"]);outward=(lat-boundary)*sign
        domain=new[idx]&(outward>=16)&(outward%8==0)&(outward<=row["outward_reach_vox"])
        errors=np.asarray(row["signed_differences"]);assert len(errors)==domain.sum()
        samples[(row["plane"]-vlo)//32,(lat[domain]-ulo)//8]=abs(errors)<=3
    square=samples[:-1,:-1]&samples[1:,:-1]&samples[:-1,1:]&samples[1:,1:]
    return square,np.array([ulo,vlo]),swap


def bounded_faces(points,good,lower,spacing,select):
    projected=(points[:,:,select]-lower)/np.array(spacing)
    lo=np.floor(projected.min(1)).astype(int);hi=np.floor(projected.max(1)).astype(int)
    keep=np.zeros(len(points),bool)
    for it,((x0,y0),(x1,y1)) in enumerate(zip(lo,hi)):
        if x0>=0 and y0>=0 and x1<good.shape[1] and y1<good.shape[0]:
            keep[it]=good[y0:y1+1,x0:x1+1].all()
    return keep


def faces(case,out,ref,stock):
    good,lower,swap=ref
    folder=next(case.glob("*.tifxyz"))
    m=np.stack([tifffile.imread(folder/(axis+".tif")).astype(float) for axis in "xyz"],-1)
    mm=(m>=0).all(-1)
    valid=mm[:-1,:-1]&mm[:-1,1:]&mm[1:,:-1]&mm[1:,1:]
    vv,uu=np.where(valid)
    p=np.stack([m[vv+a,uu+b]for a,b in ((0,0),(0,1),(1,1),(1,0))],1)
    select=[2,0] if swap else [0,2]
    keep=bounded_faces(p,good,lower,[8.,32.],select)
    quad=np.zeros_like(valid);quad[vv[keep],uu[keep]]=True
    area=(np.linalg.norm(np.cross(p[:,1]-p[:,0],p[:,2]-p[:,0]),axis=-1)+np.linalg.norm(np.cross(p[:,2]-p[:,0],p[:,3]-p[:,0]),axis=-1))*.5*(9.362/10000)**2
    labels,n=label(quad);cc=np.bincount(labels[vv[keep],uu[keep]],weights=area[keep],minlength=2)[1:]
    stored=dict(qualified_quads=quad)
    extra={}
    if case.name=="east":
        with np.load(stock/"qualified_quads.npz")as sf:stock_ok=sf["mask"]
        xm=tifffile.imread(stock/"x.tif");zm=tifffile.imread(stock/"z.tif")
        sy,sx=np.where(xm>=0)
        xo=np.unique(xm[sy,sx]-sx*2);zo=np.unique(zm[sy,sx]-sy*2)
        assert len(xo)==len(zo)==1
        paired=keep&bounded_faces(p,stock_ok,np.array([xo[0],zo[0]]),[2.,2.],[0,2])
        sc=np.zeros_like(valid);sc[vv[paired],uu[paired]]=True
        stored["also_inside_stock_agreement_quads"]=sc
        extra["also_bracketed_and_stock_agreement_cm2"]=float(area[paired].sum())
    np.savez_compressed(out/(case.name+"_faces.npz"),**stored)
    digest={axis:hashlib.sha256((folder/(axis+".tif")).read_bytes()).hexdigest()for axis in "xyz"}
    return dict(exported_faces=int(valid.sum()),qualifying_faces=int(keep.sum()),native_area_cm2=float(area.sum()),
                bracketed_native_area_cm2=float(area[keep].sum()),largest_bracketed_component_cm2=float(cc.max()),
                components=int(n),source_mesh_axis_sha256=digest,**extra)


def main(root,out):
    out.mkdir(exist_ok=True,parents=True)
    src=root/"cases";refs={d:reference(src/d,d)for d in ("east","west","north","south")}
    stock=root/"stock/oriented8_loss_candidate_agreement.tifxyz"
    rows={f.name:faces(f,out,refs[f.name.split("_")[0]],stock)for f in sorted(src.iterdir()) if f.is_dir()}
    total=sum(r["bracketed_native_area_cm2"]for r in rows.values())
    base=sum(r["native_area_cm2"]for r in rows.values())
    report=dict(cases=rows,total_cm2=total,total_original_cm2=base,fraction=total/base,
                number_of_full_cells_by_direction={d:int(r[0].sum())for d,r in refs.items()},
                scientific_scope="All four corners of every 8-by-32-voxel reference cell touched by the bounding rectangle of an accepted flattened source quad are independently carried across the old-to-new boundary by unique stored organizer curves. Each original curve starts in the old area and agrees within 3 Y voxels at its sampled new coordinate. Shared M7 source, interpolated interior interval, no manual new winding label.")
    (out/"bracketed_geometry.json").write_text(json.dumps(report,indent=2))
    lines=["# PHerc. 125 mesh faces between corroborating organizer curves","",
        f"The seven native meshes provide {base:.6f} cm². Of that area, **{total:.6f} cm² ({total/base:.1%})** meets the four-corner control below. These faces form a subset of the reported area.","",
        "| Export | All mesh area (cm²) | Bracketed area (cm²) | Largest bracketed component (cm²) |",
        "|---|---:|---:|---:|"]
    for d,row in rows.items():lines.append(f"| {d} | {row['native_area_cm2']:.6f} | {row['bracketed_native_area_cm2']:.6f} | {row['largest_bracketed_component_cm2']:.6f} |")
    lines+=["",f"Of the exported east mesh, **{rows['east']['also_bracketed_and_stock_agreement_cm2']:.6f} cm²** also falls completely inside the largest connected oriented-stock agreement mask. Every two-voxel stock face touched by an included quad's native XZ bounding box is qualified at the three-voxel tolerance. This yields a restricted subset where both the original curves and a separate official GrowPatch run corroborate the coordinates using the same source M7. See also_inside_stock_agreement_quads in east_faces.npz.",""]
    lines+=["","A retained quad lies entirely within a union of 8-by-32-native-voxel rectangles, each with an original organizer curve at every corner. Each curve was chosen solely from its unique agreement with the older surface through a 128-voxel interior band. Its original stored path continues across the boundary without a large reversal. The carried height agrees with the new estimate within three native Y voxels at each tested corner. We require every rectangle touched by the quad's entire projected bounding box to pass. This intentionally excludes both unobserved border strips and cells touching a failed control. Adjacent controls on the same plane may lie on the same stored polyline.","",
        "The controls measure boundary continuity at discrete positions. Surface identity between adjacent source planes and the winding identity of the common M7 prediction still require expert review. Every qualifying face is also in the previously reported seven-export total. Geometry areas use native 3D triangles at 9.362 µm per voxel.","",
        "Each *_faces.npz contains a two-dimensional qualified_quads Boolean mask indexed by the upper-left vertex of each cell in the corresponding addendum TIFXYZ. The report binds every mask to its three source TIFF SHA-256 digests. For a source-supported CT overlay, expand the mask eight times along both axes. Reproduce from the unpacked addendum with the following command.","",
        "```text","python growbench_recover_unread_brackets.py /path/to/growbench_pherc125_extensions /path/to/output","```",""]
    (out/"README.md").write_text("\n".join(lines),encoding="utf8");print(json.dumps(report,indent=2))


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("package",type=Path);p.add_argument("output",type=Path)
    a=p.parse_args();main(a.package,a.output)
