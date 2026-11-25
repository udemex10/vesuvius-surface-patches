"""Matched single-anchor stock GrowPatch check of the PHerc.125 east extension."""
import argparse
import json
import re
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator
from scipy.ndimage import distance_transform_edt, label
import tifffile

from growbench_geometry import faces
from growbench_stock import ROOT, digest, save
from r3_chunkvol import CACHE,ChunkVolume


EXT = ROOT / "results/growbench/recover/pherc125_extension"
OUT = ROOT / "results/growbench/stock_pherc125"
ANCHOR = EXT / "route_06/00/source.json"
OLD = ROOT / "results/first_letters_deep/PHerc0125_y3310/geometry_nodes.npz"
SITE = "pherc125_y3310_east"
# Fixed physical crop, tied to center of the previously released old sheet.
LO = np.array([3584, 3008, 10176])  # inclusive global XYZ
HI = np.array([4992, 3776, 10752])  # exclusive global XYZ
PITCH = 9.362


def extract():
    source = json.loads(ANCHOR.read_text())
    seed = source["site"]["seed_xyz"]
    normal = np.array(source["site"]["normal_xyz"])
    u = np.array([1., 0., 0.])
    u -= normal * np.dot(u, normal)
    u /= np.linalg.norm(u)
    v = np.cross(u, normal)
    record = dict(id=SITE, split="application", seed_xyz=seed, normal_xyz=normal.tolist(),
                  tangent_u_xyz=u.tolist(), tangent_v_xyz=v.tolist(), origin_xyz=LO.tolist())
    folder = OUT / "inputs" / SITE
    folder.mkdir(parents=True, exist_ok=True)
    path = folder / "m7.npy"
    expected = tuple(map(int, (HI-LO)[::-1]))
    ready = False
    if path.exists():
        try:
            probe = np.load(path, mmap_mode="r")
            ready = probe.shape==expected and probe.dtype==np.uint8
            del probe
        except ValueError:
            pass
    if not ready:
        CACHE.mkdir(parents=True,exist_ok=True)
        vol = ChunkVolume(source["cue"], max_bytes=125*1024**2, workers=8, allow_missing=True)
        lower, upper = LO[::-1], HI[::-1]
        a0, a1 = lower//vol.chunk, (upper-1)//vol.chunk
        keys = [tuple(map(int, a0+offset)) for offset in np.ndindex(tuple(a1-a0+1))]
        temp = folder / "m7.partial.npy"
        dest = np.lib.format.open_memmap(temp, mode="w+", dtype=np.uint8, shape=expected)
        for index in range(0, len(keys), 12):
            arrays = vol.chunks(keys[index:index+12])
            for key, data in arrays.items():
                start = np.asarray(key)*vol.chunk
                a, b = np.maximum(start, lower), np.minimum(start+vol.chunk, upper)
                dest[tuple(map(slice, a-lower, b-lower))] = data[tuple(map(slice, a-start, b-start))]
        dest.flush()
        del dest
        temp.replace(path)
        assert np.load(path, mmap_mode="r").shape == expected
        save(folder/"m7.json", dict(url=source["cue"],global_xyz_half_open=[LO.tolist(), HI.tolist()],
            decoded_shape_zyx=(upper-lower).tolist(),shared_readthrough_cache="data/cache",
            sha256_npy=digest(path),chunk_shape_zyx=vol.chunk.tolist(),newly_downloaded_compressed_bytes=vol.downloaded,
            absent_sparse_keys=[list(map(int,x)) for x in sorted(vol.missing)],chunk_keys_zyx=keys))
    save(OUT/"seeds.json",dict(frame="PHerc.125 native level-zero XYZ, 9.362 um",area_radius_voxels=512,
         anchor_source=str(ANCHOR.relative_to(ROOT)),anchor_source_sha256=digest(ANCHOR),sites=[record]))
    print("native crop ready", path, flush=True)


def stock_profile(mesh, x, z):
    with np.load(mesh) as data:
        triangles = faces(data["xyz"], data["valid"])[0].astype(np.float64)[:, :, [0, 2, 1]]
    span = (len(x), len(z))
    low = np.full((span[1], span[0]), np.inf)
    high = np.full(low.shape, -np.inf)
    for tri in triangles:
        xa, za = np.maximum(0, np.ceil((tri[:, :2].min(0)-[x[0],z[0]])/2-1e-8).astype(int))
        xb, zb = np.minimum(np.array(span)-1,np.floor((tri[:, :2].max(0)-[x[0],z[0]])/2+1e-8).astype(int))
        if xa>xb or za>zb:
            continue
        ba, ca = tri[1]-tri[0], tri[2]-tri[0]
        det = ba[0]*ca[1]-ca[0]*ba[1]
        if abs(det)<1e-8:
            continue
        dz = z[za:zb+1, None]-tri[0, 1]
        dx = x[None, xa:xb+1]-tri[0, 0]
        s, t = (dx*ca[1]-dz*ca[0])/det, (ba[0]*dz-ba[1]*dx)/det
        inside = (s>=-1e-8)&(t>=-1e-8)&(s+t<=1+1e-8)
        y = tri[0,2]+s*ba[2]+t*ca[2]
        lo = low[za:zb+1, xa:xb+1]
        hi = high[za:zb+1, xa:xb+1]
        np.minimum(lo, y, out=lo, where=inside)
        np.maximum(hi, y, out=hi, where=inside)
    return low, high


def route_six(x, z):
    height = np.full((len(z),len(x)), np.nan)
    best = np.zeros_like(height)
    folder = EXT/"route_06"
    for i, step in enumerate(json.loads((folder/"track.json").read_text())["steps"]):
        if "valid_nodes" not in step:
            continue
        ix = np.searchsorted(x, step["cx"]-96)
        iz = np.searchsorted(z, 10467-96)
        region = np.s_[iz:iz+97, ix:ix+97]
        with np.load(folder/f"{i:02d}"/"supported.npz") as data:
            weight = data["q"]*(1-np.abs(np.arange(97)-48)[None,:]/120)
            keep = data["v"]&(weight>best[region])
            best[region][keep] = weight[keep]
            height[region][keep] = data["y"][keep]
    return height


def physical_quads(y):
    a,b,c,d = (y[s] for s in (np.s_[:-1,:-1],np.s_[:-1,1:],np.s_[1:,1:],np.s_[1:,:-1]))
    return (np.sqrt((2*(b-a))**2+16+(2*(c-b))**2)+
            np.sqrt((2*(c-d))**2+16+(2*(d-a))**2))*.5*(PITCH/10000)**2


def agreement_patch(x,z,y,mask,variant):
    cells=mask[:-1,:-1]&mask[:-1,1:]&mask[1:,1:]&mask[1:,:-1]
    if not cells.any():
        return None
    area=physical_quads(y)
    components,_=label(cells)
    areas=np.bincount(components[cells],weights=area[cells])
    largest=int(np.argmax(areas[1:])+1)
    faces_ok=components==largest
    nodes=np.zeros_like(mask)
    for iz,ix in np.ndindex(2,2):
        nodes[iz:iz+cells.shape[0],ix:ix+cells.shape[1]]|=faces_ok
    zi,xi=np.where(nodes)
    sl=np.s_[zi.min():zi.max()+1,xi.min():xi.max()+1]
    xx,zz=np.meshgrid(x,z)
    xyz=np.stack((xx,y,zz),-1)[sl].astype("f4")
    xyz[~nodes[sl]]=-1
    final=OUT/(variant+"_candidate_agreement.tifxyz")
    final.mkdir(exist_ok=True)
    for k,key in enumerate("xyz"):
        tifffile.imwrite(final/(key+".tif"),xyz[:,:,k])
    bbox=[xyz[nodes[sl]].min(0).tolist(),xyz[nodes[sl]].max(0).tolist()]
    save(final/"meta.json",dict(type="seg",format="tifxyz",uuid=final.stem,name=final.stem,
         scale=[.5,.5],bbox=bbox,qualification="candidate consensus and native stock same-cue geometry agree within 3 Y voxels on interior east samples"))
    np.savez_compressed(final/"qualified_quads.npz",mask=faces_ok[zi.min():zi.max(),xi.min():xi.max()])
    return dict(path=final.name,largest_connected_cm2=float(areas[largest]),total_cm2=float(area[cells].sum()),
                largest_connected_quads=int(faces_ok.sum()),bbox_xyz=bbox,
                sha256={f.name:digest(f) for f in sorted(final.iterdir())},
                qualified_quads="qualified_quads.npz contains the exact measured faces; 4-connected component, 2-voxel node spacing.")


def review_regions(x,z,target,stock,mask):
    components,n=label(mask)
    if not n: return []
    order=np.argsort(np.bincount(components.ravel(),minlength=n+1)[1:])[::-1]+1
    places=[]
    for key in order[:5]:
        component=components==key
        iz,ix=np.unravel_index(np.argmax(distance_transform_edt(component)),component.shape)
        places.append(dict(connected_disagreement_rays=int(component.sum()),
            candidate_xyz=[float(x[ix]),float(target[iz,ix]),float(z[iz])],
            stock_xyz=[float(x[ix]),float(stock[iz,ix]),float(z[iz])],
            median_absolute_delta_y_vox=float(np.median(np.abs(target[component]-stock[component])))))
    return places


def ceiling_check(variant):
    configs=[]
    counts=[]
    for arm in ("oriented8_loss",variant):
        base=OUT/arm/SITE
        configs.append(json.loads((base/"params.json").read_text()))
        counts.append(re.findall(r"gen (\d+) processing (\d+) fringe cands \(total done (\d+) fringe: (\d+)\)",
                                 (base/"grow.log").read_text()))
    differences={k:[configs[0].get(k),configs[1].get(k)] for k in set(configs[0])|set(configs[1])
                 if configs[0].get(k)!=configs[1].get(k)}
    divergence=next((int(a[0]) for a,b in zip(*counts) if a!=b),None)
    return dict(native_configuration_differences=differences,first_distinct_native_frontier_count_generation=divergence,
                last_logged_generation=int(counts[-1][-1][0]),
                qualification="Each native run starts at the original seed. The native work-grid size is a function of the generation ceiling, so the larger run is a separate stock sensitivity.")


def coverage_scope(xx,zz,y):
    eligible=np.isfinite(y)&(xx>=4056)
    xyz=np.stack((xx,y,zz),-1)
    area=physical_quads(y)
    def measured(mask):
        full=mask[:-1,:-1]&mask[:-1,1:]&mask[1:,1:]&mask[1:,:-1]
        return float(area[full].sum())
    whole=measured(eligible)
    inside=measured(eligible&((xyz>=LO)&(xyz<HI)).all(-1))
    interior=measured(eligible&((xyz>=LO+32)&(xyz<HI-32)).all(-1))
    return dict(total_east_candidate_cm2=whole,wholly_inside_input_cm2=inside,
                wholly_inside_scored_interior_cm2=interior,outside_or_crossing_input_boundary_cm2=whole-inside,
                inside_input_but_outside_scored_collar_cm2=inside-interior)


def compare(variant):
    path = OUT / variant / SITE / "surface.npz"
    with np.load(EXT/"consensus.npz") as consensus:
        x,z = consensus["x"],consensus["z"]
        target = np.where(consensus["valid"],consensus["y"],np.nan)
    low,high = stock_profile(path,x,z)
    stock = np.isfinite(low)
    center = np.full_like(low, np.nan)
    center[stock] = (low[stock]+high[stock])/2
    xx,zz = np.meshgrid(x,z)
    east = (xx>=4056)&(xx<HI[0]-32)&(zz>=LO[2]+32)&(zz<HI[2]-32)
    centerline = np.abs(zz-10467)<=80
    references = {"central_route_06_east":(route_six(x,z),east&centerline),
                  "consensus_central_corridor_east":(target,east&centerline),
                  "consensus_all_interior_east":(target,east)}
    with np.load(OLD) as original:
        old_y = np.where(original["qualified"],original["fit"],np.nan)
        old = RegularGridInterpolator((original["z"],original["x"]),old_y,bounds_error=False,fill_value=np.nan)
        references["previously_released_sheet"]=(old((zz,xx)),(xx<4056)&centerline)
    output,requested_review = {},[]
    for name,(ref,roi) in references.items():
        support = roi&np.isfinite(ref)&(ref>=LO[1]+32)&(ref<HI[1]-32)
        common = support&stock
        # Sum actual 3-D areas of two Y(X,Z) triangles per 2x2-voxel reference cell.
        area = physical_quads(ref)
        def summary(mask):
            cells=mask[:-1,:-1]&mask[:-1,1:]&mask[1:,1:]&mask[1:,:-1]
            return dict(rays=int(mask.sum()),area_cm2=float(area[cells].sum()))
        delta = np.abs(center-ref)
        match = common&((high-low)<=1)
        output[name]=dict(reference=summary(support),stock_intersects=summary(common),
            same_height_3vox=summary(match&(delta<=3)),same_height_6vox=summary(match&(delta<=6)),
            folded_rays=int((common&((high-low)>1)).sum()),max_east_stock_x=int(xx[common].max()) if common.any() else None,
            median_y_difference_vox=float(np.median(delta[common])) if common.any() else None)
        if name=="consensus_all_interior_east":
            patch=agreement_patch(x,z,ref,match&(delta<=3),variant)
            if variant=="oriented8_g160_loss":
                requested_review=review_regions(x,z,ref,center,match&(delta>3))
    denominator = stock&east
    save(OUT/(variant+"_application.json"),dict(stock_mesh_sha256=digest(path),proposed_mesh_sha256=digest(EXT/"consensus.npz"),
         pitch_um=PITCH,east_guard_x=4056,central_z_half_width=80,interior_crop_margin_vox=32,
         stock_native_east_max_x=int(xx[denominator].max()) if denominator.any() else None,sections=output,
         largest_candidate_stock_agreement_patch=patch,
         **({"physical_review_locations":requested_review,"ceiling_check":ceiling_check(variant),
             "scope":coverage_scope(xx,zz,target)} if variant=="oriented8_g160_loss" else {}),
         interpretation="Agreement of algorithms sharing M7 and an initial seed. The candidate extension has no independent external winding labels here. The central-route test matches a single seed. The multi-route consensus has 13 seeds."))
    np.savez_compressed(OUT/(variant+"_raster.npz"),x=x,z=z,low=low,high=high)
    print(variant, output["central_route_06_east"],flush=True)


def figure():
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.patches import Patch

    with np.load(EXT/"consensus.npz") as data:
        x,z,y,mask=(data[k] for k in ("x","z","y","new"))
    extent=((x[0]-3880-1)*PITCH/1000,(x[-1]-3880+1)*PITCH/1000,
            (z[0]-10467-1)*PITCH/1000,(z[-1]-10467+1)*PITCH/1000)
    xx,zz=np.meshgrid(x,z)
    interior=(xx<HI[0]-32)&(zz>=LO[2]+32)&(zz<HI[2]-32)&(y>=LO[1]+32)&(y<HI[1]-32)
    colors=["#ffffff","#dce1e6","#c8ae72","#147d73","#b94e3e"]
    fig,axes=plt.subplots(1,2,figsize=(10.5,5.2),sharex=True,sharey=True)
    for ax,variant,title in zip(axes,("default20_loss","oriented8_loss"),("20-voxel point seed","8-voxel oriented seed")):
        with np.load(OUT/(variant+"_raster.npz")) as data:
            low,high=data["low"],data["high"]
        common=mask&interior&np.isfinite(low)
        agree=np.zeros_like(common)
        agree[common]=((high[common]-low[common])<=1)&(np.abs((high[common]+low[common])/2-y[common])<=3)
        image=np.zeros_like(mask,dtype=np.uint8)
        image[mask]=1
        image[mask&interior]=2
        image[common]=4
        image[agree]=3
        ax.imshow(image,cmap=ListedColormap(colors),vmin=-.5,vmax=4.5,extent=extent,origin="lower",interpolation="nearest")
        stats=json.loads((OUT/(variant+"_application.json")).read_text())["sections"]["consensus_all_interior_east"]
        ax.set_title(f"{title}\n{stats['same_height_3vox']['area_cm2']:.3f} of {stats['reference']['area_cm2']:.3f} cm² reproduced",fontsize=11)
        ax.axvline((4056-3880)*PITCH/1000,ls=":",lw=1,color="#42454b")
        ax.plot([0],[0],"+",markersize=10,color="#20242a")
        ax.set_xlabel("East of original seed (mm)")
        ax.spines[["top","right"]].set_visible(False)
    axes[0].set_ylabel("Native Z from seed (mm)")
    labels=["Candidate beyond comparison crop","Candidate only inside crop","Agrees within 3 Y voxels","Overlaps and exceeds tolerance"]
    fig.legend([Patch(facecolor=c) for c in colors[1:]],labels,loc="lower center",ncol=2,frameon=False,fontsize=9)
    fig.suptitle("PHerc. 125 east extension compared with stock GrowPatch",fontsize=15)
    fig.subplots_adjust(bottom=.23,top=.84,wspace=.1)
    fig.savefig(OUT/"east_stock_comparison.png",dpi=180,bbox_inches="tight")
    plt.close(fig)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compare",choices=("default20_loss","oriented8_loss","oriented8_g160_loss"))
    parser.add_argument("--figure",action="store_true")
    args=parser.parse_args()
    if args.compare: compare(args.compare)
    elif args.figure: figure()
    else: extract()
