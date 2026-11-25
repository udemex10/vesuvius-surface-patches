"""Native CT review strips at disagreements between the two PHerc.125 meshes."""
import hashlib
import json
import argparse
from zipfile import ZipFile, ZIP_DEFLATED

import matplotlib
matplotlib.use("Agg")
from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
import numpy as np

from growbench_stock import ROOT,digest,save
from growbench_stock_unread import EXT,OLD,OUT
from r3_chunkvol import CACHE,ChunkVolume


def main():
    source=json.loads((EXT/"flat/meta.json").read_text())["native_volume_url"]
    result=json.loads((OUT/"oriented8_g160_loss_application.json").read_text())
    options=result.get("physical_review_locations",[])
    selected=[]
    for region in options:
        pt=np.asarray(region["candidate_xyz"])
        if all(np.linalg.norm(pt-np.asarray(other["candidate_xyz"]))>=45 for other in selected):
            selected.append(region)
        if len(selected)==3: break
    if not selected: return
    destination=OUT/"longer_run_ct_review"
    destination.mkdir(exist_ok=True)
    CACHE.mkdir(parents=True,exist_ok=True)
    volume=ChunkVolume(source,allow_missing=False,max_bytes=100*1024**2)
    with np.load(EXT/"consensus.npz") as data:
        x,z=data["x"],data["z"]
        candidate=np.where(data["new"],data["y"],np.nan)
    with np.load(OUT/"oriented8_g160_loss_raster.npz") as data:
        lo,hi=data["low"],data["high"]
        good=np.isfinite(lo)&((hi-lo)<=1)
        stock=np.full(lo.shape,np.nan)
        stock[good]=(lo[good]+hi[good])/2
    fig,axes=plt.subplots(2,len(selected),figsize=(3.7*len(selected),4.8),squeeze=False,sharex="col",sharey="col")
    report=[]
    for index,region in enumerate(selected):
        cx,cy,cz=region["candidate_xyz"]
        sy=region["stock_xyz"][1]
        center=int(round((cy+sy)/2))
        half_y=max(38,int(np.ceil(abs(cy-sy)/2))+22)
        xs=np.arange(int(cx)-64,int(cx)+65)
        ys=np.arange(center-half_y,center+half_y+1)
        xx,yy=np.meshgrid(xs,ys)
        raw=volume.gather(np.full(xx.shape,int(cz)),yy,xx)
        vmin,vmax=np.percentile(raw[raw>0],[2,98])
        iz=int(np.flatnonzero(z==cz)[0])
        take=(x>=xs[0])&(x<=xs[-1])
        target=candidate[iz,take]
        baseline=stock[iz,take]
        file=destination/f"native_section_{index+1:02d}.npz"
        np.savez_compressed(file,ct_yx=raw,ct_x=xs,ct_y=ys,ct_z=int(cz),trace_x=x[take],
                            candidate_y=target,stock_y=baseline)
        for ax in axes[:,index]:
            ax.imshow(raw,cmap="gray",vmin=vmin,vmax=vmax,origin="lower",interpolation="nearest",
                      extent=[xs[0]-.5,xs[-1]+.5,ys[0]-.5,ys[-1]+.5],aspect="equal")
            ax.ticklabel_format(useOffset=False)
            ax.tick_params(axis="both",labelsize=7)
        axes[0,index].set_title(f"Native Z {cz:.0f}  ·  X {cx:.0f}",fontsize=10)
        axes[1,index].plot(x[take],target,color="#ffbf00",lw=1.1)
        axes[1,index].plot(x[take],baseline,color="#00cbe5",lw=1.1,ls="--")
        axes[1,index].set_xlabel("Native X (voxels)",fontsize=8)
        report.append(dict(**region,section=file.name,ct_z=int(cz),native_ct_window=[float(vmin),float(vmax)],
             raw_ct_sha256=hashlib.sha256(raw.tobytes()).hexdigest()))
    axes[0,0].set_ylabel("Native CT\nY (voxels)",fontsize=9)
    axes[1,0].set_ylabel("Trace overlay\nY (voxels)",fontsize=9)
    fig.suptitle("PHerc. 125 east  ·  locations of persistent height disagreement",fontsize=12)
    fig.legend([Line2D([0],[0],color="#b98000"),Line2D([0],[0],color="#0093b0",ls="--")],
               ["Multi-route candidate","160-generation stock GrowPatch"],loc="lower center",ncol=2,frameon=False)
    fig.subplots_adjust(bottom=.16,top=.87,hspace=.08,wspace=.24)
    fig.savefig(destination/"native_ct_disagreements.png",dpi=180,bbox_inches="tight")
    plt.close(fig)
    save(destination/"locations.json",dict(native_ct_url=source,pitch_um=9.362,
        purpose="Raw physical CT profiles for inspection; no independently assigned sheet identity at these coordinates.",
        selection="Up to three spatially separated centers of the largest 4-connected disagreement components in common support. Centers maximize distance to the component boundary; Y differences exceed three voxels.",
        sites=report))


def bundle():
    arm="oriented8_g160_loss"
    package=OUT/"PHerc125_east_extended_stock_check.zip"
    selected=[OUT/"README.md",OUT/"seeds.json",OUT/"default20_loss_application.json",
              OUT/"oriented8_loss_application.json",OUT/(arm+"_application.json"),OUT/"east_stock_comparison.png",
              EXT/"consensus.npz",EXT/"extension.json",EXT/"flat/meta.json",EXT/"route_06/track.json",
              EXT/"route_06/00/source.json",OLD,OUT/"inputs/pherc125_y3310_east/m7.json",
              ROOT/"results/growbench/stock/README.md",
              ROOT/"scripts/growbench_stock.py",ROOT/"scripts/growbench_stock_unread.py",
              ROOT/"scripts/growbench_stock_unread_sections.py",ROOT/"scripts/r3_chunkvol.py",
              ROOT/"scripts/growbench_geometry.py"]
    selected.extend((OUT/(arm+"_candidate_agreement.tifxyz")).glob("*"))
    selected.extend((EXT/"route_06").glob("*/supported.npz"))
    selected.extend((OUT/"longer_run_ct_review").glob("*"))
    for mode in ("default20_loss","oriented8_loss",arm):
        selected.extend((OUT/mode/"pherc125_y3310_east"/name) for name in ("params.json","record.json","surface.npz","grow.log","grow_command.json"))
    manifest=[]
    with ZipFile(package,"w",compression=ZIP_DEFLATED,compresslevel=6) as archive:
        intro="""# PHerc. 125 east — stock GrowPatch follow-up

Read results/growbench/stock_pherc125/README.md. The longer-ceiling test uses an unmodified official binary on the identical native M7 crop. Its area is conditional on the previously released Y3310 seed. See oriented8_g160_loss_application.json for metrics. Open longer_run_ct_review/native_ct_disagreements.png for raw physical cross-sections and both projected traces.

The XYZ coordinate TIFFs and exact admitted face mask are in oriented8_g160_loss_candidate_agreement.tifxyz. The official mesh before comparison, local configuration, runtime receipt and binary hashes are in oriented8_g160_loss/pherc125_y3310_east. The public M7 input can be downloaded again with scripts/growbench_stock_unread.py. Install NumPy, SciPy, Pillow, tifffile, numcodecs, requests, filelock and matplotlib under Python 3.12 for replay. The native release runner uses Windows x64.
"""
        archive.writestr("README.md",intro)
        manifest.append(hashlib.sha256(intro.encode()).hexdigest()+"  README.md")
        for path in sorted(selected):
            name=path.relative_to(ROOT).as_posix()
            archive.write(path,name)
            manifest.append(digest(path)+"  "+name)
        archive.writestr("SHA256SUMS","\n".join(manifest)+"\n")
    with ZipFile(package) as archive:
        assert archive.testzip() is None
        for line in archive.read("SHA256SUMS").decode().splitlines():
            signature,name=line.split("  ",1)
            assert hashlib.sha256(archive.read(name)).hexdigest()==signature
    print(package.name,package.stat().st_size,digest(package),flush=True)


if __name__=="__main__":
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--bundle",action="store_true")
    if parser.parse_args().bundle: bundle()
    else: main()
