"""Draw a native-coordinate map and coverage of the connected source-polyline checks."""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.collections import PolyCollection
from matplotlib.patches import Patch
import numpy as np
import tifffile


def draw(root):
    pitch=.009362
    old=np.load(root/"previous_geometry/PHerc0125_y3310.npz")["mesh"]
    center=np.array([3400,10467])
    fig,(ax,bx)=plt.subplots(1,2,figsize=(10.6,6.6),gridspec_kw={"width_ratios":[1.3,1]})
    def add_mesh(mesh,color):
        ok=(mesh>=0).all(axis=-1)
        mask=ok[:-1,:-1]&ok[:-1,1:]&ok[1:,:-1]&ok[1:,1:]
        a,b,c,d=mesh[:-1,:-1],mesh[:-1,1:],mesh[1:,1:],mesh[1:,:-1]
        polygons=(np.stack([t[mask][:,[0,2]]for t in (a,b,c,d)],1)-center)*pitch
        ax.add_collection(PolyCollection(polygons,facecolor=color,edgecolor="none",antialiased=False))
    add_mesh(old,"#ccd3d6")
    for case in sorted((root/"cases").iterdir()):
        folder=next(case.glob("*.tifxyz"))
        mesh=np.stack([tifffile.imread(folder/(a+".tif")) for a in "xyz"],-1)
        add_mesh(mesh,"#1f807b")
    ax.autoscale();ax.set_aspect("equal")
    ax.set_xlabel("Native X relative to the original seed (mm)")
    ax.set_ylabel("Native Z relative to the original seed (mm)")
    ax.set_title("Exported interior surfaces",loc="left",weight="bold",fontsize=12)
    ax.legend(handles=[Patch(color="#ccd3d6",label="Previous Y3310 surface"),Patch(color="#1f807b",label="New disjoint exports")],loc="lower right",frameon=False,fontsize=9)
    rows=[]
    for direction in ("east","north","south","west"):
        report=json.loads((root/"cases"/direction/"organizer_continuity.json").read_text())
        rows.append([report[k]for k in ("new_consensus_nodes","new_nodes_on_single_original_polyline","within_three_voxels")])
    entries=np.asarray(rows);y=np.arange(4);bx.set_xlim(0,1);colors=["#e1e4e5","#d7a84e","#1f807b"]
    for index in range(3):bx.barh(y,entries[:,index]/entries[:,0],height=.50,color=colors[index])
    for i,(all_nodes,on_line,close)in enumerate(entries):
        bx.text(0,i+.42,f"{close:,} agree  /  {on_line:,} reached  /  {all_nodes:,} tested",fontsize=8,color="#38464b",va="center")
    bx.set_yticks(y,["East","North","South","West"]);bx.set_ylim(3.7,-.55)
    bx.set_xticks([0,.25,.5,.75,1],["0","25","50","75","100%"])
    bx.set_xlabel("Share of tested extension points")
    bx.set_title("One continuous organizer curve",loc="left",weight="bold",fontsize=12)
    bx.legend(handles=[Patch(color=colors[2],label="Within 3 voxels"),Patch(color=colors[1],label="Reached, outside tolerance"),Patch(color=colors[0],label="No unique anchored curve")],loc="lower left",bbox_to_anchor=(0,-.27),frameon=False,fontsize=9)
    for chart in (ax,bx):
        chart.spines[["top","right"]].set_visible(False);chart.tick_params(color="#9aa5a8")
    fig.suptitle("PHerc. 125  |  Surface extensions from the Y3310 region",x=.075,ha="left",weight="bold",fontsize=16)
    fig.subplots_adjust(left=.085,right=.98,top=.86,bottom=.19,wspace=.3)
    fig.savefig(root/"coverage.png",dpi=165,facecolor="white");plt.close(fig)


if __name__=="__main__":
    import argparse
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("root",type=Path);draw(p.parse_args().root)
