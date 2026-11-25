"""Render all sixteen fixed external regions using their paired native sheet-ID assignments."""

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap
from matplotlib.patches import Patch
import numpy as np


COLORS = ["#f5f2ec", "#c9cac7", "#455a6c", "#21b5a0", "#f1aa38", "#dc374c", "#70528c"]
KEY = ["Outside scored reference", "Other scored position", "Target retained", "Target gained",
       "Target lost to unassigned", "New different sheet", "Different sheet retained"]


def render(cases, output):
    bg = "#f5f2ec"
    fig = plt.figure(figsize=(10.6, 11.5), facecolor=bg)
    fig.text(.07,.955,"Where the blend recovered target material",fontsize=24,fontweight="bold",color="#1e2e36")
    fig.text(.07,.924,"All sixteen external sites  ·  Official GrowPatch threshold 128 and the fixed eight-voxel blend",
             fontsize=11,color="#384b52")
    fig.text(.07,.900,"Each map spans 1.52 mm in the seed tangent plane. Values give the change in connected correct area.",
             fontsize=10,color="#384b52")
    area = fig.add_gridspec(4,4,top=.858,bottom=.14,left=.07,right=.94,hspace=.30,wspace=.16)
    for index,(name,stock,trial,weights,delta_mm2) in enumerate(cases):
        chart=np.ones(stock.shape,dtype=np.uint8)
        chart[(stock==3)&(trial==3)]=2
        chart[(stock!=3)&(trial==3)]=3
        chart[(stock==3)&(trial!=3)]=4
        chart[(stock!=4)&(trial==4)]=5
        chart[(stock==4)&(trial==4)]=6
        chart[weights==0]=0
        ax=fig.add_subplot(area[index//4,index%4],facecolor=bg)
        ax.imshow(chart,cmap=ListedColormap(COLORS),vmin=0,vmax=6,interpolation="nearest",origin="lower")
        ax.set_axis_off()
        ax.set_title(f"{name.split('_')[-1]}     {delta_mm2:+.3f} mm²",fontsize=10,loc="center",pad=2,color="#213640")
    legend=[Patch(facecolor=c,edgecolor="#b8b6b0" if idx==0 else "none",label=k) for idx,(c,k) in enumerate(zip(COLORS,KEY))]
    fig.legend(legend,[h.get_label() for h in legend],loc="lower center",bbox_to_anchor=(.5,.065),
               ncol=4,frameon=False,fontsize=9,handletextpad=.5,columnspacing=1.2)
    fig.text(.07,.039,"Colors classify paired samples of the native manual labels. The seed collar and unsupported positions are excluded.",
             fontsize=9,color="#384b52")
    fig.savefig(output,dpi=180,facecolor=bg)
    plt.close(fig)
