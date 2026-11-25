"""Export all guarded consensus components spanning at least 4,000 fitted nodes."""
import argparse
import numpy as np
from scipy.ndimage import label
import growbench_recover_unread_export as export


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument("direction",choices=["east","west","north","south"]);a=p.parse_args()
    d=a.direction
    directory=export.DEST.parent/("pherc125_"+("extension" if d=="east" else d))
    export.DEST=directory;export.DIRECTION=d
    with np.load(directory/"consensus.npz") as src:
        ids,_=label(src["new"]);count=int((np.bincount(ids.ravel())[1:]>=4000).sum())
    for rank in range(1,count+1):
        export.COMPONENT_RANK=rank;export.EXPORT=directory/("flat" if rank==1 else f"flat_{rank}")
        if rank>1:
            folder=export.mesh()
            export.sample(folder)
        export.evidence()
