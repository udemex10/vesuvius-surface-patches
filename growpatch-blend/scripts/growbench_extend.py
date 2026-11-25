"""Append a locally supported prediction to a GrowPatch boundary in the common seed frame."""
from pathlib import Path

import numpy as np
from scipy.ndimage import binary_dilation, convolve

from growbench_geometry import Projection, STEP, U, V, frame


def extend(stock, candidate, threshold):
    base = np.isfinite(stock.low)
    reliable = np.isfinite(candidate.low)&(candidate.confidence+1e-9>=threshold)&((candidate.high-candidate.low)<=1)
    joint = base & reliable
    agrees = joint & ((stock.high-stock.low)<=1) & (np.abs(stock.center-candidate.center)<=2)
    seam = agrees & (convolve(agrees.astype(np.uint8),np.ones((3,3),np.uint8),mode="constant")>=3)
    conflict = binary_dilation(joint&~agrees, iterations=2)
    allowed = ~base & reliable & ~conflict
    new = seam & ~conflict
    # Thirty-two native voxels along a connected candidate path from the stock boundary.
    for _ in range(32//STEP):
        new |= binary_dilation(new) & allowed
    new &= allowed
    return Projection(np.where(new,candidate.low,stock.low),np.where(new,candidate.high,stock.high),
                      np.ones(base.shape),np.where(new,candidate.area_factor,stock.area_factor))


def export_grid(path, projection, site):
    valid = np.isfinite(projection.low)&((projection.high-projection.low)<=1)
    pts = np.stack((U,V,projection.center),axis=-1) @ frame(site) + site["seed_xyz"]
    path = Path(path)
    path.parent.mkdir(parents=True,exist_ok=True)
    np.savez_compressed(path,xyz=pts.astype(np.float32),valid=valid)
