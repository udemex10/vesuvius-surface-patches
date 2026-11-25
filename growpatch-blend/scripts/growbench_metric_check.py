"""Analytic controls for area, acceptance, folds, boundaries and a stock-first extension."""
import json

import numpy as np

from growbench_extend import extend
from growbench_geometry import AXIS, PITCH_UM, R2, STEP, U, V, faces, project, score


def main():
    site=dict(seed_xyz=[400,400,400],normal_xyz=[0,0,1],tangent_u_xyz=[1,0,0],tangent_v_xyz=[0,1,0])
    grid=np.stack((U+400,V+400,np.full(U.shape,400)),axis=-1).astype(float)
    def raster(g,mask=None,quality=None):
        tri,q=faces(g,mask,quality)
        return project(tri,site,q)
    ref=raster(grid)
    flat=score(ref,ref,0)
    expected=float(np.count_nonzero((R2<=96**2)&(R2>12**2))*(STEP*PITCH_UM/10000)**2)
    assert abs(flat["connected"]-expected)<1e-12 and flat["incorrect"]==0
    tilted=grid.copy()
    tilted[:,:,2]+=U*.75
    incline=raster(tilted)
    checked=score(incline,incline,0)
    assert abs(checked["connected"]-1.25*expected)<1e-12
    displaced=grid.copy()
    displaced[:,:,2]+=3
    wrong=raster(displaced)
    assert abs(score(ref,wrong,0)["incorrect"]-expected)<1e-12
    assert abs(score(ref,wrong,0,4)["connected"]-expected)<1e-12
    lo,_=faces(grid)
    hi,_=faces(displaced)
    stacked=project(np.concatenate((lo,hi)),site)
    assert abs(score(ref,stacked,0)["incorrect"]-expected)<1e-12
    assert score(stacked,ref,0)["reference"]==0
    disconnected=raster(grid,(U<10)|(U>22))
    islands=score(ref,disconnected,0)
    assert 0<islands["connected"]<islands["correct"]<expected
    smaller=raster(grid,U<=0)
    fused=extend(smaller,ref,0)
    assert score(ref,smaller,0)["connected"]<score(ref,fused,0)["connected"]<expected
    assert score(ref,extend(smaller,wrong,0),0)==score(ref,smaller,0)
    graded=raster(grid,quality=np.ones(R2.shape)*.4)
    assert score(ref,graded,.5)["accepted"]==0
    print(json.dumps(dict(status="passed",flat_disc_annulus_cm2=expected,
                          inclined_area_ratio=checked["connected"]/expected,
                          cases=["area","incline","displacement","double intersection","ambiguous reference",
                                 "disconnected island","safe seam","conflicting seam","abstention"])))


if __name__=="__main__":
    main()
