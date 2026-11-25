"""Qualify reciprocal PHerc0125 geometry and sample the native scan at its 9.362 µm pitch."""
import json,sys
import numpy as np
import requests,tifffile
from scipy.ndimage import distance_transform_edt,map_coordinates,gaussian_filter,gaussian_filter1d
from first_letters_0125_surface import F,ROOT,XC,ZC,DX
from first_letters_0125 import CT,GRID
from first_letters_grid_cases import CASE,SPECIMEN
from first_letters_native_sample import sample_native,sfile
def main(seed):
    f=np.load(F/f'solution_{seed}.npz')
    a=f['axial'][:,::2];b=f['sagittal'][::2];av=f['axial_valid'][:,::2];bv=f['sagittal_valid'][::2]
    agree=av&bv&(abs(a-b)<=4);disagree=av&bv&(abs(a-b)>4)
    dist,idx=distance_transform_edt(~agree,return_indices=True)
    y=(a+b)/2
    for hi in [1,2]:
        good=agree.copy()
        q=gaussian_filter(np.where(good,y,0),hi)/np.maximum(gaussian_filter(good.astype(float),hi),1e-8)
        if hi==1:fit=q
        else:fit=np.where(dist<1.8,fit,q)
    # For singly observed knots near an agreed crossing include the observed
    # trace if it fits the local consensus. Never fill an explicit disagreement.
    solo=(av^bv)&(dist<=2.4)
    direct=np.where(av,a,b)
    fgood=agree | (solo&(abs(direct-fit)<=3.5))
    fit=np.where(agree,.78*y+.22*fit,np.where(fgood,.7*direct+.3*fit,fit))
    # A tiny hole can be bridged by interpolation backed on every side. Keep
    # separate direct and interpolated knot masks, radius and discrepancy.
    valid=(~disagree)&(dist<=2.4)
    fit[~valid]=fit[tuple(idx)][~valid]
    dp=D=DX[::2]
    shape=(len(D)-1)*16;gz,gx=np.mgrid[:shape,:shape].astype('f4')/16
    pts=np.array([gz,gx]);h=map_coordinates(fit,pts,order=1)
    mask=map_coordinates(valid.astype('f4'),pts,order=1)>.99999
    zv,xv=np.mgrid[:shape,:shape];z=zv+D[0]+ZC;x=xv+D[0]+XC
    hy,hx=np.gradient(h);n=np.stack([hx,-np.ones_like(h),hy],-1);n/=np.linalg.norm(n,axis=2,keepdims=True)
    xyz=np.stack([x,h,z],-1)
    dest=ROOT/f'results/first_letters_deep/{SPECIMEN}_y{seed}'
    mesh=dest/'mesh.tifxyz';mesh.mkdir(parents=True,exist_ok=True)
    xm,zm=np.meshgrid(D+XC,D+ZC)
    for c,mat in zip('xyz',(xm,fit,zm)):tifffile.imwrite(mesh/f'{c}.tif',np.where(valid,mat,-1).astype('f4'))
    (mesh/'meta.json').write_text(json.dumps(dict(format='tifxyz',type='seg',scale=[1/16,1/16],uuid=dest.name)))
    np.savez_compressed(dest/'geometry_nodes.npz',fit=fit,axial=a,sagittal=b,axial_observed=av,sagittal_observed=bv,direct_agreement=agree,qualified=valid,x=DX[::2]+XC,z=DX[::2]+ZC)
    info=dict(specimen=SPECIMEN,volume_id=CASE['id'],field=dest.name,
              flat_origin=dict(x=int(x[0,0]),z=int(z[0,0])),projection='Y(X,Z), columns increasing native x and rows increasing native z; forward depth along (~hx,-1,~hz)',
              organizer_grid_url=GRID,source_grids=F.name+'/xy and yz, immutable underlying polylines',mesh='mesh.tifxyz',
              surface_sha256={c:sfile(mesh/(c+'.tif'))for c in 'xyz'},node_diagnostics=dict(fraction_independent_agreement=float(agree.mean()),eligible_nodes=int(valid.sum()),total_nodes=int(valid.size),explicit_disagreements=int(disagree.sum()),knots_direct=int(fgood.sum())),
              pitch_quantiles_xy_um=[np.percentile(CASE['pitch']*np.sqrt(1+g[mask]**2),[5,50,95]).tolist()for g in (hx,hy)],
              seed_native_xyz=[XC,seed,ZC],source_layers=[-8,8],physical_depth_um=[-8*CASE['pitch'],8*CASE['pitch']])
    sample_native(dest,CT,xyz,n,mask,CASE['pitch'],info)
if __name__=='__main__':main(int(sys.argv[1]))