"""Render the regular intrinsic tifxyz mesh without applying a display warp."""
import sys,json
import numpy as np,tifffile
from first_letters_0125 import CT
from first_letters_0125_surface import F
from first_letters_grid_cases import CASE,SPECIMEN
from first_letters_native_sample import sample_native
def main(seed):
    base=F.parent/f'{SPECIMEN}_y{seed}';dest=base/'intrinsic';mesh=dest/(base.name+'.tifxyz')
    g=np.stack([tifffile.imread(mesh/(k+'.tif'))for k in 'xyz'],axis=2)
    h,w=g.shape[:2];rr,cc=np.mgrid[:(h-1)*8,:(w-1)*8]
    i,j=rr//8,cc//8;v=(rr%8)[...,None]/8;u=(cc%8)[...,None]/8
    a,b,c,d=[g[i+dy,j+dx]for dy,dx in ((0,0),(0,1),(1,0),(1,1))]
    valid=(np.minimum.reduce([a,b,c,d])>0).all(2)
    p=(1-v)*((1-u)*a+u*b)+v*((1-u)*c+u*d)
    du=(1-v)*(b-a)+v*(d-c);dv=(1-u)*(c-a)+u*(d-b)
    normal=np.cross(du,dv);nlen=np.linalg.norm(normal,axis=2);valid&=nlen>.5
    normal/=np.maximum(nlen[...,None],1e-8)
    info=dict(specimen=SPECIMEN,field=base.name,mesh=mesh.name,native_volume_id=CASE['id'],
              normal='cross(increasing regular intrinsic x,increasing regular intrinsic y), approximately -native Y',
              geometry='strict four valid tifxyz corners, eight native pixels per output mesh knot',mesh_info='distortion.json')
    sample_native(dest,CT,p,normal,valid,CASE['pitch'],info)
if __name__=='__main__':main(int(sys.argv[1]))