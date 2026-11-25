"""Intrinsic as-rigid-as-possible flattening and regular tifxyz export."""
import json,sys
from pathlib import Path
import numpy as np
from scipy import sparse
from scipy.sparse.linalg import factorized
from scipy.ndimage import map_coordinates
import tifffile
from first_letters_0125_surface import F,ROOT
from first_letters_grid_cases import CASE,SPECIMEN
from first_letters_native_sample import sfile

def intrinsic(p,uv,faces):
    tri=p[faces];old=uv[faces]
    e1=tri[:,1]-tri[:,0];e2=tri[:,2]-tri[:,0]
    l1=np.linalg.norm(e1,axis=1)
    parallel=np.sum(e1*e2,axis=1)/l1;perp=np.sqrt(np.maximum(0,np.sum(e2**2,axis=1)-parallel**2))
    tangent=old[:,1]-old[:,0];tangent/=np.linalg.norm(tangent,axis=1)[:,None]
    normal=np.column_stack([-tangent[:,1],tangent[:,0]])
    sign=np.sign(np.sum(normal*(old[:,2]-old[:,0]),axis=1))
    return np.stack([np.zeros_like(tangent),l1[:,None]*tangent,parallel[:,None]*tangent+(sign*perp)[:,None]*normal],axis=1)

def distort(p,uv,faces):
    world=p[faces];flat=uv[faces]
    A=world[:,1:]-world[:,:1];B=flat[:,1:]-flat[:,:1]
    ar=B[:,0,0]*B[:,1,1]-B[:,0,1]*B[:,1,0]
    svi=np.linalg.svd(np.matmul(np.linalg.inv(B),A),compute_uv=False)
    log=np.max(abs(np.log(svi)),axis=1)
    return dict(triangles=len(faces),flips=int((ar<=0).sum()),p50_p90_p95_max_stretch=np.percentile(np.exp(log),[50,90,95,100]).tolist(),
                rms_log_distortion=float(np.sqrt(np.mean(log**2))),fraction_within_5_percent=float((np.exp(log)<1.05).mean()),
                fraction_within_10_percent=float((np.exp(log)<1.10).mean()))
def flatten(p,uv,faces,iters=35):
    intrinsic_tris=intrinsic(p,uv,faces)
    steps=np.roll(intrinsic_tris,-1,axis=1)-intrinsic_tris
    # Each physical triangle contributes all three edges. Weak similarity to
    # original positions removes the null motion of disconnected components.
    i=faces.ravel();j=np.roll(faces,-1,axis=1).ravel();n=len(uv)
    e=sparse.coo_matrix((np.r_[-np.ones(len(i)),np.ones(len(i))],
                             (np.tile(np.arange(len(i)),2),np.r_[i,j])),shape=(len(i),n)).tocsr()
    weight=1e-5;solve=factorized((e.T@e+weight*sparse.eye(n)).tocsc())
    out=uv.copy()
    for _ in range(iters):
        now=np.roll(out[faces],-1,axis=1)-out[faces]
        cc=np.einsum('ijk,ijk->i',steps,now)
        ss=np.sum(steps[:,:,0]*now[:,:,1]-steps[:,:,1]*now[:,:,0],axis=1)
        scale=np.hypot(cc,ss);cs=cc/scale;sn=ss/scale
        rotated=np.stack([steps[:,:,0]*cs[:,None]-steps[:,:,1]*sn[:,None],steps[:,:,0]*sn[:,None]+steps[:,:,1]*cs[:,None]],axis=2)
        rhs=e.T@rotated.reshape(-1,2)+weight*uv
        out=solve(rhs)
    return out

def raster(nodes,faces,uv,step=8):
    origin=np.floor(uv.min(0)/step)*step;ext=np.ceil((uv.max(0)-origin)/step).astype(int)+1
    shape=tuple(ext[::-1]);sumxyz=np.zeros((*shape,3));count=np.zeros(shape,int);clash=np.zeros(shape,bool)
    ux= (uv-origin)/step
    for pos,world in zip(ux[faces],nodes[faces]):
        lo=np.maximum(0,np.ceil(pos.min(0)-1e-10).astype(int));hi=np.minimum(ext,np.floor(pos.max(0)+1e-10).astype(int)+1)
        if np.any(hi<=lo):continue
        gy,gx=np.mgrid[lo[1]:hi[1],lo[0]:hi[0]]
        grid=np.stack([gx,gy],axis=2)
        bc=(grid-pos[0])@np.linalg.inv(pos[1:]-pos[0]);w=np.dstack([1-bc.sum(2),bc])
        inside=w.min(2)>=-1e-7
        ys,xs=gy[inside],gx[inside]
        xyz=w[inside]@world
        previous=count[ys,xs]>0
        if previous.any():
            yp,xp=ys[previous],xs[previous]
            different=np.linalg.norm(sumxyz[yp,xp]/count[yp,xp,None]-xyz[previous],axis=1)>.1
            clash[yp[different],xp[different]]=True
        count[ys,xs]+=1;sumxyz[ys,xs]+=xyz
    valid=(count>0)&~clash
    rr=sumxyz/np.maximum(1,count)[...,None]
    return rr,valid,origin,clash

def main(seed):
    source=F.parent/f'{SPECIMEN}_y{seed}'
    g=np.load(source/'geometry_nodes.npz')
    x,z=np.meshgrid(g['x'],g['z']);xyz=np.stack([x,g['fit'],z],axis=2)
    uv=np.stack([x-x[0,0],z-z[0,0]],axis=2).reshape(-1,2).astype(float)
    ok=g['qualified'];quad=ok[:-1,:-1]&ok[:-1,1:]&ok[1:,:-1]&ok[1:,1:]
    vv,uu=np.where(quad);q=vv*x.shape[1]+uu
    faces=np.vstack([np.stack([q,q+1,q+x.shape[1]+1],axis=1),np.stack([q,q+x.shape[1]+1,q+x.shape[1]],axis=1)])
    i=np.unique(faces);table=np.empty(x.size,int);table[i]=np.arange(len(i))
    faces=table[faces];uv=uv[i];p=xyz.reshape(-1,3)[i]
    flat=flatten(p,uv,faces);before=distort(p,uv,faces);after=distort(p,flat,faces)
    assert before['flips']==0 and after['flips']==0
    rr,valid,origin,clash=raster(p,faces,flat,8)
    # Ensure a regular output quad cannot bridge a hole in the physical surface:
    # cross-test every exported cell at its midpoint in original chart space.
    orig=(rr[:,:,[2,0]]-np.array([g['z'][0],g['x'][0]]))/16
    # Mark a node unusable if its immediate output neighbors lie far away in
    # source space or its cell midpoint lacks a complete source geometry cell.
    cells=valid[:-1,:-1]&valid[:-1,1:]&valid[1:,:-1]&valid[1:,1:]
    mid=(orig[:-1,:-1]+orig[1:,1:]+orig[:-1,1:]+orig[1:,:-1])/4
    goodmid=map_coordinates(quad.astype(float),(mid-.5).transpose(2,0,1),order=0,mode='constant')>.5
    bad=cells&~goodmid
    for k in [(0,0),(0,1),(1,0),(1,1)]:
        dy,dx=k;valid[dy:dy+bad.shape[0],dx:dx+bad.shape[1]]&=~bad
    out=source/'intrinsic';mesh=out/(source.name+'.tifxyz');mesh.mkdir(parents=True,exist_ok=True)
    for j,k in enumerate('xyz'):tifffile.imwrite(mesh/(k+'.tif'),np.where(valid,rr[:,:,j],-1).astype('f4'))
    shape=valid.shape;bounds=[rr[valid].min(0).tolist(),rr[valid].max(0).tolist()]
    (mesh/'meta.json').write_text(json.dumps(dict(format='tifxyz',type='seg',uuid=source.name,name=source.name,scale=[.125,.125],bbox=bounds)))
    # Distortion of regularized tifxyz after inverting and resampling.
    qq=valid[:-1,:-1]&valid[1:,1:]&valid[:-1,1:]&valid[1:,:-1];iy,ix=np.where(qq);j=iy*shape[1]+ix
    ff=np.vstack([np.stack([j,j+1,j+shape[1]+1],axis=1),np.stack([j,j+shape[1]+1,j+shape[1]],axis=1)])
    v2,u2=np.mgrid[:shape[0],:shape[1]]
    d2=distort(rr.reshape(-1,3),np.stack([u2*8,v2*8],axis=2).reshape(-1,2),ff)
    meta=dict(source_grid_step=16,output_grid_step=8,fit_iterations=35,source_projection=before,intrinsic_triangle_solution=after,exported_mesh=d2,
              source_valid_quads=int(quad.sum()),nodes_exported=int(valid.sum()),regular_valid_quads=int(qq.sum()),rejected_overlaps=int(clash.sum()),rejected_hole_bridges=int(bad.sum()),
              original_unsigned_surface_mm2=float(np.sum(np.linalg.norm(np.cross(p[faces][:,1]-p[faces][:,0],p[faces][:,2]-p[faces][:,0]),axis=1))*.5*(CASE['pitch']*.001)**2),
              mesh_hashes={k:sfile(mesh/(k+'.tif'))for k in 'xyz'})
    np.savez_compressed(out/'embedding.npz',xyz=p,source_uv=uv,flat_uv=flat,faces=faces,regular_origin=origin,valid=valid)
    (out/'distortion.json').write_text(json.dumps(meta,indent=2));print(source.name,before['p50_p90_p95_max_stretch'],d2['p50_p90_p95_max_stretch'],flush=True)
if __name__=='__main__':main(int(sys.argv[1]))