"""Reconstruct independent transverse traces of organizer surface predictions on PHerc0125."""
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import json
import numpy as np
from scipy.ndimage import gaussian_filter,distance_transform_edt,map_coordinates
from first_letters_0125 import F,ROOT,getgrid
from first_letters_grid_cases import CASE,SPECIMEN
XC=CASE['xc']
ZC=CASE['zc']
RAD=640
STEP=8
DX=np.arange(-RAD,RAD+1,STEP,dtype=int);N=len(DX);MID=N//2
SEEDS=CASE['seeds']
def edges(arg):
    axis,v=arg
    out=F/('edges_'+axis)/f'{v}.npz';out.parent.mkdir(exist_ok=True)
    if out.exists():return
    axisindex=0 if axis=='xy' else 1
    middle=XC if axis=='xy' else ZC
    paths=getgrid(axis,v)
    part=[]
    for i,p in enumerate(paths):
        local=p[:,[axisindex,1-axisindex]]
        if local.min(0)[1]>CASE['band'][1] or local.max(0)[1]<CASE['band'][0] or local[:,0].max()<middle-RAD or local[:,0].min()>middle+RAD:continue
        a,b=local[:-1],local[1:];d=b-a
        keep=(abs(d[:,0])>0)&(abs(d[:,1])<=1.35*abs(d[:,0]))&(np.maximum(a[:,0],b[:,0])>=middle-RAD)&(np.minimum(a[:,0],b[:,0])<=middle+RAD)
        if keep.any():part.append(np.column_stack([a[keep],b[keep],np.full(keep.sum(),i)]))
    s=np.vstack(part).astype('f4');np.savez_compressed(out,edges=s)
def samples(axis,v):
    s=np.load(F/('edges_'+axis)/f'{v}.npz')['edges']
    a,b,c,d,ids=s.T;lo=np.minimum(a,c);hi=np.maximum(a,c)
    spots=[]
    for u in (XC if axis=='xy' else ZC)+DX:
        q=(lo<=u)&(hi>u)
        slope=(d[q]-b[q])/(c[q]-a[q]);y=b[q]+(u-a[q])*slope
        spots.append((y,ids[q].astype(int),slope))
    return spots
# The small candidate beam stays on the same physical connected polyline where
# possible. An explicit unsupported state allows a small local break to be
# bridged without silently declaring that interval measured.
def path(info,seed):
    y0,c0,t0=info[MID];i=np.argmin(abs(y0-seed));offset=abs(y0[i]-seed)
    outcome=np.empty(N);good=np.zeros(N,bool)
    outcome[MID]=y0[i];good[MID]=offset<=6.
    for direct in (-1,1):
        paths=[(0.,float(y0[i]),float(t0[i]),c0[i],0,[float(y0[i])],[True])]
        for n in range(1,MID+1):
            k=MID+n*direct;y,c,t=info[k]
            opts={}
            for old,yold,told,cold,gap,h,hg in paths:
                yp=yold+direct*STEP*told
                ids=np.flatnonzero(abs(y-yp)<=7.)
                for idx in ids:
                    err=abs(y[idx]-yp)
                    quality=1.-.17*err-.35*min(2,abs(t[idx]-told))+(1.45 if c[idx]==cold else 0.)
                    value=old+quality
                    if idx not in opts or value>opts[idx][0]:
                        opts[idx]=(value,float(y[idx]),.6*float(t[idx])+.4*((float(y[idx])-yold)/(STEP*direct)),c[idx],0,h+[float(y[idx])],hg+[True])
                if gap<6:
                    key=('gap',int(round(yp)),int(cold))
                    val=old-.65-.1*gap
                    if key not in opts or val>opts[key][0]:
                        opts[key]=(val,yp,.8*told,cold,gap+1,h+[float(yp)],hg+[False])
            if not opts:
                hp=paths[0];delta=MID+1-len(hp[5])
                paths=[(*hp[:5],hp[5]+[hp[5][-1]]*delta,hp[6]+[False]*delta)]
                break
            paths=sorted(opts.values(),key=lambda q:-q[0])[:6]
        best=paths[0]
        ids=MID+np.arange(1,MID+1)*direct
        outcome[ids]=best[5][1:];good[ids]=np.asarray(best[6][1:])&(offset<=6.)
    return outcome,good
def run():
    # Interleaved 16-voxel native plane spacing, with validation delegated to
    # independent sagittal contours and omitted midplane intersections.
    offsets=DX[::2]
    jobs=[(axis,int(c+i))for axis,c in [('xy',ZC),('yz',XC)]for i in offsets]
    with ThreadPoolExecutor(max_workers=8)as ex:list(ex.map(edges,jobs))
    sX=samples('xy',ZC);sZ=samples('yz',XC)
    anchor={a:(path(sX,a),path(sZ,a)) for a in SEEDS}
    surfaces={v:[[],[],[],[]]for v in SEEDS}
    for i,d in enumerate(offsets):
        xx=samples('xy',ZC+d);zz=samples('yz',XC+d)
        for v,pairs in anchor.items():
            t1,g1=path(xx,pairs[1][0][2*i]);t2,g2=path(zz,pairs[0][0][2*i])
            for target,arr in zip(surfaces[v],[t1,g1 & pairs[1][1][2*i],t2,g2 & pairs[0][1][2*i]]):
                target.append(arr)
    for v,vals in surfaces.items():
        a,g1,b,g2=map(np.asarray,vals);g2=g2.T;b=b.T
        np.savez_compressed(F/f'solution_{int(v)}.npz',axial=a,axial_valid=g1,sagittal=b,sagittal_valid=g2,axis=DX)
        diff=abs(a[:,::2]-b[::2]);common=g1[:,::2]&g2[::2]
        print(int(v),round(common.mean(),2),np.round(np.percentile(diff[common],[50,75,90,95]),1),round(np.mean(common&(diff<=4)),2),flush=True)
if __name__=='__main__':run()