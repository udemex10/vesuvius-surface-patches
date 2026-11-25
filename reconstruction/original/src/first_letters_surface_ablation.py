"""Compare two one-direction surfaces and the cross-checked fit at excluded sections."""
import json
import numpy as np
from scipy.ndimage import map_coordinates
from first_letters_0125_surface import F,ROOT,XC,ZC,DX,samples,path,edges
from first_letters_grid_cases import CASE,SPECIMEN

def run(seed):
    out=F.parent/f'{SPECIMEN}_y{seed}';g=np.load(out/'geometry_nodes.npz');rows=[]
    center_anchors={ax:path(samples(ax,c),seed) for ax,c in [('xy',ZC),('yz',XC)]}
    for axis,base,other in [('xy',ZC,'yz'),('yz',XC,'xy')]:
        for idx,d in enumerate(DX[1:-1:8]):
            edges((axis,int(base+d)));s=samples(axis,int(base+d))
            anchor,flag=center_anchors[other];seed_i=int(np.where(DX==d)[0][0])
            ref,refok=path(s,anchor[seed_i]);refok&=flag[seed_i]
            arr=np.stack([np.full_like(DX,d),DX] if axis=='xy' else [DX,np.full_like(DX,d)])/16-DX[0]/16
            for branch,key,mask in [('axial','axial','axial_observed'),('sagittal','sagittal','sagittal_observed'),('crosschecked','fit','qualified')]:
                y=map_coordinates(g[key],arr,order=1)
                ok=map_coordinates(g[mask].astype(float),arr,order=1)>.9999
                distance=np.array([np.min(abs(pp[0]-v)) if pp[0].size else np.nan for v,pp in zip(y,s)])
                for j in np.flatnonzero(ok):
                    rows.append((axis,int(d),int(j),branch,float(distance[j]),float(abs(y[j]-ref[j])) if refok[j] else np.nan))
    keys=set((a,b,c)for a,b,c,br,e,r in rows if br=='crosschecked')
    sets=[set((a,b,c)for a,b,c,br,e,r in rows if br==k)for k in ('axial','sagittal')]
    common=keys.intersection(*sets)
    metrics={}
    for group in ['own','common']:
        for branch in ['axial','sagittal','crosschecked']:
            subset=[r for r in rows if r[3]==branch and (group=='own' or tuple(r[:3]) in common)]
            v=np.array([r[-2:]for r in subset]);q={}
            for i,k in enumerate(['distance_any_organizer_curve','distance_connected_trace']):
                vals=v[:,i];vals=vals[np.isfinite(vals)]
                q[k]=dict(n=len(vals),median=float(np.median(vals)),p90=float(np.percentile(vals,90)),p95=float(np.percentile(vals,95)),over4=int((vals>4).sum()),over8=int((vals>8).sum()),within2=float((vals<=2).mean()))
            metrics[group+'_'+branch]=q
    (out/'ablation.json').write_text(json.dumps(metrics,indent=2))
    print(seed,[(k,v['distance_any_organizer_curve']['n'],round(v['distance_any_organizer_curve']['p95'],2),round(v['distance_connected_trace']['p90'],2))for k,v in metrics.items()],flush=True)
if __name__=='__main__':
    import sys;run(int(sys.argv[1]))