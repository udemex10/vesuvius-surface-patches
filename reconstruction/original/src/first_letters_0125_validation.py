"""Audit held-out intersections of the PHerc0125 grid and physical CT texture."""
import json
from concurrent.futures import ThreadPoolExecutor
import numpy as np
from scipy.ndimage import map_coordinates,gaussian_filter
from PIL import Image,ImageDraw
from first_letters_0125_surface import edges,samples,XC,ZC,DX,ROOT,F
from first_letters_native_sample import sfile
from first_letters_grid_cases import CASE,SPECIMEN
def main():
    centers=DX[1:-1:8]
    jobs=[(ax,int(base+d))for ax,base in [('xy',ZC),('yz',XC)]for d in centers]
    with ThreadPoolExecutor(max_workers=8) as e:list(e.map(edges,jobs))
    fields=[(SPECIMEN+'_y'+str(s),s) for s in CASE['seeds'] if (F.parent/(SPECIMEN+'_y'+str(s))/'meta.json').is_file()]
    for name,seed in fields:
        dest=F.parent/name;geom=np.load(dest/'geometry_nodes.npz');report={}
        for ax,base in [('xy',ZC),('yz',XC)]:
            rows=[];row_dists=[]
            for d in centers:
                data=samples(ax,int(base+d));q=np.zeros((len(DX),2))
                if ax=='xy':q[:,0]=d;q[:,1]=DX
                else:q[:,0]=DX;q[:,1]=d
                q=(q-DX[0])/16
                known=map_coordinates(geom['fit'],q.T,order=1)
                mask=map_coordinates(geom['qualified'].astype(float),q.T,order=1)>.9999
                for y,ok,(candidates,ids,slopes) in zip(known,mask,data):
                    if not ok or not len(candidates):continue
                    rows.append([float(np.min(np.abs(candidates-y-off))) for off in [0,-5,5,-9,9,-16,16,-23,23]])
            a=np.array(rows)
            report[ax]=dict(heldout_scans=len(centers),intersections=len(a),offsets_in_voxels=[0,-5,5,-9,9,-16,16,-23,23],median_dy=np.median(a,axis=0).tolist(),
                            fraction_intersection_within_2vox=(a<=2).mean(axis=0).tolist(),nominal_dy_q50_q90_q95_vox=np.percentile(a[:,0],[50,90,95]).tolist())
        vol=np.load(dest/'ct17.npy',mmap_mode='r');mask=np.load(dest/'mask.npy')
        report['ct_median_by_signed_normal_plane']=[float(np.median(a[mask]))for a in vol]
        report['ct_p90_by_signed_normal_plane']=[float(np.percentile(a[mask],90))for a in vol]
        (dest/'validation.json').write_text(json.dumps(report,indent=2))
        print(name,[(k,v['intersections'],np.round(v['nominal_dy_q50_q90_q95_vox'],2),np.round(v['fraction_intersection_within_2vox'][:5],2))for k,v in report.items()if isinstance(v,dict)],flush=True)
    # Show the actual central CT layer with two flanking depths under one intensity scale.
    fig=Image.new('RGB',(3*490,2*525),'#222222');d=ImageDraw.Draw(fig)
    for j,(name,_) in enumerate(fields):
        ff=F.parent/name;vol=np.load(ff/'ct17.npy',mmap_mode='r');mask=np.load(ff/'mask.npy');lo,hi=np.percentile(vol[8,mask],[1,99.7])
        for k,offset in enumerate([-4,0,4]):
            im=Image.fromarray(np.where(mask,np.clip((vol[8+offset].astype(float)-lo)*255/(hi-lo),0,255),0).astype('u1')).resize((480,480))
            fig.paste(im,(k*490,j*525+26));d.text((k*490+2,j*525+4),f'{name} | CT normal offset {offset*CASE["pitch"]:+.1f} um',fill='white')
    fig.save(F/'raw_qualification.jpg',quality=93)
if __name__=='__main__':main()