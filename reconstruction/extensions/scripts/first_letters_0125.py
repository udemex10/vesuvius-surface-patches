"""Use organizer-provided axial and sagittal connected polylines to reconstruct a local PHerc0125 surface."""
import hashlib,json,struct,threading
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import numpy as np
import requests
import tifffile
from scipy.ndimage import gaussian_filter,map_coordinates
ROOT=Path(__file__).resolve().parents[1]
from first_letters_grid_cases import CASE,SPECIMEN,SCAN_CASE
BASE='https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/'+SPECIMEN
GRID=BASE+'/representations/predictions/surfaces/'+CASE['id']+'-surface-20260413222639-surface-m7-L0-th0.2.normal-grids'
CT=BASE+'/volumes/'+CASE['scan']+'.zarr/0'
F=ROOT/'results/first_letters_deep'/(SCAN_CASE+'_survey')
F.mkdir(parents=True,exist_ok=True)
LOC=threading.local()
def obj(url,dest):
    if not dest.is_file():
        if not hasattr(LOC,'s'):LOC.s=requests.Session()
        for i in range(3):
            try:
                r=LOC.s.get(url,timeout=(10,45));r.raise_for_status()
                dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes(r.content);break
            except requests.RequestException:
                if i==2:raise
    return dest.read_bytes()
def getgrid(axis,val):
    rel=axis+f'/{val:06d}.grid';s=obj(GRID+'/'+rel,F/rel)
    if not s:return []
    h=struct.unpack_from('>13I',s)
    assert h[:2]==(0x56434753,3)
    off=h[10];paths=[]
    for i in range(h[8]):
        x,y,n=struct.unpack_from('>3I',s,off);p=np.frombuffer(s,np.int8,count=n,offset=off+12).reshape(-1,2)
        if x>2**31:x-=2**32
        if y>2**31:y-=2**32
        paths.append(np.vstack(([x,y],[x,y]+np.cumsum(p,axis=0,dtype=int))))
        off+=12+n
    assert off==h[11]
    return paths
if __name__=='__main__':
    for axis,v in [('xy',10467),('yz',3400),('xy',10600)]:
        lines=getgrid(axis,v);print(axis,v,len(lines),'length top',sorted([len(p) for p in lines],reverse=True)[:12]);print([(p.min(0).tolist(),p.max(0).tolist()) for p in sorted(lines,key=len,reverse=True)[:4]])