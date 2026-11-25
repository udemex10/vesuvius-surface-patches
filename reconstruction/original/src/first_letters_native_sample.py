"""Sample a physical XYZ writing surface directly from a public uncompressed native CT."""
import os,hashlib,itertools,json,threading
from pathlib import Path
from concurrent.futures import ThreadPoolExecutor
import numpy as np
import requests
from scipy.ndimage import map_coordinates
from first_letters_0125 import ROOT
CACHE=Path(os.environ.get('CROSS_SECTIONS_RAW_CACHE',ROOT/'results/first_letters_deep/source_ranges'))
LOCAL=threading.local()
def sha(a):return hashlib.sha256(a).hexdigest()
def sfile(p):return hashlib.file_digest(p.open('rb'),'sha256').hexdigest()
def sample_native(dest,url,xyz,norm,support,pitch,info):
    dest.mkdir(parents=True,exist_ok=True)
    M=requests.get(url+'/.zarray',timeout=20).json();assert M['chunks']==[128]*3 and M['dtype']=='|u1' and not M['compressor'] and M['order']=='C' and M['dimension_separator']=='/'
    h,w=support.shape
    requests_by={};tiles=[]
    # Exact outer extents per small spatial block. Source lengths are clipped to
    # the actual z range touched inside each block, conserving range transfers.
    for y,x in itertools.product(range(0,h,64),range(0,w,64)):
        sl=np.s_[y:min(h,y+64),x:min(w,x+64)]
        if not support[sl].any():continue
        p=xyz[sl][support[sl]]; n=norm[sl][support[sl]]
        low=np.floor(np.min(p-8*abs(n),axis=0)).astype(int)[::-1]
        high=np.floor(np.max(p+8*abs(n),axis=0)).astype(int)[::-1]+2
        assert np.all(low>=0) and np.all(high<=M['shape']),(low,high)
        keys=list(itertools.product(*(range(a//128,(b-1)//128+1)for a,b in zip(low,high))))
        for key in keys:
            lo,hi=requests_by.get(key,(128,0))
            start=max(0,int(low[0]-128*key[0]));end=min(128,int(high[0]-128*key[0]))
            requests_by[key]=(min(lo,start),max(hi,end))
        tiles.append((sl,low,high,keys))
    print(dest.name,'range count',len(requests_by),'MiB',sum((b-a)*128**2 for a,b in requests_by.values())/2**20,flush=True)
    def fetch(item):
        k,(a,b)=item;link=url+'/'+'/'.join(map(str,k))
        key=sha(link.encode()); d=CACHE/key;d.mkdir(parents=True,exist_ok=True)
        # Reuse a covering range across surfaces of the same scan.
        ranges=[q for q in d.glob('*.bin') if (t:=list(map(int,q.stem.split('_'))))[0]<=a and t[1]>=b]
        if ranges:
            p=ranges[0];offset=int(p.stem.split('_')[0])
            with p.open('rb')as f:f.seek((a-offset)*128**2);raw=f.read((b-a)*128**2)
        else:
            if not hasattr(LOCAL,'session'):LOCAL.session=requests.Session()
            start=a*128**2;end=b*128**2-1
            for i in range(4):
                try:
                    r=LOCAL.session.get(link,headers={'Range':f'bytes={start}-{end}'},timeout=(15,70))
                    if r.status_code in (403,404):return k,(None,None),dict(key=list(k),url=link,missing=True)
                    r.raise_for_status();assert r.status_code==206 and r.headers['Content-Range'].startswith(f'bytes {start}-{end}/')
                    raw=r.content;assert len(raw)==end-start+1;break
                except requests.RequestException:
                    if i==3:raise
            p=d/f'{a}_{b}.bin';p.write_bytes(raw)
        assert len(raw)==(b-a)*128**2
        return k,(np.frombuffer(raw,np.uint8).reshape(-1,128,128),a),dict(key=list(k),url=link,l0_z_range_in_chunk=[a,b],sha256=sha(raw),bytes=len(raw))
    records=[];objects={}
    with ThreadPoolExecutor(12)as ex:
        for key,dat,rec in ex.map(fetch,requests_by.items()):objects[key]=dat;records.append(rec)
    stack=np.zeros((17,h,w),np.uint8)
    source=support.copy();bad=np.zeros_like(source)
    for sl,lo,hi,keys in tiles:
        block=np.zeros(hi-lo,np.uint8);present=np.zeros_like(block,bool)
        for k in keys:
            arr,zbase=objects[k]
            if arr is None:continue
            origin=128*np.array(k)+[zbase,0,0];a=np.maximum(lo,origin);b=np.minimum(hi,origin+arr.shape)
            idx=tuple(map(slice,a-lo,b-lo));block[idx]=arr[tuple(map(slice,a-origin,b-origin))];present[idx]=True
        selected=support[sl];point=xyz[sl][selected][:,::-1];n=norm[sl][selected][:,::-1]
        missing=np.zeros(len(point),bool);zero=missing.copy()
        for i,off in enumerate(range(-8,9)):
            q=point+off*n-lo
            stack[i,sl[0],sl[1]][selected]=map_coordinates(block,q.T,order=1,prefilter=False)
            b0=np.floor(q).astype(int)
            for c in itertools.product((0,1),repeat=3):
                j=tuple((b0+c).T);missing|=~present[j];zero|=block[j]==0
        bad[sl][selected]=zero;source[sl][selected]=~missing
    np.save(dest/'ct17.npy',stack);np.save(dest/'mask.npy',source)
    np.savez_compressed(dest/'support.npz',geometry=support,source_and_geometry=source,all_nonzero=source&~bad)
    with open(dest/'source_ranges.json','w')as f:json.dump(dict(metadata=M,chunks=records),f)
    info.update(shape=list(stack.shape),dtype='uint8',pitch_um=pitch,depth_pitch_um=pitch,native_volume_url=url,center_index=8,
                depth_order='xyz + unit normal times (depth index - 8) in native voxels',support=int(source.sum()),
                all_corners_nonzero=int((source&~bad).sum()),missing_objects=sum(x.get('missing',False)for x in records),
                ct17_sha256=sfile(dest/'ct17.npy'),mask_sha256=sfile(dest/'mask.npy'))
    (dest/'meta.json').write_text(json.dumps(info,indent=2))
    print('READY',dest.name,stack.shape,source.sum(),flush=True)