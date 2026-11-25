"""Bounded, cached Zarr v2 HTTP reads with explicit missing-chunk policy/provenance."""
import argparse
import concurrent.futures
import hashlib
import itertools
import json
from pathlib import Path
import urllib.request
import urllib.error

import os

import numpy as np
import numcodecs

def get_bytes(url):
    with urllib.request.urlopen(url, timeout=90) as r:
        return r.read()

def read_crop(url, bounds, cache, max_bytes=512*1024**2, missing_fill=False):
    # missing_fill=True is only for sparse *prediction* arrays, where Zarr defines an absent chunk as
    # fill_value; raw CT keeps the default and raises on any missing chunk.
    url=url.rstrip('/')
    meta=json.loads(get_bytes(url+'/.zarray'))
    if meta.get('zarr_format')!=2 or len(meta['shape'])!=3:
        raise ValueError('Expected 3D Zarr v2 array')
    start=np.array(bounds[:3],dtype=int); end=np.array(bounds[3:],dtype=int)
    shape=np.array(meta['shape']); chunk=np.array(meta['chunks'])
    if np.any(start<0) or np.any(end>shape) or np.any(end<=start):
        raise ValueError(f'Bounds {bounds} outside {shape.tolist()}')
    dtype=np.dtype(meta['dtype']); coords=list(itertools.product(*[range(a//c,(b-1)//c+1) for a,b,c in zip(start,end,chunk)]))
    raw_bytes=len(coords)*int(np.prod(chunk))*dtype.itemsize
    if raw_bytes>max_bytes:
        raise ValueError(f'Chunk read budget exceeded: {raw_bytes} > {max_bytes}')
    dest=Path(cache); dest.mkdir(parents=True,exist_ok=True)
    metadata=[]
    out=np.zeros(tuple(end-start),dtype=dtype)
    def fetch(coord):
        name=meta.get('dimension_separator','.').join(map(str,coord))
        address=url+'/'+name
        key=hashlib.sha256(address.encode()).hexdigest()
        path=dest/key
        cached=path.exists()
        if not cached:
            # Missing chunks raise: never silently interpret HTTP errors as blank papyrus.
            try:
                data=get_bytes(address)
            except urllib.error.HTTPError as e:
                if not (missing_fill and e.code in (403,404)):
                    raise
                fill=np.full(tuple(chunk),meta.get('fill_value') or 0,dtype=dtype)
                return coord,fill,{'coordinate':list(coord),'url':address,'bytes':0,'sha256':None,'cache_hit':False,'missing_filled':True}
            path.write_bytes(data)
        data=path.read_bytes()
        record={'coordinate':list(coord),'url':address,'bytes':len(data),'sha256':hashlib.sha256(data).hexdigest(),'cache_hit':cached}
        decoded=data
        if meta.get('compressor'):
            decoded=numcodecs.get_codec(meta['compressor']).decode(decoded)
        for filt in reversed(meta.get('filters') or []):
            decoded=numcodecs.get_codec(filt).decode(decoded)
        a=np.frombuffer(decoded,dtype=dtype).reshape(tuple(chunk),order=meta.get('order','C'))
        return coord,a,record
    with concurrent.futures.ThreadPoolExecutor(max_workers=int(os.environ.get('VC_FETCH_WORKERS','6'))) as pool:
        for coord,a,record in pool.map(fetch,coords):
            origin=np.array(coord)*chunk; lo=np.maximum(start,origin); hi=np.minimum(end,origin+chunk)
            out[tuple(slice(int(x),int(y)) for x,y in zip(lo-start,hi-start))]=a[tuple(slice(int(x),int(y)) for x,y in zip(lo-origin,hi-origin))]
            metadata.append(record)
    return out,{'url':url,'bounds_zyx':list(map(int,bounds)),'array_metadata':meta,'uncompressed_chunk_bytes':raw_bytes,'downloaded_bytes':sum(r['bytes'] for r in metadata if not r['cache_hit']),'chunks':metadata,'array_sha256':hashlib.sha256(out.tobytes()).hexdigest()}

if __name__=='__main__':
    p=argparse.ArgumentParser();p.add_argument('--url',required=True);p.add_argument('--bounds',nargs=6,type=int,required=True);p.add_argument('--out',required=True);p.add_argument('--cache',default='data/cache');p.add_argument('--max-mb',type=int,default=512)
    a=p.parse_args();volume,manifest=read_crop(a.url,a.bounds,a.cache,a.max_mb*1024**2)
    out=Path(a.out);out.parent.mkdir(parents=True,exist_ok=True);np.save(out,volume)
    out.with_suffix('.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(json.dumps({'shape':volume.shape,'min':int(volume.min()),'max':int(volume.max()),'downloaded_bytes':manifest['downloaded_bytes'],'out':str(out)}))
