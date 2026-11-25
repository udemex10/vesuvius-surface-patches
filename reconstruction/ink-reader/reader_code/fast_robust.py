"""Exact histogram implementation of released robust normalization for uint8 CT.

It implements only the no-mask, 1/99 percentile defaults used in our frozen
inference pipeline. Integer input is mandatory. This is not a new model.
"""
import numpy as np


def weighted_quantile(values, counts, q):
    c=np.cumsum(counts)
    rank=(int(c[-1])-1)*q
    a=int(np.floor(rank));b=int(np.ceil(rank));f=rank-a
    ia=np.searchsorted(c,a,side='right');ib=np.searchsorted(c,b,side='right')
    return float(values[ia]*(1-f)+values[ib]*f)


def normalize_uint8(image):
    a=np.asarray(image)
    if a.dtype==np.uint8:
        u=a
    else:
        u=a.astype(np.uint8)
        if not np.array_equal(a,u):raise ValueError('Fast normalizer requires uint8-valued CT')
    counts=np.bincount(u.ravel(),minlength=256)
    values=np.arange(256,dtype=np.float64)
    lo=weighted_quantile(values,counts,.01);hi=weighted_quantile(values,counts,.99)
    med=float(np.float32(weighted_quantile(values,counts,.5)))
    clipped=np.clip(values.astype(np.float32),lo,hi)
    deviations=np.abs(clipped-np.float32(med))
    order=np.argsort(deviations)
    mad=float(np.float32(weighted_quantile(deviations[order],counts[order],.5)))
    scale=1.4826*mad
    if not np.isfinite(scale) or scale<1e-6:
        std=float(np.std(u.astype(np.float32)))
        scale=next((v for v in [std,abs(hi-lo)/2] if np.isfinite(v) and v>=1e-6),1.)
    out=u.astype(np.float32)
    np.clip(out,lo,hi,out=out)
    out-=med;out/=scale
    return out


if __name__=='__main__':
    import json,time
    from pathlib import Path
    from surface_pretrained import normalize_robust,pool_native,ROOT
    rng=np.random.default_rng(601)
    cases=[]
    for n in [127,128,1000,17*128*128]:
        for kind in ['uniform','normal','sparse','constant']:
            if kind=='uniform':a=rng.integers(0,256,n,dtype=np.uint8)
            elif kind=='normal':a=rng.normal(130,7,n).clip(0,255).astype(np.uint8)
            elif kind=='sparse':a=np.where(rng.random(n)<.001,240,70).astype(np.uint8)
            else:a=np.full(n,70,np.uint8)
            cases.append((kind+'_'+str(n),a))
    v,_=pool_native(np.load(ROOT/'data/positive_w010027.npy'))
    for y in [0,128,256]:
        for x in [0,128,256]:cases.append((f'real_{y}_{x}',v[:17,y:y+128,x:x+128]))
    report=[]
    for name,a in cases:
        t=time.perf_counter();ref=normalize_robust(a.astype(np.float32));rt=time.perf_counter()-t
        t=time.perf_counter();fast=normalize_uint8(a);ft=time.perf_counter()-t
        delta=float(np.max(np.abs(ref-fast)))
        report.append({'name':name,'max_absolute_difference':delta,'reference_seconds':rt,'fast_seconds':ft})
        if delta!=0:raise AssertionError((name,delta))
    out=ROOT/'results/round2_fast_normalization.json'
    out.write_text(json.dumps({'all_bitwise_equal':True,'cases':report},indent=2))
    print(json.dumps({'cases':len(cases),'all_bitwise_equal':True,
        'reference_seconds':sum(x['reference_seconds'] for x in report),
        'fast_seconds':sum(x['fast_seconds'] for x in report)}))
