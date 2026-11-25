import numpy as np
import torch
from fast_robust import normalize_uint8
from types import SimpleNamespace

def hann96():
    w = 0.5 - 0.5 * np.cos(2 * np.pi * (np.arange(96) + 0.5) / 96)
    return (w[:, None] * w[None, :]).astype(np.float32)

studio=SimpleNamespace(hann96=hann96)

def supported_starts(mask):
    h,w=mask.shape
    if min(h,w)<128:
        return []
    sam=lambda n: sorted(set(range(0,n-127,64))|{n-128})
    return [(y,x) for y in sam(h) for x in sam(w) if mask[y:y+128,x:x+128].all()]

def score(models,volume,starts,mirror):
    shape=volume.shape[1:]
    scores=np.zeros((2,*shape),np.float32)
    counts=np.zeros(shape,np.float32)
    k=studio.hann96()
    with torch.inference_mode():
        for first in range(0,len(starts),4):
            batch=starts[first:first+4]
            raw=np.stack([normalize_uint8(volume[:,y:y+128,x:x+128]) for y,x in batch])
            tensor=torch.from_numpy(raw).cuda()[:,None]
            for sign in range(2):
                t=tensor.flip((2,)) if sign else tensor
                accum=0
                transforms=[(),(3,),(4,),(3,4)] if mirror else [()]
                for model in models:
                    for axes in transforms:
                        with torch.autocast('cuda',dtype=torch.float16):
                            logits=model(t.flip(axes) if axes else t)
                        im=logits.float().sigmoid()
                        # Model removes the depth axis: input H/W are dimensions 3/4, output 2/3.
                        accum=accum+(im.flip(tuple(a-1 for a in axes)) if axes else im)
                maps=(accum[:,0,16:112,16:112]/(len(models)*len(transforms))).cpu().numpy()
                if not np.isfinite(maps).all():
                    raise ValueError('Nonfinite transfer output')
                for (y,x),a in zip(batch,maps):
                    section=np.s_[y+16:y+112,x+16:x+112]
                    scores[sign][section]+=a*k
                    if sign==0:
                        counts[section]+=k
    return np.divide(scores,counts,out=np.full_like(scores,np.nan),where=counts>0)
