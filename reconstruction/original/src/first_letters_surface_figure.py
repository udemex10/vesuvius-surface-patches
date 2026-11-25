"""Build a paired original-CT axial and flattened-fibre review figure."""
import hashlib,json
import numpy as np
from PIL import Image,ImageDraw
from scipy.ndimage import map_coordinates
from first_letters_0125 import CT,obj
from first_letters_0125_surface import XC,ZC,F,DX,samples,path
from first_letters_grid_cases import CASE,SPECIMEN
from first_letters_native_sample import sfile
import requests
def raw_xy(z,lo,hi):
    out=np.zeros(hi-lo,np.uint8);ranges=[]
    for y in range(lo[0]//128,(hi[0]-1)//128+1):
        for x in range(lo[1]//128,(hi[1]-1)//128+1):
            k=f'{z//128}/{y}/{x}';start=int(z%128)*128**2
            link=CT+'/'+k
            P=F/'evidence_ranges'/f'{z}_{y}_{x}.dat';P.parent.mkdir(exist_ok=True)
            if not P.is_file():
                r=requests.get(link,headers={'Range':f'bytes={start}-{start+128**2-1}'},timeout=30)
                if r.status_code in (403,404):
                    ranges.append(dict(key=k,missing=True));continue
                r.raise_for_status();assert len(r.content)==128**2 and r.status_code==206;P.write_bytes(r.content)
            raw=P.read_bytes();a=np.maximum(lo,np.array([y,x])*128);b=np.minimum(hi,np.array([y,x])*128+128)
            out[tuple(map(slice,a-lo,b-lo))]=np.frombuffer(raw,np.uint8).reshape(128,128)[tuple(map(slice,a%128,(a%128)+(b-a)))]
            ranges.append(dict(url=link,start=start,count=len(raw),sha256=hashlib.sha256(raw).hexdigest()))
    return out,ranges
def main(seed):
    field=F.parent/f'{SPECIMEN}_y{seed}';g=np.load(field/'geometry_nodes.npz');z=ZC+8
    feat=np.stack([np.full_like(DX,8),DX])/16-DX[0]/16
    fit=map_coordinates(g['fit'],feat,order=1);valid=map_coordinates(g['qualified'].astype(float),feat,order=1)>.999
    a=np.flatnonzero(valid&(abs(DX)<=392))
    lo=np.array([int(np.floor(fit[a].min()))-38,XC-400]);hi=np.array([int(np.ceil(fit[a].max()))+39,XC+401]);raw,digests=raw_xy(z,lo,hi)
    (field/'section_sources.json').write_text(json.dumps(digests,indent=1))
    rlo,rhi=np.percentile(raw[raw>0],[4,98.5])
    gray=Image.fromarray(np.uint8(np.clip((raw.astype(float)-rlo)/(rhi-rlo)*255,0,255))).convert('RGB')
    overlay=gray.copy();d=ImageDraw.Draw(overlay)
    for i in a:
        if i+1<len(DX) and valid[i+1] and DX[i+1]<=392:
            d.line([(int(XC+DX[j]-lo[1]),float(fit[j]-lo[0]))for j in (i,i+1)],fill=(255,209,36),width=1)
    d.line([(15,gray.height-10),(15+round(1000/CASE['pitch']),gray.height-10)],fill='white',width=3)
    dest=field/'intrinsic';vol=np.load(dest/'ct17.npy',mmap_mode='r');ok=np.load(dest/'mask.npy')
    rr=np.percentile(vol[8][ok],[1,99]);board=Image.new('RGB',(1130,590),'#222');drawer=ImageDraw.Draw(board)
    def add(im,box):
        im.thumbnail((box[2],box[3]),Image.Resampling.LANCZOS);board.paste(im,box[:2])
    drawer.text((10,6),f'{SPECIMEN} | native XYZ seed ({XC}, {seed}, {ZC}) | organizer M7 surface curves',fill='white')
    drawer.text((10,34),f'Original CT | held-out axial z {z}',fill='white');add(gray.copy(),(10,57,545,205))
    drawer.text((10,290),'Same section | fitted mesh intersection in yellow | white bar = 1 mm',fill='white');add(overlay,(10,314,545,205))
    drawer.text((575,34),f'Intrinsic mesh | directly sampled center-plane CT | {CASE["pitch"]:.3f} um pixels',fill='white')
    img=np.uint8(np.where(ok,np.clip((vol[8].astype(float)-rr[0])/(rr[1]-rr[0])*255,0,255),0));add(Image.fromarray(img),(575,57,545,510))
    board.save(field/'evidence.png')
if __name__=='__main__':
    import sys;main(int(sys.argv[1]))