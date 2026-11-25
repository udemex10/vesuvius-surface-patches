"""View real registered raw CT, the public model input and independently named sheet instances."""
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_erosion
from PIL import Image, ImageDraw, ImageFont
from growbench_labels import OUT, write


def font(sz):
    for name in ("C:/Windows/Fonts/segoeui.ttf","DejaVuSans.ttf"):
        try:return ImageFont.truetype(name,sz)
        except OSError:pass
    return ImageFont.load_default()


def main():
    manual=OUT/"manual_ids";seeds={s["id"]:s for s in json.loads((manual/"seeds.json").read_text())["sites"]}
    refs={r["id"]:r for r in json.loads((manual/"refs.json").read_text())["sites"]}
    cases=["test_03","test_01"]
    size=365;pad=20;gap=16;head=95;rowh=425
    sheet=Image.new("RGB",(2*pad+3*size+2*gap,head+2*rowh+40),"#171b20")
    draw=ImageDraw.Draw(sheet)
    draw.text((pad,12),"How sheet identity is verified",font=font(29),fill="white")
    draw.text((pad,56),"Each row shows one XY section at the same position and scale across all three panels.",font=font(17),fill="#cfd9dc")
    meta=[]
    for i,sid in enumerate(cases):
        s,r=seeds[sid],refs[sid];origin=np.asarray(s["origin_xyz"]);x,y,z=np.rint(np.asarray(s["seed_xyz"])-origin).astype(int)
        ct=np.load(manual/"inputs"/sid/"ct.npy",mmap_mode="r")[z];m7=np.load(manual/"inputs"/sid/"m7.npy",mmap_mode="r")[z]
        labels=np.load(manual/"truth"/sid/"instance_ids.npy",mmap_mode="r")[z]
        target=r["target_id"];oth=np.argwhere((labels>0)&(labels!=target))
        iy,ix=oth[np.argmin(np.linalg.norm(oth-[y,x],axis=1))];adj=int(labels[iy,ix])
        low,high=np.percentile(ct,[1,99]);gray=np.uint8(np.clip((ct.astype(float)-low)/max(high-low,1),0,1)*255)
        overlay=np.repeat(gray[...,None],3,axis=2)
        for num,col in ((target,(64,244,245)),(adj,(252,179,65))):
            mask=labels==num;edge=mask & ~binary_erosion(mask)
            overlay[edge]=col
        top=head+i*rowh
        draw.text((pad,top),f"{sid}   •   raw Z {origin[2]+z}   •   target ID {target}, nearest neighboring ID on this slice {adj}",font=font(18),fill="white")
        headers=["Original CT","Common surface input","Manual sheet outlines on CT"]
        for j,array in enumerate((gray,m7,overlay)):
            x0=pad+j*(size+gap);draw.text((x0,top+32),headers[j],font=font(16),fill="#cfd9dc")
            sheet.paste(Image.fromarray(array).resize((size,size),Image.Resampling.NEAREST),(x0,top+57))
        meta.append(dict(id=sid,plane="XY",global_z=int(origin[2]+z),target_id=target,nearest_other_id_in_section=adj,
                         ct_display_percentiles=[float(low),float(high)]))
    top=head+2*rowh+5
    draw.line((pad,top+11,pad+round(500/7.91*size/256),top+11),fill="white",width=3)
    draw.text((pad+105,top),"0.5 mm",font=font(15),fill="white")
    draw.text((pad+230,top),"Cyan   target sheet",font=font(15),fill="#40f4f5")
    draw.text((pad+428,top),"Amber   nearby sheet",font=font(15),fill="#fcb341")
    sheet.save(OUT/"manual_identity_evidence.jpg",quality=94)
    write(OUT/"manual_identity_evidence.json",dict(cases=meta,source="Two prespecified representative-test cubes chosen for contrasting checked neighbor distances, before using method outcomes. Each image shows a distinct manually harmonized sheet-ID boundary. The panel illustrates identity, not algorithm performance."))


if __name__=="__main__":main()
