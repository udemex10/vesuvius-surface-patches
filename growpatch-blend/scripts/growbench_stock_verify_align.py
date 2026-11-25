"""Exercise the public tifxyz alignment CLI on all eight real, held-out stock meshes."""
import json
from pathlib import Path
import subprocess
import sys
from tempfile import TemporaryDirectory

import numpy as np
import tifffile
from growbench_compare import pair_summary
from growbench_geometry import faces, project, score
from growbench_stock import ROOT, digest, save


DATA=ROOT/"results/growbench"


def evaluate():
    manifest=DATA/"labels/seeds.json"
    report=DATA/"comparison/phase_test.json"
    cli=ROOT/"scripts/growbench_seed_align.py"
    expected=json.loads(report.read_text())
    sites=[s for s in json.loads(manifest.read_text())["sites"] if s["split"]=="test"]
    before={t:{} for t in (1,2,4)}
    after={t:{} for t in (1,2,4)}
    records=[]
    with TemporaryDirectory(prefix="grow_align_") as staging:
        for site in sites:
            name=site["id"]
            source=DATA/"stock/oriented8"/name/"surface.npz"
            raw=Path(staging)/(name+"_input");raw.mkdir()
            result=Path(staging)/(name+"_aligned")
            with np.load(source) as data:
                xyz,valid=data["xyz"],data["valid"]
            for j,k in enumerate("xyz"):
                tifffile.imwrite(raw/(k+".tif"),np.where(valid,xyz[...,j],-1).astype("f4"))
            metadata=json.loads((source.parent/"surface/meta.json").read_text())
            metadata["bbox"]=[xyz[valid].min(0).tolist(),xyz[valid].max(0).tolist()]
            save(raw/"meta.json",metadata)
            subprocess.run([sys.executable,str(cli),str(raw),str(result),"--site",str(manifest),
                            "--id",name],check=True,stdout=subprocess.DEVNULL)
            moved=np.stack([tifffile.imread(result/(k+".tif")) for k in "xyz"],axis=-1)
            preserved=np.array_equal((moved>=0).all(-1),valid)
            info=json.loads((result/"seed_alignment.json").read_text())
            ref=project(np.load(DATA/"labels/truth"/name/"triangles.npy"),site)
            p0=project(faces(xyz,valid)[0],site)
            p1=project(faces(moved,valid)[0],site)
            deviations={}
            for t in before:
                before[t][name]=score(ref,p0,0,t)
                after[t][name]=score(ref,p1,0,t)
                deviations[t]={k:v-expected["by_tolerance"][str(t)]["adjusted_stock"][name][k] for k,v in after[t][name].items()}
            records.append(dict(id=name,input_npz_sha256=digest(source),seed_rays=info["seed_rays"],
                shift_vox=info["shift_along_seed_normal_voxels"],projection_shift_vox=-expected["offsets"][name],
                mask_preserved=bool(preserved),cli_minus_projection=deviations,
                aligned_tiff_sha256={k:digest(result/(k+".tif")) for k in "xyz"}))
    comparisons={}
    for t in after:
        pair=pair_summary(before[t],after[t])
        original=expected["by_tolerance"][str(t)]["adjusted_vs_stock"]
        comparisons[t]={"cli":pair,"projection_only":original,"cli_minus_projection":{
            k:pair[k]-original[k] for k in ("connected_delta_cm2","connected_gain_percent","risk_difference_points")}}
    stages=dict(seeds_sha256=digest(manifest),reference_projection_report_sha256=digest(report),
          cli_sha256=digest(cli),file_qualification="Converted valid global XYZ from real, frozen stock oriented8 NPZ to float32 tifxyz, preserving its official/in-cube mask. The CLI reads and writes native tifxyz. Re-rasterized output geometry was compared with the earlier ideal continuous seed-normal translation.",
          cases=records,paired_by_tolerance=comparisons)
    save(DATA/"stock/seed_alignment_cli_verification.json",stages)
    assert all(r["mask_preserved"] and abs(r["shift_vox"]-r["projection_shift_vox"])<1e-10 for r in records)
    for t in comparisons:
        print(t,comparisons[t]["cli_minus_projection"],flush=True)


if __name__=="__main__":
    evaluate()
