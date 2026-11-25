"""Assemble a portable review copy of new PHerc. 125 surface extensions."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import zipfile

import numpy as np
import tifffile
from growbench_recover_unread_figure import draw

from growbench_recover_verify_unread import sha,verify
from growbench_recover import ROOT


DEST=ROOT/"outputs/growbench_pherc125_extensions"
SOURCE=ROOT/"results/growbench/recover"
START=Path("results/first_letters_deep/PHerc0125_y3310")
SCRIPTS=("growbench_recover","growbench_recover_phase","growbench_recover_unread","growbench_recover_unread_components","growbench_recover_unread_polylines",
         "growbench_recover_unread_export","growbench_recover_unread_directions","r3_chunkvol","zarr_crop",
         "first_letters_intrinsic","first_letters_native_sample","first_letters_0125",
         "first_letters_0125_surface","first_letters_grid_cases","growbench_stock","growbench_stock_unread","growbench_geometry","growbench_recover_unread_figure")


def assemble_case(direction,rank=1):
    source=SOURCE/("pherc125_"+("extension"if direction=="east"else direction))
    suffix="" if rank==1 else "_"+str(rank)
    flat=source/("flat"+suffix);dest=DEST/"cases"/(direction+suffix);dest.mkdir(parents=True,exist_ok=True)
    folder=next(flat.glob("*.tifxyz"))
    shutil.copytree(folder,dest/folder.name,dirs_exist_ok=True)
    for name in ("meta.json","support.npz","native_evidence.jpg","native_transects.json","source_ranges.json","distortion.json"):
        shutil.copy2(flat/name,dest/name)
    np.savez_compressed(dest/"native_ct.npz",ct17=np.load(flat/"ct17.npy"),mask=np.load(flat/"mask.npy"))
    if rank!=1:return
    shutil.copy2(source/"extension.json",dest/"route_summary.json")
    shutil.copy2(source/"consensus.npz",dest/"route_agreement.npz")
    shutil.copy2(source/"organizer_continuity.json",dest/"organizer_continuity.json")
    tracks={};cues=[]
    for path in sorted(source.glob("route_*/track.json")):
        track=json.loads(path.read_text())
        tracks[path.parent.name]=track
        for i,step in enumerate(track["steps"]):
            if "valid_nodes" not in step:continue
            raw=json.loads((path.parent/f"{i:02d}"/"source.json").read_text())
            cues.append(dict(route=path.parent.name,tile=i,**raw))
    (dest/"route_provenance.json").write_text(json.dumps(dict(tracks=tracks,accepted_tiles=cues),separators=(",",":")))


def docs(report):
    lines=["# PHerc. 125 interior surface extensions","",
        "This addendum contains disjoint extensions of the earlier PHerc. 125 Y3310 regional reconstruction in four outward directions. Each case includes an original-coordinate TIFXYZ mesh, a 17-plane native CT stack and its support mask, a source manifest, and a grayscale comparison plate. Numbered suffixes denote separated components in the same direction.","",
        "| Direction | Native mesh area (cm²) | Largest connected part (cm²) | Native CT pixels | 95th percentile stretch |",
        "|---|---:|---:|---:|---:|"]
    for case,row in report["cases"].items():
        lines.append(f"| {case} | {row['triangulated_cm2']:.6f} | {row['largest_connected_triangulated_cm2']:.6f} | {row['source_supported_pixels']:,} | {(row['distortion_p95_stretch']-1)*100:.2f}% |")
    lines+=["",f"The {len(report['cases'])} exports cover **{report['total_triangulated_cm2']:.6f} cm²** of triangle area in the original 3D scan. The largest single connected piece covers **{report['largest_single_connected_export_cm2']:.6f} cm²**. Counting supported flattened pixels at the nominal pitch gives {report['total_nominal_cm2']:.6f} cm² across all exports. Every native footprint lies beyond the previous mesh guard. All pairs of native bounding rectangles are separated by more than 18 voxels, including the secondary components.","",
        "Multiple spatially separated routes agree on these coordinates. They use the same M7 volume. Native CT verifies the measured attenuation and makes the geometry inspectable. There are no independent sheet-identity annotations for the extensions. A winding switch shared by multiple routes remains possible. The table measures exported geometry, with no claim of ink or text.","",
        "## Inspect the files","",
        "Open the three TIFF coordinate maps in each case's TIFXYZ directory with compatible mesh tools. Values are global level-zero XYZ voxel coordinates of the original scroll. A value of −1 marks an invalid vertex. The files use eight native voxels per grid edge. Four valid corners form a usable quad.","",
        "Load the CT and mask with NumPy. The central depth index is 8. All 17 planes are original attenuation values sampled one native voxel apart. Exclude pixels outside the mask. The surface normal follows increasing flattened column crossed with increasing flattened row. Pixel size is 9.362 µm.","",
        '```python','import numpy as np','data = np.load("cases/east/native_ct.npz")','stack, mask = data["ct17"], data["mask"]','surface = np.ma.array(stack[8], mask=~mask)','```',"",
        "Each case includes three axial CT transects with an unmarked slice and the computed trace over the identical slice. Native bounds and hashes are in the JSON records. The exported masks preserve holes and all disconnected islands.","",
        "## Reproduction","",
        "Install the versions in requirements.txt using Python 3.12. Run the following from this folder. The first command checks all payload digests, geometry, coordinates, and nonoverlap. The second additionally rereads original CT chunks for 55 depth-pixels per direction. A one-level uint8 rounding tolerance is allowed.","",
        "```text","python verify_geometry.py","python verify_geometry.py --online","```","",
        "To rebuild from the supplied starting geometry and upstream M7, first create data/cache. Run the following from the same folder with network access. Missing entries in the sparse M7 have the published fill value. Large intermediate route files and local caches are written to results and data. Start with an empty results/growbench/recover directory for an independent reproduction. The supplied verification report records checks on this review copy.","",
        "```text","python scripts/growbench_recover_unread.py --threads 2","python scripts/growbench_recover_unread.py --combine","python scripts/growbench_recover_unread_export.py","python scripts/growbench_recover_unread_directions.py west","python scripts/growbench_recover_unread_directions.py north","python scripts/growbench_recover_unread_directions.py south","```","",
        "## Reconstruction and area accounting","",
        "Thirteen routes per direction start 80 native voxels apart, wherever the old qualified footprint supplies a seed and local tangent. They advance at intervals of 80 voxels through 193-voxel lateral windows within 257³ M7 volumes. Each tile follows peaks of the public M7 surface field using tangent and curvature costs, then preserves the local offset to the starter plane. A retained node requires a route score of at least 0.58. The next tile needs a supported local starter, at least 240 overlap nodes and at most 30% of checked neighbors disagreeing by more than three voxels. Each route can stop early. Competing routes must predict heights within three voxels; surviving regions retain a path to the original anchor strip.","",
        "The old regional reconstruction spans native X 2760–4040 and Z 9827–11107. The outward new coordinate cutoff is X ≥ 4056 east, X ≤ 2744 west, Z ≥ 11123 north, and Z ≤ 9811 south. The transverse coordinate stays inside the earlier footprint's projected range. Direction names denote increasing or decreasing scan axes. Independent rotations and reflections transform native M7 into the same tracking chart and transform output back to native XYZ.","",
        "The largest component in each new guarded region is eroded by an eight-voxel radius before eight-voxel regridding. A 35-step local/global as-rigid-as-possible flattening is rasterized in physical coordinates. Quad midpoints must retain support. The conservative export can fragment at narrow bridges. Summed triangle areas are calculated in 3D from the delivered TIFXYZ and native pitch. The CT uses trilinear interpolation over original native voxels, with a separate all-corners-nonzero mask available in support.npz.","",
        "The route_agreement.npz files also retain the original fitted domain and route count. Within those files, only the new mask marks the guarded extension. The TIFXYZ mesh and native_ct.npz already apply the guard. Positive numbers in the full y array away from valid support are unusable.","",
        "## Native GrowPatch cross-check","",
        "Stock GrowPatch was compared on the east side. The common-start central route and the larger multi-start region have different denominators. The check uses the same M7 input and evaluates agreement between two solvers. No human winding label was available. Details and original measurements are in stock/README.md.","",
        "## Sources and attribution","",
        "PHerc. 125 is held by the Biblioteca Nazionale di Napoli ‘Vittorio Emanuele III’, Officina dei Papiri Ercolanesi. Vesuvius Challenge distributes the ESRF scan, volume 20250821151825, and the level-zero M7 surface prediction, model 20260413222639, at 9.362 µm. The official catalog lists zero released segments for this scroll. Original scan values were resampled here, and upstream surface predictions were traced, flattened and cropped. See the official dataset citation for Giorgio Angelotti and colleagues. [PHerc. 125 catalog](https://scrollprize.org/data_browser/PHerc0125) · [Dataset citation](https://scrollprize.org/data#how-to-cite) · [CC BY-NC 4.0](https://creativecommons.org/licenses/by-nc/4.0/)","",
        "The organizer's noncommercial terms apply to its data and derived materials. This is a private review copy. The supplied analysis code has no public release license assigned in this package.",""]
    lines=[s.replace("55 depth-pixels per direction","55 depth-pixels per export").replace(
        "The largest component in each new guarded region is eroded","Each component spanning at least 4,000 two-voxel-grid nodes in a new guarded region is eroded") for s in lines]
    i=lines.index("python scripts/growbench_recover_unread_directions.py south")+1
    lines[i:i]=["python scripts/growbench_recover_unread_components.py "+d for d in ("east","west","north","south")]+["python scripts/growbench_recover_unread_polylines.py"]
    ix=lines.index("## Native GrowPatch cross-check")
    section=["## Continuity of organizer curves","","For each direction, a single stored organizer polyline had to match the old qualified surface through a 128-voxel-wide interior band. Ambiguous or short anchors were rejected before the new positions were compared. Following that same curve outward gives a separate continuity check using the common M7 source. The following counts are on 8-voxel spaced new points along 32-voxel spaced planes. Unmatched points remain unresolved.","","| Direction | New candidate points | Reached by a unique anchored curve | Within 3 native Y voxels |","|---|---:|---:|---:|"]
    for d in ("east","west","north","south"):
        item=json.loads((DEST/"cases"/d/"organizer_continuity.json").read_text())
        section.append(f"| {d} | {item['new_consensus_nodes']:,} | {item['new_nodes_on_single_original_polyline']:,} | {item['within_three_voxels']:,} |")
    lines[ix:ix]=section+[""]
    ix=lines.index("## Inspect the files")
    old=report["previous_interior_cases"]
    gap=min(g for item in old.values() for g in item["overlap_checks"].values())
    sep=report["previous_pherc125_native_y_separation_vox"]
    lines[ix:ix]=["## Relationship to the earlier three interior patches","",
        f"The previous package reported 2.913 cm² using its count of supported CT pixels times nominal pixel area. Recomputing all three source masks gives {report['previous_interior_nominal_cm2']:.6f} cm². On the same convention, the combined interior coverage across both scrolls is **{report['combined_interior_nominal_cm2']:.6f} cm²**. Computing actual triangle area on every supplied mesh instead gives {report['previous_interior_triangulated_cm2']:.6f} cm² previously and **{report['combined_interior_triangulated_cm2']:.6f} cm²** combined.","",
        f"The old interior cases are PHerc. 125 Y3310 and Y3706, and PHerc. 268 Y2991. Each extension was tested separately against the two different PHerc. 125 meshes. The minimum separating native X or Z gap across all fourteen pairs is {gap:.1f} voxels. The two older PHerc. 125 surface estimates are {sep['minimum']:.1f}–{sep['maximum']:.1f} native Y voxels apart across {sep['shared_qualified_nodes']:,} shared qualified chart nodes. PHerc. 268 occupies a different scan. Its exterior Y2193 control remains outside all interior totals. The decoded original masks, meshes and their prior-package source digests are included under previous_geometry for independent recalculation.","","![Native surface footprints and organizer-curve continuity](coverage.png)",""]
    (DEST/"README.md").write_text("\n".join(lines),encoding="utf8")


def stock():
    target=DEST/"stock";target.mkdir(exist_ok=True)
    lines=["# Stock GrowPatch comparison on the east extension","","These are comparisons against the preceding method's recovered coordinates. Stock and challenger used the same M7 information. The default and oriented runs share a center starter with challenger route 06. The broader comparison uses the challenger's separate neighbor routes and the interior of the common crop.","",
           "| Configuration | Region | Evaluated area (cm²) | Stock intersects (cm²) | Matches within 3 voxels (cm²) | Median height difference (voxels) |",
           "|---|---|---:|---:|---:|---:|"]
    for path in sorted((ROOT/"results/growbench/stock_pherc125").glob("*_application.json")):
        data=json.loads(path.read_text());shutil.copy2(path,target/path.name)
        for key in ("central_route_06_east","consensus_all_interior_east"):
            row=data["sections"][key]
            lines.append(f"| {path.stem.removesuffix('_application')} | {key} | {row['reference']['area_cm2']:.5f} | {row['stock_intersects']['area_cm2']:.5f} | {row['same_height_3vox']['area_cm2']:.5f} | {row['median_y_difference_vox']:.3f} |")
    upstream=ROOT/"results/growbench/stock_pherc125"
    shutil.copy2(upstream/"seeds.json",target/"seeds.json")
    for f in upstream.glob("*candidate_agreement.tifxyz"):shutil.copytree(f,target/f.name,dirs_exist_ok=True)
    for f in upstream.glob("*comparison.png"):shutil.copy2(f,target/f.name)
    for variant in ("default20_loss","oriented8_loss"):
        folder=target/variant;folder.mkdir(exist_ok=True)
        params=json.loads(next((upstream/variant).rglob("params.json")).read_text())
        params["normal_grid_path"]=Path(params["normal_grid_path"]).relative_to(ROOT).as_posix()
        (folder/"params.json").write_text(json.dumps(params,indent=2))
        shutil.copy2(next((upstream/variant).rglob("surface.npz")),folder/"surface.npz")
    lines+=["","Areas in this table use the original, fine consensus grid within the comparator's crop, before intrinsic regridding and erosion. The reported conditional agreement applies only where the two meshes intersected. The unmatched area remains unresolved.",""]
    lines+=["The oriented run's contiguous agreement footprint is in oriented8_loss_candidate_agreement.tifxyz. Its qualified_quads.npz marks the exact native faces. This is a subset of the east evidence and adds no unique area to the four-direction total. Saved configurations use project-relative grid paths; the wrapper resolves them at run time.","",
            "The comparison used the official VC3D Windows x64 release e875441. The stock binary archive SHA-256 was dfdebbdf480f97da455ae821d2bafa8c4402d30006a37aff60a22237bc53f1ea. The associated Python wrapper fetches the pinned upstream release from the organizers. [Release archive](https://api.github.com/repos/ScrollPrize/villa/releases/assets/787782221) · [GrowPatch source](https://github.com/ScrollPrize/villa/blob/e87544118ed046811635f62a47d04539e2a92c12/volume-cartographer/core/src/GrowPatch.cpp)","",
            "After rebuilding the east candidate from the addendum root, use the following on Windows x64 to repeat the same-input comparison. The figure is optional.","","```text","python scripts/growbench_stock_unread.py"]
    for v in ("default20","oriented8"):
        lines+=["python scripts/growbench_stock.py --manifest results/growbench/stock_pherc125/seeds.json --output results/growbench/stock_pherc125 --variant "+v+" --pitch-um 9.362 --diagnostics","python scripts/growbench_stock_unread.py --compare "+v+"_loss"]
    lines+=["python scripts/growbench_stock_unread.py --figure","```",""]
    (target/"README.md").write_text("\n".join(lines),encoding="utf8")


def main(online):
    DEST.mkdir(exist_ok=True,parents=True)
    for d in ("east","west","north","south"):
        source=SOURCE/("pherc125_"+("extension" if d=="east" else d))
        from scipy.ndimage import label
        with np.load(source/"consensus.npz")as data:
            ids,_=label(data["new"]);count=int((np.bincount(ids.ravel())[1:]>=4000).sum())
        for rank in range(1,count+1):assemble_case(d,rank)
    (DEST/"scripts").mkdir(exist_ok=True)
    for script in SCRIPTS:shutil.copy2(ROOT/"scripts"/(script+".py"),DEST/"scripts")
    shutil.copy2(ROOT/"scripts/growbench_recover_verify_unread.py",DEST/"verify_geometry.py")
    starter=DEST/START;starter.mkdir(exist_ok=True,parents=True)
    for name in ("geometry_nodes.npz","meta.json","validation.json","section_sources.json"):shutil.copy2(ROOT/START/name,starter/name)
    previous=DEST/"previous_geometry";previous.mkdir(exist_ok=True);provenance={}
    for case in ("PHerc0125_y3310","PHerc0125_y3706","PHerc0268_y2991"):
        original=ROOT/"outputs/progress_cross_sections/cases"/case
        mesh_dir=next(original.rglob(case+".tifxyz"))
        mesh=np.stack([tifffile.imread(mesh_dir/(axis+".tif")).astype(float) for axis in "xyz"],-1)
        with np.load(original/"native.npz")as f:mask=f["mask"]
        meta=json.loads((original/"native_meta.json").read_text())
        np.savez_compressed(previous/(case+".npz"),mesh=mesh,mask=mask,pitch_um=meta["pitch_um"])
        provenance[case]={str(p.relative_to(ROOT)).replace("\\","/"):sha(p) for p in [original/"native.npz",original/"native_meta.json"]+[mesh_dir/(axis+".tif")for axis in "xyz"]}
    chart={}
    for y in (3310,3706):
        path=ROOT/f"results/first_letters_deep/PHerc0125_y{y}/geometry_nodes.npz"
        with np.load(path)as d:chart["y"+str(y)]=d["fit"];chart["qualified"+str(y)]=d["qualified"]
        provenance["PHerc0125_y"+str(y)][path.relative_to(ROOT).as_posix()]=sha(path)
    np.savez_compressed(previous/"old_shared_grid.npz",**chart)
    (previous/"source_digests.json").write_text(json.dumps(provenance,indent=2))
    (DEST/"requirements.txt").write_text("numpy==2.5.4\nscipy==1.18.1\nnumcodecs==0.17.1\nrequests==2.34.2\ntifffile==2026.10.15\npillow==13.0.0\nfilelock==4.2.5\nmatplotlib==3.11.2\n")
    report=verify(DEST,online);stock();docs(report);draw(DEST)
    (DEST/"VERIFICATION.json").write_text(json.dumps(report,indent=2))
    files=sorted(p for p in DEST.rglob("*") if p.is_file() and p.name!="SHA256SUMS" and "__pycache__" not in p.parts)
    lock=DEST/"SHA256SUMS";lock.write_text("".join(sha(p)+"  "+p.relative_to(DEST).as_posix()+"\n"for p in files))
    archive=DEST.with_suffix(".zip")
    with zipfile.ZipFile(archive,"w",compression=zipfile.ZIP_DEFLATED,compresslevel=5) as z:
        for p in files+[lock]:z.write(p,p.relative_to(DEST.parent))
    with zipfile.ZipFile(archive) as z:assert z.testzip() is None
    digest=sha(archive)
    archive.with_suffix(".zip.sha256").write_text(digest+"  "+archive.name+"\n")
    print(digest,archive.stat().st_size,report["total_triangulated_cm2"],flush=True)


if __name__=="__main__":
    p=argparse.ArgumentParser(description=__doc__);p.add_argument("--online",action="store_true")
    main(p.parse_args().online)
