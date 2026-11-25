"""Freeze the four preidentified gap-heavy sites from the original pressure candidate pool."""
import argparse
from concurrent.futures import ThreadPoolExecutor
import json
import shutil

from growbench_labels import CT, M7, OUT as BASE, digest, extract, write

POOL=BASE/"pressure"
OUT=BASE/"interrupted"
EXPECTED=["pool_37","pool_17","pool_24","pool_04"]


def select():
    if (OUT/"seeds.json").exists():raise SystemExit("Already locked")
    p=json.loads((POOL/"pool_public.json").read_text())["sites"]
    refs=json.loads((POOL/"pool_refs.json").read_text())["sites"]
    scores=json.loads((POOL/"input_scores.json").read_text())["sites"]
    used={r["score"]["id"] for r in json.loads((POOL/"refs.json").read_text())["sites"]}
    selected=sorted([r for r in scores if r["eligible_seed"] and r["id"] not in used],key=lambda r:-r["gaps"])[:4]
    assert [r["id"] for r in selected]==EXPECTED
    sites=[];provenance=[]
    for i,row in enumerate(selected,1):
        j=next(j for j,s in enumerate(p) if s["id"]==row["id"])
        sid=f"gap_{i:02d}";target=OUT/"truth"/sid;target.mkdir(exist_ok=True,parents=True)
        for n in ("ref_xyz.npy","triangles.npy"):
            shutil.copyfile(POOL/"pool_truth"/row["id"]/n,target/n)
        sites.append(dict(p[j],id=sid,split="test"));provenance.append(dict(refs[j],id=sid,pool_id=row["id"],input_score=row))
    write(OUT/"refs.json",dict(source=POOL.relative_to(BASE.parent).as_posix(),selected_by="Most absent near-plane M7 peaks among anchored eligible unused candidates in the original physically disjoint forty-site pool. Four source IDs fixed before their solver runs. Local M7 absence has multiple causes, including a surface curving away from the projected seed plane.",source_pool_sha256=digest((POOL/"pool_public.json").read_bytes()),source_scores_sha256=digest((POOL/"input_scores.json").read_bytes()),sites=provenance))
    write(OUT/"seeds.json",dict(benchmark="Paris4-M7-interrupted-v1",frame="global XYZ 9.6um=CT level 2",ct=CT,m7=M7,area_radius_voxels=160,excluded_seed_radius_voxels=12,sites=sites))
    print("GAP LOCK",digest((OUT/"seeds.json").read_bytes()),flush=True)


if __name__=="__main__":
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument("action",choices=["select","ct","m7"]);a=ap.parse_args()
    if a.action=="select":select()
    else:
        sites=json.loads((OUT/"seeds.json").read_text())["sites"]
        with ThreadPoolExecutor(4) as ex:list(ex.map(lambda s:extract(s,a.action,M7 if a.action=="m7" else CT,OUT),sites))
