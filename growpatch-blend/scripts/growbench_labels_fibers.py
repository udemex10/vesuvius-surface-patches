"""Audit sparse evaluated fiber control anchors local to the frozen Paris4 sites."""
import json
from concurrent.futures import ThreadPoolExecutor
from xml.etree import ElementTree

import numpy as np
import requests
from scipy.spatial import cKDTree
from growbench_labels import OUT, SPIRAL, fetch, digest, write


def main():
    seeds=json.loads((OUT/"seeds.json").read_text())["sites"]
    prefix=SPIRAL.split(".com/")[1]+"eval_fibers/"
    bucket=SPIRAL.split(".com/")[0]+".com/"
    xml=ElementTree.fromstring(requests.get(bucket,params={"list-type":2,"prefix":prefix,"max-keys":1000},timeout=30).content)
    assert xml.find("{*}IsTruncated").text=="false"
    names=[c.find("{*}Key").text for c in xml.findall("{*}Contents") if c.find("{*}Key").text.endswith(".json")]
    def anchors(name):
        data=fetch(bucket+name);obj=json.loads(data)
        if obj.get("vc_open_data_coordinate_space")!="PHercParis4/20260411134726@L0":
            return dict(file=name.split("/")[-1],sha256=digest(data),skipped="No declared matching coordinate frame",clicks_l2=[])
        return dict(file=name.split("/")[-1],sha256=digest(data),mode=obj.get("optimization_mode"),
                    clicks_l2=[(np.array(c["position"])/4).tolist() for c in obj["control_points"]])
    with ThreadPoolExecutor(15) as ex:
        all_lines=list(ex.map(anchors,names))
    evidence=[]
    for site in seeds:
        seed=np.asarray(site["seed_xyz"])
        grid=np.load(OUT/"truth"/site["id"]/"ref_xyz.npy");xyz=grid.reshape(-1,3);xyz=xyz[np.isfinite(xyz).all(1)]
        tree=cKDTree(xyz)
        records=[]
        for line in all_lines:
            p=np.array(line["clicks_l2"])
            if p.size==0:continue
            inside=np.linalg.norm(p-seed,axis=1)<=96
            if not inside.any():continue
            d,i=tree.query(p[inside])
            for ix,dist in zip(np.flatnonzero(inside).tolist(),d):
                records.append(dict(file=line["file"],index=ix,xyz_l2=p[ix].tolist(),nearest_reference_voxels=float(dist)))
        evidence.append(dict(id=site["id"],anchors=records))
    write(OUT/"fiber_anchor_index.json",dict(source=SPIRAL+"eval_fibers/",files=len(names),traces=all_lines))
    write(OUT/"fiber_anchor_audit.json",dict(dataset="Only original control_points positions; excluded model-interpolated line_points and segment_to_next.",scope="Within 96 voxel 3D sphere of locked seed. Nearness alone does not identify a fiber as belonging to target surface; controls may lie near either face.",sites=evidence))
    print(len(names),sum(len(line["clicks_l2"]) for line in all_lines),[(s["id"],len(s["anchors"]),len([r for r in s["anchors"] if r["nearest_reference_voxels"]<4])) for s in evidence])


if __name__=="__main__":main()
