"""Rebuild a frozen benchmark case directly from public data and verify its array digests."""

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np

from growbench_labels import CT, M7, OUT, extract, patch, raster, triangulate
from growbench_labels_manual import choose, common_input, mesh, original


COHORTS = {
    "primary": "",
    "manual": "manual_ids",
    "pressure": "pressure",
    "interrupted": "interrupted",
    "manual_external": "manual_external",
}


def file_hash(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def replay_manual(case, reference, output, cache):
    source_name = reference["source_cube_zyx"]
    ids, instance_info = original(source_name, "instance.nrrd", cache)
    if instance_info["sha256"] != reference["source"]["sha256"]:
        raise ValueError(f"Instance-label release has changed for {source_name}.")
    target_id = reference["target_id"]
    excluded = set(np.unique(ids)) - {target_id}
    target = choose(ids, excluded)
    if target is None or target[0] != target_id:
        raise ValueError(f"Frozen target {target_id} was not recovered in {source_name}.")
    origin = np.asarray(case["origin_xyz"])
    xyz, triangles = mesh(target[-1], origin)
    labels = output / "truth" / case["id"]
    np.save(labels / "ref_xyz.npy", xyz)
    np.save(labels / "triangles.npy", triangles)
    np.save(labels / "instance_ids.npy", ids)

    semantic, _ = original(source_name, "semantic.nrrd", cache)
    ct, _ = original(source_name, "volume.nrrd", cache)
    items = output / "inputs" / case["id"]
    np.save(items / "m7.npy", common_input(semantic != 0))
    np.save(items / "ct.npy", ct)


def replay_vetted(case, reference, output, cache):
    vertices, _, source_hashes = patch(reference["name"], cache)
    if source_hashes != reference["hashes"]:
        raise ValueError(f"Published mesh files changed for {reference['name']}.")
    origin = np.asarray(case["origin_xyz"])
    size = case["cube_shape_zyx"][0]
    triangles = triangulate(vertices)
    in_cube = (triangles.max(1) >= origin) & (triangles.min(1) <= origin + size - 1)
    triangles = triangles[in_cube.all(1)]
    y, _ = raster(triangles, origin, size)
    z, x = np.mgrid[:size, :size]
    xyz = np.stack([x + origin[0], y, z + origin[2]], -1).astype("f4")
    xyz[~np.isfinite(y)] = np.nan
    labels = output / "truth" / case["id"]
    np.save(labels / "ref_xyz.npy", xyz)
    np.save(labels / "triangles.npy", triangles.astype("f4"))
    extract(case, "m7", M7, output, cache)
    extract(case, "ct", CT, output, cache)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("cohort", choices=COHORTS)
    ap.add_argument("case", help="Frozen ID, for example test_03, external_01 or pressure_01")
    ap.add_argument("--output-root", type=Path, required=True, help="Separate replay directory")
    ap.add_argument("--manifests", type=Path, default=OUT, help="Directory with frozen seeds, refs and replay_hashes.json")
    args = ap.parse_args()
    manifest_root = args.manifests.resolve()
    output_root = args.output_root.resolve()
    if output_root == manifest_root or manifest_root in output_root.parents or output_root in manifest_root.parents:
        ap.error("Select a separate replay directory outside the frozen source tree.")
    manifest_dir = manifest_root / COHORTS[args.cohort]
    seeds = json.loads((manifest_dir / "seeds.json").read_text())["sites"]
    references = json.loads((manifest_dir / "refs.json").read_text())["sites"]
    frozen = json.loads((manifest_root / "replay_hashes.json").read_text())[args.cohort]
    sites = {s["id"]: s for s in seeds}
    if args.case not in sites:
        ap.error(f"Case IDs for {args.cohort}: {', '.join(sites)}")
    case = sites[args.case]
    reference = next(r for r in references if r["id"] == args.case)
    output = output_root / args.cohort
    for part in ("inputs", "truth"):
        (output / part / args.case).mkdir(parents=True, exist_ok=True)
    cache = output_root / "public_sources"
    if args.cohort in ("manual", "manual_external"):
        replay_manual(case, reference, output, cache)
    else:
        replay_vetted(case, reference, output, cache)
    expected = frozen[args.case]
    for name, expected_hash in expected.items():
        folder = "inputs" if name in ("m7.npy", "ct.npy") else "truth"
        actual = file_hash(output / folder / args.case / name)
        if actual != expected_hash:
            raise ValueError(f"Hash mismatch for {name}. Expected {expected_hash}, got {actual}.")
    print(f"{args.cohort}/{args.case}: {len(expected)} array files matched frozen SHA-256 digests.")


if __name__ == "__main__":
    main()
