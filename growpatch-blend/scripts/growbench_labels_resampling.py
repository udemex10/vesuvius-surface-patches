"""Independent lookup and paired uncertainty audit of the frozen external graph comparison."""

import argparse
import json
from collections import Counter

import numpy as np
from scipy.ndimage import binary_propagation
from scipy.sparse.csgraph import connected_components
from scipy.spatial.distance import cdist

from growbench_geometry import R2, STEP, U, V, Projection, frame, load_grid, project
from growbench_labels import OUT, write
from growbench_labels_replay import file_hash


ROOT = OUT.parent
MANUAL = OUT / "manual_external"
SOURCE = ROOT / "comparison/named_risk/oriented8_loss_external.json"
BLEND_SOURCE = ROOT / "comparison/regional/manual_external_oriented8_th128_loss.json"
METRICS = ["named_wrong_difference_points", "strict_error_difference_points", "connected_difference_cm2_at_16_sites",
           "named_wrong_per_accepted_difference_points", "named_wrong_area_difference_cm2_at_16_sites"]
KEYS = ["known_other", "correct", "accepted", "incorrect", "connected"]
ASSIGNMENTS = ["abstained", "ambiguous", "unassigned", "target", "different_named"]


def measurement(site, reference, trial, volume, target, confidence):
    support = np.isfinite(reference.low) & (reference.high - reference.low <= 0.5)
    support &= (reference.area_factor <= 2) & (R2 <= 96 ** 2)
    weight = np.where(support & (R2 > 12 ** 2), reference.area_factor * (STEP * 7.91e-4) ** 2, 0)
    accepted = support & np.isfinite(trial.low)
    if confidence is not None:
        accepted &= trial.confidence + 1e-9 >= confidence
    unique = accepted & (trial.high - trial.low <= 1)
    iy, ix = np.nonzero(unique)
    coordinates = np.column_stack((U[iy, ix], V[iy, ix], trial.center[iy, ix]))
    native = coordinates @ frame(site) + (np.asarray(site["seed_xyz"]) - site["origin_xyz"])
    voxels = np.floor(native + 0.5).astype(int)[:, ::-1]
    valid = np.all((voxels >= 0) & (voxels < volume.shape), axis=1)
    sampled = np.zeros(accepted.shape, dtype=volume.dtype)
    sampled[iy[valid], ix[valid]] = volume[tuple(voxels[valid].T)]
    correct = unique & (sampled == target)
    wrong = unique & (sampled > 0) & (sampled != target)
    center_component = binary_propagation(correct & (R2 <= 4 ** 2), mask=correct)
    masks = (wrong, correct, accepted, accepted & ~correct, center_component)
    state = np.zeros(accepted.shape, dtype=np.uint8)
    state[accepted & ~unique] = 1
    state[unique & (sampled == 0)] = 2
    state[correct] = 3
    state[wrong] = 4
    return [float(weight[m].sum()) for m in masks], state, weight


def summary(counts, data):
    # (resample, method, [wrong, correct, accepted, failed, connected])
    pooled = (counts @ data.reshape(len(data), -1)).reshape(-1, 2, len(KEYS))
    wrong, correct, accepted, failed, connected = np.moveaxis(pooled, 2, 0)
    n = counts.sum(1)[:, None]
    stats = (100 * wrong / (wrong + correct), 100 * failed / accepted, 16 * connected / n,
             100 * wrong / accepted, 16 * wrong / n)
    return np.stack([stat[:, 1] - stat[:, 0] for stat in stats], axis=1)


def resample(data, names, groups):
    group_values = np.unique(groups)
    membership = (group_values[:, None] == groups).astype(float)
    rng = np.random.default_rng(20261124)
    blocks = []
    for _ in range(20):
        multiplicity = rng.multinomial(len(group_values),
            np.full(len(group_values), 1 / len(group_values)), size=10_000)
        blocks.append(summary(multiplicity @ membership, data))
    intervals = np.quantile(np.concatenate(blocks), [0.025, 0.975], axis=0).T
    leave = summary(1 - membership, data)
    return {
        "units": [[names[j] for j in np.flatnonzero(mask)] for mask in membership],
        "paired_percentile_95": {k: v.tolist() for k, v in zip(METRICS, intervals)},
        "leave_one_out_range": {k: [float(v.min()), float(v.max())] for k, v in zip(METRICS, leave.T)},
        "leave_one_out_differences": {key: val.tolist() for key, val in zip(METRICS, leave.T)},
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--blend128", action="store_true", help="Audit the existing ungated blend_8 paired with official threshold 128")
    parser.add_argument("--figure", action="store_true", help="Render the paired assignment maps for the threshold 128 blend")
    args = parser.parse_args()
    if args.figure and not args.blend128:
        parser.error("--figure requires --blend128")
    seeds = json.loads((MANUAL / "seeds.json").read_text())["sites"]
    refs = {r["id"]: r for r in json.loads((MANUAL / "refs.json").read_text())["sites"]}
    source = BLEND_SOURCE if args.blend128 else SOURCE
    records = json.loads(source.read_text())
    if not args.blend128:
        assert records["threshold"]["graph"] == 0.575
    data = []
    transitions = []
    map_cases = []
    names = []
    max_discrepancy = 0
    for site in seeds:
        name = site["id"]
        truth = MANUAL / "truth" / name
        surface = project(np.load(truth / "triangles.npy"), site)
        volume = np.load(truth / "instance_ids.npy", mmap_mode="r")
        graph = load_grid(ROOT / "recover/manual_external" / name / "graph.npz", site)
        preset = "oriented8_th128_loss" if args.blend128 else "oriented8_loss"
        stock = load_grid(ROOT / "stock_manual_external" / preset / name / "surface.npz", site)
        if args.blend128:
            use = np.isfinite(stock.low) & np.isfinite(graph.low) & (stock.high - stock.low <= 1) & (graph.high - graph.low <= 1)
            delta = np.zeros(stock.low.shape)
            delta[use] = graph.center[use] - stock.center[use]
            shift = np.where(use & (abs(delta) <= 8), 0.5 * delta, 0)
            trial = Projection(stock.low + shift, stock.high + shift, stock.confidence, stock.area_factor)
            fields = [(None, stock, records["rows"]["stock"][name]), (None, trial, records["rows"]["blend_8"][name])]
        else:
            fields = [(None, stock, records["baselines"][name]), (0.575, graph, records["rows"]["graph"][name])]
        paired = []
        assignments = []
        for cutoff, field, expected in fields:
            actual, states, weight = measurement(site, surface, field, volume, refs[name]["target_id"], cutoff)
            max_discrepancy = max(max_discrepancy, max(abs(v - expected[k]) for v, k in zip(actual, KEYS)))
            paired.append(actual)
            assignments.append(states)
        n = len(ASSIGNMENTS)
        codes = assignments[0].ravel() * n + assignments[1].ravel()
        transitions.append(np.bincount(codes, weights=weight.ravel(), minlength=n*n).reshape(n, n))
        if args.figure:
            map_cases.append((name, assignments[0], assignments[1], weight, (paired[1][-1] - paired[0][-1]) * 100))
        data.append(paired)
        names.append(name)
    assert max_discrepancy < 1e-11, max_discrepancy
    data = np.asarray(data)
    centers = np.asarray([s["origin_xyz"] for s in seeds], float) + 127.5
    _, neighbors = connected_components(cdist(centers, centers) <= 512, directed=False)
    slabs = np.floor(centers[:, 2] / 4096).astype(int)
    singleton = np.eye(len(names))
    differences = summary(singleton, data)
    records_per_site = {}
    for name, row in zip(names, differences):
        native = row.copy()
        native[[2, 4]] /= 16
        records_per_site[name] = dict(zip([k.replace("_at_16_sites", "") for k in METRICS], native.tolist()))
    baseline, candidate = data.sum(axis=0)
    point = summary(np.ones((1, len(names))), data)[0]
    audit = {
        "methods": ["complete official oriented8 threshold 128", "existing ungated blend_8"] if args.blend128 else ["complete official oriented8 mesh", "graph confidence >=0.575"],
        "comparison_table_sha256": file_hash(source),
        "declared_sensitivity_rule_sha256": file_hash(OUT / "external_resampling_rule.md"),
        "n_sites": len(names), "paired_bootstrap_samples": 200000, "seed": 20261124,
        "maximum_independent_site_area_discrepancy_cm2": max_discrepancy,
        "sum_stock": dict(zip(KEYS, baseline.tolist())), "sum_candidate": dict(zip(KEYS, candidate.tolist())),
        "difference_candidate_minus_stock": dict(zip(METRICS, point.tolist())),
        "site_sign_counts": {k: dict(Counter("positive" if d > 1e-11 else "negative" if d < -1e-11 else "tie" for d in col)) for k, col in zip(METRICS, differences.T)},
        "per_site": records_per_site,
        "paired_ray_transition_cm2": {
            "classes": ASSIGNMENTS, "rows": "official", "columns": "candidate",
            "pooled": np.sum(transitions, axis=0).tolist(),
            "per_site": {k: v.tolist() for k, v in zip(names, transitions)},
        },
        "site_bootstrap": resample(data, names, np.arange(len(names))),
        "neighbor_group_bootstrap_512voxel_centers": resample(data, names, neighbors),
        "coarse_z_slab_diagnostic_4096voxels": resample(data, names, slabs),
    }
    write(OUT / ("external_resampling_blend128.json" if args.blend128 else "external_resampling.json"), audit)
    if args.figure:
        from growbench_labels_transition_figure import render
        render(map_cases, OUT / "external_blend_changes.png")
    print("Max independent discrepancy", max_discrepancy)
    print("Candidate minus stock", dict(zip(METRICS, point)))
    for key in ("site_bootstrap", "neighbor_group_bootstrap_512voxel_centers", "coarse_z_slab_diagnostic_4096voxels"):
        print(key, len(audit[key]["units"]), audit[key]["paired_percentile_95"], audit[key]["leave_one_out_range"])
    print(audit["site_sign_counts"])


if __name__ == "__main__":
    main()
