"""Replay frozen manual-sheet comparisons and export their seed-centered coordinate charts."""
import argparse
import hashlib
import json
from pathlib import Path
from tempfile import TemporaryDirectory
from contextlib import ExitStack

import numpy as np
import tifffile

from growbench_geometry import Projection, U, V, faces, frame, project
from growbench_identity import identities, summarize
from growbench_recover import solve


def read_json(path):
    return json.loads(path.read_text(encoding="utf-8"))


def write_json(path, data):
    path.write_text(json.dumps(data, indent=2, allow_nan=False) + "\n", encoding="utf-8")


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def project_mesh(data, site, quality=False):
    tri, confidence = faces(data["xyz"], data["valid"], data["quality"] if quality else None)
    return project(tri, site, confidence)


def blend_projection(stock, graph):
    gc, sc = graph.center, stock.center
    shared = np.isfinite(gc) & np.isfinite(sc) & (graph.high-graph.low <= 1) & (stock.high-stock.low <= 1)
    gap = np.zeros(sc.shape)
    gap[shared] = gc[shared] - sc[shared]
    shift = np.where(shared & (abs(gap) <= 8), .5*gap, 0.)
    return Projection(stock.low+shift, stock.high+shift, np.ones(sc.shape), stock.area_factor)


def export_chart(out, site, depth, selected):
    """Use global XYZ at two-voxel spacing. Original projections can contain multiple hits."""
    out.mkdir(parents=True, exist_ok=False)
    xyz = np.full((*U.shape, 3), -1, dtype=np.float32)
    aligned = np.stack((U[selected], V[selected], depth[selected]), axis=-1)
    xyz[selected] = aligned @ frame(site) + site["seed_xyz"]
    for k, a in enumerate("xyz"):
        tifffile.imwrite(out / (a + ".tif"), xyz[:, :, k], compression="deflate")
    meta = dict(type="seg", format="tifxyz", uuid=out.name, name=out.name, scale=[0.5, 0.5],
                coordinate_units="global XYZ in input voxels", source="growbench")
    if selected.any():
        meta["bbox"] = [xyz[selected].min(0).tolist(), xyz[selected].max(0).tolist()]
    write_json(out / "meta.json", meta)
    reopened = np.stack([tifffile.imread(out / (a + ".tif")) for a in "xyz"], axis=-1)
    if not np.array_equal(reopened, xyz):
        raise ValueError("Coordinate TIFF round-trip changed a vertex.")
    tri, _ = faces(reopened)
    return project(tri, site)


def verify_values(actual, expected):
    for key in ("reference", "accepted", "correct", "connected", "known_other", "unassigned", "ambiguous"):
        if not np.isclose(actual[key], expected[key], atol=2e-10, rtol=0):
            raise ValueError(f"{key}: expected {expected[key]}, got {actual[key]}")


def save_clipped_obj(folder, graph, site, threshold):
    """Clip individual native triangles at their linear confidence crossing."""
    tri, q = faces(graph["xyz"], graph["valid"], graph["quality"])
    inside = q + 1e-9 >= threshold
    output = [piece for piece in tri[inside.all(axis=1)]]
    for vertices, values, contains in zip(tri[inside.any(1) & ~inside.all(1)].astype(float),
                                          q[inside.any(1) & ~inside.all(1)], inside[inside.any(1) & ~inside.all(1)]):
        polygon = []
        for b in range(3):
            a = (b-1) % 3
            if contains[a] != contains[b]:
                part = (threshold-1e-9-values[a]) / (values[b]-values[a])
                polygon.append(vertices[a] + part*(vertices[b]-vertices[a]))
            if contains[b]:
                polygon.append(vertices[b])
        for b in range(1,len(polygon)-1):
            output.append([polygon[0], polygon[b], polygon[b+1]])
    verts, indices = np.unique(np.asarray(output).reshape(-1,3).round(8), axis=0, return_inverse=True)
    path = folder / "graph-confidence-cut.obj"
    with path.open("w", encoding="ascii") as stream:
        stream.write("# Global XYZ, original scan voxel units. Confidence-clipped native graph.\n")
        np.savetxt(stream, verts, fmt="v %.8f %.8f %.8f")
        np.savetxt(stream, indices.reshape(-1,3)+1, fmt="f %d %d %d")
    with path.open(encoding="ascii") as stream:
        vertices, faces_out = [], []
        for line in stream:
            if line.startswith("v "):
                vertices.append([float(x) for x in line.split()[1:]])
            if line.startswith("f "):
                faces_out.append([int(x)-1 for x in line.split()[1:]])
    vertices = np.asarray(vertices, dtype=float).reshape(-1,3)
    indices = np.asarray(faces_out, dtype=int).reshape(-1,3)
    return project(vertices[indices], site)


def review_case(folder, out, expected, source_root=None):
    meta = read_json(folder / "case.json")
    site = meta["site"]
    with ExitStack() as opened:
        scratch = Path(opened.enter_context(TemporaryDirectory()))
        if source_root is None:
            data = opened.enter_context(np.load(folder / "input.npz"))
        else:
            local = source_root / "manual_external"
            data = {name: np.load(local / group / site["id"] / file, mmap_mode="r")
                    for name, group, file in (("foreground", "inputs", "m7.npy"),
                    ("reference_triangles", "truth", "triangles.npy"), ("instance_labels", "truth", "instance_ids.npy"))}
        try:
            solve(data["foreground"], site, scratch)
        except ValueError as exc:
            if "cue voxels" not in str(exc) and "No " not in str(exc):
                raise
            np.savez_compressed(scratch / "graph.npz", xyz=np.full((*U.shape, 3), np.nan, np.float32),
                                valid=np.zeros(U.shape, bool), quality=np.zeros(U.shape, np.float32))
        out.mkdir(parents=True)
        with np.load(scratch / "graph.npz") as graph:
            graph_projection = project_mesh(graph, site, quality=True)
            clipped = save_clipped_obj(out, graph, site, .575)
        with np.load(folder / "stock_1.npz") as stock:
            full = project_mesh(stock, site)
        with np.load(folder / "stock_128.npz") as stock:
            half = project_mesh(stock, site)
        truth = project(data["reference_triangles"], site)
        ids = data["instance_labels"]
        def measure(prediction, threshold=0):
            return identities(truth, prediction, threshold, site, ids, meta["target_id"])
        blend = blend_projection(half, graph_projection)
        cases = {"stock_1": (full, 0), "graph": (graph_projection, .575),
                 "stock_128": (half, 0), "blend": (blend, 0)}
        rows = {"graph_obj": measure(clipped)}
        for name, (prediction, threshold) in cases.items():
            rows[name] = measure(prediction, threshold)
            verify_values(rows[name], expected[name][site["id"]])
            unique = np.isfinite(prediction.low) & (prediction.high - prediction.low <= 1)
            accepted = unique & (prediction.confidence + 1e-9 >= threshold)
            reopened = export_chart(out / name, site, prediction.center, accepted)
            rows[name + "_export"] = measure(reopened)
    return rows


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("package", type=Path, help="Extracted comparison directory with cases/, SHA256SUMS and expected.json")
    p.add_argument("output", type=Path, help="New directory for global-XYZ meshes and replay measurements")
    p.add_argument("--case", action="append", help="Replay one or more case IDs; default is all 16")
    p.add_argument("--source-root", type=Path, help="Use cases rebuilt with growbench_labels_replay.py in this directory")
    a = p.parse_args()
    for line in (a.package / "SHA256SUMS").read_text().splitlines():
        hex_digest, path = line.split("  ", 1)
        if digest(a.package / path) != hex_digest:
            raise ValueError(f"File digest differs for {path}")
    expected = read_json(a.package / "expected.json")
    ids = a.case or sorted(expected["graph"])
    if any(i not in expected["graph"] for i in ids) or len(ids) != len(set(ids)):
        p.error("Specify distinct case IDs from expected.json")
    a.output.mkdir(parents=True, exist_ok=False)
    rows = {}
    for i in ids:
        amounts = review_case(a.package / "cases" / i, a.output / i, expected, a.source_root)
        for k, value in amounts.items():
            rows.setdefault(k, {})[i] = value
        print(f"{i} reproduced", flush=True)
    pairs = {}
    for base, candidate in (("stock_1", "graph"), ("stock_128", "blend")):
        pairs[candidate] = summarize(rows[base], rows[candidate])
        pairs[candidate + "_export"] = summarize(rows[base + "_export"], rows[candidate + "_export"])
        pairs[candidate + "_export_vs_native_stock"] = summarize(rows[base], rows[candidate + "_export"])
    pairs["graph_obj_vs_native_stock"] = summarize(rows["stock_1"], rows["graph_obj"])
    write_json(a.output / "replay.json", dict(cases=ids, summary=pairs, rows=rows))
    print("Wrote original and TIFF round-trip comparisons to", a.output / "replay.json")


if __name__ == "__main__":
    main()
