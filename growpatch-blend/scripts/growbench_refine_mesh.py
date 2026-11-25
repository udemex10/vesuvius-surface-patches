"""Produce a local papyrus chart from a GrowPatch mesh and an aligned surface-foreground cube."""
import argparse
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np

from growbench_geometry import U, faces, project
from growbench_recover import solve
from growbench_review import blend_projection, export_chart, project_mesh, read_json, save_clipped_obj, write_json
from growbench_seed_align import load_surface


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("stock", type=Path, help="GrowPatch tifxyz folder, or global XYZ/valid NPZ")
    p.add_argument("foreground", type=Path, help="ZYX surface-foreground NPY, or review input NPZ containing foreground")
    p.add_argument("site", type=Path, help="Seed JSON with seed_xyz, normal, tangent axes and origin_xyz")
    p.add_argument("output", type=Path, help="New folder for the output charts")
    p.add_argument("--id", help="Case ID when site is a multi-case manifest")
    p.add_argument("--stock-local", action="store_true", help="Coordinates in the stock file start at the foreground cube origin")
    args = p.parse_args()
    site = read_json(args.site)
    if "sites" in site:
        site = next((s for s in site["sites"] if s["id"] == args.id), None)
        if site is None:
            p.error("Select a listed site with --id")
    site = site.get("site", site)
    if args.stock.is_dir():
        xyz, valid = load_surface(args.stock)
    else:
        with np.load(args.stock) as stock:
            xyz, valid = stock["xyz"], stock["valid"]
    if args.stock_local:
        xyz = xyz.copy()
        xyz[valid.astype(bool)] += np.asarray(site["origin_xyz"])
    stock = project(faces(xyz, valid)[0], site)
    if args.foreground.suffix.lower() == ".npz":
        with np.load(args.foreground) as data:
            foreground = data["foreground"]
    else:
        foreground = np.load(args.foreground, mmap_mode="r")
    args.output.mkdir(parents=True, exist_ok=False)
    with TemporaryDirectory() as tmp:
        tmp = Path(tmp)
        try:
            info = solve(foreground, site, tmp)
        except ValueError as exc:
            if "cue voxels" not in str(exc) and "No " not in str(exc):
                raise
            info = {"abstained": str(exc)}
            np.savez_compressed(tmp / "graph.npz", xyz=np.full((*U.shape,3),np.nan,np.float32),
                                valid=np.zeros(U.shape,bool),quality=np.zeros(U.shape,np.float32))
        with np.load(tmp / "graph.npz") as graph:
            graph_projection = project_mesh(graph, site, quality=True)
            save_clipped_obj(args.output, graph, site, .575)
    blend = blend_projection(stock, graph_projection)
    accept = np.isfinite(blend.low) & (blend.high-blend.low <= 1)
    export_chart(args.output / "local-blend.tifxyz", site, blend.center, accept)
    write_json(args.output / "run.json", dict(site=site, valid_blend_nodes=int(accept.sum()), graph=info))


if __name__ == "__main__":
    main()
