"""Whole-mesh local blend: tile the validated 8-voxel GrowPatch blend over an entire tifxyz mesh.

Each tile repeats the single-seed rule exactly. The unmodified graph solver runs at a seed on the
mesh. Where stock and graph are both single-valued and within 8 voxels along the tile normal, stock
moves halfway toward the graph; elsewhere it stays. Tile shifts are transferred to a 2-voxel
resampling of the stock mesh and combined with cos^2 weights that vanish at the 96-voxel tile edge.
See research/growpatch-tiled-blend-protocol.md.

    python growbench_tiled_blend.py STOCK_TIFXYZ FOREGROUND.npy OUT_DIR --origin X Y Z [--stock-local]
"""
import argparse
import hashlib
import json
import os
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from tempfile import TemporaryDirectory

import numpy as np
import tifffile
from scipy.ndimage import map_coordinates

from growbench_geometry import RADIUS, STEP, faces, frame, project
from growbench_recover import solve

LIMIT = 8.0          # validated blend agreement limit (voxels)
VISIBLE_TOL = 3.0    # vertex must lie this close to the tile's stock depth on its nearest ray
NEIGHBOR_RADIUS = 24.0
N_RAYS = 2 * (RADIUS // STEP) + 1


# ----------------------------------------------------------------------------- geometry helpers
def grid_valid(xyz):
    return np.isfinite(xyz).all(-1) & (xyz >= 0).all(-1)


def upsample(xyz, valid, k):
    """Bilinear k-fold resampling of every complete quad. Shared edges get identical values."""
    xyz = np.asarray(xyz, dtype=np.float64)
    h, w, _ = xyz.shape
    full = valid[:-1, :-1] & valid[1:, :-1] & valid[:-1, 1:] & valid[1:, 1:]
    out = np.full(((h - 1) * k + 1, (w - 1) * k + 1, 3), -1.0)
    ok = np.zeros(out.shape[:2], bool)
    I, J = np.nonzero(full)
    a00, a01, a10, a11 = xyz[I, J], xyz[I, J + 1], xyz[I + 1, J], xyz[I + 1, J + 1]
    for a in range(k + 1):
        for b in range(k + 1):
            s, t = a / k, b / k
            p = (1 - s) * (1 - t) * a00 + (1 - s) * t * a01 + s * (1 - t) * a10 + s * t * a11
            out[I * k + a, J * k + b] = p
            ok[I * k + a, J * k + b] = True
    return out, ok


def grid_step(xyz, valid):
    both = valid[:, 1:] & valid[:, :-1]
    d = np.linalg.norm(xyz[:, 1:] - xyz[:, :-1], axis=-1)[both]
    return float(np.median(d)) if d.size else float("nan")


def tile_seeds(xyz, valid, spacing_vox):
    step = grid_step(xyz, valid)
    s = max(1, int(round(spacing_vox / step)))
    rows = np.arange(s // 2, xyz.shape[0], s)
    cols = np.arange(s // 2, xyz.shape[1], s)
    pts = xyz[valid]
    seeds = []
    for i in rows:
        for j in cols:
            if not valid[i, j]:
                continue
            p = xyz[i, j]
            near = pts[np.linalg.norm(pts - p, axis=1) <= NEIGHBOR_RADIUS]
            if len(near) < 6:
                continue
            centered = near - near.mean(0)
            _, vecs = np.linalg.eigh(centered.T @ centered)
            normal = vecs[:, 0]
            # column direction of the grid as the u tangent
            jj = j + 1 if j + 1 < xyz.shape[1] and valid[i, j + 1] else j - 1
            if jj < 0 or not valid[i, jj]:
                continue
            u = (xyz[i, jj] - p) * (1 if jj > j else -1)
            u = u - np.dot(u, normal) * normal
            if np.linalg.norm(u) < 1e-6:
                continue
            u /= np.linalg.norm(u)
            seeds.append(dict(grid_ij=[int(i), int(j)], seed_xyz=p.tolist(), normal_xyz=normal.tolist(),
                              tangent_u_xyz=u.tolist(), tangent_v_xyz=np.cross(normal, u).tolist()))
    return seeds, s, step


# ----------------------------------------------------------------------------- per-tile work
_VOLUME = None


def _init_worker(volume_path):
    global _VOLUME
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    _VOLUME = np.load(volume_path, mmap_mode="r")


def tile_shift(task):
    """Return the validated blend's shift field for one tile, or the reason it abstained."""
    site, tri = task["site"], task["triangles"]
    stock = project(tri, site)
    with TemporaryDirectory() as tmp:
        try:
            info = solve(_VOLUME, site, Path(tmp))
        except ValueError as exc:
            if "cue voxels" in str(exc) or "No " in str(exc):
                return dict(index=task["index"], abstained=str(exc))
            raise
        with np.load(Path(tmp) / "graph.npz") as g:
            gtri, _ = faces(g["xyz"], g["valid"])
    graph = project(gtri, site)
    unique_s = np.isfinite(stock.low) & (stock.high - stock.low <= 1)
    unique_g = np.isfinite(graph.low) & (graph.high - graph.low <= 1)
    both = unique_s & unique_g
    gap = np.zeros(stock.low.shape)
    gap[both] = graph.center[both] - stock.center[both]
    shared = both & (np.abs(gap) <= LIMIT)
    shift = np.where(shared, 0.5 * gap, 0.0)
    return dict(index=task["index"], shift=shift.astype(np.float32), center=np.where(unique_s, stock.center, np.nan).astype(np.float32),
                unique=unique_s, shared=shared, graph_nodes=int(info.get("graph", 0)), anchor_offset=float(info["anchor_offset"]))


def transfer(points, results, sites, mode):
    """Weighted combination of tile shifts on resampled vertices (protocol steps 5-6)."""
    acc = np.zeros_like(points)
    wsum = np.zeros(len(points))
    tiles_used = np.zeros(len(points), np.int32)
    for res in results:
        if "shift" not in res:
            continue
        site = sites[res["index"]]
        F = frame(site)
        seed = np.asarray(site["seed_xyz"], float)
        lo, hi = seed - RADIUS * 1.75, seed + RADIUS * 1.75
        cand = np.flatnonzero(((points >= lo) & (points <= hi)).all(1))
        if not len(cand):
            continue
        loc = (points[cand] - seed) @ F.T
        r = np.hypot(loc[:, 0], loc[:, 1])
        inside = r < RADIUS
        cand, loc, r = cand[inside], loc[inside], r[inside]
        fu, fv = (loc[:, 0] + RADIUS) / STEP, (loc[:, 1] + RADIUS) / STEP
        iu = np.clip(np.rint(fu).astype(int), 0, N_RAYS - 1)
        iv = np.clip(np.rint(fv).astype(int), 0, N_RAYS - 1)
        center = res["center"][iv, iu]
        visible = res["unique"][iv, iu] & np.isfinite(center) & (np.abs(loc[:, 2] - np.nan_to_num(center)) <= VISIBLE_TOL)
        if mode == "shared":
            visible &= res["shared"][iv, iu]
        if not visible.any():
            continue
        cand, fu, fv, r = cand[visible], fu[visible], fv[visible], r[visible]
        s = map_coordinates(res["shift"], [fv, fu], order=1, mode="nearest")
        w = np.cos(np.pi * r / (2 * RADIUS)) ** 2
        acc[cand] += (w * s)[:, None] * F[2]
        wsum[cand] += w
        tiles_used[cand] += 1
    disp = np.zeros_like(points)
    has = wsum > 0
    disp[has] = acc[has] / wsum[has, None]
    return disp, tiles_used


def solve_tiles(xyz, valid, volume_path, origin_xyz, spacing_vox, workers=1):
    sites, lattice_nodes, step = tile_seeds(xyz, valid, spacing_vox)
    for k, s in enumerate(sites):
        s["id"] = f"tile_{k:04d}"
        s["origin_xyz"] = [float(c) for c in origin_xyz]
    tri_all, _ = faces(xyz, valid)
    centroids = tri_all.mean(1)
    tasks = []
    for k, s in enumerate(sites):
        near = np.linalg.norm(centroids - np.asarray(s["seed_xyz"]), axis=1) <= RADIUS * 1.75 + 2 * step
        tasks.append(dict(index=k, site=s, triangles=tri_all[near]))
    if workers > 1 and len(tasks) > 1:
        with ProcessPoolExecutor(workers, initializer=_init_worker, initargs=(str(volume_path),)) as pool:
            results = list(pool.map(tile_shift, tasks, chunksize=1))
    else:
        _init_worker(str(volume_path))
        results = [tile_shift(t) for t in tasks]
    return sites, results, dict(lattice_nodes=lattice_nodes, grid_step_vox=step)


def tiled_blend(xyz, valid, volume_path, origin_xyz, spacing_vox=32, mode="visible", upsample_k=None, workers=1,
                solved=None):
    """Return the refined resampled mesh, its validity, the displacement field and a report."""
    xyz = np.asarray(xyz, dtype=np.float64)
    if solved is None:
        solved = solve_tiles(xyz, valid, volume_path, origin_xyz, spacing_vox, workers)
    sites, results, lattice = solved
    k = upsample_k or max(1, int(round(lattice["grid_step_vox"] / STEP)))
    up, ok = upsample(xyz, valid, k)
    pts = up[ok]
    disp, used = transfer(pts, results, sites, mode)
    out = up.copy()
    out[ok] = pts + disp
    field = np.zeros(up.shape)
    field[ok] = disp
    moved = np.linalg.norm(disp, axis=1)
    report = dict(mode=mode, spacing_vox=spacing_vox, upsample=k, tiles=len(sites),
                  tiles_solved=sum("shift" in r for r in results),
                  tiles_abstained=sum("abstained" in r for r in results), resampled_vertices=int(ok.sum()),
                  vertices_with_tile=int((used > 0).sum()), vertices_moved=int((moved > 1e-6).sum()),
                  displacement_median_vox=float(np.median(moved[moved > 1e-6])) if (moved > 1e-6).any() else 0.0,
                  displacement_p95_vox=float(np.percentile(moved[moved > 1e-6], 95)) if (moved > 1e-6).any() else 0.0,
                  **lattice)
    return out, ok, field, report


# ----------------------------------------------------------------------------- CLI
def load_tifxyz(folder):
    xyz = np.stack([tifffile.imread(Path(folder) / f"{a}.tif") for a in "xyz"], axis=-1).astype(np.float64)
    return xyz, grid_valid(xyz)


def write_tifxyz(folder, xyz, ok, scale, extra):
    folder.mkdir(parents=True, exist_ok=False)
    arr = np.where(ok[..., None], xyz, -1.0).astype(np.float32)
    for k, a in enumerate("xyz"):
        tifffile.imwrite(folder / f"{a}.tif", arr[..., k], compression="deflate")
    meta = dict(type="seg", format="tifxyz", uuid=folder.name, name=folder.name, scale=[scale, scale],
                bbox=[arr[ok].min(0).tolist(), arr[ok].max(0).tolist()], source="growpatch tiled local blend", **extra)
    (folder / "meta.json").write_text(json.dumps(meta, indent=1), encoding="utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("stock", type=Path, help="GrowPatch tifxyz folder")
    p.add_argument("foreground", type=Path, help="ZYX surface-prediction cube (.npy) covering the mesh")
    p.add_argument("output", type=Path, help="New output folder")
    p.add_argument("--origin", type=float, nargs=3, required=True, metavar=("X", "Y", "Z"),
                   help="Global XYZ of the cube's first voxel")
    p.add_argument("--stock-local", action="store_true", help="Mesh coordinates are relative to the cube origin")
    p.add_argument("--spacing", type=float, default=None, help="Tile lattice spacing in voxels (default: locked value)")
    p.add_argument("--mode", choices=("visible", "shared"), default=None, help="Tile combination (default: locked value)")
    p.add_argument("--workers", type=int, default=max(1, (os.cpu_count() or 2) - 2))
    a = p.parse_args()
    lock = Path(__file__).with_name("tiled_blend_lock.json")
    locked = json.loads(lock.read_text()) if lock.exists() else {}
    spacing = a.spacing or locked.get("spacing_vox", 32)
    mode = a.mode or locked.get("mode", "visible")
    xyz, valid = load_tifxyz(a.stock)
    if a.stock_local:
        xyz[valid] += np.asarray(a.origin)
    started = time.time()
    out, ok, field, report = tiled_blend(xyz, valid, a.foreground, a.origin, spacing, mode, workers=a.workers)
    meta_in = json.loads((a.stock / "meta.json").read_text()) if (a.stock / "meta.json").exists() else {}
    scale = float(meta_in.get("scale", [1 / report["grid_step_vox"]])[0]) * report["upsample"]
    write_tifxyz(a.output / "tiled-blend.tifxyz", out, ok, scale, dict(coordinate_units="global XYZ in input voxels"))
    np.save(a.output / "displacement_xyz.npy", field.astype(np.float32))
    report.update(elapsed_seconds=round(time.time() - started, 1), stock=str(a.stock.name),
                  foreground_sha256=hashlib.file_digest(open(a.foreground, "rb"), "sha256").hexdigest(),
                  script_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest())
    (a.output / "report.json").write_text(json.dumps(report, indent=1), encoding="utf-8")
    print(json.dumps(report, indent=1))


if __name__ == "__main__":
    main()
