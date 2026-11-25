"""Write the data for the held-out region explorer (docs/index.html).

    python tools/explorer_meshes.py --data <project>/results/growbench --out <work>/explorer-meshes
    python tools/build_explorer.py --data <project>/results/growbench --meshes <work>/explorer-meshes

For each of the 16 held-out PHerc. Paris. 4 regions it writes, under docs/data/<region>/:

- ct.webp: 65 CT cross-sections through the seed frame (planes v = -128, -124, ..., 128; u across, the seed normal
  up), stacked vertically, transparent outside the cube. Contrast is stretched per region between the 1st and
  99.7th CT percentiles of the cube.
- labels.png: the manual instance labels on the same planes. Aqua = the region's target sheet, orange = any other
  labeled sheet. Interiors are faint and boundaries strong.
- traces.js: where the official GrowPatch mesh and the tiled blend cross each plane, for the three cutoffs
  (marching squares on the mesh grid, simplified to 0.2 voxels).
- chart_<cutoff>_<mesh>.png: the scorer's ray classes on the 2-voxel chart within 128 voxels, computed exactly as
  `growbench_tiled_eval.wide_measure` does. The builder checks the connected area against the published JSON.
- diff_<cutoff>.png: what the blend changed on the same chart. Bright aqua = newly counted target sheet, faint aqua
  = counted before and after, bone = counted before but not after, bright orange = newly on a wrong sheet, faint
  orange = on a wrong sheet before and after.
- tex_<cutoff>_<mesh>.webp and tex_ref.webp: CT sampled on each surface along the seed-normal rays (1-voxel grid,
  a [1, 2, 1] / 4 average across the surface), and on the labeled target-sheet surface.

docs/data/manifest.js holds the per-region scores from evidence/tiled/heldout_manual_external.json.
"""
import argparse
import json
import math
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "growpatch-blend" / "scripts"))

PRESETS = {"oriented8_loss": "cut1", "oriented8_th128_loss": "cut128", "oriented8_th192_loss": "cut192"}
COHORT = "manual_external"
WIDE = 128
SLICES = list(range(-WIDE, WIDE + 1, 4))
PITCH = 7.91
TARGET_RGB = (0x19, 0x9E, 0x70)   # aqua, palette slot 3
OTHER_RGB = (0xD9, 0x59, 0x26)    # orange, palette slot 2
CLASS_RGBA = {                     # chart classes (see classes())
    1: (0x2A, 0x29, 0x26, 255),   # counted ray, no surface
    2: (0x55, 0x53, 0x4E, 255),   # ambiguous (surface folds back over the ray)
    3: (0x8E, 0x8C, 0x85, 255),   # unlabeled voxel
    4: OTHER_RGB + (255,),        # other labeled sheet
    5: TARGET_RGB + (110,),       # target sheet, not connected to the seed
    6: TARGET_RGB + (255,),       # target sheet, seed-connected (the scored area)
}


def local(xyz, site, F):
    return (np.asarray(xyz, float) - site["seed_xyz"]) @ F.T


def stretch(ct):
    lo, hi = np.percentile(ct, [1, 99.7])
    return float(lo), float(hi)


def to_u8(a, lo, hi):
    return np.clip((a - lo) / (hi - lo) * 255, 0, 255).astype(np.uint8)


def contour_traces(xyz, valid, site, F, v0, wmax):
    """Polylines where the mesh grid crosses the plane v = v0, in slice pixel coordinates (x = u + 128, y = wmax - w)."""
    from skimage.measure import approximate_polygon, find_contours
    loc = np.full(xyz.shape, np.nan)
    loc[valid] = local(xyz[valid], site, F)
    f = np.where(valid, loc[..., 1] - v0, 0.0)
    out = []
    for c in find_contours(f, 0.0, mask=valid):
        pts = []
        for r, col in c:
            r0, c0 = int(math.floor(r + 1e-9)), int(math.floor(col + 1e-9))
            fr, fc = r - r0, col - c0
            if fr > 1e-6:      # on a vertical grid edge
                a, b, t = loc[r0, c0], loc[min(r0 + 1, f.shape[0] - 1), c0], fr
            else:
                a, b, t = loc[r0, c0], loc[r0, min(c0 + 1, f.shape[1] - 1)], fc
            p = a + (b - a) * t
            pts.append((p[0] + WIDE, wmax - p[2]))
        pts = np.asarray(pts)
        if len(pts) < 2 or not np.isfinite(pts).all():
            continue
        pts = approximate_polygon(pts, 0.2)
        out.append([int(round(x * 10)) for x in pts.ravel()])
    return out


def classes(site, F, ref, p, target, volume):
    """Ray classes exactly as growbench_tiled_eval.wide_measure computes them (r128)."""
    from scipy.ndimage import label
    axis = np.arange(-WIDE, WIDE + 1, 2)
    u, v = np.meshgrid(axis, axis)
    rr = u * u + v * v
    known = np.isfinite(ref.low) & (ref.high - ref.low <= .5) & (ref.area_factor <= 2) & (rr <= WIDE ** 2)
    a = np.where(known, ref.area_factor, 0) * (2 * PITCH / 10000) ** 2
    use = known & np.isfinite(p.low)
    unique = use & (p.high - p.low <= 1)
    xyz = np.stack((u[unique], v[unique], p.center[unique]), -1) @ F
    pt = np.floor(xyz + np.asarray(site["seed_xyz"]) - site["origin_xyz"] + .5).astype(int)[:, ::-1]
    inside = ((pt >= 0) & (pt < volume.shape)).all(1)
    val = np.zeros(len(pt), np.int32)
    val[inside] = volume[tuple(pt[inside].T)]
    ids = np.zeros(known.shape, np.int32)
    ids[unique] = val
    correct = unique & (ids == target)
    other = unique & (ids != 0) & (ids != target)
    labs, _ = label(correct)
    roots = np.unique(labs[correct & (rr <= 4 ** 2)])
    connected = correct & np.isin(labs, roots[roots > 0])
    a[rr <= 12 ** 2] = 0
    cls = np.zeros(known.shape, np.uint8)
    cls[known] = 1
    cls[use & ~unique] = 2
    cls[unique & ~correct & ~other] = 3
    cls[other] = 4
    cls[correct] = 5
    cls[connected] = 6
    return cls, dict(connected=float(a[connected].sum()), known_other=float(a[other].sum()))


DIFF_RGBA = {
    1: (0x2A, 0x29, 0x26, 255),   # scored ray, nothing notable
    2: TARGET_RGB + (92,),        # counted before and after
    3: TARGET_RGB + (255,),       # newly counted
    4: (0xE8, 0xE2, 0xD4, 255),   # counted before, not after
    5: OTHER_RGB + (255,),        # newly on a wrong sheet
    6: OTHER_RGB + (92,),         # on a wrong sheet before and after
}


def diff_classes(a, b):
    d = np.zeros(a.shape, np.uint8)
    d[a > 0] = 1
    d[(a == 6) & (b == 6)] = 2
    d[(b == 6) & (a != 6)] = 3
    d[(a == 6) & (b != 6)] = 4
    d[(b == 4) & (a == 4)] = 6
    d[(b == 4) & (a != 4)] = 5
    img = np.zeros(d.shape + (4,), np.uint8)
    for k, c in DIFF_RGBA.items():
        img[d == k] = c
    return img[::-1]


def up2(a):
    n = a.shape[0]
    out = np.full((2 * n - 1, 2 * n - 1), np.nan)
    out[::2, ::2] = a
    out[1::2, ::2] = (a[:-1] + a[1:]) / 2
    out[::2, 1::2] = (a[:, :-1] + a[:, 1:]) / 2
    out[1::2, 1::2] = (a[:-1, :-1] + a[1:, :-1] + a[:-1, 1:] + a[1:, 1:]) / 4
    return out


def texture(ct, site, F, p, lo, hi):
    """CT on the surface along the seed-normal rays, 1-voxel grid, rows = v from +128 down, cols = u."""
    from scipy.ndimage import map_coordinates
    w = np.where(np.isfinite(p.low) & (p.high - p.low <= 1), p.center, np.nan)
    w = up2(w)
    axis = np.arange(-WIDE, WIDE + 1)
    u, v = np.meshgrid(axis, axis)
    ok = np.isfinite(w)
    acc = np.zeros(w.shape)
    for d, wt in ((-1, .25), (0, .5), (1, .25)):
        P = np.asarray(site["seed_xyz"]) + u[..., None] * F[0] + v[..., None] * F[1] + np.where(ok, w + d, 0)[..., None] * F[2]
        idx = (P - site["origin_xyz"])[..., ::-1].reshape(-1, 3).T
        acc += wt * map_coordinates(ct, idx, order=1, mode="constant", cval=np.nan).reshape(w.shape)
    ok &= np.isfinite(acc)
    rgba = np.zeros(w.shape + (4,), np.uint8)
    g = to_u8(np.nan_to_num(acc), lo, hi)
    rgba[..., 0] = rgba[..., 1] = rgba[..., 2] = g
    rgba[..., 3] = np.where(ok, 255, 0)
    return rgba[::-1]


def rgba_classes(cls):
    img = np.zeros(cls.shape + (4,), np.uint8)
    for k, c in CLASS_RGBA.items():
        img[cls == k] = c
    return img[::-1]


def build_region(job):
    from PIL import Image
    from scipy.ndimage import map_coordinates
    from growbench_geometry import faces, frame
    from growbench_regional import region_project

    data, meshes, out, site, target = job
    i = site["id"]
    lab = Path(data) / "labels" / COHORT
    F = frame(site)
    ct = np.load(lab / "inputs" / i / "ct.npy").astype(np.float32)
    ids = np.load(lab / "truth" / i / "instance_ids.npy")
    lo, hi = stretch(ct)
    dest = Path(out) / i
    dest.mkdir(parents=True, exist_ok=True)

    surfaces = {}
    for preset, key in PRESETS.items():
        with np.load(Path(meshes) / preset / f"{i}.npz") as d:
            surfaces[key] = {"stock": (d["stock_xyz"], d["stock_valid"].astype(bool)),
                             "tiled": (d["tiled_xyz"], d["tiled_valid"].astype(bool))}
    # Cross-section window: every mesh vertex inside the 128-voxel square, plus a margin.
    ws = []
    for per in surfaces.values():
        for xyz, ok in per.values():
            q = local(xyz[ok], site, F)
            m = (np.abs(q[:, 0]) <= WIDE) & (np.abs(q[:, 1]) <= WIDE)
            ws.append(q[m, 2])
    ws = np.concatenate(ws)
    wmax = int(min(WIDE, 4 * math.ceil((ws.max() + 20) / 4)))
    wmin = int(max(-WIDE, 4 * math.floor((ws.min() - 20) / 4)))
    H = wmax - wmin + 1

    U, W = np.meshgrid(np.arange(-WIDE, WIDE + 1), np.arange(wmax, wmin - 1, -1))
    ct_rows, lab_rows = [], []
    for v0 in SLICES:
        P = np.asarray(site["seed_xyz"]) + U[..., None] * F[0] + v0 * F[1] + W[..., None] * F[2]
        idx = (P - site["origin_xyz"])[..., ::-1].reshape(-1, 3).T
        img = map_coordinates(ct, idx, order=1, mode="constant", cval=np.nan).reshape(U.shape)
        rgba = np.zeros(U.shape + (4,), np.uint8)
        rgba[..., :3] = to_u8(np.nan_to_num(img), lo, hi)[..., None]
        rgba[..., 3] = np.where(np.isfinite(img), 255, 0)
        ct_rows.append(rgba)
        lid = map_coordinates(ids, idx, order=0, mode="constant", cval=0).reshape(U.shape).astype(np.int64)
        cls = np.where(lid == target, 1, np.where(lid > 0, 2, 0))
        edge = np.zeros(cls.shape, bool)
        edge[1:] |= cls[1:] != cls[:-1]
        edge[:-1] |= cls[:-1] != cls[1:]
        edge[:, 1:] |= cls[:, 1:] != cls[:, :-1]
        edge[:, :-1] |= cls[:, :-1] != cls[:, 1:]
        rgba = np.zeros(cls.shape + (4,), np.uint8)
        for k, c, inner, rim in ((1, TARGET_RGB, 72, 240), (2, OTHER_RGB, 26, 150)):
            m = cls == k
            rgba[m, :3] = c
            rgba[m, 3] = np.where(edge[m], rim, inner)
        lab_rows.append(rgba)
    Image.fromarray(np.concatenate(ct_rows), "RGBA").save(dest / "ct.webp", quality=84, method=6)
    Image.fromarray(np.concatenate(lab_rows), "RGBA").save(dest / "labels.png", optimize=True)

    ref_tri = np.load(lab / "truth" / i / "triangles.npy")
    ref = region_project(ref_tri, site, WIDE)
    Image.fromarray(texture(ct, site, F, ref, lo, hi), "RGBA").save(dest / "tex_ref.webp", quality=86, method=6)
    traces, checks = {}, {}
    for key, per in surfaces.items():
        traces[key] = {}
        cls_by = {}
        for mesh, (xyz, ok) in per.items():
            traces[key][mesh] = [contour_traces(xyz, ok, site, F, v0, wmax) for v0 in SLICES]
            p = region_project(faces(xyz, ok)[0], site, WIDE)
            cls, areas = classes(site, F, ref, p, target, ids)
            cls_by[mesh] = cls
            checks[f"{key}/{mesh}"] = areas
            Image.fromarray(rgba_classes(cls), "RGBA").save(dest / f"chart_{key}_{mesh}.png", optimize=True)
            Image.fromarray(texture(ct, site, F, p, lo, hi), "RGBA").save(dest / f"tex_{key}_{mesh}.webp", quality=86, method=6)
        Image.fromarray(diff_classes(cls_by["stock"], cls_by["tiled"]), "RGBA").save(dest / f"diff_{key}.png", optimize=True)
    payload = dict(id=i, wmax=wmax, wmin=wmin, height=H, width=2 * WIDE + 1, slices=SLICES, traces=traces)
    (dest / "traces.js").write_text(f"window.__explorerRegion({json.dumps(payload, separators=(',', ':'))});\n", encoding="utf-8")
    return i, checks, dict(wmax=wmax, wmin=wmin, ct_stretch=[lo, hi])


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", required=True)
    ap.add_argument("--meshes", required=True)
    ap.add_argument("--out", default=str(REPO / "docs" / "data"))
    ap.add_argument("--workers", type=int, default=16)
    a = ap.parse_args()
    lab = Path(a.data) / "labels" / COHORT
    seeds = json.loads((lab / "seeds.json").read_text(encoding="utf-8"))["sites"]
    refs = {r["id"]: r for r in json.loads((lab / "refs.json").read_text(encoding="utf-8"))["sites"]}
    published = json.loads((REPO / "evidence" / "tiled" / "heldout_manual_external.json").read_text(encoding="utf-8"))
    lock = json.loads((REPO / "growpatch-blend" / "scripts" / "tiled_blend_lock.json").read_text(encoding="utf-8"))
    tkey = f"tiled_{lock['spacing_vox']}_{lock['mode']}"
    jobs = [(a.data, a.meshes, a.out, s, int(refs[s["id"]]["target_id"])) for s in seeds]
    regions, worst = {}, 0.0
    with ProcessPoolExecutor(a.workers) as pool:
        for i, checks, extra in pool.map(build_region, jobs, chunksize=1):
            regions[i] = extra
            for c in published["cases"]:
                if c["id"] != i:
                    continue
                key = PRESETS[c["preset"]]
                for mesh, row in (("stock", "stock"), ("tiled", tkey)):
                    for m in ("connected", "known_other"):
                        worst = max(worst, abs(checks[f"{key}/{mesh}"][m] - c["rows"][row]["r128"][m]))
            print(f"{i} written", flush=True)
    if worst > 1e-12:
        sys.exit(f"Chart classes do not reproduce the published r128 areas (max diff {worst}).")

    cm = lambda x: round(x * 100, 4)   # cm2 -> mm2
    manifest = dict(source="evidence/tiled/heldout_manual_external.json", lock=dict(spacing_vox=lock["spacing_vox"], mode=lock["mode"]),
                    slices=SLICES, cutoffs={k: int(k[3:]) for k in PRESETS.values()}, regions=[], pooled={})
    for preset, key in PRESETS.items():
        s = published["summary"][preset][tkey]
        manifest["pooled"][key] = {z: dict(stock_mm2=cm(s[z]["base"]["connected"]), tiled_mm2=cm(s[z]["candidate"]["connected"]),
                                           gain_percent=round(s[z]["connected_gain_percent"], 2), better=s[z]["better"], tie=s[z]["tie"],
                                           worse=s[z]["worse"], interval_mm2=[cm(x) for x in s[z]["interval_95_cm2"]],
                                           other_share_stock=round(100 * s[z]["base"]["other_share_named"], 2),
                                           other_share_tiled=round(100 * s[z]["candidate"]["other_share_named"], 2),
                                           off_target_stock=round(100 * s[z]["base"]["strict_error"], 1),
                                           off_target_tiled=round(100 * s[z]["candidate"]["strict_error"], 1))
                                   for z in ("r96", "r128", "annulus")}
    for s in seeds:
        i, r = s["id"], refs[s["id"]]
        cube = json.loads((lab / "inputs" / i / "ct.json").read_text(encoding="utf-8"))["url"].rsplit("/", 1)[-1].replace("_volume.nrrd", "")
        entry = dict(id=i, split=s["split"], seed_xyz=s["seed_xyz"], origin_xyz=s["origin_xyz"], cube=cube, target_id=int(r["target_id"]),
                     wmax=regions[i]["wmax"], wmin=regions[i]["wmin"], scores={})
        for c in published["cases"]:
            if c["id"] != i:
                continue
            key = PRESETS[c["preset"]]
            st, ti = c["rows"]["stock"], c["rows"][tkey]
            rep = c["reports"][tkey]
            entry["scores"][key] = dict(
                stock=dict(r128=cm(st["r128"]["connected"]), r96=cm(st["r96"]["connected"]), ring=cm(st["annulus"]["connected"]), other=cm(st["r128"]["known_other"])),
                tiled=dict(r128=cm(ti["r128"]["connected"]), r96=cm(ti["r96"]["connected"]), ring=cm(ti["annulus"]["connected"]), other=cm(ti["r128"]["known_other"])),
                tiles=rep["tiles"], tiles_solved=rep["tiles_solved"], move_median=round(rep["displacement_median_vox"], 2),
                move_p95=round(rep["displacement_p95_vox"], 2))
        manifest["regions"].append(entry)
    Path(a.out).mkdir(parents=True, exist_ok=True)
    (Path(a.out) / "manifest.js").write_text("window.EXPLORER = " + json.dumps(manifest, indent=1) + ";\n", encoding="utf-8")
    print(f"Wrote {len(seeds)} regions to {a.out}. Chart classes reproduce the published r128 areas (max diff {worst}).")


if __name__ == "__main__":
    main()
