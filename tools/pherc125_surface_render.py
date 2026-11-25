"""Render the PHerc. 125 scan along the whole refined GrowPatch mesh, and along the original GrowPatch mesh.

    python tools/pherc125_surface_render.py --work ../render-work

Inputs come from the repository: the official GrowPatch run in examples/pherc125-blend-demo/stock-growpatch and the
tiled-blend output in examples/pherc125-blend-demo/whole-mesh-output. Only the part of the mesh inside the
prediction cube is rendered, because that is the part the blend could refine (41% of the mesh). The scan is read
with the same sampler as `reconstruction/review.py render`, straight from the public Vesuvius Challenge bucket:
17 layers along the surface normal, one voxel apart, on the mesh's own 2-voxel grid.

It writes <work>/pherc125-tiled/{tiled,stock}/ct17.npy and figures/pherc125_tiled_surface.jpg.
"""
import argparse
import json
import os
import sys
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
DEMO = REPO / "examples" / "pherc125-blend-demo"
sys.path.insert(0, str(REPO / "growpatch-blend" / "scripts"))
URL = "https://vesuvius-challenge-open-data.s3.us-east-1.amazonaws.com/PHerc0125/volumes/20250821151825-9.362um-1.2m-113keV-masked.zarr/0"
PITCH = 9.362


def normals(xyz, ok):
    """Unit normals from grid differences; zero where a neighbor is missing."""
    du = np.zeros_like(xyz)
    dv = np.zeros_like(xyz)
    du[:, 1:-1] = xyz[:, 2:] - xyz[:, :-2]
    dv[1:-1] = xyz[2:] - xyz[:-2]
    good = ok.copy()
    good[:, 1:-1] &= ok[:, 2:] & ok[:, :-2]
    good[1:-1] &= ok[2:] & ok[:-2]
    good[[0, -1], :] = False
    good[:, [0, -1]] = False
    n = np.cross(du, dv)
    length = np.linalg.norm(n, axis=-1)
    good &= length > 1e-6
    n /= np.maximum(length[..., None], 1e-12)
    return n, good


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--work", type=Path, required=True)
    ap.add_argument("--figure-only", action="store_true", help="redraw the figure from an earlier run")
    a = ap.parse_args()
    if a.figure_only:
        return figure({m: a.work / "pherc125-tiled" / m for m in ("tiled", "stock")}, a.work)
    os.environ["CROSS_SECTIONS_WORK"] = str(a.work.resolve())
    os.environ.setdefault("CROSS_SECTIONS_RAW_CACHE", str((a.work / "raw-cache").resolve()))
    os.environ["FIRST_LETTERS_GRID_CASE"] = "PHerc0125"
    sys.path.insert(0, str(REPO / "reconstruction" / "original" / "src"))
    from first_letters_native_sample import sample_native
    from growbench_tiled_blend import load_tifxyz, upsample

    cube = json.loads((DEMO / "m7_cube.json").read_text(encoding="utf-8"))
    lo, hi = (np.asarray(b, float) for b in cube["global_xyz_half_open"])
    tiled, tok = load_tifxyz(DEMO / "whole-mesh-output" / "tiled-blend.tifxyz")
    stock, sok = load_tifxyz(DEMO / "stock-growpatch")
    stock[sok] += lo                                  # the stock mesh is stored relative to the cube origin
    report = json.loads((DEMO / "whole-mesh-output" / "report.json").read_text(encoding="utf-8"))
    stock, sok = upsample(stock, sok, report["upsample"])
    assert stock.shape == tiled.shape, (stock.shape, tiled.shape)
    inside = sok & tok & ((stock >= lo) & (stock < hi)).all(-1)
    rows, cols = np.nonzero(inside)
    crop = np.s_[rows.min():rows.max() + 1, cols.min():cols.max() + 1]
    out = {}
    for name, xyz, ok in (("tiled", tiled, tok), ("stock", stock, sok)):
        g = xyz[crop]
        n, good = normals(g, ok[crop])
        good &= inside[crop]
        dest = a.work / "pherc125-tiled" / name
        sample_native(dest, URL, g, n, good, PITCH, {"field": f"pherc125-{name}"})
        out[name] = dest
    figure(out, a.work)


def figure(out, work):
    from PIL import Image, ImageDraw, ImageFont
    import matplotlib
    font_dir = Path(matplotlib.get_data_path()) / "fonts" / "ttf"
    f1 = ImageFont.truetype(str(font_dir / "DejaVuSans-Bold.ttf"), 22)
    f2 = ImageFont.truetype(str(font_dir / "DejaVuSans.ttf"), 15)
    views = {}
    for name, dest in out.items():
        ct = np.load(dest / "ct17.npy").astype(np.float32)
        mask = np.load(dest / "mask.npy")
        img = ct[7:10].mean(0)                        # the surface and one voxel either side
        views[name] = (img, mask)
    img, mask = views["tiled"]
    vlo, vhi = np.percentile(img[mask], (1, 99.5))
    g = np.uint8(np.clip((img - vlo) / (vhi - vlo), 0, 1) * 255)
    bg = (24, 22, 19)
    rgb = np.where(mask[..., None], np.repeat(g[..., None], 3, -1), np.array(bg, np.uint8))
    # One mesh-grid cell is 2 voxels (18.7 um). Show it at 1 px per cell, enlarged 2x for viewing.
    surf = Image.fromarray(rgb).resize((rgb.shape[1] * 2, rgb.shape[0] * 2), Image.LANCZOS)
    pad, top, bottom = 24, 86, 52
    W = surf.width + 2 * pad
    canvas = Image.new("RGB", (W, surf.height + top + bottom), bg)
    canvas.paste(surf, (pad, top))
    d = ImageDraw.Draw(canvas)
    d.text((pad, 20), "PHerc. 125: the scan read along the refined GrowPatch surface", font=f1, fill=(240, 236, 226))
    d.text((pad, 52), "This is the 41% of the mesh inside the prediction cube, the part the blend could refine. About 14 x 6 mm.",
           font=f2, fill=(184, 179, 168))
    mm = 1000 / PITCH / 2 * 2                         # pixels per mm at 2 px per 2-voxel cell
    x1 = pad + surf.width
    d.rectangle([x1 - mm, top + surf.height + 18, x1, top + surf.height + 22], fill=(240, 236, 226))
    d.text((x1 - mm - 50, top + surf.height + 12), "1 mm", font=f2, fill=(240, 236, 226))
    d.text((pad, top + surf.height + 14), "Bright, fibrous texture is papyrus. Mean of the surface layer and one voxel either side.",
           font=f2, fill=(184, 179, 168))
    path = REPO / "figures" / "pherc125_tiled_surface.jpg"
    canvas.save(path, quality=88, optimize=True)
    print(f"wrote {path.relative_to(REPO)} ({canvas.width} x {canvas.height})")


if __name__ == "__main__":
    main()
