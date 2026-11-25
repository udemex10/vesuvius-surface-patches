"""Map of the tiled blend's displacement over the whole PHerc. 125 GrowPatch mesh (grid view).

Usage: python tools/publish_pherc125_tiled_map.py STOCK_TIFXYZ TILED_TIFXYZ CUBE_SHAPE_ZYX_JSON OUT_PNG
"""
import json, sys
from pathlib import Path
import numpy as np, tifffile
import matplotlib; matplotlib.use("Agg"); import matplotlib.pyplot as plt

stock_dir, tiled_dir, cube_json, out = map(Path, sys.argv[1:5])
cube = json.loads(cube_json.read_text())
origin = np.array(cube["global_xyz_half_open"][0]); end = np.array(cube["global_xyz_half_open"][1])
t = np.stack([tifffile.imread(tiled_dir / f"{a}.tif") for a in "xyz"], -1).astype(float)
s = np.stack([tifffile.imread(stock_dir / f"{a}.tif") for a in "xyz"], -1).astype(float)
ok = (t >= 0).all(-1)
# stock resampled the same way (k=4 bilinear) to measure displacement per vertex
k = 4; h, w, _ = s.shape; sv = (s >= 0).all(-1)
up = np.full(t.shape, np.nan)
full = sv[:-1, :-1] & sv[1:, :-1] & sv[:-1, 1:] & sv[1:, 1:]
I, J = np.nonzero(full)
for a in range(k + 1):
    for b in range(k + 1):
        p, q = a / k, b / k
        up[I*k+a, J*k+b] = (1-p)*(1-q)*s[I, J] + (1-p)*q*s[I, J+1] + p*(1-q)*s[I+1, J] + p*q*s[I+1, J+1]
up += origin
disp = np.linalg.norm(t - up, axis=-1)
inside = ok & ((t >= origin) & (t < end)).all(-1)
img = np.full(ok.shape + (3,), 1.0)
img[ok] = (0.86, 0.86, 0.84)
img[inside] = (0.93, 0.95, 0.98)
fig, ax = plt.subplots(figsize=(8, 8.4), facecolor="#fcfcfb")
ax.imshow(img, interpolation="nearest")
m = np.where(ok & (disp > 1e-3), disp, np.nan)  # TIFF float32 rounding is below 1e-3
im = ax.imshow(m, cmap="Blues", vmin=0, vmax=3, interpolation="nearest")
ax.set_axis_off()
cb = fig.colorbar(im, ax=ax, fraction=0.035, pad=0.02); cb.set_label("Vertex displacement (voxels, 9.362 µm)")
ax.set_title("Tiled blend on a whole official GrowPatch mesh, PHerc. 125", loc="left", fontsize=12, fontweight="bold")
ax.text(0, -0.035, f"Blue: vertices moved by more than 0.001 voxels ({(m > 0).sum():,} of {ok.sum():,}). Pale: inside the prediction cube GrowPatch was given. "
        "Gray: outside it (18 edge vertices moved).", transform=ax.transAxes, fontsize=8.5, color="#52514e", va="top", wrap=True)
fig.savefig(out, dpi=130, facecolor="#fcfcfb", bbox_inches="tight")
print(out, "moved", int((m > 0).sum()), "valid", int(ok.sum()), "inside", int(inside.sum()), "median moved", float(np.nanmedian(m)))
