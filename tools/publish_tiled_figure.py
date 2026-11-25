"""Held-out figure for the whole-mesh tiled blend: area by method and where moved surface went.

Reads evidence/tiled/heldout_manual_external.json and evidence/tiled/transitions_manual_external.json,
writes figures/tiled_blend_heldout.png.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
EV = ROOT / "evidence/tiled"
OUT = ROOT / "figures/tiled_blend_heldout.png"
PRESETS = [("oriented8_loss", "Cutoff 1"), ("oriented8_th128_loss", "Cutoff 128"), ("oriented8_th192_loss", "Cutoff 192")]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
SHADES = {"stock": "#c9d9ef", "single": "#7fa9e0", "tiled": "#2a78d6"}
TARGET, OTHER = "#2a78d6", "#eb6834"

held = json.loads((EV / "heldout_manual_external.json").read_text())["summary"]
trans = json.loads((EV / "transitions_manual_external.json").read_text())["report"]

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, (a, b) = plt.subplots(1, 2, figsize=(14, 6.2), facecolor=SURFACE, gridspec_kw=dict(width_ratios=[1.15, 1]))
fig.subplots_adjust(left=0.09, right=0.98, top=0.78, bottom=0.19, wspace=0.32)

# Panel A: connected correct area within 128 voxels, pooled over 16 regions.
rows = []
for key, name in PRESETS:
    z = held[key]
    stock = z["tiled_32_shared"]["r128"]["base"]["connected"] * 100
    single = z["single_seed_blend"]["r128"]["candidate"]["connected"] * 100
    tiled = z["tiled_32_shared"]["r128"]["candidate"]["connected"] * 100
    gain = z["tiled_32_shared"]["r128"]["connected_gain_percent"]
    rows.append((name, stock, single, tiled, gain, z["tiled_32_shared"]["r128"]))
y = 0
ticks, labels = [], []
for name, stock, single, tiled, gain, c in rows:
    for k, (lab, val, col) in enumerate((("Official GrowPatch", stock, SHADES["stock"]),
                                          ("Single-seed blend", single, SHADES["single"]),
                                          ("Whole-mesh tiled blend", tiled, SHADES["tiled"]))):
        a.barh(y - k * 0.27, val, height=0.24, color=col, edgecolor=SURFACE, linewidth=1)
        a.text(val + 0.4, y - k * 0.27, f"{val:.1f}", va="center", fontsize=8.5, color=INK)
    a.text(max(stock, single, tiled) + 4.2, y - 0.27, f"{gain:+.1f}%\n{c['better']}/{c['tie']}/{c['worse']}",
           va="center", fontsize=9.5, color=INK, fontweight="bold", linespacing=1.3)
    ticks.append(y - 0.27)
    labels.append(name)
    y -= 1.25
a.set_yticks(ticks, labels)
a.set_xlim(0, 48)
a.set_xlabel("Seed-connected area on the labeled sheet, within 128 voxels (mm², 16 regions)", color=MUTED)
a.set_title("A · Correct-sheet area recovered", loc="left", fontsize=12, fontweight="bold", color=INK, pad=26)
a.text(0, 1.02, "Bold: tiled vs official GrowPatch, regions better/tie/worse", transform=a.transAxes, fontsize=9, color=MUTED)
handles = [plt.Rectangle((0, 0), 1, 1, color=SHADES[k]) for k in ("stock", "single", "tiled")]
a.legend(handles, ["Official GrowPatch", "Single-seed blend", "Whole-mesh tiled blend"], loc="upper center",
         bbox_to_anchor=(0.5, -0.12), ncol=3, frameon=False, fontsize=9)

# Panel B: where surface that changed label went (stock -> tiled), pooled mm2.
y = 0
ticks = []
for key, name in PRESETS:
    m = trans[key]["transition_mm2"]  # rows: from; cols: to (target, other, unassigned, ambiguous, absent)
    u2t, u2o, t2o = m[2][0], m[2][1], m[0][1]
    b.barh(y, u2t, height=0.5, color=TARGET, edgecolor=SURFACE, linewidth=2)
    b.barh(y, u2o, left=u2t, height=0.5, color=OTHER, edgecolor=SURFACE, linewidth=2)
    share = 100 * u2t / (u2t + u2o)
    b.text(u2t / 2, y, f"{u2t:.2f}", ha="center", va="center", color="white", fontsize=9, fontweight="bold")
    b.text(u2t + u2o + 0.08, y, f"{u2o:.2f}  ({share:.0f}% to target)", va="center", fontsize=9, color=INK)
    b.text(0, y - 0.42, f"target → other sheet: {t2o:.3f} mm²", fontsize=8.5, color=MUTED, va="center")
    ticks.append(y)
    y -= 1.25
b.set_yticks(ticks, [n for _, n in PRESETS])
b.set_xlim(0, 8.2)
b.set_ylim(-3.25, 0.55)
b.set_xlabel("Area that moved off unlabeled voxels (mm², 16 regions, 12–128 voxels)", color=MUTED)
b.set_title("B · Where the moved surface landed", loc="left", fontsize=12, fontweight="bold", color=INK, pad=26)
b.text(0, 1.02, "Unlabeled (ID 0) → target sheet (blue) or a different named sheet (orange)", transform=b.transAxes, fontsize=9, color=MUTED)

for ax in (a, b):
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=INK, length=0)
fig.text(0.01, 0.955, "Whole-mesh tiling: more correct-sheet area on 16 held-out regions, mostly by snapping unlabeled surface onto the target",
         fontsize=13.5, fontweight="bold", color=INK)
fig.text(0.01, 0.915, "Locked setting (32-voxel tiles, shared-only), chosen on 16 older development regions. Some unlabeled surface also lands on a neighboring "
         "sheet, which raises the different-sheet share.", fontsize=9.5, color=MUTED)
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print(OUT)
