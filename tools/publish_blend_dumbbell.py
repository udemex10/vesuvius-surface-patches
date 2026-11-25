"""Per-region dumbbell figure: official GrowPatch vs GrowPatch + 8-voxel local blend at three contour cutoffs.

Reads the frozen regional comparison records in evidence/held-out-regions and writes
figures/blend_vs_growpatch.png in the publication repository. Values are seed-connected area on the
manually labeled target sheet, in mm^2, for the 16 held-out PHerc. Paris. 4 regions.
"""
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
REG = ROOT / "evidence/held-out-regions"
OUT = ROOT / "figures/blend_vs_growpatch.png"

RUNS = [
    ("cutoff1_regional.json", "Cutoff 1", "Most stock area · blend chosen on dev · fails wrong-sheet gate"),
    ("cutoff128_regional.json", "Cutoff 128", "Locked threshold-128 comparison"),
    ("cutoff192_regional.json", "Cutoff 192", "Post-hoc sensitivity · dev chose another rule (+1.0%)"),
]
SURFACE, INK, MUTED, GRID = "#fcfcfb", "#0b0b0b", "#52514e", "#e4e2dc"
STOCK, BLEND = "#a9c4e8", "#2a78d6"

plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 10})
fig, axes = plt.subplots(1, 3, figsize=(14, 7.2), sharey=True, facecolor=SURFACE)
fig.subplots_adjust(left=0.075, right=0.985, top=0.75, bottom=0.14, wspace=0.12)
for ax, (name, title, role) in zip(axes, RUNS):
    d = json.loads((REG / name).read_text())
    rows, comp = d["rows"], d["comparisons"]["blend_8"]
    for i, s in enumerate(sorted(rows["stock"])):
        y = 16 - i
        a = rows["stock"][s]["connected"] * 100
        b = rows["blend_8"][s]["connected"] * 100
        ax.plot([a, b], [y, y], color=STOCK, lw=2.5, solid_capstyle="round", zorder=1)
        ax.scatter([a], [y], s=50, color=STOCK, edgecolor=SURFACE, linewidth=1.5, zorder=2)
        ax.scatter([b], [y], s=50, color=BLEND, edgecolor=SURFACE, linewidth=1.5, zorder=3)
    ax.text(0, 1.165, f"GrowPatch {title.lower()}", transform=ax.transAxes, fontsize=12.5, fontweight="bold", color=INK)
    ax.text(0, 1.115, role, transform=ax.transAxes, fontsize=9, color=MUTED, style="italic")
    ax.text(0, 1.015,
            f"{comp['connected_gain_percent']:+.1f}% area · {comp['improved']}/{comp['same']}/{comp['worse']} better/tie/worse\n"
            f"strict error {comp['stock']['risk']*100:.1f}% → {comp['candidate']['risk']*100:.1f}%",
            transform=ax.transAxes, fontsize=9.5, color=INK, linespacing=1.35)
    ax.set_facecolor(SURFACE)
    ax.grid(axis="x", color=GRID, lw=0.8)
    ax.set_axisbelow(True)
    for sp in ("top", "right", "left"):
        ax.spines[sp].set_visible(False)
    ax.spines["bottom"].set_color(MUTED)
    ax.tick_params(colors=MUTED, length=0)
    ax.set_xlim(-0.05, 1.9)
    ax.set_xlabel("Connected area on the labeled sheet (mm²)", color=MUTED)
axes[0].set_yticks(range(16, 0, -1), [f"region {i:02d}" for i in range(1, 17)])
h1 = axes[0].scatter([], [], s=50, color=STOCK, label="Official GrowPatch (Villa e875441)")
h2 = axes[0].scatter([], [], s=50, color=BLEND, label="GrowPatch + 8-voxel local blend")
fig.legend(handles=[h1, h2], loc="lower center", ncol=2, frameon=False, fontsize=10.5, bbox_to_anchor=(0.5, 0.0))
fig.text(0.01, 0.955, "A local blend adds connected correct-sheet area to official GrowPatch in 15 of 16 held-out regions",
         fontsize=14, fontweight="bold", color=INK)
fig.text(0.01, 0.915, "16 held-out PHerc. Paris. 4 regions scored against manual sheet IDs. No region lost connected correct-sheet area. "
         "A dot at 0 means the seed's sheet was never connected by either method.", fontsize=9.5, color=MUTED)
fig.savefig(OUT, dpi=150, facecolor=SURFACE)
print(OUT)
