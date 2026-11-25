"""Plot individual held-out named-sheet results from the frozen comparison tables."""
from pathlib import Path
import json

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np


def main():
    dest = Path(__file__).resolve().parents[1] / "results/growbench/comparison"
    graph = json.loads((dest / "named_risk/oriented8_loss_external.json").read_text())
    blend = json.loads((dest / "regional/manual_external_oriented8_th128_loss.json").read_text())["rows"]
    sets = [(graph["baselines"], graph["rows"]["graph"]), (blend["stock"], blend["blend_8"])]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.7), constrained_layout=True)
    fig.suptitle("Recovering the manually identified sheet on 16 fresh Paris.4 regions", fontsize=15)
    for ax, (baseline, candidate), title in zip(axes, sets, ("Graph against GrowPatch at cutoff 1", "Local blend against GrowPatch at cutoff 128")):
        labels = sorted(baseline)
        delta = np.asarray([(candidate[i]["connected"]-baseline[i]["connected"])*100 for i in labels])
        ax.axhline(0, color="#333333", linewidth=1)
        ax.bar(range(1,17), delta, color=["#327463" if v>=0 else "#ae5946" for v in delta], width=.74)
        ax.set(xlabel="Fixed region ID", title=title, xticks=[1,4,8,12,16], ylim=(-.65,1.7))
        ax.grid(axis="y", alpha=.18)
        ax.spines[["top","right"]].set_visible(False)
    low,high = min(p.get_height() for a in axes for p in a.patches),max(p.get_height() for a in axes for p in a.patches)
    for ax in axes:
        ax.set_ylim(low-.08,high+.10)
    axes[0].set_ylabel("Change in connected correct-sheet area (mm²)")
    fig.savefig(dest / "prize_comparison.png", dpi=170)
    plt.close(fig)


if __name__ == "__main__":
    main()
