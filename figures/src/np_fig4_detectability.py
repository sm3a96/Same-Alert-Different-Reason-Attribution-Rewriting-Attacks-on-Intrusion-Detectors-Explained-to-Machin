"""Fig. 4 (paper) -- how far the shown set moved against how well the best single
check sees it, one point per (corpus, attack).

x: top-5 Jaccard distance between the clean and attacked shown sets (Eq. corrupt).
y: AUROC of the best single check for that (corpus, attack), chosen once on the
   seed-clustered mean with the fused score excluded (eval.metrics.best_single_auroc).
Error bars are 95% intervals clustered by seed. Same cells as the checks table.

Reads results/matrix/raw.json. Writes figures/out/fig4_detectability.pdf.
"""
from __future__ import annotations

import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
from np_style import (ATTACK_COLOR, ATTACK_LABEL, COL, DATASET_LABEL, DATASET_MARKER,  # noqa: E402
                      apply_style)
from avert.eval.metrics import best_single_auroc, seed_clustered_ci  # noqa: E402

RAW = REPO / "results" / "matrix" / "raw.json"
OUT = REPO / "figures" / "out" / "fig4_detectability.pdf"


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    rows = json.loads(RAW.read_text())
    best = best_single_auroc(rows)
    cells = defaultdict(lambda: defaultdict(list))
    for r in rows:
        cells[(r["dataset"], r["attack"])]["corrupt"].append(r["corrupt"])
        cells[(r["dataset"], r["attack"])]["seed"].append(r["seed"])

    fig, ax = plt.subplots(figsize=(COL, 2.3))
    ax.axhline(0.5, color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.995, 0.5 + 0.006, "chance", transform=ax.get_yaxis_transform(),  # layout
            ha="right", va="bottom", fontsize=7, color="0.35")
    for (ds, atk), m in sorted(cells.items()):
        x, xe = seed_clustered_ci(np.array(m["corrupt"]), np.array(m["seed"]))
        y, ye = seed_clustered_ci(np.array(best[(ds, atk)]["per_cell"]),
                                  np.array(best[(ds, atk)]["seeds"]))
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none", ecolor=ATTACK_COLOR[atk],
                    elinewidth=0.9, alpha=0.6, zorder=2)
        ax.plot(x, y, marker=DATASET_MARKER[ds], ms=6.5, mfc=ATTACK_COLOR[atk], mec="white",
                mew=1.1, ls="none", zorder=3)
        print(f"  {DATASET_LABEL[ds]:12s} {ATTACK_LABEL[atk]:22s} J={x:.2f}+/-{xe:.2f} "
              f"AUROC={y:.2f}+/-{ye:.2f} ({best[(ds, atk)]['signal']})")
    ax.set_xlabel("corruption of the shown set, $J(x,x')$")
    ax.set_ylabel("AUROC of the best single check")
    ax.set_xlim(-0.04, 0.9)
    ax.set_ylim(0.3, 1.0)
    handles = [plt.Line2D([], [], marker="o", ls="none", ms=6, mfc=ATTACK_COLOR[a], mec="white",
                          mew=1.0, label=ATTACK_LABEL[a]) for a in ATTACK_COLOR]
    handles += [plt.Line2D([], [], marker=DATASET_MARKER[d], ls="none", ms=6, mfc="0.55",
                           mec="white", mew=1.0, label=DATASET_LABEL[d]) for d in DATASET_MARKER]
    ax.legend(handles=handles, loc="upper right", ncol=2, frameon=False, fontsize=7,
              handletextpad=0.3, columnspacing=0.8, borderpad=0.2, labelspacing=0.3)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT, bbox_inches="tight")
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
