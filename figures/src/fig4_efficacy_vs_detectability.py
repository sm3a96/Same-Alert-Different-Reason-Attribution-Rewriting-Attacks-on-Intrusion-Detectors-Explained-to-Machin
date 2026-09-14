"""Fig4 — attack efficacy against detectability.

The figure that carries C3 on its own. Each point is one (dataset, attack class) cell:
how much of the shown explanation the attack rewrites, against how well the best
integrity signal separates it from clean. Corruption runs along x and detectability along
y, so the paper's problem statement is the BOTTOM-RIGHT corner — maximum damage, chance
detection — and displacement attacks live there on every dataset. (This docstring said
top-left until 2026-08-15, which is the corner an axis swap would put it in and the kind of
error a reader trusts because the prose sounds specific.)

Reads results/matrix/raw.json; writes figures/out/fig4_efficacy_vs_detectability.pdf.

  python figures/src/fig4_efficacy_vs_detectability.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from avert.eval.metrics import best_single_auroc

from style import COL_SINGLE, PALETTE, apply_style

REPO = Path(__file__).resolve().parents[2]
RAW = REPO / "results" / "matrix" / "raw.json"
OUT = REPO / "figures" / "out" / "fig4_efficacy_vs_detectability.pdf"

# Fixed hue order — never cycled, never reassigned when a series is filtered out.
ATTACK_COLOR = {"A1_misdirection": PALETTE[0], "A1_displacement": PALETTE[1],
                "A3_scaffolding": PALETTE[2]}
ATTACK_LABEL = {"A1_misdirection": "A1a rank promotion",
                "A1_displacement": "A1b cause displacement",
                "A3_scaffolding": "A3 explainer scaffolding"}
# Shape carries dataset, so identity survives greyscale and print.
DATASET_MARKER = {"fiveg_nidd": "o", "ciciot2023": "s", "ciciomt2024": "^"}
DATASET_LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023",
                 "ciciomt2024": "CICIoMT2024"}


def mean_ci(v, seeds=None, ci: float = 0.95) -> tuple[float, float]:
    """95% CI over SEED means. The three classes inside a seed share one fitted detector,
    so error bars over all nine cells would be up to 2.3x too narrow."""
    from avert.eval.metrics import seed_clustered_ci
    v = np.asarray(v, dtype=float)
    s = np.arange(len(v)) if seeds is None else np.asarray(seeds)
    return seed_clustered_ci(v, s, ci)


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    rows = json.loads(RAW.read_text())
    cells = defaultdict(lambda: defaultdict(list))
    # Detectability is the best SINGLE check chosen once per (corpus, attack) on its
    # seed-clustered mean, fused row excluded -- never the per-cell maximum, which is
    # selection bias over correlated channels (see eval.metrics.best_single_auroc).
    best = best_single_auroc(rows)
    for r in rows:
        cells[(r["dataset"], r["attack"])]["corrupt"].append(r["corrupt"])
        cells[(r["dataset"], r["attack"])]["seed"].append(r["seed"])
    for k, b in best.items():
        cells[k]["auroc"] = b["per_cell"]

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))

    # Chance line first so it sits behind the data and stays recessive. The label and the
    # annotation anchor are offset from CHANCE rather than written as literals, so they
    # cannot drift away from the line they refer to.
    CHANCE = 0.5
    ax.axhline(CHANCE, color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.30, CHANCE + 0.005, "chance", transform=ax.get_yaxis_transform(),
            ha="left", va="bottom", fontsize=6, color="0.35")

    for (ds, attack), m in sorted(cells.items()):
        x, xe = mean_ci(m["corrupt"], m["seed"])
        y, ye = mean_ci(m["auroc"], m["seed"])
        ax.errorbar(x, y, xerr=xe, yerr=ye, fmt="none",
                    ecolor=ATTACK_COLOR[attack], elinewidth=1.0, alpha=0.55, zorder=2)
        ax.plot(x, y, marker=DATASET_MARKER[ds], ms=6.5, mfc=ATTACK_COLOR[attack],
                mec="white", mew=1.2, ls="none", zorder=3)   # 2px-equivalent surface ring

    ax.set_xlabel("explanation corrupted (Jaccard distance, top-$k$)")
    ax.set_ylabel("detectability (best signal AUROC)")
    ax.set_xlim(-0.04, 0.82)
    ax.set_ylim(0.33, 1.06)

    # One annotation, on the region that is the paper's point — not a label per marker.
    ax.annotate("severe and undetectable", xy=(0.58, CHANCE - 0.015), xytext=(0.30, 0.60),   # layout
                fontsize=6.5, color="0.25", ha="center",
                arrowprops=dict(arrowstyle="->", color="0.45", lw=0.8,
                                connectionstyle="arc3,rad=-0.25"))

    handles = [plt.Line2D([], [], marker="o", ls="none", ms=6, mfc=c, mec="white", mew=1.0,
                          label=ATTACK_LABEL[a]) for a, c in ATTACK_COLOR.items()]
    handles += [plt.Line2D([], [], marker=mk, ls="none", ms=6, mfc="0.55", mec="white",
                           mew=1.0, label=DATASET_LABEL[d]) for d, mk in DATASET_MARKER.items()]
    ax.legend(handles=handles, loc="upper left", ncol=2, frameon=False,
              handletextpad=0.3, columnspacing=0.8, borderpad=0.2, labelspacing=0.25)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"wrote {OUT}")

    print("\ncell values (mean +/- 95% CI over classes x seeds):")
    for (ds, attack), m in sorted(cells.items()):
        x, xe = mean_ci(m["corrupt"], m["seed"])
        y, ye = mean_ci(m["auroc"], m["seed"])
        print(f"  {DATASET_LABEL[ds]:12s} {ATTACK_LABEL[attack]:17s} "
              f"corrupt {x:.2f}+/-{xe:.2f}   auroc {y:.2f}+/-{ye:.2f}  n={len(m['auroc'])}")


if __name__ == "__main__":
    main()
