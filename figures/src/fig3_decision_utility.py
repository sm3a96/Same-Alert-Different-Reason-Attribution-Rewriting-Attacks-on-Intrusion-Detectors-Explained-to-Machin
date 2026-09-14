"""Fig3 — the headline: does the attack change what the agent DOES?

Decision accuracy against XInt-Bench's interventional ground truth, per condition, per
attack class, one panel per judge. Four bars per group:

  clean            what the agent gets right from an honest explanation
  attacked         the same alert after a prediction-preserving explanation attack
  repaired         the re-derivation arm, an identity with `attacked`: same explainer, same
                   moved flow, same answer. Nothing is claimed from it; the column exists only
                   to show that a quarter of the run measured x == x.
  ranking withheld same ten features, importance values removed -- the floor

The fourth bar is what makes the rest interpretable. If it sits level with clean, the
ranking was never load-bearing for this agent and the attack has nothing to corrupt; the
gap between them is the room an attack has to work in.

It is labelled "ranking withheld" rather than "no explanation" because it holds feature
selection fixed: the ten features on screen are still the ones the clean explanation
chose. That makes it a conservative floor -- it flatters the baseline -- and the caption
has to say so.

Two rows, and the difference between them is itself a finding.

The top row is every cell. There the attack appears to *improve* accuracy on one dataset,
which is an artifact: on CICIoT2023 the judges answer correctly 7% of the time from an
honest explanation, below the 10% a blind guess would score. An agent that is
systematically wrong gets moved toward chance by any perturbation of its input, and
chance is better than systematically wrong. That is not a defense.

The bottom row is cells where the judge clears the clean-accuracy floor declared before
the run. There the direction is consistent and the effect is real. Showing both rows costs
one figure and pre-empts the obvious objection, which is cheaper than being asked.

Reads results/decision_utility/summary/per_cell.json.

  python figures/src/fig3_decision_utility.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from style import COL_DOUBLE, PALETTE, apply_style

REPO = Path(__file__).resolve().parents[2]
CELLS = REPO / "results" / "decision_utility" / "summary" / "per_cell.json"
OUT = REPO / "figures" / "out" / "fig3_decision_utility.pdf"

CONDITIONS = ["clean", "attacked", "repaired", "no_explanation"]  # retired-ok: "repaired" is the arm's key in the raw data; it is an identity and not claimed
COND_COLOR = dict(zip(CONDITIONS, PALETTE[:4]))
COND_LABEL = {"clean": "clean", "attacked": "attacked", "repaired": "re-derived (identity)",  # retired-ok: labels the arm as the identity it is
              "no_explanation": "ranking withheld"}
ATTACK_LABEL = {"A1_misdirection": "A1a rank\npromotion",
                "A1_displacement": "A1b cause\ndisplacement",
                "A3_scaffolding": "A3 explainer\nscaffolding"}
ATTACK_ORDER = ["A1_misdirection", "A1_displacement", "A3_scaffolding"]


def mean_ci(v, ci=0.95):
    v = np.asarray([x for x in v if not np.isnan(x)], dtype=float)
    if len(v) < 2:
        return (float(v.mean()) if len(v) else float("nan")), 0.0
    from scipy import stats
    return float(v.mean()), float(stats.sem(v) * stats.t.ppf(0.5 + ci / 2, len(v) - 1))


FLOOR = 0.5      # pre-declared clean-accuracy floor


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    if not CELLS.exists():
        raise SystemExit(f"{CELLS} not found — run scripts/run_decision_utility.py first")
    cells = json.loads(CELLS.read_text())
    judges = sorted({c["judge"] for c in cells})

    def aggregate(subset):
        agg = defaultdict(lambda: defaultdict(list))
        for c in subset:
            for cond in CONDITIONS:
                if f"{cond}__acc" in c:
                    agg[(c["judge"], c["attack"])][cond].append(c[f"{cond}__acc"])
        return agg

    competent = [c for c in cells if c.get("clean__acc", 0) >= FLOOR]
    panels = [("all cells", cells), (f"judge competent (clean $\\geq$ {FLOOR})", competent)]

    fig, axes = plt.subplots(len(panels), len(judges), figsize=(COL_DOUBLE, 4.4),
                             sharey=True, sharex=True)
    axes = np.atleast_2d(axes)
    width = 0.2
    xs = np.arange(len(ATTACK_ORDER))

    print("decision accuracy vs interventional ground truth (mean +/- 95% CI over cells):")
    for row, (row_label, subset) in enumerate(panels):
        agg = aggregate(subset)
        print(f"\n  --- {row_label} ({len(subset)} cells) ---")
        for col, judge in enumerate(judges):
            ax = axes[row, col]
            for i, cond in enumerate(CONDITIONS):
                mus, hws = [], []
                for atk in ATTACK_ORDER:
                    mu, hw = mean_ci(agg[(judge, atk)].get(cond, []))
                    mus.append(mu)
                    hws.append(hw)
                    print(f"    {judge:10s} {atk:17s} {COND_LABEL[cond]:17s} {mu:.3f}+/-{hw:.3f}")
                ax.bar(xs + (i - 1.5) * width, mus, width * 0.9, yerr=hws, capsize=1.6,
                       color=COND_COLOR[cond], edgecolor="white", linewidth=0.6,
                       error_kw=dict(elinewidth=0.8, ecolor="0.3"),
                       label=COND_LABEL[cond] if (row == 0 and col == 0) else None, zorder=3)
            if row == 0:
                ax.set_title(judge)
            if col == 0:
                ax.set_ylabel(f"{row_label}\ndecision accuracy", fontsize=7)
            ax.set_ylim(0, 1.09)
            ax.margins(x=0.06)
        axes[row, 0].set_xticks(xs)
        axes[row, 0].set_xticklabels([ATTACK_LABEL[a] for a in ATTACK_ORDER])

    axes[0, 0].legend(loc="upper left", ncol=2, frameon=False, handletextpad=0.4,
                      columnspacing=0.9, borderpad=0.2, labelspacing=0.25, fontsize=6.5)
    fig.align_ylabels(axes[:, 0])

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
