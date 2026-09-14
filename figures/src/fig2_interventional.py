"""Fig2 — the figure that proves XInt-Bench's ground truth is real.

A benchmark whose labels come from the generator is only a claim about the data. The
labels are claims about *attributions*, so they have to be tied to the detector: ablating
a ground-truth causal feature toward its benign value must drag the detection down, and
ablating a non-causal feature must not. Sweeping the ablation fraction shows the whole
curve rather than a single before/after pair, which is harder to fake and easier for a
reviewer to check.

If the two curves ever converge, the labels do not describe this detector and the
benchmark is measuring the wrong thing.

Reads results/c3_artifacts_*/raw/ablation.csv.

  python figures/src/fig2_interventional.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from style import COL_DOUBLE, PALETTE, apply_style

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "out" / "fig2_interventional.pdf"

KIND_COLOR = {"causal": PALETTE[1], "non_causal": PALETTE[0]}
KIND_LABEL = {"causal": "ground-truth causal", "non_causal": "non-causal"}
DATASET_LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023",
                 "ciciomt2024": "CICIoMT2024"}


def load():
    """detect_drop per dataset per feature-kind per ablation fraction."""
    out = defaultdict(lambda: defaultdict(lambda: defaultdict(list)))
    for path in sorted((REPO / "results").glob("c3_artifacts_*/raw/ablation.csv")):
        for row in csv.DictReader(open(path)):
            out[row["dataset"]][row["feature_kind"]][float(row["ablation_frac"])].append(
                float(row["detect_drop"]))
    return out


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt
    from scipy import stats

    data = load()
    if not data:
        raise SystemExit("no ablation.csv found — run scripts/run_c3_artifacts.py first")

    order = [d for d in ("fiveg_nidd", "ciciot2023", "ciciomt2024") if d in data]
    fig, axes = plt.subplots(1, len(order), figsize=(COL_DOUBLE, 2.3), sharey=True)
    axes = np.atleast_1d(axes)

    print("detection drop when a feature is ablated toward benign (mean +/- 95% CI):")
    for ax, ds in zip(axes, order):
        for kind in ("causal", "non_causal"):
            by_frac = data[ds].get(kind)
            if not by_frac:
                continue
            fracs = sorted(by_frac)
            mu = np.array([np.mean(by_frac[f]) for f in fracs])
            hw = np.array([stats.sem(by_frac[f]) * stats.t.ppf(0.975, max(len(by_frac[f]) - 1, 1))
                           if len(by_frac[f]) > 1 else 0.0 for f in fracs])
            ax.plot(fracs, mu, color=KIND_COLOR[kind], lw=2.0, marker="o", ms=4,
                    mec="white", mew=1.0, label=KIND_LABEL[kind], zorder=3)
            ax.fill_between(fracs, mu - hw, mu + hw, color=KIND_COLOR[kind],
                            alpha=0.18, lw=0, zorder=2)
            # Four significant figures, not three. The non-causal drop is genuinely tiny
            # but it is NOT zero, and printing "+0.000 +/- 0.000" reads as too good to be
            # true -- which invites exactly the suspicion the figure exists to remove.
            print(f"  {DATASET_LABEL[ds]:12s} {KIND_LABEL[kind]:20s} "
                  f"full ablation {mu[-1]:+.5f} +/- {hw[-1]:.5f}  "
                  f"(n={len(by_frac[fracs[-1]])}, max |drop| "
                  f"{max(abs(x) for x in by_frac[fracs[-1]]):.5f})")
        ax.axhline(0, color="0.45", lw=0.8, zorder=1)
        ax.set_title(DATASET_LABEL[ds])
        ax.set_xlabel("ablation toward benign")
        ax.set_xlim(-0.03, 1.03)

    axes[0].set_ylabel("drop in detection probability")
    axes[0].legend(loc="upper left", frameon=False, handletextpad=0.5,
                   borderpad=0.2, labelspacing=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
