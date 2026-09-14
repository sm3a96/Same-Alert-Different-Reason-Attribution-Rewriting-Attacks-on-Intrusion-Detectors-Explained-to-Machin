"""Fig6 — the false-alarm guarantee, verified.

Split-conformal calibration on clean traffic promises P(p < alpha) <= alpha under
exchangeability. This sweeps alpha over the saved clean-held-out p-values and plots the
empirical violation rate against it. Every dataset should track the diagonal from below.
This is the correctness check in the pre-writing gate: if a line runs above y = x, the
guarantee the paper claims does not hold and nothing else matters.

Reads results/c3_artifacts_*/raw/conformal.csv.

  python figures/src/fig6_conformal_calibration.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from style import COL_SINGLE, PALETTE, apply_style

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "out" / "fig6_conformal_calibration.pdf"

DATASET_COLOR = {"fiveg_nidd": PALETTE[0], "ciciot2023": PALETTE[1], "ciciomt2024": PALETTE[2]}
DATASET_LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023",
                 "ciciomt2024": "CICIoMT2024"}
ALPHAS = np.linspace(0.005, 0.20, 40)


def clopper_pearson(k, n, conf=0.05):
    """Exact binomial CI. Vectorised over k; the interval is what decides whether an
    empirical rate above alpha is a real violation or finite-sample noise."""
    from scipy.stats import beta
    k = np.asarray(k, dtype=float)
    lo = np.where(k > 0, beta.ppf(conf / 2, np.maximum(k, 1e-9), n - k + 1), 0.0)
    hi = np.where(k < n, beta.ppf(1 - conf / 2, k + 1, np.maximum(n - k, 1e-9)), 1.0)
    return lo, hi


def load() -> dict[str, dict[int, list[float]]]:
    """p-values per dataset per seed. Seeds stay separate so the band is over seeds."""
    out: dict[str, dict[int, list[float]]] = defaultdict(lambda: defaultdict(list))
    for path in sorted((REPO / "results").glob("c3_artifacts_*/raw/conformal.csv")):
        for row in csv.DictReader(open(path)):
            out[row["dataset"]][int(row["seed"])].append(float(row["p_value"]))
    return out


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    data = load()
    if not data:
        raise SystemExit("no conformal.csv found — run scripts/run_c3_artifacts.py first")

    fig, ax = plt.subplots(figsize=(COL_SINGLE, 2.7))
    ax.plot([0, 0.20], [0, 0.20], color="0.45", lw=0.9, ls=(0, (4, 3)), zorder=1)
    ax.text(0.185, 0.192, "$y=x$", fontsize=6.5, color="0.35", ha="right", va="top")   # layout

    print("empirical false-alarm rate vs nominal alpha (Clopper-Pearson 95% CI):")
    for ds, by_seed in sorted(data.items()):
        # Pool the seeds. The quantity is a Binomial proportion over held-out clean flows,
        # so its uncertainty is binomial — a normal-approx band over three seed-curves has
        # 2 degrees of freedom and produces a band wide enough to be meaningless.
        pv = np.array([p for seed_pv in by_seed.values() for p in seed_pv])
        n = len(pv)
        hits = np.array([int((pv < a).sum()) for a in ALPHAS])
        mu = hits / n
        lo, hi = clopper_pearson(hits, n)
        ax.plot(ALPHAS, mu, color=DATASET_COLOR[ds], lw=2.0,
                label=f"{DATASET_LABEL[ds]} ($n$={n})", zorder=3)
        ax.fill_between(ALPHAS, lo, hi, color=DATASET_COLOR[ds], alpha=0.18, lw=0, zorder=2)

        # The guarantee is one-sided: the rate must not sit ABOVE alpha. A point estimate
        # over alpha is only a violation if the whole interval clears alpha.
        exceed = [(a, m, cl) for a, m, cl in zip(ALPHAS, mu, lo) if cl > a]
        at05 = float(np.interp(0.05, ALPHAS, mu))
        verdict = ("holds at every alpha tested" if not exceed else
                   f"CI ABOVE alpha at {len(exceed)} of {len(ALPHAS)} points "
                   f"(worst alpha={exceed[0][0]:.3f}: {exceed[0][1]:.3f})")
        print(f"  {DATASET_LABEL[ds]:12s} seeds={len(by_seed)}  n={n:4d}  "
              f"FA@0.05 = {at05:.3f}  -> {verdict}")

    ax.set_xlabel(r"nominal $\alpha$")
    ax.set_ylabel("empirical false-alarm rate")
    ax.set_xlim(0, 0.20)
    ax.set_ylim(0, 0.20)
    ax.set_aspect("equal", adjustable="box")
    ax.legend(loc="upper left", frameon=False, handletextpad=0.5, borderpad=0.2,
              labelspacing=0.3)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
