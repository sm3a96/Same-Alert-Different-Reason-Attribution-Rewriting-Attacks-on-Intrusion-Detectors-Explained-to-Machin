"""Fig5 — why the certified-stability signal cannot discriminate.

An ECDF rather than a histogram, because the point of this figure is where the
probability mass sits relative to the estimator ceiling R_max = sigma * Phi^-1(conf^(1/n)).
A radius cannot exceed that however stable the attribution is, since p_lo is a
Clopper-Pearson bound.

Two things this figure shows and one it does not. It shows that on the tree pipeline the
mass piles against the ceiling or at zero, leaving little in the range that can rank; and
that the clean and attacked curves lie close enough together to explain the chance-level
AUROC. It does NOT show that saturation is the general mechanism — a gradient pipeline on
the same data does not saturate (see scripts/run_generality.py), and the signal fails there
for a different reason. The bound is general; the piling is a property of this pipeline.

This also corrects the correction. The June account read the failure as clean attributions
being inherently unstable; a five-seed re-measurement put 86-94% of clean flows certifying  # retired-ok: names the withdrawn figure in order to withdraw it
and retired that reading -- but it ran on corpora carrying a row counter and a capture clock. On
the cleaned feature sets 1.7-21.0% certify and essentially nothing sits on the ceiling, so the
June direction was right and both recorded explanations for it were wrong.

Reads results/c3_artifacts_*/raw/radii.csv.

  python figures/src/fig5_certified_radius.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np

from style import COL_DOUBLE, PALETTE, apply_style

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "figures" / "out" / "fig5_certified_radius.pdf"

COND_COLOR = {"clean": PALETTE[6], "A1_misdirection": PALETTE[0],
              "A1_displacement": PALETTE[1], "A3_scaffolding": PALETTE[2]}
COND_LABEL = {"clean": "clean", "A1_misdirection": "A1a rank promotion",
              "A1_displacement": "A1b cause displacement",
              "A3_scaffolding": "A3 explainer scaffolding"}
DATASET_LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023",
                 "ciciomt2024": "CICIoMT2024"}
def ceiling_for(run_dir: Path) -> float:
    """R_max for this run, read from its own metadata rather than written in here.

    A literal would silently draw the wrong line the moment anyone changes sigma or the
    number of smoothing draws, and the whole point of the figure is where the mass sits
    relative to that line.
    """
    import json
    from scipy.stats import norm

    from avert.signals.certified_stability import clopper_pearson_lower

    metas = sorted((run_dir / "meta").glob("run_seed*.json"))
    if not metas:
        raise SystemExit(f"no metadata in {run_dir}; cannot determine the radius ceiling")
    cfg = json.loads(metas[0].read_text())["config"]
    n, sigma = int(cfg["stab_n"]), float(cfg["stab_sigma"])
    return sigma * float(norm.ppf(clopper_pearson_lower(n, n, 0.05)))


def load():
    """Radii per dataset per condition, plus that dataset's own ceiling."""
    out = defaultdict(lambda: defaultdict(list))
    ceilings = {}
    for path in sorted((REPO / "results").glob("c3_artifacts_*/raw/radii.csv")):
        run = path.parent.parent
        ds = run.name.replace("c3_artifacts_", "")
        ceilings[ds] = ceiling_for(run)
        for row in csv.DictReader(open(path)):
            out[row["dataset"]][row["condition"]].append(float(row["certified_radius"]))
    return out, ceilings


def main() -> None:
    apply_style()
    import matplotlib.pyplot as plt

    data, ceilings = load()
    if not data:
        raise SystemExit("no radii.csv found — run scripts/run_c3_artifacts.py first")

    order = [d for d in ("fiveg_nidd", "ciciot2023", "ciciomt2024") if d in data]
    # 2.5 rather than 2.3: the figure-level legend takes the top band, so the extra
    # height keeps the panels at the height the old in-axes-legend version had.
    fig, axes = plt.subplots(1, len(order), figsize=(COL_DOUBLE, 2.5), sharey=True)
    axes = np.atleast_1d(axes)

    print("certified-radius mass (fraction of samples):")
    for ax, ds in zip(axes, order):
        ceiling = ceilings[ds]
        for cond in ("clean", "A1_misdirection", "A1_displacement", "A3_scaffolding"):
            r = np.sort(np.array(data[ds].get(cond, [])))
            if not len(r):
                continue
            ax.step(r, np.arange(1, len(r) + 1) / len(r), where="post",
                    color=COND_COLOR[cond], lw=1.9 if cond == "clean" else 1.3,
                    ls="-" if cond == "clean" else (0, (3, 1.5)),
                    label=COND_LABEL[cond], zorder=3 if cond == "clean" else 2)
            print(f"  {DATASET_LABEL[ds]:12s} {COND_LABEL[cond]:17s} n={len(r):4d}  "
                  f"at-zero {np.mean(r == 0):.2f}  at-ceiling {np.mean(np.isclose(r, ceiling, rtol=1e-3)):.2f}")
        ax.axvline(ceiling, color="0.45", lw=0.9, ls=(0, (2, 2)), zorder=1)
        ax.set_title(DATASET_LABEL[ds])
        ax.set_xlabel("certified radius $R$")
        ax.set_xlim(-0.004, ceiling * 1.18)   # layout: small left pad so R=0 is visible

    axes[0].set_ylabel("cumulative fraction")
    axes[0].set_ylim(0, 1.02)
    axes[-1].text(ceilings[order[-1]], 0.06, r" $R_{\max}$", fontsize=6.5, color="0.35",
                  ha="left", va="bottom")
    # A per-axes legend in the first panel sits exactly where the ECDF mass is (most
    # radii are 0, so the curves start high on the left). One figure-level legend above
    # the panels keeps every curve visible.
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False,
               handletextpad=0.5, columnspacing=1.2, bbox_to_anchor=(0.5, 1.02))
    fig.subplots_adjust(top=0.78)

    OUT.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT)
    print(f"\nwrote {OUT}")


if __name__ == "__main__":
    main()
