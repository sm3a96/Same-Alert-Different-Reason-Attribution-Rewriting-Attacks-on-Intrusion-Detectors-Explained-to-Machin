"""Is the between-seed spread in certified-stability AUROC real, or its own sampling noise?

The deployability claim rests on the same dataset and pipeline giving very different AUROCs
depending only on the training seed. That claim is worth nothing until the spread is shown
against the uncertainty each per-seed estimate carries on its own: every seed's AUROC is
computed on that seed's few hundred flows, so if a single seed's interval is +/-0.15 then a
0.37-to-0.90 range is exactly what noise looks like.

Two things are computed here.

  Per-seed DeLong intervals. The claim survives if the extreme seeds' intervals are actually
  separated, and dies if they overlap.

  A between-versus-within variance split. Between-seed variance is the spread of the seed
  estimates; within-seed variance is the mean DeLong variance of those estimates. If between
  is not comfortably larger than within, the seeds are not disagreeing -- they are being
  measured imprecisely.

Also reports the flow counts behind each seed, because an extreme value resting on a thin
split is a different object from one resting on a full one.

  python scripts/check_seed_variance.py
"""
from __future__ import annotations

import csv
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024"}


def delong_var(y, s) -> float:
    """DeLong variance of an AUROC estimate."""
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return float("nan")
    v10 = np.array([(np.sum(neg < p) + 0.5 * np.sum(neg == p)) / n for p in pos])
    v01 = np.array([(np.sum(pos > q) + 0.5 * np.sum(pos == q)) / m for q in neg])
    return float(v10.var(ddof=1) / m + v01.var(ddof=1) / n)


def main() -> None:
    for run in sorted((REPO / "results").glob("c3_artifacts_*")):
        f = run / "raw" / "radii.csv"
        if not f.exists():
            continue
        ds = run.name.replace("c3_artifacts_", "")
        by = defaultdict(lambda: defaultdict(list))
        for row in csv.DictReader(open(f)):
            by[row["condition"]][int(row["seed"])].append(float(row["certified_radius"]))

        for cond in ("A1_displacement", "A1_misdirection"):
            if cond not in by:
                continue
            print(f"\n{LABEL.get(ds, ds)} — {cond}")
            print(f"  {'seed':>5s} {'n clean':>8s} {'n attacked':>11s} {'AUROC':>7s} "
                  f"{'DeLong 95% CI':>18s}")
            ests, varis = [], []
            for sd in sorted(by["clean"]):
                c = np.array(by["clean"][sd])
                a = np.array(by[cond].get(sd, []))
                if len(c) < 2 or len(a) < 2:
                    continue
                y = np.r_[np.zeros(len(c)), np.ones(len(a))]
                sc = np.r_[-c, -a]
                auc = float(roc_auc_score(y, sc))
                v = delong_var(y, sc)
                se = float(np.sqrt(v)) if v == v else float("nan")
                ests.append(auc)
                varis.append(v)
                print(f"  {sd:5d} {len(c):8d} {len(a):11d} {auc:7.3f} "
                      f"  [{auc - 1.96 * se:6.3f}, {auc + 1.96 * se:6.3f}]")

            if len(ests) < 3:
                continue
            ests = np.array(ests)
            varis = np.array(varis)
            between = float(ests.var(ddof=1))
            within = float(np.nanmean(varis))
            ratio = between / within if within > 0 else float("inf")
            print(f"\n    between-seed variance {between:.5f} | mean within-seed (DeLong) "
                  f"{within:.5f} | ratio {ratio:.1f}x")

            hi_i, lo_i = int(np.argmax(ests)), int(np.argmin(ests))
            hi_lo_bound = ests[hi_i] - 1.96 * np.sqrt(varis[hi_i])
            lo_hi_bound = ests[lo_i] + 1.96 * np.sqrt(varis[lo_i])
            separated = hi_lo_bound > lo_hi_bound
            print(f"    extreme seeds: highest {ests[hi_i]:.3f} (lower bound "
                  f"{hi_lo_bound:.3f}) vs lowest {ests[lo_i]:.3f} (upper bound "
                  f"{lo_hi_bound:.3f})")
            if separated and ratio > 2:
                print("    -> The seeds genuinely disagree: the extreme intervals do not")
                print("       overlap and between-seed variance dominates within-seed noise.")
            elif separated:
                print("    -> Extremes separate, but between-seed variance is not much larger")
                print("       than the measurement noise. Report the spread, claim it weakly.")
            else:
                print("    -> NOT SEPARATED. The spread is consistent with per-seed sampling")
                print("       noise, and cannot carry a claim about seed dependence.")


if __name__ == "__main__":
    main()
