"""How large is the displacement perturbation, against the radius the certificate can cover?

Section V-D reports that the certified-stability check inverts against cause displacement --
it reads BELOW chance, so it separates the two arms with the wrong sign -- and the Discussion
called that unexplained. It is not unexplained once the two magnitudes are put side by side.

The certificate covers a radius of at most sigma * Phi^-1(alpha_c^(1/N)) in units of the
calibration-flow standard deviation: 0.131 sigma_cal on the matrix path. If the attack moves
the flow much further than that, the certificate was never a bound on it, and the check is not
being evaded so much as asked a question it cannot answer. Worse for the check: the attack
searches 160 draws for the one whose shown set moves most, which is the draw whose shown set
was least stable to begin with -- so the flows it returns are selected to look stable under
re-smoothing, and the inversion is what that selection predicts.

This measures the first half of that: the L2 norm of x' - x over the FREE coordinates, in units
of each feature's standard deviation on that cell's calibration flows. Free coordinates only,
because the derived ones are recomputed by the extractor rather than moved by the adversary,
and counting them would credit the attack with motion it did not choose.

The two units are not close, and that is the finding rather than an artefact. The attack budget
is 0.15 sigma_train, a standard deviation taken over the whole training draw; the certificate's
sigma_cal is taken over one cell's calibration flows, which are all of ONE attack class. On
5G-NIDD HTTP flood the median free coordinate has sigma_cal/sigma_train = 0.20 and the extreme
one 0.000186, so a perturbation of 0.49 sigma_train is 470 sigma_cal on that flow. Ten of the
27 free coordinates are exactly constant on the calibration flows, so sigma_cal is zero and the
ratio is undefined there; those coordinates are dropped from the norm and counted, because a
certificate calibrated on a constant column has no scale to offer at all.

The pooled MEAN is therefore dominated by a handful of near-degenerate columns. The median over
flows is reported beside it and is what the paper quotes; the mean, the zero-sigma count and the
sigma_train norm are all recorded so the reader can see what the ratio is made of.

Reads the attacked flows the release ships
(results/reference_decomposition/raw/attacked_flows.json) and rebuilds each cell's calibration
draw to get sigma_cal. Writes results/attack_magnitude/.

  python scripts/report_attack_magnitude.py
  python scripts/report_attack_magnitude.py --datasets fiveg_nidd --seeds 0
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
from scipy.stats import norm

from avert.benchmark import XIntBench
from avert.benchmark.harness import target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.metrics import seed_clustered_ci
from avert.eval.runner import RESULTS_ROOT, RunResult, run_experiment

log = logging.getLogger("magnitude")

FLOWS = RESULTS_ROOT / "reference_decomposition" / "raw" / "attacked_flows.json"
DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
SEEDS = [0, 1, 2, 3, 4]
N_CLASSES, N_TEST, TOP_K = 3, 40, 5


def clopper_pearson_lower(k: int, n: int, alpha: float) -> float:
    from scipy.stats import beta
    return float(beta.ppf(alpha, k, n - k + 1)) if k else 0.0


def ceiling(sigma: float, n: int, alpha_c: float = 0.05) -> float:
    """The largest radius the certificate can return when all N draws agree."""
    return float(sigma * norm.ppf(clopper_pearson_lower(n, n, alpha_c)))


def sweep(ctx) -> RunResult:
    args = ctx.config["args"]
    payload = json.loads(FLOWS.read_text())
    by_cell = defaultdict(dict)
    for r in payload["flows"]:
        if r["attack"] != args["attack"]:
            continue
        by_cell[(r["dataset"], r["class"], r["seed"])][r["sample_id"]] = r["x_prime"]

    rows = []
    for ds in args["datasets"]:
        data = load_dataset(ds, load_config(f"datasets/{ds}"))
        classes, _ = target_classes(data, N_CLASSES, n_cal=100, n_test=N_TEST, seed=0,
                                    min_count=2000, seeds=SEEDS)
        for c in classes:
            cname = data.label_names[c] if data.label_names else str(c)
            for seed in args["seeds"]:
                flows = by_cell.get((ds, cname, seed))
                if not flows:
                    log.warning("no exported flows for %s/%s/seed%s", ds, cname, seed)
                    continue
                bench = XIntBench(data, n_cal=100, n_test=N_TEST, top_k=TOP_K, perm_samples=10,
                                  stability_n=30, seed=seed)
                st = bench._prepare(target=c)
                free = np.asarray(st.frag.bounds.free_idx, dtype=int)
                # sigma_cal: the per-feature standard deviation of the cell's calibration
                # flows, which is the unit the certificate's sigma is expressed in.
                cal = np.stack([s.features for s in st.cal]).astype(float)
                sd_cal = cal.std(axis=0)
                sd_tr = st.X[st.tr].std(axis=0)
                zero = int((sd_cal[free] == 0).sum())
                sd_cal = np.where(sd_cal == 0, np.nan, sd_cal)   # no scale, not a huge scale
                per_cal, per_train = [], []
                for s in st.test:
                    xp = flows.get(s.sample_id)
                    if xp is None:
                        continue
                    delta = (np.asarray(xp, dtype=float) - s.features.astype(float))[free]
                    per_cal.append(float(np.sqrt(np.nansum((delta / sd_cal[free]) ** 2))))
                    with np.errstate(divide="ignore", invalid="ignore"):
                        per_train.append(float(np.sqrt(np.nansum((delta / np.where(
                            sd_tr[free] == 0, np.nan, sd_tr[free])) ** 2))))
                if not per_cal:
                    continue
                with np.errstate(divide="ignore", invalid="ignore"):
                    ratio = np.nanmedian(sd_cal[free] / np.where(sd_tr[free] == 0, np.nan,
                                                                 sd_tr[free]))
                rows.append({"dataset": ds, "class": cname, "seed": seed,
                             "n_flows": len(per_cal), "n_free": int(len(free)),
                             "n_free_sigma_cal_zero": zero,
                             "median_sigma_cal_over_sigma_train": float(ratio),
                             "l2_sigma_cal_mean": float(np.mean(per_cal)),
                             "l2_sigma_cal_median": float(np.median(per_cal)),
                             "l2_sigma_cal_max": float(np.max(per_cal)),
                             "l2_sigma_train_mean": float(np.mean(per_train))})
                log.info("%s class=%s seed=%s: median ||x'-x|| = %.1f sigma_cal "
                         "(%.3f sigma_train), %d of %d free coordinates constant on the "
                         "calibration flows", ds, cname, seed,
                         rows[-1]["l2_sigma_cal_median"], rows[-1]["l2_sigma_train_mean"],
                         zero, len(free))

    seeds = np.array([r["seed"] for r in rows])
    # The MEDIAN per-cell value is the headline: the mean is dominated by the few free
    # coordinates whose calibration-flow spread is near zero.
    mu, hw = seed_clustered_ci(np.array([r["l2_sigma_cal_median"] for r in rows]), seeds)
    mean_mu, _ = seed_clustered_ci(np.array([r["l2_sigma_cal_mean"] for r in rows]), seeds)
    train_mu, _ = seed_clustered_ci(np.array([r["l2_sigma_train_mean"] for r in rows]), seeds)
    per_corpus = {}
    for ds in sorted({r["dataset"] for r in rows}):
        sel = [r for r in rows if r["dataset"] == ds]
        m, h = seed_clustered_ci(np.array([r["l2_sigma_cal_median"] for r in sel]),
                                 np.array([r["seed"] for r in sel]))
        per_corpus[ds] = {"median": m, "ci_lo": m - h, "ci_hi": m + h}

    ceilings = {"matrix": ceiling(args["sigma_matrix"], args["n_matrix"]),
                "artifact": ceiling(args["sigma_artifact"], args["n_artifact"])}
    summary = {"pooled": {"median": mu, "ci_lo": mu - hw, "ci_hi": mu + hw,
                          "mean": mean_mu, "l2_sigma_train_mean": train_mu},
               "per_corpus": per_corpus, "ceilings_sigma_cal": ceilings,
               "ratio_to_matrix_ceiling": mu / ceilings["matrix"], "cells": len(rows),
               "free_coords_constant_on_calibration": {
                   "total": int(sum(r["n_free_sigma_cal_zero"] for r in rows)),
                   "of_free": int(sum(r["n_free"] for r in rows)),
                   "per_cell_mean": float(np.mean([r["n_free_sigma_cal_zero"] for r in rows]))},
               "median_sigma_cal_over_sigma_train": float(np.median(
                   [r["median_sigma_cal_over_sigma_train"] for r in rows]))}

    (ctx.run_dir / "raw" / "per_cell.json").write_text(json.dumps(rows, indent=1))
    (ctx.run_dir / "summary" / "magnitude.json").write_text(json.dumps(summary, indent=1))
    md = ["# Displacement magnitude against the certificate's reach", "",
          f"Median $\\|x'-x\\|_2$ over the free coordinates, in units of sigma_cal: "
          f"**{mu:.1f}** [{mu-hw:.1f}, {mu+hw:.1f}] over {len(rows)} cells "
          f"(cell means average {mean_mu:.0f}, dominated by near-constant calibration columns). "
          f"The same perturbation is {train_mu:.2f} sigma_train, which is the unit its budget "
          f"is set in.", "",
          f"On average {summary['free_coords_constant_on_calibration']['per_cell_mean']:.1f} "
          "free coordinates per cell are exactly constant on the calibration flows, so the "
          "certificate has no scale for them at all; they are excluded from the norm. The "
          "median free coordinate has sigma_cal/sigma_train = "
          f"{summary['median_sigma_cal_over_sigma_train']:.2f}.", "",
          f"Certificate ceiling: {ceilings['matrix']:.3f} sigma_cal on the matrix path "
          f"(sigma={args['sigma_matrix']}, N={args['n_matrix']}), "
          f"{ceilings['artifact']:.3f} on the artifact runs "
          f"(sigma={args['sigma_artifact']}, N={args['n_artifact']}).", "",
          f"The perturbation is {mu / ceilings['matrix']:.0f}x the radius the certificate can "
          "cover at the matrix budget, so the certificate does not bound this attack.", "",
          "| Corpus | median | 95% CI |", "|---|---|---|"]
    for ds, v in per_corpus.items():
        md.append(f"| {ds} | {v['median']:.1f} | [{v['ci_lo']:.1f}, {v['ci_hi']:.1f}] |")
    (ctx.run_dir / "summary" / "report.md").write_text("\n".join(md))
    print("\n".join(md))

    return RunResult(
        summary=summary,
        floats=[{"float_id": "attack_magnitude", "kind": "summary", "paper_section": "detect",
                 "path": str(ctx.run_dir / "summary" / "magnitude.json"),
                 "claim": "the displacement perturbation is far larger than the radius the "
                          "certified check can cover, so the certificate does not bound it"}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--attack", default="A1_displacement")
    ap.add_argument("--sigma-matrix", type=float, default=0.1)
    ap.add_argument("--n-matrix", type=int, default=30)
    ap.add_argument("--sigma-artifact", type=float, default=0.05)
    ap.add_argument("--n-artifact", type=int, default=120)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment("attack_magnitude", sweep, config={"args": {
        "datasets": a.datasets, "seeds": a.seeds, "attack": a.attack,
        "sigma_matrix": a.sigma_matrix, "n_matrix": a.n_matrix,
        "sigma_artifact": a.sigma_artifact, "n_artifact": a.n_artifact}}, seed=0)


if __name__ == "__main__":
    main()
