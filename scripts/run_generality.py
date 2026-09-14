"""Do the C3 negatives survive a different detector family and a different attributor?

Every headline number in this work uses XGBoost with TreeSHAP. The strongest objection a
reviewer can make is that the negatives are properties of that pair rather than of the
techniques: TreeSHAP is exact and piecewise constant, trees are step functions, and a
certified radius computed over a step function may behave unlike one computed over a
smooth model.

This runs the same measurements on a differentiable detector (MLP) explained by integrated
gradients — a genuinely different mechanism on both sides — and asks whether the two
findings that carry C3 still hold:

  1. Is the certified radius still pinned to the estimator ceiling or to zero, leaving
     little mass in the range that can rank?
  2. Does the AUROC against A1 displacement still fail, and does its SIGN still move?

A negative that only exists for tree models is a much smaller claim than one that also
holds for a gradient-based pipeline, and the honest version of this paper needs to know
which it has.

  python scripts/run_generality.py --dataset fiveg_nidd --seeds 0 1 2
"""
from __future__ import annotations

import argparse
import csv
import json

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from pathlib import Path

from avert.attribution.real import IntegratedGradientsAttributor
from avert.benchmark import XIntBench
from avert.benchmark.ground_truth import causal_features
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.detectors.real import MLPDetector
from avert.eval.runner import RunResult, run_experiment
from avert.signals.certified_stability import CertifiedStabilitySignal, clopper_pearson_lower

# A1 only, and the reason is substantive rather than convenience. A3 replaces the
# deployed model with a non-differentiable scaffold, so a gradient attributor cannot
# explain it at all -- which is itself consistent with the A3 finding, since the attack
# only bites explainers that probe the model off-manifold. The A1 attacks perturb the
# input and leave the detector alone, so they transfer to any detector family.
ATTACKS = ["A1_misdirection", "A1_displacement"]
REPO = Path(__file__).resolve().parents[1]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiveg_nidd")
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--stab-n", type=int, default=120)
    ap.add_argument("--stab-sigma", type=float, default=0.05)
    args = ap.parse_args()

    data = load_dataset(args.dataset, load_config(f"datasets/{args.dataset}"))
    ceiling = args.stab_sigma * float(
        norm.ppf(clopper_pearson_lower(args.stab_n, args.stab_n, 0.05)))

    def experiment(ctx):
        radii_rows, auroc_rows = [], []
        for seed in args.seeds:
            bench = XIntBench(data, n_cal=80, n_test=args.n_test, top_k=5,
                              perm_samples=10, stability_n=args.stab_n, seed=seed)
            # A differentiable detector, and a gradient-based attributor to match. The MLP
            # trains on the SAME grouped training rows the harness draws for this seed. Until
            # 2026-09-09 it trained on a random permutation of the whole corpus, so it saw
            # rows from the held-out captures its calibration and test flows come from --
            # the leak the session-aware split exists to prevent. The first _prepare call
            # fits the default tree detector only to obtain the seeded split; the second
            # rebuilds the identical split around the MLP.
            X, y = data.X, data.y
            expl = IntegratedGradientsAttributor()
            tr = bench._prepare(explainer=expl).tr
            det = MLPDetector().fit(X[tr], y[tr])
            st = bench._prepare(detector=det, explainer=expl)
            assert np.array_equal(st.tr, tr), "the two _prepare calls must draw the same split"

            stability = CertifiedStabilitySignal(expl, n=args.stab_n, sigma=args.stab_sigma,
                                                 top_k=5, seed=seed)
            # Only the certified-stability signal is scored in this script -- the generality
            # question is about that signal across pipelines. A consensus signal used to be
            # constructed and calibrated here and never scored, which left a live object
            # calibrated against an empty null for the next reader to pick up. Removed.
            stability.calibrate(st.cal, [], det)

            def radius(sample, d):
                return float(stability.certified_radius(sample, d)[0])

            clean = [radius(s, det) for s in st.test]
            gt = [causal_features(s, det, st.benign_ref, 0.05, 3)[0] for s in st.test]
            for r in clean:
                radii_rows.append({"seed": seed, "condition": "clean", "radius": round(r, 6)})

            for attack in ATTACKS:
                atk = []
                for s, causal in zip(st.test, gt):
                    a, d = bench.attacked(s, attack, st, list(causal))
                    r = radius(a, d)
                    atk.append(r)
                    radii_rows.append({"seed": seed, "condition": attack, "radius": round(r, 6)})
                auroc = float(roc_auc_score([0] * len(clean) + [1] * len(atk),
                                            [-v for v in clean] + [-v for v in atk]))
                auroc_rows.append({"seed": seed, "attack": attack, "auroc": round(auroc, 4)})
                print(f"  seed {seed} {attack:17s} AUROC {auroc:.3f}", flush=True)

        raw = ctx.run_dir / "raw"
        for name, rows in (("radii.csv", radii_rows), ("auroc.csv", auroc_rows)):
            if rows:
                with open(raw / name, "w", newline="") as f:
                    w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
                    w.writeheader()
                    w.writerows(rows)

        def mass(cond):
            r = np.array([x["radius"] for x in radii_rows if x["condition"] == cond])
            return (float(np.mean(r == 0)),
                    float(np.mean(np.isclose(r, ceiling, rtol=1e-3))),
                    float(1 - np.mean(r == 0) - np.mean(np.isclose(r, ceiling, rtol=1e-3))))

        z, c, m = mass("clean")
        frag = [x["auroc"] for x in auroc_rows if x["attack"] == "A1_displacement"]
        summary = {"dataset": args.dataset, "detector": "MLP", "attributor": "integrated_gradients",
                   "stab_n": args.stab_n, "sigma": args.stab_sigma, "ceiling": round(ceiling, 6),
                   "clean_at_zero": round(z, 4), "clean_at_ceiling": round(c, 4),
                   "clean_informative": round(m, 4),
                   "displacement_auroc_mean": round(float(np.mean(frag)), 4),
                   "displacement_auroc_per_seed": frag}
        print("\n  " + json.dumps(summary), flush=True)
        # Read the tree-pipeline comparison out of the saved artifact rather than
        # hardcoding it. The first version of this line carried 5G-NIDD's numbers and
        # printed them for every dataset -- exactly the hand-copied-number failure the
        # rest of this repo is built to prevent.
        try:
            ref = json.loads((REPO / "results" / "_logs" / "signal1_cross_dataset.json").read_text())
            base = {r["condition"]: r for r in ref if r["dataset"] == args.dataset}
            if base:
                c0, f0 = base["clean"], base.get("A1_displacement", {})
                # A missing key printed as `nan` reads like a measurement that came out
                # undefined. It is not: on 2026-08-15 this line printed `displacement AUROC
                # nan` for every dataset because the saved baseline still called the attack
                # `A1_fragility`, renamed five days earlier. Say which it is, so a stale
                # baseline announces itself instead of impersonating a null result.
                auroc = f0.get("auroc")
                shown = f"{auroc:.3f}" if isinstance(auroc, (int, float)) else \
                    f"NOT IN THE SAVED BASELINE (conditions there: " \
                    f"{sorted(base)}) — re-run scripts/summarize_signal1.py"
                print(f"\n  XGBoost+TreeSHAP on {args.dataset} for comparison: "
                      f"at-zero {c0['frac_zero']:.2f}, at-ceiling {c0['frac_at_ceiling']:.2f}, "
                      f"informative {c0['frac_informative']:.2f}, "
                      f"displacement AUROC {shown}", flush=True)
            else:
                print(f"\n  (no tree-pipeline baseline saved for {args.dataset})", flush=True)
        except FileNotFoundError:
            print("\n  (run scripts/summarize_signal1.py for the tree baseline)", flush=True)
        return RunResult(summary=summary, floats=[{
            "float_id": "generality_mlp_ig", "kind": "table", "paper_section": "C3",
            "path": str(raw / "auroc.csv"),
            "claim": "the C3 negatives are not an artifact of XGBoost+TreeSHAP",
            "status": "done"}])

    run_experiment(f"generality_mlp_ig_{args.dataset}", experiment,
                   config=vars(args), seed=args.seeds[0])


if __name__ == "__main__":
    main()
