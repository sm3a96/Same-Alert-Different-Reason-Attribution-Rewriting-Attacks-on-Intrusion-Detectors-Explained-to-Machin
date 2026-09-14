"""Why does certified stability fail? Two incompatible explanations, one experiment.

An early design note (2026-06-15) records that only 7-10% of CLEAN 5G-NIDD
samples are certifiable, and concludes that clean tabular attributions are inherently
unstable so the signal has no stable baseline. A re-measurement on 2026-07-31 does not
reproduce that: 93% certify at sigma=0.05 with n=30, and 100% at n>=120.

If clean attributions actually certify, the signal's failure needs a different mechanism.
The candidate is saturation: R = sigma * Phi^-1(p_lo) is capped at sigma * Phi^-1(conf^(1/n))
because p_lo cannot exceed that even when every draw agrees. If clean AND attacked samples
both sit at that cap, the statistic is constant and carries no information — which produces
exactly the chance-level AUROC that was observed, for a completely different reason.

This measures the clean and attacked radius distributions side by side against the ceiling,
so the two explanations separate:

  unstable-baseline   -> clean radii spread low, many at zero
  saturation          -> clean AND attacked radii pile at the ceiling, AUROC ~ 0.5

  python scripts/diagnose_signal1.py --dataset fiveg_nidd --n 120
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from avert.benchmark import XIntBench
from avert.benchmark.ground_truth import causal_features
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.signals.certified_stability import CertifiedStabilitySignal, clopper_pearson_lower

ATTACKS = ["A1_misdirection", "A1_displacement", "A3_scaffolding"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiveg_nidd")
    ap.add_argument("--n", type=int, default=120, help="smoothing draws")
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.05)
    args = ap.parse_args()

    cap = args.sigma * float(norm.ppf(clopper_pearson_lower(args.n, args.n, args.conf)))
    print(f"[{args.dataset}] n={args.n} sigma={args.sigma}  ->  radius ceiling R_max = {cap:.5f}\n")

    data = load_dataset(args.dataset, load_config(f"datasets/{args.dataset}"))
    bench = XIntBench(data, n_cal=60, n_test=args.n_samples, top_k=5,
                      perm_samples=10, stability_n=args.n, seed=args.seed)
    st = bench._prepare()
    sig = CertifiedStabilitySignal(st.explainer, n=args.n, sigma=args.sigma, top_k=5,
                                   seed=args.seed, conf=args.conf)
    sig.calibrate(st.cal, [], st.detector)

    def radii(samples_dets):
        return np.array([sig.certified_radius(s, d)[0] for s, d in samples_dets])

    clean = radii([(s, st.detector) for s in st.test])
    gt = [causal_features(s, st.detector, st.benign_ref, 0.05, 3)[0] for s in st.test]

    def describe(tag, r):
        at_cap = float(np.mean(np.isclose(r, cap, rtol=1e-3)))
        print(f"  {tag:18s} mean {r.mean():.5f}  median {np.median(r):.5f}  "
              f"zero {np.mean(r == 0):.2f}  at-ceiling {at_cap:.2f}")
        return {"mean": float(r.mean()), "median": float(np.median(r)),
                "frac_zero": float(np.mean(r == 0)), "frac_at_ceiling": at_cap}

    out = {"dataset": args.dataset, "n": args.n, "sigma": args.sigma, "ceiling": cap,
           "class": st.target_class, "n_samples": len(st.test)}
    print(f"clean vs attacked radius distributions ({len(st.test)} flows, class {st.target_class}):")
    out["clean"] = describe("clean", clean)

    for attack in ATTACKS:
        atk = radii([bench.attacked(s, attack, st, list(c)) for s, c in zip(st.test, gt)])
        d = describe(attack, atk)
        d["auroc"] = round(float(roc_auc_score([0] * len(clean) + [1] * len(atk),
                                               list(-clean) + list(-atk))), 4)
        print(f"  {'':18s} AUROC (clean vs attacked, using -R as nonconformity) = {d['auroc']:.3f}")
        out[attack] = d

    verdict = ("SATURATION — clean and attacked both pile at the ceiling, so the statistic is "
               "near-constant and cannot discriminate"
               if out["clean"]["frac_at_ceiling"] > 0.5 else
               "UNSTABLE BASELINE — clean radii are spread low, matching the June account")
    out["verdict"] = verdict
    print(f"\nverdict: {verdict}")

    with open(f"results/_logs/signal1_diagnosis_{args.dataset}.json", "w") as f:
        json.dump(out, f, indent=1)


if __name__ == "__main__":
    main()
