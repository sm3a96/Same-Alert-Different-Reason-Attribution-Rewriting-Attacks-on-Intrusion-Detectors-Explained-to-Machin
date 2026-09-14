"""Is "only 7-10% of clean samples are certifiable" a property of the domain, or of n?

The June run recorded that certified attribution stability fails on tabular NIDS because
almost no CLEAN sample certifies. That is one of the paper's three claims. It was measured
with 30 smoothing draws.

The certified radius is R = sigma * Phi^-1(p_lo), where p_lo is the Clopper-Pearson lower
bound on the probability that the top-k set is the modal one, and R is defined as 0 whenever
p_lo <= 0.5. With n draws and a two-sided confidence level, p_lo cannot exceed 0.5 at all
until n is large enough -- so a small n forces R = 0 no matter how stable the attribution
actually is. That would make the finding an artifact of the measurement rather than a
statement about tabular attributions.

This sweeps n and sigma on clean flows and reports the certifiable fraction, so the two
explanations separate. It also reports the ceiling: the largest p_lo reachable at each n
even when every single draw agrees, which is the decisive number.

  python scripts/diagnose_certifiability.py --dataset fiveg_nidd
"""
from __future__ import annotations

import argparse
import json

import numpy as np
from scipy.stats import norm

from avert.benchmark import XIntBench
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.signals.certified_stability import CertifiedStabilitySignal, clopper_pearson_lower

NS = [30, 60, 120, 250, 500]
SIGMAS = [0.05, 0.1]


def ceiling(n: int, conf: float) -> tuple[float, float]:
    """p_lo and R/sigma when ALL n draws agree — the best the estimator can do at this n.

    `conf` is the TAIL probability (0.05 = a 95% lower bound), matching the helper.
    """
    p_lo = clopper_pearson_lower(n, n, conf)
    return p_lo, float(norm.ppf(p_lo)) if p_lo > 0.5 else 0.0


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiveg_nidd")
    ap.add_argument("--n-samples", type=int, default=40)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--conf", type=float, default=0.05)  # tail prob: 0.05 = a 95% lower bound
    args = ap.parse_args()

    print("Estimator ceiling — p_lo and R/sigma when every draw agrees:")
    print(f"  {'n':>5s} {'max p_lo':>10s} {'max R/sigma':>12s}   certifiable at all?")
    for n in NS:
        p, r = ceiling(n, args.conf)
        print(f"  {n:5d} {p:10.4f} {r:12.4f}   {'yes' if r > 0 else 'NO — R is 0 by construction'}")

    data = load_dataset(args.dataset, load_config(f"datasets/{args.dataset}"))
    bench = XIntBench(data, n_cal=60, n_test=args.n_samples, top_k=5,
                      perm_samples=10, stability_n=30, seed=args.seed)
    st = bench._prepare()
    print(f"\nMeasured on {len(st.test)} clean {args.dataset} flows "
          f"(class {st.target_class}), conf={args.conf}:")
    print(f"  {'sigma':>6s} {'n':>5s} {'certifiable':>12s} {'mean R':>9s} {'median R':>9s}")

    out = []
    for sigma in SIGMAS:
        for n in NS:
            sig = CertifiedStabilitySignal(st.explainer, n=n, sigma=sigma, top_k=5,
                                           seed=args.seed, conf=args.conf)
            sig.calibrate(st.cal, [], st.detector)
            radii = [sig.certified_radius(s, st.detector)[0] for s in st.test]
            frac = float(np.mean([r > 0 for r in radii]))
            row = {"sigma": sigma, "n": n, "certifiable_frac": round(frac, 4),
                   "mean_radius": round(float(np.mean(radii)), 5),
                   "median_radius": round(float(np.median(radii)), 5)}
            out.append(row)
            print(f"  {sigma:6.2f} {n:5d} {frac:12.2f} {np.mean(radii):9.4f} {np.median(radii):9.4f}",
                  flush=True)

    with open(f"results/_logs/certifiability_sweep_{args.dataset}.json", "w") as f:
        json.dump({"dataset": args.dataset, "n_samples": len(st.test), "conf": args.conf,
                   "class": st.target_class, "rows": out}, f, indent=1)


if __name__ == "__main__":
    main()
