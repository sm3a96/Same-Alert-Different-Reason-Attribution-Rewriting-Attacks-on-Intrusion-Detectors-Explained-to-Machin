"""Why does Signal 4 score AUROC 0.022 on 5G-NIDD A1 displacement?

Signal 4 is meant to be blind by construction once the attack projects onto the realizable
set: every residual is zero on both arms, every pair ties, AUROC 0.5. Six of the nine matrix
cells behave exactly that way. Three do not, and one of them -- 5G-NIDD, A1 displacement --
comes out at 0.022, which is not blindness. An AUROC that far below chance is a signal that
separates the two arms almost perfectly with its sign flipped, so a defender who inverts it
detects the displacement attack 97.8% of the time. That would puncture the paper's
load-bearing negative, and it has to be run down before Table 5 is written, not after.

Two candidate causes, and they call for opposite conclusions:

  (a) Real leak. The projection over-constrains: it produces flows that satisfy the
      extractor arithmetic more exactly than genuine captures do, and the gap is large
      enough to survive quantisation. Then the negative claim is wrong as stated and the
      attack needs fixing.
  (b) Float noise. Clean rows carry rounding at the 1e-7 level from the extractor's own
      arithmetic; the projection writes exact values; the residuals differ only in the last
      bits. AUROC sees a clean ordering where a defender sees nothing, because no monitor
      thresholds a relative violation at 1e-7.

This script measures which, by printing the two residual distributions against the 1e-3
tolerance the signal itself uses for its `realizable` flag. Run it before trusting any
Signal 4 number:

    python scripts/diagnose_feature_consistency.py --dataset fiveg_nidd --seed 0
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from avert.benchmark import XIntBench
from avert.benchmark.harness import target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.signals.feature_consistency import FeatureConsistencySignal

log = logging.getLogger("diag-fc")

OUT = Path("results/_logs/feature_consistency_diagnosis.json")
TOL = 1e-3  # the signal's own realizable threshold; a defender cannot act below it


def quantiles(a: np.ndarray) -> dict:
    a = np.asarray(a, dtype=float)
    qs = [0, 50, 90, 99, 100]
    return {"n": int(a.size), "mean": float(a.mean()), "n_exact_zero": int((a == 0.0).sum()),
            "n_above_tol": int((a > TOL).sum()),
            **{f"p{q}": float(np.percentile(a, q)) for q in qs}}


def auroc(clean: np.ndarray, atk: np.ndarray) -> float:
    from sklearn.metrics import roc_auc_score
    y = [0] * len(clean) + [1] * len(atk)
    return float(roc_auc_score(y, list(clean) + list(atk)))


def diagnose(dataset: str, seed: int, attack: str) -> dict:
    data = load_dataset(dataset, load_config(f"datasets/{dataset}"))
    classes, _ = target_classes(data, 3, n_cal=100, n_test=40, seed=seed, min_count=2000,
                                seeds=[seed])
    cells = []
    for c in classes:
        bench = XIntBench(data, n_cal=100, n_test=40, top_k=5, perm_samples=10,
                          stability_n=30, seed=seed)
        st = bench._prepare(None, None, c)
        sig = FeatureConsistencySignal(data.name, st.names)
        bound = sig._ensure(st.test[0])

        clean_X = np.stack([s.features for s in st.test]).astype(float)
        atk_X = np.stack([bench.attacked(s, attack, st, [])[0].features for s in st.test])
        r_clean, r_atk = bound.residual(clean_X), bound.residual(atk_X.astype(float))

        # The number a defender could actually act on: both arms thresholded at the signal's
        # own tolerance, which is what any deployed input validator would do.
        thr_clean = (r_clean > TOL).astype(float)
        thr_atk = (r_atk > TOL).astype(float)
        cname = data.label_names[c] if data.label_names else str(c)
        cells.append({
            "class": cname,
            "clean_residual": quantiles(r_clean), "attacked_residual": quantiles(r_atk),
            "auroc_raw": round(auroc(r_clean, r_atk), 3),
            "auroc_thresholded": round(auroc(thr_clean, thr_atk), 3),
            "max_residual_either_arm": float(max(r_clean.max(), r_atk.max())),
            "verdict": ("float-noise: neither arm violates a single constraint at the 1e-3 "
                        "tolerance the signal itself uses, so the ordering AUROC sees is in "
                        "bits no monitor can threshold"
                        if max(r_clean.max(), r_atk.max()) <= TOL else
                        "REAL SEPARATION above tolerance -- the negative claim needs revisiting"),
        })
        print(f"  {dataset} class={cname} raw AUROC={cells[-1]['auroc_raw']:.3f} "
              f"thresholded={cells[-1]['auroc_thresholded']:.3f} "
              f"max residual={cells[-1]['max_residual_either_arm']:.3e}")
    return {"dataset": dataset, "seed": seed, "attack": attack, "tolerance": TOL, "cells": cells}


def main() -> None:
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    p = argparse.ArgumentParser()
    p.add_argument("--dataset", default="fiveg_nidd")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--attack", default="A1_displacement")
    a = p.parse_args()
    rep = diagnose(a.dataset, a.seed, a.attack)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(rep, indent=2))
    print(f"\nsaved: {OUT}")


if __name__ == "__main__":
    main()
