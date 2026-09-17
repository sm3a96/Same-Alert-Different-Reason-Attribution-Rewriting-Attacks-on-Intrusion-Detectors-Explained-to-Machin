"""EIB A1 evaluation on real data (Plan Phase 1 P1-4 + the start of the
detection evaluation). For one target attack class we:

  1. establish interventional ground-truth causal features per attack flow,
  2. craft A1 attribution-evasion attacks (demote those features from the top-k while the
     prediction is preserved, protocol-valid perturbations),
  3. calibrate the AVERT monitor on CLEAN explanations of that class (within-class
     exchangeability), then measure
       - false-alarm on held-out CLEAN explanations,
       - detection rate on the A1-ATTACKED explanations,
       - attack efficacy (causal top-k overlap clean->attacked) and prediction-preservation.

  python scripts/eval_xint_a1.py --dataset fiveg_nidd

This isolates the A1 effect (corrupted vs faithful explanation), not benign-vs-attack.
"""
from __future__ import annotations

import argparse

import numpy as np

from avert.attribution.real import PermutationAttributor, TreeSHAPAttributor
from avert.benchmark.attacks import AttributionEvasion
from avert.benchmark.ground_truth import causal_features
from avert.benchmark.perturbation import bounds_for
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.detectors.real import XGBoostDetector
from avert.eval.runner import RunResult, run_experiment
from avert.pipeline import AVERT
from avert.signals.certified_stability import CertifiedStabilitySignal
from avert.signals.cross_method_consensus import CrossMethodConsensusSignal
from avert.types import FlowSample


def _samples(X, y, names, dataset, idx):
    return [FlowSample(features=X[i], feature_names=names, dataset=dataset,
                       true_label=int(y[i]), sample_id=f"s{i}") for i in idx]


def experiment(ctx) -> RunResult:
    name = ctx.config["dataset"]
    alpha = ctx.config.get("alpha", 0.05)
    top_k = ctx.config.get("top_k", 5)
    data = load_dataset(name, ctx.config)
    X, y, names = data.X, data.y, data.feature_names
    bi = data.benign_index()
    rng = np.random.default_rng(ctx.seed)

    # Detector on a balanced subsample; bounds + benign reference for attacks/ablation.
    tr = rng.permutation(len(X))[:60000]
    det = XGBoostDetector(n_estimators=150).fit(X[tr], y[tr])
    bounds = bounds_for(X[tr], data.feature_names, data.name)
    benign_ref = X[y == bi].mean(axis=0)

    # Target attack class = largest non-benign class the detector gets right.
    classes, counts = np.unique(y, return_counts=True)
    attack_classes = [c for c in classes if c != bi]
    target = max(attack_classes, key=lambda c: counts[list(classes).index(c)])

    pool = np.where(y == target)[0]
    rng.shuffle(pool)
    correct = [i for i in pool[:4000] if int(det.predict(X[i].reshape(1, -1))[0]) == target]
    n_cal, n_test = ctx.config.get("n_cal", 150), ctx.config.get("n_test", 100)
    cal_i, test_i = correct[:n_cal], correct[n_cal : n_cal + n_test]

    treeshap = TreeSHAPAttributor()
    signals = [
        CertifiedStabilitySignal(treeshap, n=15, seed=ctx.seed),
        CrossMethodConsensusSignal([PermutationAttributor(X[tr], n_samples=6), treeshap], top_k=top_k),
    ]
    avert = AVERT(det, treeshap, signals, alpha=alpha)
    avert.calibrate(_samples(X, y, names, name, cal_i))            # clean, within-class

    clean_test = _samples(X, y, names, name, test_i)
    attacker = AttributionEvasion(bounds, treeshap, top_k=top_k,
                                  n_iter=ctx.config.get("n_iter", 40),
                                  restarts=ctx.config.get("restarts", 2), seed=ctx.seed)

    clean_viol, atk_viol, overlap_clean, overlap_atk, pred_kept = [], [], [], [], []
    for s in clean_test:
        causal, _ = causal_features(s, det, benign_ref, delta_threshold=0.05, min_k=3)
        causal_set = set(causal)
        e_clean = treeshap.explain(det, s.features, names, int(det.predict(s.features.reshape(1, -1))[0]),
                                   s.sample_id, top_k=top_k)
        overlap_clean.append(len(set(e_clean.top_features()) & causal_set))
        clean_viol.append(avert.assess(s).violation)

        atk = attacker.apply(s, det, causal_idx=causal)
        c0 = int(det.predict(s.features.reshape(1, -1))[0])
        pred_kept.append(int(det.predict(atk.features.reshape(1, -1))[0]) == c0)
        e_atk = treeshap.explain(det, atk.features, names, c0, atk.sample_id, top_k=top_k)
        overlap_atk.append(len(set(e_atk.top_features()) & causal_set))
        atk_viol.append(avert.assess(atk).violation)

    summary = {
        "dataset": name, "target_class": data.label_names[target], "alpha": alpha,
        "n_test": len(clean_test),
        "false_alarm_clean": round(float(np.mean(clean_viol)), 3),
        "detection_rate_A1": round(float(np.mean(atk_viol)), 3),
        "prediction_preserved": round(float(np.mean(pred_kept)), 3),
        "causal_topk_overlap_clean": round(float(np.mean(overlap_clean)), 2),
        "causal_topk_overlap_attacked": round(float(np.mean(overlap_atk)), 2),
    }
    print("\n=== EIB A1 evaluation ===")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    print("================================\n")

    return RunResult(summary=summary, floats=[{
        "float_id": f"a1_detection_{name}", "kind": "table", "paper_section": "Evaluation",
        "path": str(ctx.run_dir / "meta"),
        "claim": "monitor detects A1 attribution-evasion above the clean false-alarm rate",
    }])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiveg_nidd")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = {"dataset": args.dataset, "alpha": 0.05}
    if args.dataset != "synthetic":
        cfg.update(load_config(f"datasets/{args.dataset}"))
        cfg["dataset"] = args.dataset
    run_experiment(f"xint_a1_{args.dataset}", experiment, cfg, seed=args.seed)
