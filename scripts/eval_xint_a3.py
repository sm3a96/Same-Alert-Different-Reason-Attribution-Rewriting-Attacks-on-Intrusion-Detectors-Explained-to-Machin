"""EIB A3 (scaffolding) evaluation. Signal 3 (cross-method consensus) is built to
catch this: a perturbation-based attributor (permutation) is fooled by the scaffold's
probe-routing while TreeSHAP reads the real model, so they diverge. We calibrate the
consensus null on the CLEAN detector and measure:
    - false-alarm: violation rate scoring clean (real) explanations,
    - detection:   violation rate scoring scaffolded explanations.

  python scripts/eval_xint_a3.py --dataset fiveg_nidd
"""
from __future__ import annotations

import argparse

import numpy as np

from avert.attribution.real import PermutationAttributor, TreeSHAPAttributor
from avert.benchmark.scaffold import make_scaffolded_detector
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.runner import RunResult, run_experiment
from avert.fusion.conformal import ConformalFuser
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

    tr = rng.permutation(len(X))[:60000]
    scaffold, real = make_scaffolded_detector(X[tr], y[tr], n_corrupt=12,
                                              contamination=0.3, seed=ctx.seed)

    classes, counts = np.unique(y, return_counts=True)
    target = max([c for c in classes if c != bi],
                 key=lambda c: counts[list(classes).index(c)])
    pool = np.where(y == target)[0]
    rng.shuffle(pool)
    correct = [i for i in pool[:5000] if int(real.predict(X[i].reshape(1, -1))[0]) == target]
    n_cal, n_test = ctx.config.get("n_cal", 150), ctx.config.get("n_test", 120)
    cal_i, test_i = correct[:n_cal], correct[n_cal : n_cal + n_test]

    permutation = PermutationAttributor(X[tr], n_samples=30, seed=ctx.seed)   # lower variance
    treeshap = TreeSHAPAttributor()
    consensus = CrossMethodConsensusSignal([permutation, treeshap], top_k=top_k)

    cal = _samples(X, y, names, name, cal_i)
    cal_expls = [treeshap.explain(real, s.features, names, target, s.sample_id, top_k) for s in cal]
    consensus.calibrate(cal, cal_expls, real)
    fuser = ConformalFuser()
    fuser.calibrate(np.array([consensus.score(s, e, real).score for s, e in zip(cal, cal_expls)]))

    test = _samples(X, y, names, name, test_i)

    def disagreement_and_violation(sample, detector):
        e = treeshap.explain(detector, sample.features, names,
                             int(detector.predict(sample.features.reshape(1, -1))[0]),
                             sample.sample_id, top_k)
        sc = consensus.score(sample, e, detector)
        return sc.evidence["disagreement"], fuser.is_violation(sc.score, alpha)

    clean_dis, clean_v, a3_dis, a3_v = [], [], [], []
    for s in test:
        d0, v0 = disagreement_and_violation(s, real)        # clean
        d1, v1 = disagreement_and_violation(s, scaffold)    # A3
        clean_dis.append(d0)
        clean_v.append(v0)
        a3_dis.append(d1)
        a3_v.append(v1)

    from sklearn.metrics import roc_auc_score
    labels = np.r_[np.zeros(len(clean_dis)), np.ones(len(a3_dis))]
    auroc = float(roc_auc_score(labels, np.r_[clean_dis, a3_dis]))
    summary = {
        "dataset": name, "target_class": data.label_names[target], "alpha": alpha,
        "n_test": len(test),
        "AUROC_clean_vs_A3": round(auroc, 3),
        "false_alarm_clean": round(float(np.mean(clean_v)), 3),
        "detection_rate_A3": round(float(np.mean(a3_v)), 3),
        "mean_disagreement_clean": round(float(np.mean(clean_dis)), 3),
        "mean_disagreement_scaffold": round(float(np.mean(a3_dis)), 3),
    }
    print("\n=== EIB A3 (scaffolding) evaluation ===")
    for k, v in summary.items():
        print(f"  {k:28s} {v}")
    print("=============================================\n")

    return RunResult(summary=summary, floats=[{
        "float_id": f"a3_detection_{name}", "kind": "table", "paper_section": "Evaluation",
        "path": str(ctx.run_dir / "meta"),
        "claim": "consensus (Signal 3) detects A3 scaffolding above the clean false-alarm rate",
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
    run_experiment(f"xint_a3_{args.dataset}", experiment, cfg, seed=args.seed)
