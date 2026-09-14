"""Unified detection evaluation with multi-signal conformal FUSION (Plan Section 3.6).

Calibrates the monitor (Signal 1 empirical stability + Signal 3 consensus) on clean
within-class traffic, then scores three conditions on held-out flows:
    clean (real detector)         -> false-alarm
    A1 attribution-evasion        -> detection
    A3 scaffolding (via scaffold) -> detection
and reports AUROC (clean vs attack) for EACH signal and for the FUSED score, plus the
fused detection rate at alpha. This is the test of whether independent weak signals fuse
into a strong detector.

  python scripts/eval_detection.py --dataset fiveg_nidd
"""
from __future__ import annotations

import argparse

import numpy as np
from sklearn.metrics import roc_auc_score

from avert.attribution.real import PermutationAttributor, TreeSHAPAttributor
from avert.benchmark.attacks import AttributionEvasion
from avert.benchmark.ground_truth import causal_features
from avert.benchmark.perturbation import bounds_for
from avert.benchmark.scaffold import make_scaffolded_detector
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.runner import RunResult, run_experiment
from avert.fusion.conformal import ConformalFuser, SignalAggregator
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

    tr = rng.permutation(len(X))[:60000]
    scaffold, real = make_scaffolded_detector(X[tr], y[tr], n_corrupt=12, contamination=0.3, seed=ctx.seed)
    bounds = bounds_for(X[tr], data.feature_names, data.name)
    benign_ref = X[y == bi].mean(axis=0)

    classes, counts = np.unique(y, return_counts=True)
    target = max([c for c in classes if c != bi], key=lambda c: counts[list(classes).index(c)])
    pool = np.where(y == target)[0]
    rng.shuffle(pool)
    correct = [i for i in pool[:5000] if int(real.predict(X[i].reshape(1, -1))[0]) == target]
    n_cal, n_test = ctx.config.get("n_cal", 120), ctx.config.get("n_test", 80)
    cal_i, test_i = correct[:n_cal], correct[n_cal : n_cal + n_test]

    treeshap = TreeSHAPAttributor()
    permutation = PermutationAttributor(X[tr], n_samples=20, seed=ctx.seed)
    signals = [
        CertifiedStabilitySignal(treeshap, n=12, seed=ctx.seed),
        CrossMethodConsensusSignal([permutation, treeshap], top_k=top_k),
    ]

    def explain(s, det):
        c = int(det.predict(s.features.reshape(1, -1))[0])
        return treeshap.explain(det, s.features, names, c, s.sample_id, top_k), c

    # Calibrate signals + aggregator + fuser on CLEAN within-class traffic (real detector).
    cal = _samples(X, y, names, name, cal_i)
    cal_e = [explain(s, real)[0] for s in cal]
    for sig in signals:
        sig.calibrate(cal, cal_e, real)
    per_sig = {sig.name: np.array([sig.score(s, e, real).score for s, e in zip(cal, cal_e)]) for sig in signals}
    agg = SignalAggregator().fit(per_sig)
    fuser = ConformalFuser()
    fuser.calibrate(np.array([agg.aggregate({sig.name: sig.score(s, e, real) for sig in signals})
                              for s, e in zip(cal, cal_e)]))

    def score_all(s, det):
        e, _ = explain(s, det)
        sc = {sig.name: sig.score(s, e, det) for sig in signals}
        return {k: v.score for k, v in sc.items()}, agg.aggregate(sc)

    test = _samples(X, y, names, name, test_i)
    attacker = AttributionEvasion(bounds, treeshap, top_k=top_k, n_iter=40, restarts=2, seed=ctx.seed)

    rows = {"clean": [], "A1": [], "A3": []}
    fused = {"clean": [], "A1": [], "A3": []}
    for s in test:
        per, fu = score_all(s, real)
        rows["clean"].append(per)
        fused["clean"].append(fu)
        causal, _ = causal_features(s, real, benign_ref, delta_threshold=0.05, min_k=3)
        atk = attacker.apply(s, real, causal_idx=causal)
        per, fu = score_all(atk, real)
        rows["A1"].append(per)
        fused["A1"].append(fu)
        per, fu = score_all(s, scaffold)
        rows["A3"].append(per)
        fused["A3"].append(fu)

    def auroc(attack):
        out = {}
        for sig in signals:
            cs = [r[sig.name] for r in rows["clean"]]
            as_ = [r[sig.name] for r in rows[attack]]
            out[sig.name.value] = round(float(roc_auc_score([0] * len(cs) + [1] * len(as_), cs + as_)), 3)
        out["FUSED"] = round(float(roc_auc_score(
            [0] * len(fused["clean"]) + [1] * len(fused[attack]), fused["clean"] + fused[attack])), 3)
        return out

    fa = float(np.mean([fuser.is_violation(f, alpha) for f in fused["clean"]]))
    det = {a: round(float(np.mean([fuser.is_violation(f, alpha) for f in fused[a]])), 3) for a in ("A1", "A3")}

    summary = {
        "dataset": name, "target_class": data.label_names[target], "n_test": len(test),
        "false_alarm_fused": round(fa, 3),
        "AUROC_A1": auroc("A1"), "AUROC_A3": auroc("A3"),
        "fused_detection@alpha": det,
    }
    print("\n=== Unified detection eval (fusion) ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print("=======================================\n")
    return RunResult(summary=summary, floats=[{
        "float_id": f"fusion_detection_{name}", "kind": "table", "paper_section": "Evaluation",
        "path": str(ctx.run_dir / "meta"), "claim": "fused vs per-signal detection AUROC for A1 and A3"}])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiveg_nidd")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = {"dataset": args.dataset, "alpha": 0.05}
    if args.dataset != "synthetic":
        cfg.update(load_config(f"datasets/{args.dataset}"))
        cfg["dataset"] = args.dataset
    run_experiment(f"detection_{args.dataset}", experiment, cfg, seed=args.seed)
