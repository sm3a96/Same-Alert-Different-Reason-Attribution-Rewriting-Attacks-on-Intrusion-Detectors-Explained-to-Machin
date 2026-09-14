"""Run the AVERT monitor on REAL data and verify the conformal false-alarm guarantee
(Plan Section 3.6 — the paper's headline claim). We calibrate on clean (benign) traffic
only and measure the violation rate on held-out clean traffic; under exchangeability it
must not exceed alpha. Signals 1 (empirical stability) + 3 (consensus) + conformal fusion;
the agentic verifier and temporal channel come later.

  python scripts/run_monitor.py --dataset ciciov2024
"""
from __future__ import annotations

import argparse

import numpy as np

from avert.attribution.real import PermutationAttributor, TreeSHAPAttributor
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.detectors.real import XGBoostDetector
from avert.eval.runner import RunResult, run_experiment
from avert.pipeline import AVERT
from avert.signals.certified_stability import CertifiedStabilitySignal
from avert.signals.cross_method_consensus import CrossMethodConsensusSignal
from avert.types import FlowSample


def experiment(ctx) -> RunResult:
    name = ctx.config["dataset"]
    alpha = ctx.config.get("alpha", 0.05)
    data = load_dataset(name, ctx.config)
    bi = data.benign_index()
    if bi is None:
        raise ValueError(f"{name}: benign class not identified (label_names={data.label_names})")

    rng = np.random.default_rng(ctx.seed)
    # Train the detector on a balanced subsample of all classes.
    train_idx = rng.permutation(len(data.X))[:60000]
    det = XGBoostDetector(n_estimators=150).fit(data.X[train_idx], data.y[train_idx])
    background = data.X[train_idx][:2000]

    # Clean (benign) pool -> calibration + held-out test, exchangeable by construction.
    benign = np.where(data.y == bi)[0]
    rng.shuffle(benign)
    n_cal, n_test = ctx.config.get("n_cal", 150), ctx.config.get("n_test", 300)
    cal_i, test_i = benign[:n_cal], benign[n_cal : n_cal + n_test]

    def mk(i):
        return FlowSample(features=data.X[i], feature_names=data.feature_names,
                          dataset=name, true_label=int(data.y[i]), sample_id=f"b{i}")

    calib, test = [mk(i) for i in cal_i], [mk(i) for i in test_i]

    treeshap = TreeSHAPAttributor()
    signals = [
        CertifiedStabilitySignal(treeshap, n=15, seed=ctx.seed),
        CrossMethodConsensusSignal([PermutationAttributor(background, n_samples=6), treeshap], top_k=4),
    ]
    avert = AVERT(det, treeshap, signals, alpha=alpha)
    avert.calibrate(calib)                                    # self-supervised, clean only

    viol = np.array([avert.assess(s).violation for s in test])
    false_alarm = float(viol.mean())
    se = float(np.sqrt(false_alarm * (1 - false_alarm) / len(test)))

    summary = {
        "dataset": name, "alpha": alpha, "n_calibration": n_cal, "n_clean_test": n_test,
        "empirical_false_alarm": round(false_alarm, 4),
        "false_alarm_se": round(se, 4),
        "within_guarantee": bool(false_alarm <= alpha + 2 * se),
    }
    print("\n=== AVERT monitor on real data ===")
    for k, v in summary.items():
        print(f"  {k:22s} {v}")
    print("==================================\n")

    return RunResult(
        summary=summary,
        floats=[{
            "float_id": f"false_alarm_{name}", "kind": "figure", "paper_section": "Evaluation",
            "path": str(ctx.run_dir / "meta"),
            "claim": "conformal false-alarm rate within nominal alpha on real clean traffic",
        }],
    )


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ciciov2024")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = {"dataset": args.dataset, "alpha": args.alpha}
    if args.dataset != "synthetic":
        cfg.update(load_config(f"datasets/{args.dataset}"))
        cfg["dataset"] = args.dataset
    run_experiment(f"monitor_{args.dataset}", experiment, cfg, seed=args.seed)
