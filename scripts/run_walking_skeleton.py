"""Walking skeleton: exercises the entire AVERT pipeline end to end on synthetic
data with toy components, so integration is proven before any real component lands
Run:  python scripts/run_walking_skeleton.py

It loads data, trains a detector, calibrates AVERT on clean traffic only, measures
the clean false-alarm rate against the conformal target, attacks test samples,
measures detection, runs diagnose+repair, and checks EIB interventional
ground truth. None of this is a research result; it is a smoke test of the wiring.
"""
from __future__ import annotations

import numpy as np

from avert.attribution.toy import CoefInputAttributor, OcclusionAttributor
from avert.benchmark.attacks import ToyAttributionEvasion
from avert.benchmark.ground_truth import interventional_check
from avert.data.datasets import SyntheticDataset
from avert.data.splits import train_calib_test_split
from avert.detectors.toy import ToyDetector
from avert.eval.runner import RunResult, run_experiment
from avert.pipeline import AVERT
from avert.signals.certified_stability import CertifiedStabilitySignal
from avert.signals.cross_method_consensus import CrossMethodConsensusSignal


def experiment(ctx) -> RunResult:
    alpha = ctx.config.get("alpha", 0.05)
    ds = SyntheticDataset(n=600, d=12, n_causal=4, seed=ctx.seed)
    data = ds.load()
    splits = train_calib_test_split(data, seed=ctx.seed)

    detector = ToyDetector().fit(splits["train"].X, splits["train"].y)
    acc = float((detector.predict(splits["test"].X) == splits["test"].y).mean())

    mean_vec = splits["train"].X.mean(axis=0)
    coef_attr = CoefInputAttributor()
    occ_attr = OcclusionAttributor(background_mean=mean_vec)

    signals = [
        CertifiedStabilitySignal(coef_attr, n=30, sigma=0.1, seed=ctx.seed),
        CrossMethodConsensusSignal([coef_attr, occ_attr], top_k=4),
    ]
    avert = AVERT(detector, coef_attr, signals, fallback_attributor=coef_attr, alpha=alpha)

    # Self-supervised calibration on CLEAN calibration traffic only.
    clean_calib = splits["calibration"].to_samples()
    avert.calibrate(clean_calib)

    # Clean held-out false-alarm rate (should be near alpha).
    clean_test = splits["test"].to_samples()
    clean_violations = np.array([avert.assess(s).violation for s in clean_test])
    false_alarm = float(clean_violations.mean())

    # Attack a subset of clean test samples and measure detection.
    attack = ToyAttributionEvasion(radius=0.6, n_tries=150, seed=ctx.seed)
    attacked = [attack.apply(s, detector, coef_attr) for s in clean_test[:40]]
    attacked_violations = np.array([avert.assess(s).violation for s in attacked])
    detection = float(attacked_violations.mean())

    # Diagnose + repair one attacked sample.
    verdict, repair_res = avert.assess_and_repair(attacked[0])

    # EIB interventional ground-truth check on one sample.
    s0 = clean_test[0]
    s0.ground_truth_causal = ds.causal_features
    iv = interventional_check(s0, detector, benign_reference=mean_vec, delta_threshold=0.02)

    summary = {
        "detector_accuracy": round(acc, 3),
        "alpha": alpha,
        "clean_false_alarm": round(false_alarm, 3),
        "attacked_detection_rate": round(detection, 3),
        "example_verdict_pvalue": round(verdict.p_value, 3),
        "example_violation": verdict.violation,
        "example_diagnosis": str(verdict.diagnosis),
        "example_repaired": repair_res.was_repaired,
        "gt_labels_valid_for_detector": iv["labels_valid_for_detector"],
    }
    print("\n=== AVERT walking skeleton ===")
    for k, v in summary.items():
        print(f"  {k:32s} {v}")
    print("==============================\n")

    return RunResult(
        summary=summary,
        floats=[{
            "float_id": "smoke", "kind": "summary", "paper_section": "n/a",
            "path": str(ctx.run_dir / "meta"), "claim": "pipeline wiring runs end to end",
        }],
    )


if __name__ == "__main__":
    run_experiment("walking_skeleton", experiment, config={"alpha": 0.05}, seed=0)
