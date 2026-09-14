"""Interventional verification of XInt-Bench ground truth (Plan Section 4).

Ground truth is valid as ATTRIBUTION ground truth only when it is tied to the
detector, not merely to the data-generating process: ablating a ground-truth feature
toward its benign value must move the detection, while ablating a non-causal feature
must not. This guard catches the case where a detector relies on features outside
the constructed causal set — an honest explanation of such a detector should never be
counted as corrupted, so those instances are flagged, not scored against the
construction labels.
"""
from __future__ import annotations

import numpy as np

from avert.detectors.base import Detector
from avert.types import FlowSample


def causal_features(
    sample: FlowSample,
    detector: Detector,
    benign_reference: np.ndarray,
    delta_threshold: float = 0.05,
    min_k: int = 3,
) -> tuple[list[int], np.ndarray]:
    """Interventional ground truth: the features whose ablation toward benign actually
    moves the detection are the ones a faithful explanation must rank highly. Returns
    (causal feature indices, per-feature detection deltas).

    A feature j is causal if replacing x[j] with the benign reference drops the
    predicted-class probability by at least `delta_threshold`. If none cross it (rare),
    fall back to the top-`min_k` by delta so every attacked sample has a ground truth.
    """
    x = sample.features
    pred = int(detector.predict(x.reshape(1, -1))[0])
    base = detector.predict_proba(x.reshape(1, -1))[0, pred]
    deltas = np.zeros(len(x))
    for j in range(len(x)):
        xj = x.copy()
        xj[j] = benign_reference[j]
        deltas[j] = base - detector.predict_proba(xj.reshape(1, -1))[0, pred]
    causal = [j for j in range(len(x)) if deltas[j] >= delta_threshold]
    if not causal:
        causal = list(np.argsort(-deltas)[:min_k])
    return causal, deltas


def interventional_check(
    sample: FlowSample,
    detector: Detector,
    benign_reference: np.ndarray,
    delta_threshold: float = 0.1,
) -> dict:
    """Return per-feature detection deltas and whether the construction labels hold
    for THIS detector (i.e. ground-truth features actually move detection)."""
    assert sample.ground_truth_causal is not None, "needs construction labels"
    x = sample.features
    pred = int(detector.predict(x.reshape(1, -1))[0])
    base = detector.predict_proba(x.reshape(1, -1))[0, pred]
    name_to_idx = {n: i for i, n in enumerate(sample.feature_names)}
    causal_idx = {name_to_idx[n] for n in sample.ground_truth_causal}

    deltas = {}
    for j, name in enumerate(sample.feature_names):
        xj = x.copy()
        xj[j] = benign_reference[j]
        deltas[name] = base - detector.predict_proba(xj.reshape(1, -1))[0, pred]

    causal_moves = all(abs(deltas[n]) >= delta_threshold for n in sample.ground_truth_causal)
    noncausal_quiet = all(
        abs(deltas[sample.feature_names[j]]) < delta_threshold
        for j in range(len(x)) if j not in causal_idx
    )
    return {
        "deltas": deltas,
        "labels_valid_for_detector": bool(causal_moves),
        "noncausal_quiet": bool(noncausal_quiet),
        "flag_detector_unfaithful": not bool(causal_moves),
    }
