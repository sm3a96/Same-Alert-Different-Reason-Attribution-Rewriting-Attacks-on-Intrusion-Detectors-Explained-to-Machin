"""Erasure faithfulness as a runtime check: does the shown top-k carry the detection?

The benchmark's ground truth (Eq. eq:erasure) says a feature is causal when erasing it toward
its benign value moves the deployed detector. This check applies the same test at runtime to
the explanation the consumer is about to read: the share of erasure-causal mass that sits
inside the shown top-k,

    F_k(x) = sum_{j in T_k(x)} max(Delta_j(x), 0)  /  sum_j max(Delta_j(x), 0),

and reports -F_k as the nonconformity. Cost is d forward passes per flow, batched.

It exists in the panel because it is the check that follows directly from the ground truth,
and because measuring it is what exposed that cause displacement is mostly a reliance shift
an attacked flow's shown set is at least as erasure-faithful
on that flow as a clean one's, so this check is blind to it for a stated reason rather than an
unexplained one. It is reported as a failure with a mechanism, never as a detector.
"""
from __future__ import annotations

import numpy as np

from avert.signals.base import IntegritySignal
from avert.types import SignalName, SignalScore


class ErasureFaithfulnessSignal(IntegritySignal):
    name = SignalName.ERASURE_FAITHFULNESS

    def __init__(self, benign_reference: np.ndarray, top_k: int = 5):
        self.benign_reference = np.asarray(benign_reference, dtype=float)
        self.top_k = top_k

    def calibrate(self, clean_samples, clean_explanations, detector):
        return None                      # nothing to fit: the reference is the benign mean

    def shown_mass_fraction(self, sample, explanation, detector) -> tuple[float, np.ndarray]:
        x = sample.features.astype(float)
        d = len(x)
        pred = int(detector.predict(x.reshape(1, -1))[0])
        base = detector.predict_proba(x.reshape(1, -1))[0, pred]
        Xe = np.repeat(x.reshape(1, -1), d, axis=0)
        Xe[np.arange(d), np.arange(d)] = self.benign_reference
        deltas = base - detector.predict_proba(Xe)[:, pred]
        pos = np.clip(deltas, 0.0, None)
        total = float(pos.sum())
        shown = [int(j) for j in explanation.top_features()[: self.top_k]]
        frac = float(pos[shown].sum() / total) if total > 0 else 0.0
        return frac, deltas

    def score(self, sample, explanation, detector):
        frac, deltas = self.shown_mass_fraction(sample, explanation, detector)
        return SignalScore(self.name, score=-frac,
                           evidence={"shown_causal_mass_fraction": frac,
                                     "max_delta": float(deltas.max())})
