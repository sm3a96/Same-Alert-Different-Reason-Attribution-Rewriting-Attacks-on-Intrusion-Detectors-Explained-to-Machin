"""Erasure faithfulness as a runtime check: does the shown top-k carry the detection?

The benchmark's ground truth (Eq. eq:erasure) says a feature is causal when erasing it toward
its benign value moves the deployed detector. This check applies the same test at runtime to
the explanation the consumer is about to read: the share of erasure-causal mass that sits
inside the shown top-k,

    F_k(x) = sum_{j in T_k(x)} max(Delta_j(x), 0)  /  sum_j max(Delta_j(x), 0),

and reports -F_k as the nonconformity. Cost is d forward passes per flow, batched.

It exists in the panel because it is the check that follows directly from the ground truth,
and it is the only one of the seven whose point estimate is above chance against cause
displacement on every corpus (AUROC 0.58 to 0.67). The mechanism is that F_k reads the SHARE
of causal mass the shown set carries, and that share falls when the causal set grows, so the
check responds to a reliance shift as well as to an unfaithful attribution -- unlike a check
that compares the explanation with the model on the same flow, which a faithful explanation of
a shifted reliance gives nothing to flag.

That is a modest signal, not a defence: 0.58 to 0.67 is what it is, and the paper reports it
as the best of a panel that mostly fails rather than as a detector to field.

This docstring said until 2026-09-22 that the check is "blind" to cause displacement and that
cause displacement is "mostly a reliance shift". Both were wrong against the results it
describes: the check is the one that sees the attack, and the reliance term survives a Holm
correction pooled but not corpus by corpus, where its interval contains zero on two of three.
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
