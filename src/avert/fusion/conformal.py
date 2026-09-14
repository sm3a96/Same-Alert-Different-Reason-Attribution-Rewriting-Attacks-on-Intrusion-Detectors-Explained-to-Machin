"""Conformal fusion (Plan Section 3.6).

The signal scores are aggregated into a single nonconformity per explanation, then
split-conformal calibration on clean data turns it into a p-value with a
distribution-free false-alarm guarantee: under exchangeability of clean data,
P(p < alpha) <= alpha, so the integrity false-alarm rate is at most alpha.

`SignalAggregator` standardises each signal against its clean mean/std and takes the
max z-score (a single anomalous channel should fire), which keeps the fused score
one-dimensional so the conformal guarantee is clean. Phase 3 may replace the
aggregation with a learned combiner, but the conformal layer stays.

Sequential / drift extensions (e-processes, adaptive conformal inference) attach
here in Phase 3 via the temporal channel; this class is the within-regime core.
"""
from __future__ import annotations

import numpy as np

from avert.types import CalibrationSet, SignalName, SignalScore


class SignalAggregator:
    def __init__(self) -> None:
        self.mean: dict[SignalName, float] = {}
        self.std: dict[SignalName, float] = {}

    def fit(self, per_signal_scores: dict[SignalName, np.ndarray]) -> "SignalAggregator":
        for sig, arr in per_signal_scores.items():
            self.mean[sig] = float(np.mean(arr))
            self.std[sig] = float(np.std(arr) + 1e-8)
        return self

    def aggregate(self, scores: dict[SignalName, SignalScore]) -> float:
        zs = []
        for sig, s in scores.items():
            if s.abstained or sig not in self.mean:
                continue
            zs.append((s.score - self.mean[sig]) / self.std[sig])
        return float(max(zs)) if zs else 0.0


class ConformalFuser:
    """Split-conformal one-sided test; higher fused score = more anomalous."""

    def __init__(self) -> None:
        self._calib: np.ndarray | None = None

    def calibrate(self, clean_fused_scores: np.ndarray, dataset: str = "default") -> CalibrationSet:
        self._calib = np.sort(np.asarray(clean_fused_scores, dtype=float))
        return CalibrationSet(self._calib, dataset)

    def p_value(self, fused_score: float) -> float:
        assert self._calib is not None, "calibrate() first"
        n = len(self._calib)
        return float((1 + np.sum(self._calib >= fused_score)) / (n + 1))

    def is_violation(self, fused_score: float, alpha: float) -> bool:
        return self.p_value(fused_score) < alpha
