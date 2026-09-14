"""Signal 3: cross-method consensus (Plan Section 3.3).

Compute attributions with mechanistically distinct methods, measure top-k rank
disagreement, and score it against a per-class null calibrated on clean data, so
the signal is the calibrated tail probability of the observed disagreement rather
than a raw correlation. Methods that disagree legitimately to different degrees on
different classes are handled by the per-class calibration.

This implementation is functionally complete for the walking skeleton; Phase 2
swaps the toy attributors for the real method->model mapping and replaces the
top-k Jaccard distance with a weighted Kendall statistic emphasising the top-k.
"""
from __future__ import annotations

from collections import defaultdict

import numpy as np

from avert.attribution.base import Attributor
from avert.signals.base import IntegritySignal
from avert.types import SignalName, SignalScore


def _mean_pairwise_rank_distance(attr_vectors: list[np.ndarray]) -> float:
    """Disagreement = 1 - mean pairwise Spearman correlation of |attributions|. Rank
    correlation over all features is far more robust than top-k set overlap (which is
    near-degenerate in high dimensions), and it is what gives the consensus signal its
    discriminative power — see the A3 diagnostic (clean ~0.66, scaffolded ~0.43)."""
    from scipy.stats import spearmanr

    if len(attr_vectors) < 2:
        return 0.0
    corrs = []
    for i in range(len(attr_vectors)):
        for j in range(i + 1, len(attr_vectors)):
            r = spearmanr(np.abs(attr_vectors[i]), np.abs(attr_vectors[j])).correlation
            corrs.append(0.0 if np.isnan(r) else float(r))
    return float(1.0 - np.mean(corrs))


class CrossMethodConsensusSignal(IntegritySignal):
    name = SignalName.CROSS_METHOD_CONSENSUS

    def __init__(self, attributors: list[Attributor], top_k: int = 5):
        self.attributors = attributors
        self.top_k = top_k
        self._null: dict[int, np.ndarray] = {}  # per predicted-class null of disagreement

    def _disagreement(self, sample, detector, predicted_class) -> float:
        vecs = []
        for attr in self.attributors:
            if not attr.applicable_to(detector):
                continue
            e = attr.explain(
                detector, sample.features, sample.feature_names,
                predicted_class, sample.sample_id or "x", top_k=self.top_k,
            )
            vecs.append(e.attributions)
        return _mean_pairwise_rank_distance(vecs)

    def calibrate(self, clean_samples, clean_explanations, detector):
        buckets: dict[int, list[float]] = defaultdict(list)
        for s, e in zip(clean_samples, clean_explanations):
            buckets[e.predicted_class].append(self._disagreement(s, detector, e.predicted_class))
        self._null = {c: np.sort(np.array(v)) for c, v in buckets.items()}

    def score(self, sample, explanation, detector):
        d = self._disagreement(sample, detector, explanation.predicted_class)
        null = self._null.get(explanation.predicted_class)
        if null is None or len(null) == 0:
            tail = 0.5
        else:
            tail = float((1 + np.sum(null >= d)) / (len(null) + 1))  # calibrated tail prob
        # Suspicious = unusually HIGH disagreement => small tail. Nonconformity = -log(tail).
        return SignalScore(
            self.name,
            score=float(-np.log(tail + 1e-9)),
            evidence={"disagreement": d, "tail_prob": tail, "n_methods": len(self.attributors)},
        )
