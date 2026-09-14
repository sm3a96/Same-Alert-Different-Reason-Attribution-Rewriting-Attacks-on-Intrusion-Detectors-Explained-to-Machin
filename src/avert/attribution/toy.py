"""Two mechanistically distinct toy attributors so the walking skeleton can run
cross-method consensus (Signal 3) end to end:

  CoefInputAttributor — coefficient x input, a faithful linear attribution.
  OcclusionAttributor  — model-agnostic; importance = drop in predicted-class
                         probability when a feature is set to its mean.

Replace with TreeSHAP / DeepSHAP / integrated gradients / permutation in Phase 2.
"""
from __future__ import annotations

import numpy as np

from avert.attribution.base import Attributor
from avert.types import Explanation


class CoefInputAttributor(Attributor):
    name = "coef_input"
    requires_differentiable = True

    def explain(self, detector, x, feature_names, predicted_class, sample_id, top_k=5):
        coef = getattr(detector, "coef", None)
        if coef is None:
            raise ValueError("CoefInputAttributor needs a linear detector with .coef")
        attr = coef * x
        return Explanation(sample_id, attr, feature_names, self.name, predicted_class, top_k)


class OcclusionAttributor(Attributor):
    name = "occlusion"
    requires_differentiable = False

    def __init__(self, background_mean: np.ndarray) -> None:
        self.mean = background_mean

    def explain(self, detector, x, feature_names, predicted_class, sample_id, top_k=5):
        base = detector.predict_proba(x.reshape(1, -1))[0, predicted_class]
        attr = np.zeros_like(x, dtype=float)
        for j in range(len(x)):
            xj = x.copy()
            xj[j] = self.mean[j]
            attr[j] = base - detector.predict_proba(xj.reshape(1, -1))[0, predicted_class]
        return Explanation(sample_id, attr, feature_names, self.name, predicted_class, top_k)
