"""A3 scaffolding attack (Slack et al. 2020 style), Plan Section 2.3.

A scaffolded model wraps the real (true) detector with an out-of-distribution detector
that recognises explainer *probe* queries (perturbation-based explainers generate
off-manifold inputs) and routes them to an innocuous surrogate, so the explanation built
from those probes reflects the surrogate, not the real model.

This is exactly what cross-method consensus (Signal 3) is built to catch: a
perturbation-based attributor (permutation importance) is fooled by the routing, while
TreeSHAP reads the real tree directly and is not — so the two methods diverge. AVERT's
consensus null is calibrated on the clean detector (methods agree); under the scaffold the
disagreement spikes and the explanation is flagged.
"""
from __future__ import annotations

import numpy as np
from sklearn.ensemble import IsolationForest

from avert.detectors.base import Detector
from avert.detectors.real import XGBoostDetector


class OODDetector:
    """Flags off-manifold inputs (explainer probes) via an isolation forest on clean data."""

    def __init__(self, contamination: float = 0.3, seed: int = 0):
        self._if = IsolationForest(contamination=contamination, random_state=seed, n_estimators=100)

    def fit(self, X: np.ndarray) -> "OODDetector":
        self._if.fit(X)
        return self

    def is_probe(self, X: np.ndarray) -> np.ndarray:
        return self._if.predict(np.atleast_2d(X)) == -1     # -1 == anomalous == probe


class ScaffoldedDetector(Detector):
    name = "scaffolded"

    def __init__(self, real: XGBoostDetector, surrogate: XGBoostDetector, ood: OODDetector):
        self.real, self.surrogate, self.ood = real, surrogate, ood

    def fit(self, X, y):
        return self

    def predict_proba(self, X):
        X = np.atleast_2d(X)
        out = self.real.predict_proba(X)
        probe = self.ood.is_probe(X)
        if probe.any():
            out[probe] = self.surrogate.predict_proba(X[probe])   # route probes to surrogate
        return out

    def predict(self, X):
        return self.predict_proba(X).argmax(axis=1)

    @property
    def is_differentiable(self) -> bool:
        return False

    @property
    def booster(self):
        return self.real.booster        # TreeSHAP reads the REAL model directly (not fooled)

    def encoded_index(self, dataset_class: int) -> int | None:
        """Delegate the dataset-space -> booster-space class mapping to the wrapped model.

        Required because `booster` above hands out the real model's booster, which speaks the
        encoded space, while `predict` here speaks the dataset space. Without this the wrapper
        silently skips the translation TreeSHAP needs and indexes a SHAP array off the end.
        """
        return self.real.encoded_index(dataset_class)


def make_scaffolded_detector(X, y, n_corrupt: int = 6, contamination: float = 0.3, seed: int = 0):
    """Build (scaffold, real). The surrogate is trained with the real model's most
    important features destroyed, so it relies on innocuous secondary features."""
    real = XGBoostDetector(n_estimators=150).fit(X, y)
    top = np.argsort(-real.booster.feature_importances_)[:n_corrupt]
    rng = np.random.default_rng(seed)
    Xs = X.copy()
    for j in top:
        Xs[:, j] = rng.permutation(Xs[:, j])        # destroy info in the real top features
    surrogate = XGBoostDetector(n_estimators=150).fit(Xs, y)
    ood = OODDetector(contamination=contamination, seed=seed).fit(X)
    return ScaffoldedDetector(real, surrogate, ood), real
