"""ToyDetector — a logistic-regression detector used by the walking skeleton and
tests so the whole pipeline runs end to end before the real detectors land.
Not a research artifact; replace with XGBoost / MLP / FT-Transformer in Phase 1.
"""
from __future__ import annotations

import numpy as np
from sklearn.linear_model import LogisticRegression

from avert.detectors.base import Detector


class ToyDetector(Detector):
    name = "toy_logreg"

    def __init__(self) -> None:
        self._clf = LogisticRegression(max_iter=200)

    def fit(self, X: np.ndarray, y: np.ndarray) -> "ToyDetector":
        self._clf.fit(X, y)
        return self

    def predict(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict(X)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return self._clf.predict_proba(X)

    @property
    def is_differentiable(self) -> bool:
        return True  # linear model exposes a usable gradient

    @property
    def coef(self) -> np.ndarray:
        return self._clf.coef_[0]
