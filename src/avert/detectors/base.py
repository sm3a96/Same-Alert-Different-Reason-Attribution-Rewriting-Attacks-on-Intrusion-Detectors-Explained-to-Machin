"""Detector interface: a flow-based NIDS classifier f: X -> Y (Plan Section 2.1)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np


class Detector(ABC):
    """Every concrete detector (XGBoost, MLP, FT-Transformer) implements this."""

    name: str = "detector"

    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray) -> "Detector": ...

    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray: ...

    @abstractmethod
    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        """Class probabilities, shape (n, n_classes)."""

    @property
    def is_differentiable(self) -> bool:
        """True for NN / FT-Transformer (enables integrated gradients);
        False for tree models, which fall back to a model-agnostic attributor.
        Drives the Signal 3 method->model mapping (Plan Section 3.3)."""
        return False
