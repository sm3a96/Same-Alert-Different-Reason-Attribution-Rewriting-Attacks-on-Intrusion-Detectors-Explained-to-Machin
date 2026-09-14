"""Attributor interface: g(x, f) -> per-feature importance (Plan Section 2.1)."""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from avert.detectors.base import Detector
from avert.types import Explanation


class Attributor(ABC):
    name: str = "attributor"
    requires_differentiable: bool = False  # e.g. integrated gradients needs a gradient path

    @abstractmethod
    def explain(
        self,
        detector: Detector,
        x: np.ndarray,
        feature_names: list[str],
        predicted_class: int,
        sample_id: str,
        top_k: int = 5,
    ) -> Explanation: ...

    def applicable_to(self, detector: Detector) -> bool:
        return detector.is_differentiable or not self.requires_differentiable
