"""IntegritySignal interface. Each signal calibrates on clean traffic only and then
scores an explanation with a nonconformity value (higher = more suspicious).
Independence across signals is the core design principle (Plan Section 3).
"""
from __future__ import annotations

from abc import ABC, abstractmethod

from avert.detectors.base import Detector
from avert.types import Explanation, FlowSample, SignalName, SignalScore


class IntegritySignal(ABC):
    name: SignalName

    @abstractmethod
    def calibrate(
        self,
        clean_samples: list[FlowSample],
        clean_explanations: list[Explanation],
        detector: Detector,
    ) -> None:
        """Fit per-class nulls / smoothing baselines on clean data only. No labels."""

    @abstractmethod
    def score(
        self,
        sample: FlowSample,
        explanation: Explanation,
        detector: Detector,
    ) -> SignalScore:
        """Nonconformity score for one explanation. Higher = more anomalous."""
