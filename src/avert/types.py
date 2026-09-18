"""Core data contracts shared across all AVERT modules.

These dataclasses are the *interfaces* between components. Detectors, attributors,
signals, fusion, repair, the benchmark, and the evaluation harness all read and
write these types and nothing else, so signals and detectors stay pluggable and a
new signal never forces a refactor elsewhere. Change a field here deliberately;
by design it ripples through the whole pipeline.

Reference: the framework and XInt-Bench sections of the paper.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from enum import Enum
from typing import Optional

import numpy as np


class AttackClass(str, Enum):
    """Explanation-integrity attack taxonomy (threat model A1-A5)."""
    CLEAN = "clean"
    A1_EVASION = "A1_attribution_evasion"
    A2_POISONING = "A2_explainer_poisoning"
    A3_SCAFFOLDING = "A3_scaffolding"
    A4_DRIFT_MASQUERADE = "A4_drift_masquerade"
    A5_MONITOR_EVASION = "A5_monitor_evasion"


class SignalName(str, Enum):
    """The integrity signals and the temporal channel."""
    CERTIFIED_STABILITY = "certified_stability"          # Signal 1
    CROSS_METHOD_CONSENSUS = "cross_method_consensus"    # Signal 3
    FEATURE_CONSISTENCY = "feature_consistency"          # Signal 4 -- the cheapest defender
    # Panel additions, September 2026:
    CERTIFIED_STABILITY_FREE = "certified_stability_free"  # Signal 1 smoothed on free coords only
    PASA_RANGE = "pasa_range"                            # Bhusal et al. 2024, their tabular spread
    PASA_STD = "pasa_std"                                # PASA at this paper's smoothing budget
    ERASURE_FAITHFULNESS = "erasure_faithfulness"        # share of erasure-causal mass shown


@dataclass
class FlowSample:
    """One network flow: feature vector plus provenance.

    `ground_truth_causal` is populated only for XInt-Bench instances, where the
    causal features are established interventionally against the detector under test
    (Plan Section 4) -- NOT generator-derived; the generators are unimplemented.
    """
    features: np.ndarray                       # shape (d,)
    feature_names: list[str]
    dataset: str
    true_label: Optional[int] = None           # NIDS class label if known
    attack_class: AttackClass = AttackClass.CLEAN
    ground_truth_causal: Optional[list[str]] = None
    sample_id: Optional[str] = None


@dataclass
class Explanation:
    """A per-feature attribution produced for one prediction."""
    sample_id: str
    attributions: np.ndarray                   # shape (d,), per-feature importance
    feature_names: list[str]
    method: str                                # "treeshap", "integrated_gradients", ...
    predicted_class: int
    top_k: int = 5

    def ranking(self) -> np.ndarray:
        """Feature indices ordered by descending |attribution|."""
        return np.argsort(-np.abs(self.attributions))

    def top_features(self) -> list[int]:
        return self.ranking()[: self.top_k].tolist()


@dataclass
class SignalScore:
    """Output of one integrity signal for one explanation.

    `score` is nonconformity-style: higher means more anomalous / more suspicious.
    `abstained` is set when a signal lacks the grounding to decide (e.g. Signal 2
    discards an ungrounded verdict rather than guessing — Plan Section 3.2).
    """
    signal: SignalName
    score: float
    abstained: bool = False
    evidence: dict = field(default_factory=dict)   # signal-specific detail, for logging + diagnosis


@dataclass
class CalibrationSet:
    """Clean-traffic nonconformity scores used for conformal calibration.

    Self-supervised: built from clean traffic only, no human labels (Plan Section 3.6).
    """
    scores: np.ndarray
    dataset: str
    regime_id: str = "default"


@dataclass
class IntegrityVerdict:
    """Fused decision for one explanation."""
    sample_id: str
    p_value: float
    violation: bool
    alpha: float
    fused_score: float
    signal_scores: dict = field(default_factory=dict)   # SignalName -> SignalScore
    diagnosis: Optional[SignalName] = None              # which channel collapsed (on violation)
    implicated_attack: Optional[AttackClass] = None


@dataclass
class RepairResult:
    """Outcome of diagnose-and-repair (Plan Section 3.7)."""
    sample_id: str
    repaired_explanation: Explanation
    trusted_signals: list[SignalName]
    conformal_confidence: float
    was_repaired: bool
