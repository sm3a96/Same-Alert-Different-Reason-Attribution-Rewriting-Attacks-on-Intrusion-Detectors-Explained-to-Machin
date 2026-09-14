"""Repair: re-derive the explanation from the channels that remain trusted
(Plan Section 3.7). The smoothed (certified) attribution is the canonical fallback
ranking; the agentic verifier's grounded verdict is attached as context. Repaired
explanations are labelled with their conformal confidence so any consumer — human
or downstream agent — always has the provenance of what it reads.
"""
from __future__ import annotations


from avert.attribution.base import Attributor
from avert.detectors.base import Detector
from avert.types import (
    Explanation,
    FlowSample,
    IntegrityVerdict,
    RepairResult,
)


def repair(
    sample: FlowSample,
    original: Explanation,
    verdict: IntegrityVerdict,
    fallback_attributor: Attributor,
    detector: Detector,
) -> RepairResult:
    """If a violation is diagnosed, re-derive from the trusted fallback channel."""
    if not verdict.violation:
        return RepairResult(
            sample.sample_id or "x", original,
            trusted_signals=list(verdict.signal_scores.keys()),
            conformal_confidence=1.0 - verdict.p_value, was_repaired=False,
        )

    compromised = verdict.diagnosis
    trusted = [s for s in verdict.signal_scores if s != compromised]
    repaired = fallback_attributor.explain(
        detector, sample.features, sample.feature_names,
        original.predicted_class, sample.sample_id or "x", top_k=original.top_k,
    )
    repaired.method = f"repaired_via_{fallback_attributor.name}"
    return RepairResult(
        sample.sample_id or "x", repaired,
        trusted_signals=trusted,
        conformal_confidence=1.0 - verdict.p_value, was_repaired=True,
    )
