"""End-to-end AVERT orchestration (Plan Section 3).

Wiring:
    detector + attributor produce an explanation
        -> each integrity signal scores it
            -> aggregator + conformal fuser -> p-value -> violation?
                -> on violation: diagnose (leave-one-signal-out) -> repair

`AVERT.calibrate` is self-supervised on clean traffic only (no labels). `AVERT.assess`
returns an IntegrityVerdict; `AVERT.assess_and_repair` adds the RepairResult.
This is the integration seam the walking skeleton exercises; concrete signals and
detectors slot in without touching this file.
"""
from __future__ import annotations

import numpy as np

from avert.attribution.base import Attributor
from avert.detectors.base import Detector
from avert.fusion.conformal import ConformalFuser, SignalAggregator
from avert.repair.diagnose import diagnose
from avert.repair.repair import repair as _repair
from avert.signals.base import IntegritySignal
from avert.types import (
    Explanation,
    FlowSample,
    IntegrityVerdict,
    RepairResult,
    SignalName,
    SignalScore,
)


class AVERT:
    def __init__(
        self,
        detector: Detector,
        primary_attributor: Attributor,
        signals: list[IntegritySignal],
        fallback_attributor: Attributor | None = None,
        alpha: float = 0.05,
    ):
        self.detector = detector
        self.primary_attributor = primary_attributor
        self.signals = signals
        self.fallback_attributor = fallback_attributor or primary_attributor
        self.alpha = alpha
        self.aggregator = SignalAggregator()
        self.fuser = ConformalFuser()

    def explain(self, sample: FlowSample) -> Explanation:
        pred = int(self.detector.predict(sample.features.reshape(1, -1))[0])
        return self.primary_attributor.explain(
            self.detector, sample.features, sample.feature_names, pred,
            sample.sample_id or "x",
        )

    def _score_all(self, sample, explanation) -> dict[SignalName, SignalScore]:
        return {s.name: s.score(sample, explanation, self.detector) for s in self.signals}

    def calibrate(self, clean_samples: list[FlowSample]) -> None:
        """Self-supervised calibration on clean traffic (Plan Section 3.6)."""
        clean_expl = [self.explain(s) for s in clean_samples]
        for sig in self.signals:
            sig.calibrate(clean_samples, clean_expl, self.detector)

        per_signal: dict[SignalName, list[float]] = {s.name: [] for s in self.signals}
        all_scores = []
        for s, e in zip(clean_samples, clean_expl):
            sc = self._score_all(s, e)
            all_scores.append(sc)
            for name, v in sc.items():
                if not v.abstained:
                    per_signal[name].append(v.score)
        self.aggregator.fit({k: np.array(v) for k, v in per_signal.items() if v})
        fused = np.array([self.aggregator.aggregate(sc) for sc in all_scores])
        self.fuser.calibrate(fused)

    def assess(self, sample: FlowSample, explanation: Explanation | None = None) -> IntegrityVerdict:
        explanation = explanation or self.explain(sample)
        scores = self._score_all(sample, explanation)
        fused = self.aggregator.aggregate(scores)
        p = self.fuser.p_value(fused)
        violation = p < self.alpha
        diag_sig, diag_attack = (diagnose(scores, self.aggregator) if violation else (None, None))
        return IntegrityVerdict(
            sample.sample_id or "x", p_value=p, violation=violation, alpha=self.alpha,
            fused_score=fused, signal_scores=scores,
            diagnosis=diag_sig, implicated_attack=diag_attack,
        )

    def assess_and_repair(self, sample: FlowSample) -> tuple[IntegrityVerdict, RepairResult]:
        explanation = self.explain(sample)
        verdict = self.assess(sample, explanation)
        rr = _repair(sample, explanation, verdict, self.fallback_attributor, self.detector)
        return verdict, rr
