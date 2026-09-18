"""Signal 4: is this feature vector one the flow extractor could have produced?

The cheapest defender in the paper, and the one the negative result has to survive. It does not
look at the explanation at all. It checks whether the *input* satisfies the arithmetic its own
extractor imposes -- `TotPkts == SrcPkts + DstPkts`, `Min <= AVG <= Max` -- and scores the worst
relative violation.

Why it exists, stated plainly because it is a correction rather than an idea. Until 2026-08-09
the A1 attacks perturbed every feature independently inside a per-feature box. Measured on the
perturbation family that search ranged over, this check separated clean from perturbed flows at
AUROC 1.000 on 5G-NIDD and 0.96 on CICIoT2023 -- so "no statistical signal detects the displacement
attack" was a claim about the signals that had been tried, not about the attack. The attack now
projects onto the realizable set (`avert.benchmark.realizability`), and this signal ships
alongside it so the claim is measured against the defender that would otherwise refute it.

What it can and cannot do, by construction:

  * Against an attacker who ignores feature dependencies it is close to perfect, and it costs
    one pass over a handful of arithmetic relations -- no model, no calibration data, no latency
    budget worth measuring.
  * Against an attacker who respects them it is exactly blind: every residual is zero on both
    the clean and the attacked flow, so the AUROC is 0.5 by construction rather than by
    measurement. That is the point. It bounds what input validation can contribute and hands
    the rest of the problem to the explanation-integrity signals.

It is therefore reported as a *detector of unrealizable inputs*, never as a detector of
explanation attacks, and its 0.5 against the projected attack is a statement about the threat
model rather than a failure of the statistic.
"""
from __future__ import annotations

import numpy as np

from avert.detectors.base import Detector
from avert.signals.base import IntegritySignal
from avert.types import Explanation, FlowSample, SignalName, SignalScore


class FeatureConsistencySignal(IntegritySignal):
    """Nonconformity = worst relative violation of the extractor's declared arithmetic."""

    name = SignalName.FEATURE_CONSISTENCY

    #: Relative violation below this is indistinguishable from extractor rounding, so the
    #: signal treats the row as realizable. It is the same threshold the `realizable`
    #: evidence flag has always reported, promoted to the thing the score itself obeys.
    TOL = 1e-3

    def __init__(self, dataset: str, feature_names: list[str] | None = None):
        # Imported inside the methods, not at module scope. `avert.benchmark.realizability`
        # lives under the benchmark package whose __init__ pulls in the harness, which pulls in
        # this module -- so a top-level import here is a cycle, and it only surfaces when a test
        # imports the scaffold before the harness.
        self.dataset = dataset
        self._names = list(feature_names) if feature_names else None
        self._bound = (None if self._names is None
                       else self._schema_for(dataset).bind(self._names))

    @staticmethod
    def _schema_for(dataset: str):
        from avert.benchmark.realizability import schema_for
        return schema_for(dataset)

    def _ensure(self, sample: FlowSample):
        if self._bound is None:
            self._names = list(sample.feature_names)
            self._bound = self._schema_for(self.dataset).bind(self._names)
        return self._bound

    @property
    def n_constraints(self) -> int:
        if self._bound is None:
            return 0
        return len(self._bound.identities) + len(self._bound.orderings)

    def calibrate(self, clean_samples, clean_explanations, detector) -> None:
        """Nothing to fit. The relations are properties of the extractor, not of the traffic.

        The clean pass is still worth doing: if a declared relation does not hold on clean data
        the signal would fire on everything, so we record the clean residual for the run log and
        let `scripts/validate_feature_identities.py` be the thing that fails the build.
        """
        if not clean_samples:
            return
        bound = self._ensure(clean_samples[0])
        X = np.stack([s.features for s in clean_samples]).astype(float)
        self.clean_residual_mean = float(np.mean(bound.residual(X)))
        self.clean_violation_rate = float(np.mean(bound.residual(X) > self.TOL))

    def score(self, sample: FlowSample, explanation: Explanation,
              detector: Detector) -> SignalScore:
        """Residual, snapped to zero below the tolerance this signal declares realizable.

        The snap is not cosmetic. Genuine captures carry rounding from the extractor's own
        arithmetic at the 1e-7 level, and the projection writes exact values, so the raw
        residual orders the two arms almost perfectly on bits no monitor could threshold --
        AUROC 0.022 on 5G-NIDD A1 displacement, measured, with every value in both arms three
        orders of magnitude below TOL. Worse, the
        fuser z-scores each signal against its clean spread, and a spread of 1e-7 divides that
        noise up to z=+3.5 on clean flows against a flat -0.56 on attacked ones, so a signal
        with no information was pushing the fused score the wrong way and inflating the false
        alarm rate. Below its own resolution the signal has nothing to say, so it says nothing:
        score zero, and `abstained` so the fuser drops it from the max rather than reading a
        floor of zero as evidence.
        """
        bound = self._ensure(sample)
        raw = float(bound.residual(sample.features.reshape(1, -1))[0])
        realizable = raw <= self.TOL
        return SignalScore(
            self.name, score=0.0 if realizable else raw, abstained=realizable,
            evidence={"max_relative_violation": raw,
                      "realizable": bool(realizable),
                      "tolerance": self.TOL,
                      "n_constraints": self.n_constraints,
                      "dataset": self.dataset})
