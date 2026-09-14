"""Explanation-integrity attacks (Plan Section 2.3, threat model A1-A5).

The walking skeleton ships ToyAttributionEvasion (A1): a gradient-free search for a
small, prediction-preserving perturbation that scrambles the top-k attribution
ranking — enough to exercise detection end to end. Phase 1/4 implement the full
taxonomy with protocol-valid perturbations:
  A1 attribution evasion  — constrained gradient-free / optimisation search
  A2 explainer poisoning  — trigger-bearing training injection
  A3 scaffolding          — Slack-style probe-detecting wrapper
  A4 drift masquerade     — slow corruption disguised as concept drift
  A5 monitor evasion      — joint signal-aware evasion + verifier prompt injection
"""
from __future__ import annotations

from abc import ABC, abstractmethod

import numpy as np

from avert.attribution.base import Attributor
from avert.benchmark.perturbation import FeatureBounds
from avert.detectors.base import Detector
from avert.types import AttackClass, FlowSample


class ExplanationAttack(ABC):
    attack_class: AttackClass

    @abstractmethod
    def apply(self, sample: FlowSample, detector: Detector, attributor: Attributor) -> FlowSample:
        """Return an attacked copy of `sample` (prediction preserved)."""


class ToyAttributionEvasion(ExplanationAttack):
    attack_class = AttackClass.A1_EVASION

    def __init__(self, radius: float = 0.5, n_tries: int = 200, seed: int = 0):
        self.radius, self.n_tries, self.rng = radius, n_tries, np.random.default_rng(seed)

    def _topk(self, detector, attributor, x, names, k=5):
        pred = int(detector.predict(x.reshape(1, -1))[0])
        e = attributor.explain(detector, x, names, pred, "x", top_k=k)
        return pred, set(e.top_features())

    def apply(self, sample, detector, attributor):
        x = sample.features
        base_pred, base_top = self._topk(detector, attributor, x, sample.feature_names)
        best_x, best_change = x, -1
        scale = self.radius * (np.abs(x) + 1e-3)
        for _ in range(self.n_tries):
            xp = x + self.rng.normal(0, scale)
            pred, top = self._topk(detector, attributor, xp, sample.feature_names)
            if pred != base_pred:
                continue  # must preserve the prediction
            change = len(base_top - top)
            if change > best_change:
                best_change, best_x = change, xp
        return FlowSample(
            features=best_x, feature_names=sample.feature_names, dataset=sample.dataset,
            true_label=sample.true_label, attack_class=self.attack_class,
            ground_truth_causal=sample.ground_truth_causal,
            sample_id=(sample.sample_id or "x") + "-A1",
        )


class DisplacementAttack(ExplanationAttack):
    """A1 displacement (Ghorbani et al. 2019 style): a SMALL, prediction-preserving
    perturbation that maximally changes the top-k attribution set, pushing x into a region
    where the attribution is hypersensitive. This is the threat certified stability
    (Signal 1) is designed to catch — a small perturbation here collapses the certified
    radius. Distinct from misdirection evasion, which is the verifier's job.
    """

    attack_class = AttackClass.A1_EVASION

    def __init__(self, attributor: Attributor, std: np.ndarray, bounds: "FeatureBounds",
                 eps: float = 0.15, n_iter: int = 100, restarts: int = 3, top_k: int = 5,
                 seed: int = 0):
        self.attributor = attributor
        self.std = np.asarray(std)
        self.bounds = bounds                # protocol-valid set; REQUIRED, not optional
        self.eps = eps                      # perturbation budget, in per-feature std units
        self.n_iter = n_iter
        self.restarts = restarts
        self.top_k = top_k
        self.rng = np.random.default_rng(seed)

    def _top(self, detector, x, names, c):
        return set(self.attributor.explain(detector, x, names, c, "x", top_k=self.top_k).top_features())

    def apply(self, sample, detector, attributor=None, **_):
        x0 = sample.features.astype(float)
        names = sample.feature_names
        c = int(detector.predict(x0.reshape(1, -1))[0])
        base_top = self._top(detector, x0, names, c)
        best_x, best_overlap = x0, len(base_top)

        # Noise goes only into the FREE features. A derived feature is not the adversary's to
        # set -- the extractor computes it -- so perturbing it directly produces a vector no
        # capture could yield. `project` then recomputes every derived feature from the moved
        # free ones, which is what keeps `TotBytes == SrcBytes + DstBytes` true of the flow that
        # is actually returned. Before this, an identity check separated attacked from clean
        # flows at AUROC 1.000 on 5G-NIDD, and C3's negative was measured against a defender
        # who had never been asked.
        free = np.asarray(self.bounds.free_idx, dtype=int)
        mask = np.zeros_like(x0)
        mask[free] = 1.0

        for _ in range(self.restarts):
            for _ in range(self.n_iter):
                # Project BEFORE evaluating anything, so the flow whose prediction and
                # attribution are checked is the flow that is returned.
                xp = self.bounds.project(x0 + mask * self.rng.normal(0.0, self.eps * self.std))
                # Recomputing a derived feature can push it outside the clean sample's range.
                # Reject rather than clip: clipping would break the identity that was the point.
                if not self.bounds.in_box(xp):
                    continue
                if int(detector.predict(xp.reshape(1, -1))[0]) != c:
                    continue
                ov = len(base_top & self._top(detector, xp, names, c))
                if ov < best_overlap:
                    best_x, best_overlap = xp, ov
        return FlowSample(features=best_x, feature_names=names, dataset=sample.dataset,
                          true_label=sample.true_label, attack_class=self.attack_class,
                          ground_truth_causal=sample.ground_truth_causal,
                          sample_id=(sample.sample_id or "x") + "-A1disp")


class AttributionEvasion(ExplanationAttack):
    """A1 (real): corrupt the attribution ranking by demoting the true causal features
    out of the top-k while the prediction is preserved, using protocol-valid perturbations
    (FeatureBounds). It promotes non-causal features so they crowd the causal ones out of
    the top-k — the misdirection variant of attribution evasion (Plan Section 2.3, A1).
    """

    attack_class = AttackClass.A1_EVASION

    def __init__(self, bounds: FeatureBounds, attributor: Attributor,
                 top_k: int = 5, n_iter: int = 60, restarts: int = 3, seed: int = 0):
        self.bounds = bounds
        self.attributor = attributor
        self.top_k = top_k
        self.n_iter = n_iter
        self.restarts = restarts
        self.rng = np.random.default_rng(seed)

    def _overlap(self, detector, x, names, c, causal_set):
        e = self.attributor.explain(detector, x, names, c, "x", top_k=self.top_k)
        return len(set(e.top_features()) & causal_set)

    def apply(self, sample, detector, attributor=None, causal_idx=None):
        if causal_idx is None:
            raise ValueError("AttributionEvasion needs causal_idx (interventional ground truth)")
        x0 = sample.features.astype(float)
        names = sample.feature_names
        c = int(detector.predict(x0.reshape(1, -1))[0])
        causal_set = set(int(j) for j in causal_idx)
        # Candidates must be non-causal AND free: a derived feature is computed by the
        # extractor, so choosing its value directly yields a flow no capture could produce.
        free = set(int(j) for j in np.asarray(self.bounds.free_idx, dtype=int))
        noncausal = [j for j in range(len(x0)) if j not in causal_set and j in free]
        if not noncausal:                       # every free feature is causal here
            noncausal = sorted(free) or list(range(len(x0)))

        best_x = x0
        best_overlap = self._overlap(detector, x0, names, c, causal_set)
        for _ in range(self.restarts):
            xp, cur = x0.copy(), best_overlap
            for _ in range(self.n_iter):
                j = int(noncausal[self.rng.integers(len(noncausal))])
                trial = xp.copy()
                trial[j] = self.bounds.sample_value(j, self.rng)
                trial = self.bounds.project(trial)   # recomputes every derived feature
                if not self.bounds.in_box(trial):
                    continue                                    # would leave the observed range
                if int(detector.predict(trial.reshape(1, -1))[0]) != c:
                    continue                                    # prediction must be preserved
                ov = self._overlap(detector, trial, names, c, causal_set)
                if ov <= cur:
                    xp, cur = trial, ov
            if cur < best_overlap:
                best_x, best_overlap = xp, cur

        return FlowSample(
            features=best_x, feature_names=names, dataset=sample.dataset,
            true_label=sample.true_label, attack_class=self.attack_class,
            ground_truth_causal=[names[j] for j in causal_idx],
            sample_id=(sample.sample_id or "x") + "-A1",
        )
