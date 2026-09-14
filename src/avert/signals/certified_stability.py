"""Signal 1: certified attribution-stability (Plan Section 3.1) — the paper's main
theoretical content.

We certify the stability of the top-k attribution SET via randomized smoothing
(Cohen et al. 2019, adapted from class prediction to the categorical "which top-k feature
set"). For a sample x we smooth with Gaussian noise in standardized feature space, draw n
Monte-Carlo samples, and let S* be the most frequent top-k set with empirical frequency
p. A Clopper-Pearson lower bound p_lo (confidence 1-conf) gives the SOUND certificate: if
p_lo > 1/2 then for every perturbation of standardized-L2 norm below

    R = sigma * Phi^{-1}(p_lo)

the smoothed top-k set provably stays S*. R is the certified attribution-stability radius.
A small radius on a sample whose neighbours certify large radii is evidence of an A1/A3
attack that has pushed the attribution into a fragile region.

The signal's nonconformity is -R (smaller radius = more suspicious). This replaces the
earlier empirical (Jaccard) placeholder, which was non-discriminative noise.

PHASE-2 refinement: replace the isotropic Gaussian with the typed, protocol-valid
smoothing distribution per feature kind (continuous / count / categorical / flag) and the
hybrid discrete-continuous certificate (ICML 2026) — the domain-constrained lemma.
"""
from __future__ import annotations

from collections import Counter

import numpy as np
from scipy.stats import beta, norm

from avert.attribution.base import Attributor
from avert.signals.base import IntegritySignal
from avert.types import SignalName, SignalScore


def clopper_pearson_lower(k: int, n: int, conf: float = 0.05) -> float:
    """One-sided (1-conf) lower confidence bound for a Binomial(n, p) proportion."""
    if k <= 0:
        return 0.0
    if k >= n:
        return conf ** (1.0 / n)            # exact for all-successes
    return float(beta.ppf(conf, k, n - k + 1))


class CertifiedStabilitySignal(IntegritySignal):
    name = SignalName.CERTIFIED_STABILITY      # overridden per instance when `bounds` is given

    def __init__(self, attributor: Attributor, n: int = 100, sigma: float = 0.5,
                 top_k: int = 5, conf: float = 0.05, certified: bool = True, seed: int = 0,
                 bounds=None):
        # `bounds` (a RealizableBounds) restricts the smoothing noise to the attacker-settable
        # coordinates and recomputes the derived ones, so the certificate is over the
        # perturbation set the threat model actually admits. Without it the signal smooths
        # every coordinate, derived fields included, which the adversary is forbidden to touch.
        # The two variants carry different names so the matrix reports both.
        self.bounds = bounds
        self.name = SignalName.CERTIFIED_STABILITY_FREE if bounds is not None else SignalName.CERTIFIED_STABILITY
        self.attributor = attributor
        self.n = n
        self.sigma = sigma
        self.top_k = top_k
        self.conf = conf
        self.certified = certified
        self.rng = np.random.default_rng(seed)
        self._mean: np.ndarray | None = None
        self._std: np.ndarray | None = None

    def calibrate(self, clean_samples, clean_explanations, detector):
        X = np.stack([s.features for s in clean_samples]).astype(float)
        self._mean = X.mean(axis=0)
        self._std = X.std(axis=0) + 1e-8        # standardize so the L2 radius is well-scaled

    def certified_radius(self, sample, detector) -> tuple[float, float]:
        """Return (certified top-k set radius R, p_lo)."""
        assert self._std is not None, "calibrate() first"
        x = sample.features.astype(float)
        mask = np.ones_like(x)
        if self.bounds is not None:
            mask[:] = 0.0
            mask[np.asarray(self.bounds.free_idx, dtype=int)] = 1.0
        sets = []
        for _ in range(self.n):
            xp = x + mask * self.rng.normal(0.0, self.sigma * self._std)     # Gaussian smoothing
            if self.bounds is not None:
                xp = self.bounds.project(xp)
            pred = int(detector.predict(xp.reshape(1, -1))[0])
            e = self.attributor.explain(detector, xp, sample.feature_names, pred,
                                        sample.sample_id or "x", top_k=self.top_k)
            sets.append(frozenset(e.top_features()))
        _, count = Counter(sets).most_common(1)[0]
        p_lo = clopper_pearson_lower(count, self.n, self.conf)
        R = float(self.sigma * norm.ppf(p_lo)) if p_lo > 0.5 else 0.0
        return R, p_lo

    def score(self, sample, explanation, detector):
        if not self.certified:                  # empirical fallback (ablation)
            return self._empirical(sample, explanation, detector)
        R, p_lo = self.certified_radius(sample, detector)
        return SignalScore(self.name, score=-R,
                           evidence={"certified_radius": R, "p_lo": p_lo, "certified": True})

    def _empirical(self, sample, explanation, detector):
        x = sample.features
        base_top = set(explanation.top_features())
        overlaps = []
        for _ in range(self.n):
            xp = x + self.rng.normal(0.0, self.sigma * self._std)
            pred = int(detector.predict(xp.reshape(1, -1))[0])
            e = self.attributor.explain(detector, xp, sample.feature_names, pred,
                                        sample.sample_id or "x", top_k=self.top_k)
            t = set(e.top_features())
            overlaps.append(len(base_top & t) / len(base_top | t) if (base_top | t) else 1.0)
        return SignalScore(self.name, score=1.0 - float(np.mean(overlaps)),
                           evidence={"mean_topk_jaccard": float(np.mean(overlaps)), "certified": False})
