"""Show the neighbourhood's attribution, not the point's. The defense C3's negative implies.

The displacement attack works by exploiting pointwise sensitivity of the attribution map: it finds
a nearby input where the same prediction comes with a different explanation. Nothing about the
explanation is a lie -- the attributor faithfully explains the flow it was handed -- which is
exactly why no integrity monitor can catch it, and why C3's negative is structural rather than a
tuning failure.

That diagnosis points at a defense, and it is not a detector. If the damage comes from
pointwise sensitivity, stop showing a pointwise quantity. Average the attribution over a small
neighbourhood and display that instead. This is SmoothGrad (Smilkov, Thorat, Kim, Viegas and
Wattenberg, arXiv:1706.03825) applied to a different end: they smooth to sharpen a saliency map
for a human, we smooth to deny an adversary the sensitivity the attack needs.

The machinery was already here and thrown away. `CertifiedStabilitySignal.certified_radius`
draws n smoothed samples, takes `Counter(sets).most_common(1)` and keeps only the count,
discarding the modal top-k set it just computed. That set is a smoothed attribution. It was
built as a detector, found useless as one, and never tried as a defense -- even though the SoK
this work answers surveys smoothing as a route to *robust* explanations rather than to detection.

Two smoothed statistics, because they answer different questions and cost the same draws:

  mean       average the attribution vector over draws. The SmoothGrad construction. Keeps a
             magnitude, so the ranked list a triage tool shows still carries scores.
  frequency  fraction of draws in which each feature lands in the top-k. Rank-based, so it is
             insensitive to a single draw with an extreme magnitude, and it is the statistic
             the certified-stability signal was already computing.

Cost is honest and must be reported with the benefit: n forward-and-explain passes per flow
instead of one, and a smoothed attribution is less faithful pointwise than the exact one it
replaces. Whether the fidelity lost is worth the harm avoided is an empirical question, which is
what `scripts/run_smoothed_defense.py` measures.
"""
from __future__ import annotations

import hashlib

import numpy as np

from avert.attribution.base import Attributor
from avert.detectors.base import Detector
from avert.types import Explanation

MODES = ("mean", "frequency")


class SmoothedAttributor(Attributor):
    """Wrap any attributor so it explains a neighbourhood instead of a point.

    `sigma` is in units of per-feature standard deviation, matching the convention used by the
    certified-stability signal, so the two are directly comparable. The noise is applied only to
    the features the caller says are free: perturbing a derived feature would hand the attributor
    a flow the extractor could not emit, which is the same mistake the attack used to make.
    """

    requires_differentiable = False

    def __init__(self, base: Attributor, std: np.ndarray, n: int = 30, sigma: float = 0.05,
                 mode: str = "mean", free_idx: np.ndarray | None = None, seed: int = 0):
        if mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {mode!r}")
        self.base = base
        self.std = np.asarray(std, dtype=float)
        self.n = n
        self.sigma = sigma
        self.mode = mode
        self.free_idx = None if free_idx is None else np.asarray(free_idx, dtype=int)
        self.seed = seed
        self.name = f"smoothed_{mode}_{base.name}"
        self.requires_differentiable = base.requires_differentiable

    def _stable_seed(self, sample_id: str) -> int:
        key = f"{self.seed}-{self.mode}-{sample_id}"
        return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16)

    def _mask(self, d: int) -> np.ndarray:
        if self.free_idx is None:
            return np.ones(d)
        m = np.zeros(d)
        m[self.free_idx] = 1.0
        return m

    def explain(self, detector: Detector, x: np.ndarray, feature_names: list[str],
                predicted_class: int, sample_id: str, top_k: int = 5) -> Explanation:
        x = np.asarray(x, dtype=float)
        d = len(x)
        # Seeded per sample so the same flow always yields the same smoothed explanation. A
        # defense whose output moves between calls is not one a SOC can operate, and a
        # nondeterministic explanation would silently contaminate the clean-vs-clean control.
        #
        # sha256, never builtin hash(): Python randomises string hashing per process, so
        # hash(sample_id) would give a different neighbourhood on every run and the defense
        # would be irreproducible from the cache. This project has already been bitten by
        # exactly that in the shuffled-order arm.
        rng = np.random.default_rng(self._stable_seed(sample_id))
        mask = self._mask(d)

        acc = np.zeros(d)
        for _ in range(self.n):
            xp = x + mask * rng.normal(0.0, self.sigma * self.std)
            e = self.base.explain(detector, xp, feature_names, predicted_class, sample_id, top_k)
            if self.mode == "mean":
                acc += np.abs(np.asarray(e.attributions, dtype=float))
            else:
                acc[np.asarray(e.top_features(), dtype=int)] += 1.0

        return Explanation(sample_id=sample_id, attributions=acc / self.n,
                           feature_names=list(feature_names), method=self.name,
                           predicted_class=predicted_class, top_k=top_k)
