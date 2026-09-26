"""PASA as a runtime integrity check: Bhusal et al., "PASA: Attack Agnostic Unsupervised
Adversarial Detection using Prediction & Attribution Sensitivity Analysis", IEEE EuroS&P 2024
(arXiv:2404.10789). Verified against the paper on 2026-09-07.

PASA adds Gaussian noise eta ~ N(0, sigma^2) to an input, with sigma = (max(x) - min(x)) * spread
per feature, and measures two L1 sensitivities (their Eq. 10 and 11):

    delta_1 = || Z(x + eta) - Z(x) ||_1          prediction sensitivity, Z = logits
    delta_2 = || A(x + eta) - A(x) ||_1          attribution sensitivity, A = the attributor

A sample is flagged when either exceeds a threshold learned from benign samples only. It is
the label-free attribution-sensitivity detector a reviewer of this paper will ask for, and it
is the published cousin of the certified-stability check: both probe the attribution with
noise, PASA reads the change, the certificate reads the modal set's frequency.

Two departures from the paper, both stated in the manuscript:
  * PASA is defined for DNN logits and Integrated Gradients. Here Z is the log-probability
    the deployed detector returns (the tree ensemble has no logit layer) and A is the
    attributor the deployment actually shows, which keeps the check label-free and generic.
  * The spread is not tuned on attacks, because there are no attacked samples at
    calibration time in this threat model. `PASA_RANGE` uses 0.0005, the value the paper
    reports for its own tabular intrusion corpus (updated CIC-IDS2017); `PASA_STD` uses noise
    of 0.1 standard deviations per feature, the smoothing budget of the certified-stability
    check, so the two probes see the same perturbation.

The score is the larger of the two clean-standardized sensitivities, which is PASA's OR rule
expressed as a single nonconformity value for the conformal fusion.
"""
from __future__ import annotations

import hashlib

import numpy as np

from avert.attribution.base import Attributor
from avert.signals.base import IntegritySignal
from avert.types import SignalName, SignalScore


def _stable_seed(sample_id: str, salt: str) -> int:
    """Per-sample seed from sha256, never from builtin hash(), which Python salts per process."""
    return int(hashlib.sha256(f"{salt}:{sample_id}".encode()).hexdigest()[:8], 16)


class PASASignal(IntegritySignal):
    def __init__(self, attributor: Attributor, mode: str = "range", spread: float = 0.0005,
                 std_scale: float = 0.1, top_k: int = 5):
        assert mode in ("range", "std")
        self.name = SignalName.PASA_RANGE if mode == "range" else SignalName.PASA_STD
        self.attributor, self.mode, self.spread, self.std_scale, self.top_k = attributor, mode, spread, std_scale, top_k
        self._sigma: np.ndarray | None = None
        self._mu = np.zeros(2)
        self._sd = np.ones(2)

    def _noise_scale(self, X: np.ndarray) -> np.ndarray:
        if self.mode == "range":
            return (X.max(axis=0) - X.min(axis=0)) * self.spread
        return X.std(axis=0) * self.std_scale

    def _logZ(self, detector, x: np.ndarray) -> np.ndarray:
        return np.log(np.clip(detector.predict_proba(x.reshape(1, -1))[0], 1e-12, None))

    def _sensitivities(self, sample, explanation, detector) -> tuple[float, float]:
        x = sample.features.astype(float)
        rng = np.random.default_rng(_stable_seed(sample.sample_id or "x", self.name.value))
        xp = x + rng.normal(0.0, self._sigma)
        pred = int(detector.predict(x.reshape(1, -1))[0])
        d1 = float(np.abs(self._logZ(detector, xp) - self._logZ(detector, x)).sum())
        e_p = self.attributor.explain(detector, xp, sample.feature_names, pred,
                                      sample.sample_id or "x", top_k=self.top_k)
        d2 = float(np.abs(e_p.attributions - explanation.attributions).sum())
        return d1, d2

    def calibrate(self, clean_samples, clean_explanations, detector):
        X = np.stack([s.features for s in clean_samples]).astype(float)
        self._sigma = self._noise_scale(X) + 1e-12
        d = np.array([self._sensitivities(s, e, detector) for s, e in zip(clean_samples, clean_explanations)])
        self._mu, self._sd = d.mean(axis=0), d.std(axis=0) + 1e-8

    def score(self, sample, explanation, detector):
        assert self._sigma is not None, "calibrate() first"
        d1, d2 = self._sensitivities(sample, explanation, detector)
        z = (np.array([d1, d2]) - self._mu) / self._sd
        return SignalScore(self.name, score=float(z.max()),
                           evidence={"prediction_sensitivity": d1, "attribution_sensitivity": d2,
                                     "z_prediction": float(z[0]), "z_attribution": float(z[1])})
