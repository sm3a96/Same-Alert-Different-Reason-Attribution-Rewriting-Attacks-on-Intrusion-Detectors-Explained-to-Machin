"""Temporal channel: drift versus manipulation (Plan Section 3.4).

Jointly monitors input-distribution drift and explanation drift with anytime-valid
sequential tests (conformal test martingales, e-processes) rather than repeated
fixed-level two-sample tests, which would inflate the false-alarm rate under
continuous monitoring.

Decision rule:
  input drift WITH proportional explanation drift  -> benign  -> recalibrate (ACI)
  explanation drift WITHOUT corresponding input drift -> A4 signature -> alarm

STUB, future work: implement the two drift statistics, the
e-process accumulator (Ramdas et al. 2023), and adaptive conformal recalibration
(Gibbs & Candes 2021). Returns a nonconformity reflecting the martingale value.
"""
from __future__ import annotations

from avert.signals.base import IntegritySignal
from avert.types import SignalName


class TemporalChannel(IntegritySignal):
    name = SignalName.TEMPORAL

    def calibrate(self, clean_samples, clean_explanations, detector):
        raise NotImplementedError("Phase 2 P2-4: fit input/explanation drift baselines on clean stream")

    def score(self, sample, explanation, detector):
        raise NotImplementedError("Phase 2 P2-4: e-process value for explanation-without-input drift")
