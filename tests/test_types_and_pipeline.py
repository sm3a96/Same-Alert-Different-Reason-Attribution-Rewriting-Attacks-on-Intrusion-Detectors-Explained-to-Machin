"""Smoke tests for the data contracts and the end-to-end pipeline wiring."""
import numpy as np

from avert.attribution.toy import CoefInputAttributor, OcclusionAttributor
from avert.data.datasets import SyntheticDataset
from avert.data.splits import train_calib_test_split
from avert.detectors.toy import ToyDetector
from avert.pipeline import AVERT
from avert.signals.certified_stability import CertifiedStabilitySignal
from avert.signals.cross_method_consensus import CrossMethodConsensusSignal
from avert.types import Explanation


def test_explanation_ranking():
    e = Explanation("s", np.array([0.1, -0.9, 0.3, 0.05]), ["a", "b", "c", "d"], "m", 1, top_k=2)
    assert e.top_features() == [1, 2]


def _build_avert(seed=0):
    data = SyntheticDataset(n=300, d=10, n_causal=3, seed=seed).load()
    sp = train_calib_test_split(data, seed=seed)
    det = ToyDetector().fit(sp["train"].X, sp["train"].y)
    coef = CoefInputAttributor()
    occ = OcclusionAttributor(sp["train"].X.mean(axis=0))
    signals = [
        CertifiedStabilitySignal(coef, n=10, seed=seed),
        CrossMethodConsensusSignal([coef, occ], top_k=3),
    ]
    av = AVERT(det, coef, signals, alpha=0.05)
    av.calibrate(sp["calibration"].to_samples())
    return av, sp


def test_pipeline_runs_and_verdict_well_formed():
    av, sp = _build_avert()
    s = sp["test"].to_samples()[0]
    v = av.assess(s)
    assert 0.0 < v.p_value <= 1.0
    assert isinstance(v.violation, bool)
    assert set(v.signal_scores.keys()) == {sig.name for sig in av.signals}


def test_clean_false_alarm_reasonable():
    av, sp = _build_avert()
    viol = np.array([av.assess(s).violation for s in sp["test"].to_samples()])
    assert viol.mean() <= 0.25  # loose smoke bound; rigorous check is test_conformal_coverage
