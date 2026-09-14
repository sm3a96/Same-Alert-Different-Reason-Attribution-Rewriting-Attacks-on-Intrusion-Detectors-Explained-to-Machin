"""September 2026 panel additions: PASA, erasure faithfulness,
the free-coordinate certified-stability variant, and the Integrated Gradients class-space fix.

Each test pins a property that a wrong implementation would violate quietly: a score outside
its range, a nondeterministic score, an attributor indexing past the module's last column.
"""
import numpy as np
import pytest

from avert.attribution.real import IntegratedGradientsAttributor, TreeSHAPAttributor
from avert.data.datasets import SyntheticDataset
from avert.data.splits import train_calib_test_split
from avert.detectors.real import MLPDetector, XGBoostDetector
from avert.signals.certified_stability import CertifiedStabilitySignal
from avert.signals.erasure_faithfulness import ErasureFaithfulnessSignal
from avert.signals.pasa import PASASignal
from avert.types import FlowSample, SignalName


def _tree_setup(seed=0):
    d = SyntheticDataset(n=400, d=10, n_causal=3, seed=seed).load()
    sp = train_calib_test_split(d, seed=seed)
    det = XGBoostDetector(n_estimators=30, n_jobs=1).fit(sp["train"].X, sp["train"].y)
    names = d.feature_names
    samples = [FlowSample(features=sp["calibration"].X[i], feature_names=names, dataset="syn",
                          true_label=int(sp["calibration"].y[i]), sample_id=f"c{i}") for i in range(20)]
    expl = TreeSHAPAttributor()
    exps = [expl.explain(det, s.features, names, int(det.predict(s.features.reshape(1, -1))[0]),
                         s.sample_id, 5) for s in samples]
    return det, expl, samples, exps, sp


def test_pasa_is_deterministic_per_sample_and_names_both_variants():
    det, expl, samples, exps, _ = _tree_setup()
    for mode, name in (("range", SignalName.PASA_RANGE), ("std", SignalName.PASA_STD)):
        sig = PASASignal(expl, mode=mode)
        sig.calibrate(samples, exps, det)
        a = sig.score(samples[3], exps[3], det)
        b = sig.score(samples[3], exps[3], det)
        assert sig.name == name and a.signal == name
        assert a.score == b.score, "PASA must seed its noise from the sample id, not a shared RNG"
        assert np.isfinite(a.score)
        assert a.evidence["prediction_sensitivity"] >= 0 and a.evidence["attribution_sensitivity"] >= 0


def test_erasure_faithfulness_fraction_is_in_unit_interval_and_matches_ground_truth_deltas():
    from avert.benchmark.ground_truth import causal_features

    det, expl, samples, exps, sp = _tree_setup()
    ref = sp["train"].X.mean(axis=0)
    sig = ErasureFaithfulnessSignal(ref, top_k=5)
    sig.calibrate(samples, exps, det)
    for s, e in zip(samples[:5], exps[:5]):
        frac, deltas = sig.shown_mass_fraction(s, e, det)
        assert 0.0 <= frac <= 1.0
        _, gt_deltas = causal_features(s, det, ref, 0.05, 3)
        assert np.allclose(deltas, gt_deltas), "the runtime check and the ground truth must erase identically"
        assert sig.score(s, e, det).score == pytest.approx(-frac)


def test_free_coordinate_stability_variant_carries_its_own_name_and_projects():
    from avert.benchmark.perturbation import bounds_for

    det, expl, samples, exps, sp = _tree_setup()
    bounds = bounds_for(sp["train"].X, samples[0].feature_names, "syn")
    plain = CertifiedStabilitySignal(expl, n=8, sigma=0.1, seed=0)
    free = CertifiedStabilitySignal(expl, n=8, sigma=0.1, seed=0, bounds=bounds)
    assert plain.name == SignalName.CERTIFIED_STABILITY
    assert free.name == SignalName.CERTIFIED_STABILITY_FREE
    for sig in (plain, free):
        sig.calibrate(samples, exps, det)
        sc = sig.score(samples[0], exps[0], det)
        assert sc.score <= 0.0 and sc.evidence["certified"]


def test_integrated_gradients_attributes_in_the_module_class_space():
    """A grouped training side can miss a class. predict_proba re-expands to the dataset label
    space; Captum indexes the raw module. Attributing dataset label 9 on a module with four
    outputs raised IndexError in the transferability run on 2026-09-07."""
    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)).astype(np.float32)
    det = MLPDetector(epochs=2).fit(X, rng.choice([0, 1, 5, 9], 300))
    names = [f"f{i}" for i in range(5)]
    ig = IntegratedGradientsAttributor(n_steps=4)
    e = ig.explain(det, X[0], names, 9, "s0", top_k=3)
    assert e.attributions.shape == (5,)
    with pytest.raises(ValueError):
        ig.explain(det, X[0], names, 2, "s0", top_k=3)      # never seen in training
