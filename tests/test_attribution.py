"""P2-2 wiring: real attributors produce well-formed explanations through the
method->model factory, and Signal 3 (cross-method consensus) runs end to end on a
real detector with them.
"""
import numpy as np

from avert.attribution.real import (
    IntegratedGradientsAttributor,
    PermutationAttributor,
    TreeSHAPAttributor,
    attributors_for,
)
from avert.data.datasets import SyntheticDataset
from avert.data.splits import train_calib_test_split
from avert.detectors.real import MLPDetector, XGBoostDetector
from avert.signals.cross_method_consensus import CrossMethodConsensusSignal


def _setup(seed=0):
    d = SyntheticDataset(n=400, d=10, n_causal=3, seed=seed).load()
    sp = train_calib_test_split(d, seed=seed)
    return sp, d.feature_names


def test_factory_picks_distinct_methods_per_family():
    sp, _ = _setup()
    xgb = XGBoostDetector(n_estimators=40, max_depth=3, n_jobs=1).fit(sp["train"].X, sp["train"].y)
    mlp = MLPDetector(hidden=(32,), epochs=20, batch=64).fit(sp["train"].X, sp["train"].y)
    bg = sp["train"].X
    xgb_methods = {a.name for a in attributors_for(xgb, bg)}
    mlp_methods = {a.name for a in attributors_for(mlp, bg)}
    assert xgb_methods == {"permutation", "treeshap"}
    assert mlp_methods == {"permutation", "integrated_gradients"}


def test_treeshap_and_ig_shapes():
    sp, names = _setup()
    xgb = XGBoostDetector(n_estimators=40, max_depth=3, n_jobs=1).fit(sp["train"].X, sp["train"].y)
    mlp = MLPDetector(hidden=(32,), epochs=20, batch=64).fit(sp["train"].X, sp["train"].y)
    x = sp["test"].X[0]
    pc = int(xgb.predict(x.reshape(1, -1))[0])
    e1 = TreeSHAPAttributor().explain(xgb, x, names, pc, "s", top_k=3)
    e2 = IntegratedGradientsAttributor().explain(mlp, x, names, pc, "s", top_k=3)
    e3 = PermutationAttributor(sp["train"].X).explain(xgb, x, names, pc, "s", top_k=3)
    for e in (e1, e2, e3):
        assert e.attributions.shape == (len(names),)
        assert len(e.top_features()) == 3


def test_consensus_signal_runs_on_real_detector():
    sp, names = _setup()
    xgb = XGBoostDetector(n_estimators=40, max_depth=3, n_jobs=1).fit(sp["train"].X, sp["train"].y)
    sig = CrossMethodConsensusSignal(attributors_for(xgb, sp["train"].X), top_k=3)

    calib = sp["calibration"].to_samples()
    calib_expl = [
        TreeSHAPAttributor().explain(
            xgb, s.features, names, int(xgb.predict(s.features.reshape(1, -1))[0]), s.sample_id, 3
        )
        for s in calib
    ]
    sig.calibrate(calib, calib_expl, xgb)

    s0 = sp["test"].to_samples()[0]
    e0 = calib_expl[0]
    score = sig.score(s0, e0, xgb)
    assert np.isfinite(score.score)
    assert "disagreement" in score.evidence


def test_treeshap_translates_the_class_space_for_every_booster_provider():
    """`predict` speaks the DATASET label space; the raw booster TreeSHAP reads speaks the
    ENCODED one. When a training subsample misses a class the two differ, and indexing SHAP
    output by a dataset label runs off the end -- "index 17 is out of bounds for axis 2 with
    size 16", which killed 12 of 45 matrix cells. Every object exposing `.booster` must also
    expose `encoded_index`, including the scaffold wrapper, which delegates.
    """
    import numpy as np

    from avert.attribution.real import TreeSHAPAttributor
    from avert.benchmark.scaffold import make_scaffolded_detector
    from avert.detectors.real import XGBoostDetector

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))
    y = rng.choice([0, 1, 5, 9], 400)          # gapped: booster has 4 outputs, labels reach 9
    names = [f"f{i}" for i in range(6)]

    det = XGBoostDetector(n_estimators=15, n_jobs=1).fit(X, y)
    assert det.encoded_index(9) == 3 and det.encoded_index(0) == 0
    assert det.encoded_index(7) is None        # never trained on, so no column to point at

    shap = TreeSHAPAttributor()
    for cls in (0, 1, 5, 9):                   # every dataset-space label must work
        e = shap.explain(det, X[0], names, cls, "s0", top_k=3)
        assert len(e.attributions) == 6 and np.all(np.isfinite(e.attributions))

    scaffold, _real = make_scaffolded_detector(X, y, n_corrupt=2, seed=0)
    assert scaffold.encoded_index(9) == 3      # the wrapper delegates rather than skipping
    pred = int(scaffold.predict(X[:1])[0])
    e = shap.explain(scaffold, X[0], names, pred, "s0", top_k=3)
    assert len(e.attributions) == 6
