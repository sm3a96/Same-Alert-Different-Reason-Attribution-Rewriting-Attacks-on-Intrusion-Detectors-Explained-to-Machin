"""P1-3 gate: the three real detectors train on synthetic data, clear chance,
expose well-formed probabilities, and report differentiability correctly.
Small epoch counts keep this fast; full training is configured per dataset.
"""
import numpy as np
import pytest

from avert.data.datasets import SyntheticDataset
from avert.data.splits import train_calib_test_split
from avert.detectors.real import XGBoostDetector, MLPDetector, FTTransformerDetector
from avert.eval.runner import set_seed


def _data(seed=0):
    set_seed(seed)
    d = SyntheticDataset(n=500, d=12, n_causal=4, seed=seed).load()
    return train_calib_test_split(d, seed=seed)


@pytest.mark.parametrize(
    "ctor,differentiable",
    [
        (lambda: XGBoostDetector(n_estimators=60, max_depth=4), False),
        (lambda: MLPDetector(hidden=(64,), epochs=60, batch=64), True),
        (lambda: FTTransformerDetector(d_token=32, n_blocks=2, n_heads=4, epochs=30, batch=64), True),
    ],
)
def test_detector_trains_and_is_wellformed(ctor, differentiable):
    sp = _data()
    det = ctor().fit(sp["train"].X, sp["train"].y)

    proba = det.predict_proba(sp["test"].X)
    n = len(sp["test"].X)
    assert proba.shape == (n, 2)
    assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-4)        # valid distribution

    acc = float((det.predict(sp["test"].X) == sp["test"].y).mean())
    assert acc > 0.7, f"{det.name} accuracy {acc} below threshold on easy synthetic"
    assert det.is_differentiable is differentiable


def test_torch_detectors_expose_module_for_attribution():
    sp = _data()
    det = MLPDetector(hidden=(64,), epochs=5).fit(sp["train"].X, sp["train"].y)
    xt = det.to_input_tensor(sp["test"].X[:4])
    logits = det.module(xt)
    assert logits.shape == (4, det.n_classes)        # Captum can attribute through this


def test_xgboost_survives_a_gapped_label_set():
    """CICIoT2023 has 34 classes and some are tiny, so a training subsample can miss one.
    XGBoost then refuses with "Invalid classes inferred from unique values of `y`" and kills the
    run. The detector encodes to contiguous codes internally and decodes on the way out, so the
    dataset's own label space survives and a missing class gets a zero probability column rather
    than shifting every other column left.
    """
    import numpy as np

    from avert.detectors.real import XGBoostDetector

    rng = np.random.default_rng(0)
    X = rng.normal(size=(400, 6))

    contiguous = XGBoostDetector(n_estimators=15, n_jobs=1).fit(X, rng.integers(0, 4, 400))
    assert contiguous.predict_proba(X).shape[1] == 4          # no-op on the usual case

    y = rng.choice([0, 1, 5, 9], 400)
    det = XGBoostDetector(n_estimators=15, n_jobs=1).fit(X, y)
    pred, proba = det.predict(X), det.predict_proba(X)

    assert set(np.unique(pred)).issubset({0, 1, 5, 9})
    assert proba.shape[1] == 10
    assert np.allclose(proba[:, [2, 3, 4, 6, 7, 8]], 0.0)
    assert np.all(np.argmax(proba, axis=1) == pred)


def test_torch_detector_survives_a_gapped_label_set():
    """The MLP head is sized from the number of distinct labels while targets are raw dataset
    codes, so a subsample missing a class kills CUDA from inside nll_loss with
    "Assertion `t >= 0 && t < n_classes` failed" -- several frames from the cause. Same
    encode/decode fix as the tree detector, and the same no-op on contiguous labels.
    """
    import numpy as np

    from avert.detectors.real import MLPDetector

    rng = np.random.default_rng(0)
    X = rng.normal(size=(300, 5)).astype(np.float32)

    contiguous = MLPDetector(epochs=2).fit(X, rng.integers(0, 4, 300))
    assert contiguous.predict_proba(X).shape[1] == 4

    det = MLPDetector(epochs=2).fit(X, rng.choice([0, 1, 5, 9], 300))
    pred, proba = det.predict(X), det.predict_proba(X)
    assert set(np.unique(pred)).issubset({0, 1, 5, 9})
    assert proba.shape[1] == 10
    assert np.allclose(proba[:, [2, 3, 4, 6, 7, 8]], 0.0)
