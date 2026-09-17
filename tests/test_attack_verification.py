"""P1-4 gate: EIB ground truth must be interventionally valid — the features it
labels causal are the ones the detector actually relies on. We verify the property on the
synthetic dataset (known causal features f0..f3): ablating the interventionally-identified
causal features toward benign drops the predicted-class probability much more than ablating
the same number of non-causal features. If this fails, the benchmark's ground truth is not
real and detection/repair numbers are meaningless.
"""
import numpy as np

from avert.benchmark.ground_truth import causal_features
from avert.data.datasets import SyntheticDataset
from avert.data.splits import train_calib_test_split
from avert.detectors.real import XGBoostDetector
from avert.types import FlowSample


def test_interventional_ground_truth_is_valid():
    data = SyntheticDataset(n=900, d=12, n_causal=4, seed=0).load()
    sp = train_calib_test_split(data, seed=0)
    det = XGBoostDetector(n_estimators=80, max_depth=4).fit(sp["train"].X, sp["train"].y)
    benign_ref = sp["train"].X[sp["train"].y == 0].mean(axis=0)   # class 0 = "benign"

    drops_causal, drops_noncausal = [], []
    n_checked = 0
    for i in np.where(sp["test"].y == 1)[0]:                       # class 1 = "attack"
        x = sp["test"].X[i]
        if int(det.predict(x.reshape(1, -1))[0]) != 1:
            continue
        s = FlowSample(features=x, feature_names=data.feature_names, dataset="synthetic",
                       true_label=1, sample_id=f"s{i}")
        causal, _ = causal_features(s, det, benign_ref, delta_threshold=0.05, min_k=2)
        noncausal = [j for j in range(len(x)) if j not in set(causal)][: len(causal)]
        c = 1
        base = det.predict_proba(x.reshape(1, -1))[0, c]

        xc = x.copy()
        for j in causal:
            xc[j] = benign_ref[j]
        drops_causal.append(base - det.predict_proba(xc.reshape(1, -1))[0, c])

        xn = x.copy()
        for j in noncausal:
            xn[j] = benign_ref[j]
        drops_noncausal.append(base - det.predict_proba(xn.reshape(1, -1))[0, c])

        n_checked += 1
        if n_checked >= 40:
            break

    assert n_checked >= 10, "not enough correctly-classified attack samples"
    # Ablating causal features must move detection substantially more than non-causal ones.
    assert np.mean(drops_causal) > 0.2, f"causal ablation too weak: {np.mean(drops_causal):.3f}"
    assert np.mean(drops_causal) > 2 * np.mean(drops_noncausal) + 0.05, (
        f"causal {np.mean(drops_causal):.3f} not >> non-causal {np.mean(drops_noncausal):.3f}"
    )


def test_every_attacked_flow_stays_inside_the_protocol_valid_set():
    """The paper claims attacks are constrained to protocol-valid traffic. That claim is
    falsifiable by dumping one attacked flow, so it gets a test rather than a sentence.

    An earlier displacement implementation added raw Gaussian noise with no projection, which
    produced negative counts and fractional flags. This fails if anyone removes the
    projection again.
    """
    import numpy as np

    from avert.benchmark.attacks import DisplacementAttack
    from avert.benchmark.perturbation import FeatureBounds
    from avert.types import FlowSample

    rng = np.random.default_rng(0)
    # two integer-valued columns (counts/flags) and one continuous
    X = np.column_stack([rng.integers(0, 50, 400), rng.integers(0, 2, 400),
                         rng.normal(5, 2, 400)]).astype(float)
    bounds = FeatureBounds(X)

    class _Det:
        def predict(self, x):
            return np.zeros(len(x), dtype=int)

        def predict_proba(self, x):
            return np.tile([0.9, 0.1], (len(x), 1))

    class _Attr:
        name = "toy"

        def explain(self, detector, x, names, cls, sid, top_k=5):
            from avert.types import Explanation
            return Explanation(sample_id=sid, attributions=np.asarray(x, dtype=float),
                               feature_names=names, method="toy", predicted_class=cls,
                               top_k=top_k)

    atk = DisplacementAttack(_Attr(), X.std(axis=0), bounds, eps=2.0, n_iter=25,
                          restarts=2, top_k=2, seed=0)
    names = ["pkts", "flag", "dur"]
    for i in range(12):
        s = FlowSample(features=X[i].copy(), feature_names=names, dataset="t", sample_id=f"s{i}")
        f = atk.apply(s, _Det()).features
        assert np.all(f >= bounds.lo - 1e-9), f"below observed minimum: {f}"
        assert np.all(f <= bounds.hi + 1e-9), f"above observed maximum: {f}"
        assert np.allclose(f[bounds.is_int], np.round(f[bounds.is_int])), \
            f"integer-valued feature became fractional: {f}"
