"""Real attribution methods (Plan Phase 2, P2-2), mapped to detector families per the
method->model applicability rule (Plan Section 3.3):

  PermutationAttributor          — model-agnostic, applies to every detector.
  TreeSHAPAttributor             — tree detectors (XGBoost), exact + fast.
  IntegratedGradientsAttributor  — differentiable detectors (MLP, FT-Transformer), Captum.

`attributors_for(detector, background)` returns the applicable, mechanistically distinct
set for a detector, so Signal 3 (cross-method consensus) always gets >=2 genuinely
different estimators. Signal 1 smooths a single fast attribution (integrated gradients on
differentiable detectors) — see certified_stability.py.
"""
from __future__ import annotations

import numpy as np

from avert.attribution.base import Attributor
from avert.detectors.base import Detector
from avert.types import Explanation


def _shap_for_class(sv, predicted_class: int) -> np.ndarray:
    """Normalize shap_values output (list / 2D / 3D) to a 1-D per-feature vector."""
    if isinstance(sv, list):
        return np.asarray(sv[predicted_class])[0]
    sv = np.asarray(sv)
    if sv.ndim == 3:                              # (n, d, n_classes) or (n_classes, n, d)
        if sv.shape[0] == 1:
            return sv[0, :, predicted_class]
        return sv[predicted_class, 0, :]
    row = sv[0]                                   # 2D (n, d): binary, values for class 1
    return row if predicted_class == 1 else -row


class TreeSHAPAttributor(Attributor):
    name = "treeshap"
    requires_differentiable = False

    def __init__(self):
        self._explainer = None
        self._det = None

    def explain(self, detector, x, feature_names, predicted_class, sample_id, top_k=5):
        import shap

        booster = getattr(detector, "booster", None)
        if booster is None:
            raise ValueError("TreeSHAPAttributor needs a tree detector exposing .booster")
        if self._explainer is None or self._det is not detector:
            self._explainer = shap.TreeExplainer(booster)
            self._det = detector
        sv = self._explainer.shap_values(np.atleast_2d(x))
        # Translate the dataset-space class into the booster's own output space. They differ
        # whenever a training subsample missed a class -- common on CICIoT2023's 34 classes and
        # on any grouped split, where whole captures are held out. Indexing SHAP output by a
        # dataset label produced "index 17 is out of bounds for axis 2 with size 16" and killed
        # 12 of 45 matrix cells.
        idx = int(predicted_class)
        if hasattr(detector, "encoded_index"):
            enc = detector.encoded_index(idx)
            if enc is None:
                raise ValueError(
                    f"{self.name}: class {idx} was absent from this detector's training set, so "
                    f"it has no attribution to give. Pick a class the detector can predict.")
            idx = enc
        attr = _shap_for_class(sv, idx)
        return Explanation(sample_id, np.asarray(attr, dtype=float), feature_names, self.name, predicted_class, top_k)


class IntegratedGradientsAttributor(Attributor):
    name = "integrated_gradients"
    requires_differentiable = True

    def __init__(self, n_steps: int = 32):
        self.n_steps = n_steps

    def explain(self, detector, x, feature_names, predicted_class, sample_id, top_k=5):
        from captum.attr import IntegratedGradients

        if not hasattr(detector, "to_input_tensor") or getattr(detector, "module", None) is None:
            raise ValueError("IntegratedGradients needs a torch detector (.module, .to_input_tensor)")
        ig = IntegratedGradients(detector.module)
        xt = detector.to_input_tensor(x).requires_grad_(True)   # scaled-space input
        # The module's output columns follow the classes seen in TRAINING; `predict_proba`
        # re-expands them to the dataset's label space, but Captum attributes the raw module.
        # When a class is absent from a grouped training side the two spaces differ, and a
        # dataset label indexes past the module's last column. Same defect class as the
        # TreeSHAP `encoded_index` fix of 2026-08-10; it surfaced 2026-09-07 on 5G-NIDD seed 1
        # in the transferability run.
        target = int(predicted_class)
        classes = getattr(detector, "_classes", None)
        if classes is not None:
            hits = np.where(np.asarray(classes).astype(int) == target)[0]
            if len(hits) == 0:
                raise ValueError(f"class {target} was not seen in training; nothing to attribute")
            target = int(hits[0])
        att = ig.attribute(xt, target=target, n_steps=self.n_steps)
        attr = att.detach().cpu().numpy().ravel()
        return Explanation(sample_id, attr.astype(float), feature_names, self.name, predicted_class, top_k)


class PermutationAttributor(Attributor):
    """Model-agnostic per-sample importance: drop in predicted-class probability when a
    feature is replaced by background reference values (averaged over samples)."""

    name = "permutation"
    requires_differentiable = False

    def __init__(self, background: np.ndarray, n_samples: int = 10, seed: int = 0):
        self.background = np.asarray(background)
        self.n_samples = n_samples
        self.rng = np.random.default_rng(seed)

    def explain(self, detector, x, feature_names, predicted_class, sample_id, top_k=5):
        x = np.asarray(x, dtype=float)
        base = detector.predict_proba(x.reshape(1, -1))[0, predicted_class]
        idx = self.rng.integers(0, len(self.background), size=self.n_samples)
        attr = np.zeros(len(x))
        for j in range(len(x)):
            drops = []
            for i in idx:
                xj = x.copy()
                xj[j] = self.background[i, j]
                drops.append(base - detector.predict_proba(xj.reshape(1, -1))[0, predicted_class])
            attr[j] = float(np.mean(drops))
        return Explanation(sample_id, attr, feature_names, self.name, predicted_class, top_k)


class LIMEAttributor(Attributor):
    """LIME-style local linear attribution: fit a proximity-weighted linear model to the
    detector's predicted-class probability over MULTI-feature perturbations around x. Being
    perturbation-based with off-manifold queries, it is the method an A3 scaffold fools —
    which is exactly why pairing it with TreeSHAP makes cross-method consensus catch A3.
    """

    name = "lime"
    requires_differentiable = False

    def __init__(self, background: np.ndarray, n_samples: int = 200, sigma: float = 0.5,
                 kernel: float = 1.0, seed: int = 0):
        self.std = np.asarray(background).std(axis=0) + 1e-9
        self.n_samples = n_samples
        self.sigma = sigma
        self.kernel = kernel
        self.rng = np.random.default_rng(seed)

    def explain(self, detector, x, feature_names, predicted_class, sample_id, top_k=5):
        from sklearn.linear_model import Ridge

        x = np.asarray(x, dtype=float)
        Z = x + self.rng.normal(0, self.sigma * self.std, size=(self.n_samples, len(x)))
        Z[0] = x
        proba = detector.predict_proba(Z)[:, predicted_class]
        dist = np.linalg.norm((Z - x) / self.std, axis=1)
        w = np.exp(-(dist ** 2) / (2 * self.kernel ** 2))
        coef = Ridge(alpha=1.0).fit(Z - x, proba, sample_weight=w).coef_
        return Explanation(sample_id, coef.astype(float), feature_names, self.name, predicted_class, top_k)


def attributors_for(detector: Detector, background: np.ndarray) -> list[Attributor]:
    """Applicable, mechanistically distinct attributors for `detector` (>=2)."""
    methods: list[Attributor] = [PermutationAttributor(background)]   # universal
    if getattr(detector, "booster", None) is not None:
        methods.append(TreeSHAPAttributor())                         # tree family
    if detector.is_differentiable and getattr(detector, "module", None) is not None:
        methods.append(IntegratedGradientsAttributor())             # differentiable family
    return methods
