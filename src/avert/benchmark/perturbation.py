"""The set of feature vectors an attack is allowed to reach.

Two nested notions, and the difference between them is a paper claim rather than an
implementation detail.

`FeatureBounds` is a **box**: stay inside each feature's observed [min, max] and keep
integer-valued features integer. That rules out negative packet counts and fractional flags. It
does not rule out a flow whose `TotBytes` disagrees with `SrcBytes + DstBytes`, because it treats
every feature as independently settable, and they are not.

`RealizableBounds` adds the extractor's own arithmetic to the box. Free features are perturbed;
derived features are recomputed from them; declared inequalities are repaired. The relations
live in `avert.benchmark.realizability`, are validated against clean traffic before use, and are
re-checked by the gate.

Why this matters beyond realism, measured on this repo on 2026-08-09: under the box alone, a
defender checking six arithmetic identities separated clean from perturbed 5G-NIDD flows at
AUROC 1.000 and CICIoT2023 at 0.96. C3's central negative -- that no statistical signal detects
the displacement attack -- was therefore being measured against a defender who had never been given
a turn. Confining the attack to the realizable set is what makes that negative mean anything.

See Pierazzi et al., IEEE S&P 2020 (doi:10.1109/SP40000.2020.00073) for the problem-space
framing this implements for flow features.
"""
from __future__ import annotations

from collections.abc import Sequence

import numpy as np

from avert.benchmark.realizability import BoundSchema, schema_for


class FeatureBounds:
    """Per-feature box: observed range plus integrality. The weaker of the two sets."""

    def __init__(self, X_clean: np.ndarray):
        X = np.asarray(X_clean, dtype=float)
        self.lo = X.min(axis=0)
        self.hi = X.max(axis=0)
        # Integer-valued features (counts, flags) stay integer.
        self.is_int = np.all(np.mod(X, 1) == 0, axis=0)
        self.d = X.shape[1]

    def project(self, x: np.ndarray) -> np.ndarray:
        """Clip to observed ranges and round integer features."""
        x = np.clip(x, self.lo, self.hi)
        x = np.where(self.is_int, np.round(x), x)
        return x

    def sample_value(self, j: int, rng: np.random.Generator) -> float:
        v = rng.uniform(self.lo[j], self.hi[j])
        return float(round(v)) if self.is_int[j] else float(v)

    # Free by default: with no schema every feature is independently settable.
    @property
    def free_idx(self) -> np.ndarray:
        return np.arange(self.d)

    @property
    def derived_names(self) -> list[str]:
        return []

    def in_box(self, x: np.ndarray) -> bool:
        """Always true after `project`, which clips. Present so the attacks can call one
        interface whether or not the dataset has a declared schema."""
        x = np.asarray(x, dtype=float)
        return bool(np.all(x >= self.lo - 1e-9) and np.all(x <= self.hi + 1e-9))

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Realizability violation. Zero for a plain box, which enforces no relations."""
        x = np.atleast_2d(np.asarray(x, dtype=float))
        return np.zeros(len(x))


class RealizableBounds(FeatureBounds):
    """The box, plus the arithmetic the flow extractor imposes on its own output.

    `project` is ordered deliberately: clip and round the free features first, so integrality
    propagates into the sums that consume them, then repair inequalities, then recompute derived
    features in topological order. Derived features are NOT re-clipped to the observed box
    afterwards -- clipping would break the identity it was just used to satisfy, and a value the
    extractor would genuinely compute is realizable whether or not this particular sample of
    clean traffic happened to contain it. Callers that also want in-distribution flows should
    test `in_box` and reject, which is what the attack search does.
    """

    def __init__(self, X_clean: np.ndarray, feature_names: Sequence[str], dataset: str):
        super().__init__(X_clean)
        self.dataset = dataset
        self.feature_names = list(feature_names)
        self.schema: BoundSchema = schema_for(dataset).bind(self.feature_names)
        derived = set(self.schema.derived)
        self._free_idx = np.array(
            [i for i, n in enumerate(self.feature_names) if n not in derived], dtype=int)

    @property
    def free_idx(self) -> np.ndarray:
        """Indices the adversary may set directly. Derived features are excluded."""
        return self._free_idx

    @property
    def derived_names(self) -> list[str]:
        return self.schema.derived

    def project(self, x: np.ndarray) -> np.ndarray:
        return self.schema.repair(super().project(x))

    def sample_value(self, j: int, rng: np.random.Generator) -> float:
        """Draw a value for feature `j`. Callers must still `project` afterwards, because
        setting a free feature moves every derived feature that consumes it."""
        return super().sample_value(j, rng)

    def in_box(self, x: np.ndarray) -> bool:
        """Whether every feature of a projected flow is still inside the observed range.

        Used to keep attacked flows on the detector's training manifold. A recomputed derived
        feature can legitimately land outside the clean sample's range; the attack search treats
        that as a rejected candidate rather than clipping it, so the identity always holds.
        """
        x = np.asarray(x, dtype=float)
        return bool(np.all(x >= self.lo - 1e-9) and np.all(x <= self.hi + 1e-9))

    def residual(self, x: np.ndarray) -> np.ndarray:
        """Worst relative violation of the declared relations, per row."""
        return self.schema.residual(x)


def bounds_for(X_clean: np.ndarray, feature_names: Sequence[str], dataset: str) -> FeatureBounds:
    """Build the tightest constraint set characterised for this dataset.

    Falls back to the plain box only where no schema is declared -- synthetic data in the
    walking skeleton has no extractor, so it has no arithmetic to respect. The fallback is
    visible rather than silent: `RealizableBounds.derived_names` is empty and the validator
    records that no relation was declared.
    """
    return RealizableBounds(X_clean, feature_names, dataset)
