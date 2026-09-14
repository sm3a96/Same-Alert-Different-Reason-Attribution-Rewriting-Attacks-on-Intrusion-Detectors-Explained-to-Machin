"""Data splits. Two splitting regimes serve two distinct purposes (Plan Phase 1):

1. DETECTOR evaluation uses `time_ordered_split` (and cross-dataset transfer) to
   avoid the optimistic bias documented for this literature, where random splitting
   and cross-train normalization inflate reported NIDS performance.
2. The CONFORMAL monitor uses `train_calib_test_split`, whose calibration partition
   must be CLEAN traffic only and exchangeable with the test stream within a regime.
   Keep it disjoint from training/test and never let an attacked instance leak in —
   that would void the false-alarm guarantee (Plan Section 3.6).
"""
from __future__ import annotations

import numpy as np

from avert.data.datasets import LoadedData


def time_ordered_split(
    data: LoadedData,
    test_frac: float = 0.2,
) -> dict[str, LoadedData]:
    """Chronological split for DETECTOR evaluation: no shuffle, test is the tail of
    the (time-ordered) stream. Use this for detector accuracy, never a random split."""
    n = len(data.X)
    cut = int((1 - test_frac) * n)

    def sub(sl):
        return LoadedData(data.X[sl], data.y[sl], data.feature_names, data.name)

    return {"train": sub(slice(0, cut)), "test": sub(slice(cut, n))}


def train_calib_test_split(
    data: LoadedData,
    calib_frac: float = 0.3,
    test_frac: float = 0.2,
    seed: int = 0,
) -> dict[str, LoadedData]:
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data.X))
    n_test = int(test_frac * len(idx))
    n_calib = int(calib_frac * len(idx))
    test_i, calib_i, train_i = idx[:n_test], idx[n_test : n_test + n_calib], idx[n_test + n_calib :]

    def sub(ii):
        return LoadedData(data.X[ii], data.y[ii], data.feature_names, data.name)

    return {"train": sub(train_i), "calibration": sub(calib_i), "test": sub(test_i)}


def clean_only(data: LoadedData, attacked_mask: np.ndarray | None = None) -> LoadedData:
    """Guard: strip any attacked instances from a would-be calibration set."""
    if attacked_mask is None:
        return data
    keep = ~attacked_mask
    return LoadedData(data.X[keep], data.y[keep], data.feature_names, data.name)
