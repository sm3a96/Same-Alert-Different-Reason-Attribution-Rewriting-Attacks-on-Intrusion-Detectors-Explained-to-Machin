"""Session-aware splitting: the fix for the leakage, and the guards that keep it honest.

Three defects made these necessary, and all three had produced published numbers:

  1. A random split put flows from one capture on both sides, so anything correlated with
     capture session was a shortcut. Detector accuracy 0.916 random against 0.860 grouped.
  2. Calibration and test flows were drawn from the whole dataset, training rows included --
     10 of 100 calibration flows were training rows, which voids the conformal guarantee's
     stated precondition.
  3. A grouping can look principled and partition nothing. CICIoT2023's distributed CSV is
     pre-shuffled, so row blocks there are random subsets; a split built on them is a random
     split wearing a better name, and the code must say so rather than pretend.
"""
from __future__ import annotations

import numpy as np
import pytest

from avert.data.grouping import (
    Grouping,
    block_purity,
    from_labels,
    from_row_blocks,
    grouped_split,
    grouping_for,
)


def _session_ordered(n_per=500, n_sessions=12, seed=0):
    """Traffic recorded capture by capture: contiguous, class-homogeneous stretches."""
    rng = np.random.default_rng(seed)
    y = np.repeat(np.arange(n_sessions) % 4, n_per)
    X = rng.normal(size=(len(y), 5)) + y[:, None]
    return X, y


def _shuffled(n=6000, seed=0):
    """The CICIoT2023 case: every capture interleaved row by row."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 4, n)
    return rng.normal(size=(n, 5)), y


# ---- what a grouping is allowed to claim ------------------------------------------

def test_row_blocks_are_kept_when_order_carries_capture_structure():
    X, y = _session_ordered()
    g = grouping_for("toy", len(y), None, y=y)
    assert g.usable and g.kind == "row_block"
    mean_pur, global_pur = block_purity(g.key, y)
    assert mean_pur > global_pur + 0.10


def test_row_blocks_are_rejected_when_the_corpus_is_pre_shuffled():
    """The guard that stops a random split from being reported as a session-aware one."""
    X, y = _shuffled()
    g = grouping_for("toy", len(y), None, y=y)
    assert not g.usable and g.kind == "none"
    assert "pre-shuffled" in g.describe()


def test_label_grouping_needs_to_be_finer_than_the_classes():
    """5G-NIDD's label gives 9 groups for 9 classes. Grouping by class would put whole attack
    types on one side -- principled-looking and fatal."""
    y = np.repeat(np.arange(9), 100)
    labels = np.array([f"class{v}" for v in y])
    g = grouping_for("toy", len(y), labels, n_classes=9, y=y)
    assert g.kind != "capture_session"


def test_label_grouping_is_used_when_it_carries_sessions():
    """CICIoMT2024's case: the label encodes class AND capture number."""
    y = np.repeat(np.arange(4), 300)
    labels = np.array([f"class{v}_session{i % 3}" for i, v in enumerate(y)])
    g = grouping_for("toy", len(y), labels, n_classes=4, y=y)
    assert g.kind == "capture_session" and g.n_groups == 12


def test_provider_split_is_detected_and_honoured():
    y = np.repeat(np.arange(3), 400)
    labels = np.array([f"class{v}_" + ("test" if i % 5 == 0 else "train")
                       for i, v in enumerate(y)])
    g = from_labels(labels)
    assert g.provider_split is not None
    split = grouped_split(g, seed=0)
    assert split.n_groups_used["honoured_provider_split"] is True
    # Every row the provider called test must be on the test side, and nowhere else.
    assert set(np.where(g.provider_split == "test")[0]) == set(split.test.tolist())


# ---- what a split must guarantee --------------------------------------------------

@pytest.mark.parametrize("seed", [0, 1, 2])
def test_no_group_straddles_the_split(seed):
    X, y = _session_ordered(seed=seed)
    g = grouping_for("toy", len(y), None, y=y)
    split = grouped_split(g, seed=seed)
    split.assert_disjoint(g.key)          # raises if any group appears on two sides


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_train_calibration_and_test_rows_never_overlap(seed):
    """Defect 2. The conformal guarantee needs calibration disjoint from training, and the
    plan claimed it was while 10% of it was training data."""
    X, y = _session_ordered(seed=seed)
    g = grouping_for("toy", len(y), None, y=y)
    s = grouped_split(g, seed=seed)
    tr, ca, te = set(s.train.tolist()), set(s.calibration.tolist()), set(s.test.tolist())
    assert not (tr & ca) and not (tr & te) and not (ca & te)
    assert len(tr) + len(ca) + len(te) == len(y)


def test_straddling_split_is_caught_rather_than_tolerated():
    g = from_row_blocks(1000, target_blocks=10)
    bad = grouped_split(g, seed=0)
    leaky = type(bad)(train=np.arange(0, 600), calibration=np.arange(600, 800),
                      test=np.arange(500, 1000),        # deliberately overlaps train
                      grouping_kind="row_block", n_groups_used={})
    with pytest.raises(AssertionError, match="span more than one side"):
        leaky.assert_disjoint(g.key)


def test_unusable_grouping_falls_back_but_flags_itself():
    """CICIoT2023. A random split is the only option, and the metadata has to admit it so no
    downstream report can call the result session-aware."""
    X, y = _shuffled()
    g = grouping_for("toy", len(y), None, y=y)
    s = grouped_split(g, seed=0)
    assert s.grouping_kind == "none"
    assert "optimistic" in s.n_groups_used["WARNING"]
    tr, ca, te = set(s.train.tolist()), set(s.calibration.tolist()), set(s.test.tolist())
    assert not (tr & ca) and not (tr & te) and not (ca & te)


def test_grouping_survives_row_subsetting():
    """The loader dedups after grouping is derived, so the key must be subsettable."""
    X, y = _session_ordered()
    g = grouping_for("toy", len(y), None, y=y)
    keep = np.sort(np.random.default_rng(0).choice(len(y), len(y) // 2, replace=False))
    sub = Grouping(g.key[keep], g.kind, g.names, None)
    grouped_split(sub, seed=0).assert_disjoint(sub.key)


def test_target_classes_honours_the_cap():
    """`n` caps the number of targets, and dropping that cap is invisible until a run overruns.

    Deduplicating run_matrix's and run_decision_utility's copies of the selection rule on
    2026-08-10 lost the `[:n]` both originals had. 5G-NIDD hid it -- only three of its classes
    survive the split -- while CICIoT2023 ran 14 targets against a design of 3, and the matrix
    reached 84 cells of an intended 45 before anyone noticed, nine hours in.
    """
    import numpy as np

    from avert.benchmark.harness import classes_supported_by_split, target_classes
    from avert.data.datasets import LoadedData
    from avert.data.grouping import grouping_for

    # Twelve well-populated classes over session-ordered captures: every one is supportable,
    # so any cap that is not applied shows up immediately.
    rng = np.random.default_rng(0)
    y = np.repeat(np.arange(12), 3000)
    X = rng.normal(size=(len(y), 4)) + y[:, None]
    names = [f"c{i}" for i in range(12)]
    data = LoadedData(X=X, y=y, feature_names=["a", "b", "c", "d"], name="toy",
                      label_names=names, grouping=grouping_for("toy", len(y), None, y=y))

    supported, _ = classes_supported_by_split(data, n_cal=10, n_test=10)
    assert len(supported) > 3, "fixture must offer more classes than the cap, or it proves nothing"

    for n in (1, 3, 5):
        keep, _ = target_classes(data, n, n_cal=10, n_test=10)
        assert len(keep) == n, f"asked for {n} targets, got {len(keep)}"
        assert keep == supported[:n], "the cap must keep the most common supported classes"


def test_target_classes_requires_support_at_every_seed():
    """The grouped split is seeded, so which captures land in calibration changes with the seed.

    Selecting targets at seed 0 alone cost four CICIoMT2024 cells on 2026-08-11: classes 1 and 17
    were comfortably supported at seed 0 and had ZERO calibration flows at seeds 2, 3 and 4. That
    left one corpus aggregating over 11 cells while the others had 15 -- unequal replication,
    which is the same defect as unequal class counts wearing a different hat.
    """
    import numpy as np

    from avert.benchmark.harness import target_classes
    from avert.data.datasets import LoadedData
    from avert.data.grouping import grouping_for

    rng = np.random.default_rng(0)
    # Two abundant classes plus one confined to a narrow stretch of captures, so which side it
    # lands on genuinely depends on the seed.
    y = np.concatenate([np.repeat([0, 1], 8000), np.full(2200, 2)])
    X = rng.normal(size=(len(y), 4)) + y[:, None]
    data = LoadedData(X=X, y=y, feature_names=list("abcd"), name="toy",
                      label_names=["c0", "c1", "rare"],
                      grouping=grouping_for("toy", len(y), None, y=y))

    seeds = [0, 1, 2, 3, 4]
    keep_all, _ = target_classes(data, 3, n_cal=100, n_test=40, seeds=seeds)

    # Whatever survives must survive at EVERY seed on its own, or the replicates differ.
    for s in seeds:
        keep_s, _ = target_classes(data, 3, n_cal=100, n_test=40, seeds=[s])
        assert set(keep_all).issubset(set(keep_s)), (
            f"class selected across seeds is unsupported at seed {s}: "
            f"{set(keep_all) - set(keep_s)}")
