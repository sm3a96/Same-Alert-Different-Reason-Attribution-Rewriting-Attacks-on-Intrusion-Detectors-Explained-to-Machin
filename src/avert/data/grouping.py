"""Capture-session grouping, so a session never spans train and test.

Dropping a leaking column treats the symptom. The cause is the split. Every corpus here records
each attack in its own capture session, so a random train/test partition puts flows from the
same session on both sides, and *any* feature that correlates with session becomes a shortcut.
`IAT` was the most blatant -- an exact-value lookup table on it predicted held-out CICIoT2023
rows at 1.000 -- but removing it leaves the mechanism intact for whatever correlates next.

The fix is the one the malware community settled on: split by group, never by row. Pendlebury,
Pierazzi, Jordaney, Kinder and Cavallaro name the two failures precisely (TESSERACT, USENIX
Security 2019, arXiv:1807.07838) -- spatial bias, where the evaluation distribution is not the
deployment one, and temporal bias, where training data postdates test data. A random split over
session-structured captures commits both.

Why it matters more here than in an ordinary NIDS paper. XInt-Bench's ground truth is
*detector-relative*: a feature is causal if ablating it moves the deployed detector. A detector
trained across a leaking split learns the session fingerprint, so the fingerprint enters the
causal set and the benchmark scores attributions against an artifact. Leakage does not merely
inflate accuracy here; it corrupts the labels everything else is built on.

What each corpus offers, in descending order of strength:

  CICIoMT2024   the provider's OWN train/test partition, 7,160,831 / 1,614,182 rows, encoded as
                a `_train` / `_test` suffix on the label -- plus 44 numbered capture sessions
                (TCP_IP-DDoS-ICMP1..4). Both were being discarded by `label_strip` and replaced
                with a random split. Group = the full raw label, so a session cannot span sides.
  CICIoT2023    no session marker survives into the cached table. Fall back to contiguous
                row-order blocks: the cache preserves concatenation order, so a block
                corresponds to a stretch of one capture.
  5G-NIDD       same fallback. The row counter and Argus sequence number are dropped as
                FEATURES because a tree reads them at 0.806 -- but row order is exactly the
                right thing to SPLIT on. Use the index to partition, never to learn.

`GroupedSplit` assigns whole groups, and `scripts/check_split_leakage.py` then tries to tell
train rows from test rows adversarially. A split that survives that check is one where no
per-row shortcut remains, which is the property the column audit cannot establish on its own.
"""
from __future__ import annotations

import re
from dataclasses import dataclass

import numpy as np

# Roughly this many groups when falling back to row-order blocks. Enough that a grouped split
# has granularity to hit its target proportions, few enough that each block spans a meaningful
# stretch of one capture rather than a handful of adjacent flows.
TARGET_BLOCKS = 200

PROVIDER_SPLIT = re.compile(r"_(train|test)$", re.I)

# A row-block grouping must beat the global class share by at least this much to count as
# having captured anything. 5G-NIDD scores 0.940 against 0.355 and CICIoMT2024 0.930 against
# 0.371; CICIoT2023 scores 0.140 against 0.136 and is correctly rejected.
MIN_PURITY_GAIN = 0.10


@dataclass(frozen=True)
class Grouping:
    """Per-row group ids, plus the provider's own partition where one exists."""

    key: np.ndarray                       # group id per row, contiguous ints
    kind: str                             # "capture_session" | "row_block"
    names: list[str]                      # group id -> human-readable name
    provider_split: np.ndarray | None     # per row: "train" / "test", or None

    @property
    def n_groups(self) -> int:
        return len(self.names)

    @property
    def usable(self) -> bool:
        """False when no session structure could be recovered, so a grouped split is impossible
        and any result on this corpus is under a random split and therefore optimistic."""
        return self.kind != "none"

    def describe(self) -> str:
        if self.kind == "none":
            return ("NO usable grouping -- the distributed corpus is pre-shuffled, so a "
                    "session-aware split cannot be constructed and results are optimistic")
        extra = ""
        if self.provider_split is not None:
            n_te = int((self.provider_split == "test").sum())
            extra = (f"; provider partition present "
                     f"({len(self.provider_split) - n_te:,} train / {n_te:,} test)")
        return f"{self.n_groups} {self.kind} groups{extra}"


def from_labels(raw_labels: np.ndarray) -> Grouping:
    """Group by the raw label string, which on CICIoMT2024 carries class, capture-session
    number and the provider's partition all at once."""
    raw = np.asarray(raw_labels, dtype=object).astype(str)
    names = sorted(set(raw))
    index = {n: i for i, n in enumerate(names)}
    split = None
    if any(PROVIDER_SPLIT.search(n) for n in names):
        split = np.array([(PROVIDER_SPLIT.search(v).group(1).lower()
                           if PROVIDER_SPLIT.search(v) else "train") for v in raw])
    return Grouping(np.array([index[v] for v in raw], dtype=int),
                    "capture_session", names, split)


def from_row_blocks(n_rows: int, target_blocks: int = TARGET_BLOCKS) -> Grouping:
    """Contiguous blocks of the stored row order -- a temporal split in TESSERACT's sense.

    The cached table preserves the order the source files were concatenated in, so a block is a
    stretch of one capture rather than a random scatter across all of them.
    """
    size = max(1, n_rows // max(1, target_blocks))
    key = np.arange(n_rows) // size
    key = np.minimum(key, key.max())
    return Grouping(key.astype(int), "row_block",
                    [f"block{i:03d}" for i in range(int(key.max()) + 1)], None)


def block_purity(key: np.ndarray, y: np.ndarray) -> tuple[float, float]:
    """Mean within-group majority-class share, against the global majority share.

    The test for whether a grouping has captured anything. Traffic recorded capture by capture
    is class-homogeneous in stretches, so real groups are far purer than the corpus as a whole.
    A grouping whose purity equals the global share has partitioned nothing, and a split built
    on it is a random split wearing a principled name.
    """
    pur = [float(np.bincount(y[key == b]).max() / max(1, (key == b).sum()))
           for b in np.unique(key)]
    return float(np.mean(pur)), float(np.bincount(y).max() / len(y))


def grouping_for(dataset: str, n_rows: int, raw_labels: np.ndarray | None,
                 n_classes: int | None = None, y: np.ndarray | None = None) -> Grouping:
    """The strongest grouping this corpus supports.

    A label-derived grouping is only used when it is STRICTLY FINER than the class labels, or
    when it carries the provider's own partition. Otherwise the "groups" are just the classes,
    and grouping by class would put whole attack types on one side of the split -- a degenerate
    partition that looks principled and destroys the experiment. 5G-NIDD's label yields exactly
    9 groups for 9 classes and must fall through to row blocks; CICIoMT2024's yields 72 raw
    labels over ~19 classes and must not.
    """
    if raw_labels is not None and len(raw_labels) == n_rows:
        cand = from_labels(raw_labels)
        finer = n_classes is None or cand.n_groups > n_classes
        if cand.provider_split is not None or (finer and cand.n_groups >= 8):
            return cand

    blocks = from_row_blocks(n_rows)
    if y is None:
        return blocks

    # Row blocks are only meaningful if the stored order carries capture structure. On the
    # Kaggle mirror of CICIoT2023 it does not, and no rebuild can recover it: the distributed
    # `ciciot23.csv` is pre-shuffled row by row -- mean label run length 1.10, all 34 classes
    # inside the first 200k rows. Block purity there is 0.140 against a global majority share of
    # 0.136 at every block size. Say so instead of returning groups that partition nothing.
    mean_pur, global_pur = block_purity(blocks.key, y)
    if mean_pur < global_pur + MIN_PURITY_GAIN:
        return Grouping(np.zeros(n_rows, dtype=int), "none", ["all"], None)
    return blocks


@dataclass(frozen=True)
class GroupedSplit:
    """Row indices for train / calibration / test, with no group on more than one side."""

    train: np.ndarray
    calibration: np.ndarray
    test: np.ndarray
    grouping_kind: str
    n_groups_used: dict

    def assert_disjoint(self, groups: np.ndarray) -> None:
        a, b, c = (set(groups[self.train]), set(groups[self.calibration]), set(groups[self.test]))
        overlap = (a & b) | (a & c) | (b & c)
        if overlap:
            raise AssertionError(
                f"{len(overlap)} group(s) span more than one side of the split: "
                f"{sorted(overlap)[:5]} -- the split leaks by construction")


def grouped_split(grouping: Grouping, test_frac: float = 0.2, calibration_frac: float = 0.3,
                  seed: int = 0, honour_provider: bool = True) -> GroupedSplit:
    """Assign whole groups, so no capture session appears on two sides.

    When the provider shipped its own partition we honour it for the test side rather than
    inventing one: their split is the closest thing to a deployment boundary that exists, and
    replacing it with our own would discard information and invite the obvious objection.
    Calibration is then carved out of the remaining training groups, because the conformal
    guarantee needs calibration data exchangeable with the test stream but disjoint from it.
    """
    rng = np.random.default_rng(seed)
    g = grouping.key
    all_groups = np.unique(g)

    if not grouping.usable:
        # No session structure could be recovered, so there is no grouped split to make. Fall
        # back to a random one and SAY SO in the metadata rather than returning something that
        # looks principled. Every number produced under this split is optimistic, and the flag
        # is what makes downstream reporting able to admit it.
        perm = rng.permutation(len(g))
        n_test = max(1, int(round(test_frac * len(perm))))
        rest = perm[n_test:]
        n_cal = max(1, int(round(calibration_frac * len(rest))))
        return GroupedSplit(
            train=np.sort(rest[n_cal:]), calibration=np.sort(rest[:n_cal]),
            test=np.sort(perm[:n_test]), grouping_kind="none",
            n_groups_used={"train": 0, "calibration": 0, "test": 0,
                           "honoured_provider_split": False,
                           "WARNING": "no usable grouping; random split, results optimistic"})

    if honour_provider and grouping.provider_split is not None:
        is_test = np.array([grouping.provider_split[g == gid][0] == "test" for gid in all_groups])
        test_groups = all_groups[is_test]
        rest = all_groups[~is_test]
    else:
        shuffled = rng.permutation(all_groups)
        n_test = max(1, int(round(test_frac * len(shuffled))))
        test_groups, rest = shuffled[:n_test], shuffled[n_test:]

    rest = rng.permutation(rest)
    n_cal = max(1, int(round(calibration_frac * len(rest))))
    cal_groups, train_groups = rest[:n_cal], rest[n_cal:]

    where = lambda gs: np.where(np.isin(g, gs))[0]  # noqa: E731
    split = GroupedSplit(
        train=where(train_groups), calibration=where(cal_groups), test=where(test_groups),
        grouping_kind=grouping.kind,
        n_groups_used={"train": int(len(train_groups)), "calibration": int(len(cal_groups)),
                       "test": int(len(test_groups)),
                       "honoured_provider_split": bool(
                           honour_provider and grouping.provider_split is not None)})
    split.assert_disjoint(g)
    return split
