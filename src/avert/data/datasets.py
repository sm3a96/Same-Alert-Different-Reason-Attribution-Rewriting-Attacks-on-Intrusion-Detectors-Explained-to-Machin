"""Dataset loading. The walking skeleton uses SyntheticDataset; Phase 1 implements
the real loaders against the same interface.

Datasets (Plan Phase 1, per Goldschmidt & Chuda 2025 survey):
  CICIoT2023, CICIoV2024, 5G-NIDD  — modern, lead with these.
  CICIDS2017 (Engelen 2021 corrected) — legacy bridge only.
  NetFlow-v2 standardised variants (Sarhan 2022) — for cross-dataset transfer.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from avert.data.grouping import Grouping, grouping_for
from avert.types import AttackClass, FlowSample


@dataclass
class LoadedData:
    X: np.ndarray
    y: np.ndarray
    feature_names: list[str]
    name: str
    label_names: list[str] | None = None      # class index -> original label string
    grouping: Grouping | None = None          # capture sessions, so a split cannot straddle one

    def benign_index(self) -> int | None:
        """Index of the benign/normal class (for clean-traffic calibration), if known."""
        if not self.label_names:
            return None
        for i, nm in enumerate(self.label_names):
            low = str(nm).strip().lower()
            if "benign" in low or "normal" in low or low in ("0", "background"):
                return i      # substring match catches e.g. CICIoT2023's "BenignTraffic"
        return None

    def to_samples(self) -> list[FlowSample]:
        return [
            FlowSample(
                features=self.X[i], feature_names=self.feature_names, dataset=self.name,
                true_label=int(self.y[i]), sample_id=f"{self.name}-{i}",
            )
            for i in range(len(self.X))
        ]


class SyntheticDataset:
    """Linearly-separable-ish synthetic flows with known causal features. Toy only."""

    def __init__(self, n: int = 600, d: int = 12, n_causal: int = 4, seed: int = 0):
        self.n, self.d, self.n_causal, self.seed = n, d, n_causal, seed

    def load(self) -> LoadedData:
        rng = np.random.default_rng(self.seed)
        X = rng.normal(0, 1, size=(self.n, self.d))
        w = np.zeros(self.d)
        w[: self.n_causal] = rng.uniform(1.5, 3.0, size=self.n_causal)  # first features causal
        logits = X @ w + rng.normal(0, 0.3, size=self.n)
        y = (logits > np.median(logits)).astype(int)
        names = [f"f{j}" for j in range(self.d)]
        return LoadedData(X, y, names, "synthetic")

    @property
    def causal_features(self) -> list[str]:
        return [f"f{j}" for j in range(self.n_causal)]


_LABEL_CANDIDATES = ["label", "Label", "attack", "Attack", "class", "Class", "Label_", "type"]
_REPO = Path(__file__).resolve().parents[3]


def load_csv_directory(
    name: str,
    label_col: str | None = None,
    drop_cols: tuple[str, ...] = (),
    subdir: str | None = None,
    sample_frac: float | None = None,
    dedupe: bool = True,
    label_strip: str | None = None,
    raw_root: Path | None = None,
) -> LoadedData:
    """Load a CIC-style dataset (a directory of CSVs) into LoadedData, standardizing the
    schema: numeric feature columns + an integer-encoded label. Caches to parquet so the
    expensive CSV concat happens once. `subdir` restricts to one representation when a
    dataset ships several with different schemas (e.g. CICIoV2024 decimal/ vs binary/).
    `sample_frac` chunk-subsamples huge CSVs (e.g. CICIoT2023's 13.7 GB) without OOM.
    """
    import pandas as pd

    raw_root = raw_root or (_REPO / "data")
    processed = raw_root / "processed" / f"{name}.parquet"
    raw_dir = raw_root / "raw" / name / subdir if subdir else raw_root / "raw" / name

    if processed.exists():
        df = pd.read_parquet(processed)
    else:
        csvs = sorted(raw_dir.rglob("*.csv"))
        if not csvs:
            raise FileNotFoundError(
                f"no CSVs under {raw_dir}. Run: python scripts/download_data.py --dataset {name}"
            )
        if sample_frac:                                   # chunked subsample for huge files
            # `.sort_index()` is not cosmetic. `DataFrame.sample` returns rows in the order it
            # drew them, so the cached table came out with capture order destroyed inside every
            # chunk. Measured on CICIoT2023: mean class purity within a row block is 0.140 at
            # every block size, against a global majority share of 0.136 -- the blocks are
            # random subsets, so a row-order grouping cannot separate captures and the
            # session-aware split silently degrades to a random one.
            #
            # Sorting restores the original order and selects exactly the same rows, so this
            # changes the arrangement of the cache and not its contents.
            frames = [
                chunk.sample(frac=sample_frac, random_state=0).sort_index()
                for c in csvs
                for chunk in pd.read_csv(c, chunksize=1_000_000, low_memory=False)
            ]
            df = pd.concat(frames, ignore_index=True)
        else:
            df = pd.concat((pd.read_csv(c, low_memory=False) for c in csvs), ignore_index=True)
        df.columns = [str(c).strip() for c in df.columns]
        processed.parent.mkdir(parents=True, exist_ok=True)
        df.to_parquet(processed)

    label_col = label_col or next((c for c in _LABEL_CANDIDATES if c in df.columns), None)
    if label_col is None:
        raise ValueError(f"{name}: no label column found; set label_col in the dataset config")

    y_raw = df[label_col].astype(str)
    # Keep the label BEFORE stripping. On CICIoMT2024 it carries the provider's own train/test
    # partition and 44 numbered capture sessions, and `label_strip` was discarding both -- so a
    # random split then put flows from one capture on both sides and every session-correlated
    # feature became a shortcut. The stripped copy is the learning target; the raw copy is how
    # we group. See avert.data.grouping.
    raw_for_grouping = y_raw.to_numpy(copy=True)
    if label_strip:                               # e.g. strip "_train"/"_test" leaked into labels
        y_raw = y_raw.str.replace(label_strip, "", regex=True)
    codes, uniques = y_raw.factorize()
    # A drop_cols entry that matches nothing is a silent failure with real consequences: the
    # identifier it was meant to remove stays in the feature matrix and every downstream number
    # is computed with it. On 2026-08-09 that cost 5G-NIDD a row-index column a depth-8 tree
    # could read at 0.806 held-out. Fail loudly instead. (YAML gotcha: a name containing a colon
    # must be quoted, or it parses as a nested mapping rather than a string.)
    bad_types = [c for c in drop_cols if not isinstance(c, str)]
    if bad_types:
        raise TypeError(
            f"{name}: drop_cols entries must be strings, got {bad_types!r}. A column name "
            f"containing ':' needs quoting in the dataset YAML.")
    missing = [c for c in drop_cols if c not in df.columns]
    if missing:
        raise KeyError(
            f"{name}: drop_cols names {missing} are not columns of this dataset "
            f"(have {list(df.columns)[:12]}...). Fix the config -- an unmatched drop_cols "
            f"entry silently leaves the column in the features.")
    feat_df = df.drop(columns=[label_col, *drop_cols])
    feat_df = feat_df.select_dtypes(include="number").replace([np.inf, -np.inf], np.nan).fillna(0.0)

    X = feat_df.to_numpy(dtype=np.float32)
    y = codes.astype(int)
    names = list(feat_df.columns)
    groups_raw = raw_for_grouping

    # Drop constant (zero-variance) features — they carry no signal and break scaling.
    keep = X.std(axis=0) > 0
    X, names = X[:, keep], [n for n, k in zip(names, keep) if k]

    # Deduplicate full rows (features+label). CIC-style datasets contain heavy row
    # duplication; left in, identical rows leak across train/test (inflating accuracy)
    # and shrink the effective sample behind the conformal exchangeability assumption.
    if dedupe:
        _, ui = np.unique(np.column_stack([X, y]), axis=0, return_index=True)
        ui.sort()                       # keeps first occurrences in stored order, so row order
        X, y = X[ui], y[ui]             # -- and therefore the row-block grouping -- survives
        groups_raw = groups_raw[ui]

    grouping = grouping_for(name, len(X), groups_raw, n_classes=len(uniques), y=y)
    return LoadedData(X=X, y=y, feature_names=names, name=name,
                      label_names=[str(u) for u in uniques], grouping=grouping)


def load_dataset(name: str, cfg: dict) -> LoadedData:
    """Dispatch: synthetic for the walking skeleton, CSV-directory loader for real datasets
    (downloaded by scripts/download_data.py)."""
    if name == "synthetic":
        return SyntheticDataset(**cfg.get("synthetic", {})).load()
    return load_csv_directory(
        name,
        label_col=cfg.get("label_col"),
        drop_cols=tuple(cfg.get("drop_cols", ())),
        subdir=cfg.get("subdir"),
        sample_frac=cfg.get("sample_frac"),
        dedupe=cfg.get("dedupe", True),
        label_strip=cfg.get("label_strip"),
    )
