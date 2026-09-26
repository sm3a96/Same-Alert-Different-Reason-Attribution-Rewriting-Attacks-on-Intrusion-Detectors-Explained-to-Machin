"""Data-quality audit (rigor gate before any dataset is used for results).

Reports, per dataset: shape, class balance, duplicate rows, constant features,
identifier-like columns, and per-feature label leakage (a depth-1 stump that alone
predicts the label is a leakage red flag). This is what justifies trusting the
downstream explanations and the conformal exchangeability assumption.

  python scripts/audit_data.py --dataset ciciov2024
  python scripts/audit_data.py --all          # every configured dataset, incl. the excluded one

Each run saves results/_logs/audit_<dataset>.json. Without that artifact an evaluator has to
take the exclusion of CICIoV2024 on trust, which is exactly what a rigor gate must not ask.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

from avert.config import load_config
from avert.data.datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]

# A single feature that beats the majority class by this much, held out, is a leakage suspect.
# 0.20 is deliberately low: the cost of investigating a false positive is minutes, and the cost
# of a missed one is every number computed on that dataset.
LEAK_MARGIN = 0.20

# An exact-value lookup table that transfers to held-out rows this well is memorising an
# identifier, not learning a decision boundary. Deliberately near-perfect: genuine features top
# out around 0.6 on this test even when they are strongly predictive, so the gap is wide and the
# threshold does not need tuning. CIC's IAT sits at 0.999.
FINGERPRINT_ACC = 0.95


def audit(name: str, cfg: dict, sample: int = 100_000, seed: int = 0) -> None:
    data = load_dataset(name, cfg)
    X, y = data.X, data.y
    n, d = X.shape
    rng = np.random.default_rng(seed)
    idx = rng.permutation(n)[:sample]
    Xs, ys = X[idx], y[idx]

    print(f"\n===== AUDIT: {name} =====")
    print(f"rows={n:,}  features={d}")

    # Class balance
    vals, counts = np.unique(y, return_counts=True)
    names = data.label_names or [str(v) for v in vals]
    print("class balance:")
    for v, c in zip(vals, counts):
        print(f"   {names[v][:24]:24s} {c:>10,}  ({100*c/n:5.2f}%)")
    print(f"   benign class index: {data.benign_index()}")

    # Duplicate rows, measured PRE-dedup. `load_dataset` already drops duplicates, so
    # counting them on its output always returns zero -- which is how a dataset that is
    # 99.7% duplicates can look clean to its own audit. Read the stored table instead.
    raw_path = REPO / "data" / "processed" / f"{name}.parquet"
    if raw_path.exists():
        import pandas as pd
        df = pd.read_parquet(raw_path)
        n_raw, n_uniq = len(df), len(df.drop_duplicates())
        dup_frac = 1 - n_uniq / n_raw
        print(f"duplicates (pre-dedup, {n_raw:,} stored rows): {n_raw-n_uniq:,} duplicates "
              f"({100*dup_frac:.2f}%), dedups to {n_uniq:,} unique; the loader hands "
              f"{n:,} rows downstream")
    else:
        n_raw, n_uniq, dup_frac = n, n, 0.0
        print(f"no stored table at {raw_path} -- duplicate rate not measured")

    # Constant + identifier-like features
    stds = Xs.std(axis=0)
    const = [data.feature_names[j] for j in range(d) if stds[j] == 0]
    nuniq = np.array([len(np.unique(Xs[:, j])) for j in range(d)])
    ident = [data.feature_names[j] for j in range(d) if nuniq[j] > 0.99 * len(Xs)]

    # Row-index detection, added 2026-08-09. The near-uniqueness rule above missed 5G-NIDD's
    # `Unnamed: 0`: the corpus is several CSVs concatenated, each restarting its index at 0, so
    # the column is only 60% unique and slipped under the 0.99 threshold while still being a
    # pure row counter. Catch it by shape instead: integer-valued, non-negative, spanning
    # roughly 0..(number of distinct values), and dense inside that span.
    # Density is measured on the FULL column, never the subsample: a row index sampled down to
    # 100k rows out of 1.2M looks sparse inside its own span and the test silently passes.
    index_like = []
    for j in range(d):
        col = X[:, j]
        if col.std() == 0 or not np.all(np.mod(col, 1) == 0) or col.min() < 0:
            continue
        span = float(col.max() - col.min() + 1)
        card = len(np.unique(col))
        if card >= 1000 and span > 0 and card / span > 0.95:
            index_like.append(data.feature_names[j])

    print(f"constant features ({len(const)}): {const[:10]}")
    print(f"identifier-like features ({len(ident)}): {ident[:10]}")
    print(f"row-index-like features ({len(index_like)}): {index_like[:10]}")

    # Per-feature leakage. Two changes on 2026-08-09, both because this check passed 5G-NIDD
    # while `Unnamed: 0` alone predicted the label at 0.81 held-out against a 0.39 baseline:
    #   1. depth 1 -> also depth 8. A stump makes one split and cannot see block-structured
    #      leakage, which is exactly what a row index in a file sorted by attack class gives.
    #   2. score on held-out folds. The old check fit and scored on the SAME rows, so its
    #      number was a training accuracy and the threshold was tuned against a fiction.
    from sklearn.model_selection import cross_val_score
    from sklearn.tree import DecisionTreeClassifier

    base = counts.max() / n
    sub = rng.choice(len(Xs), min(60_000, len(Xs)), replace=False)
    leaks = []
    for j in range(d):
        if stds[j] == 0:
            continue
        accs = {}
        for depth in (1, 8):
            accs[depth] = float(cross_val_score(
                DecisionTreeClassifier(max_depth=depth, random_state=0),
                Xs[sub][:, [j]], ys[sub], cv=3, scoring="accuracy").mean())
        best = max(accs.values())
        # Relative to the majority baseline, NOT a fixed 0.95. The old `max(0.95, base+0.2)`
        # floor made the relative term dead on any dataset with a baseline below 0.75, which is
        # every dataset here -- 5G-NIDD's row index reached 0.81 against a 0.39 baseline and was
        # waved through because 0.81 < 0.95.
        if best > base + LEAK_MARGIN:
            leaks.append((data.feature_names[j], round(accs[1], 3), round(accs[8], 3)))
    leaks.sort(key=lambda t: -max(t[1], t[2]))

    # Session fingerprints, added 2026-08-09. A feature can be observable at prediction time and
    # still be an identifier: CIC's `IAT` holds values around 8.4e7 -- 84 seconds, implausible as
    # an inter-arrival time and perfect as a capture-clock offset -- and every modal value is
    # 100% pure to one class, because each attack was captured in its own session.
    #
    # Held-out accuracy alone cannot separate that from genuine signal, so test the mechanism
    # instead: build an EXACT-VALUE lookup table on a train half and apply it to a test half. A
    # smooth predictive feature generalises poorly this way because held-out rows carry unseen
    # values; an identifier generalises almost perfectly, because the same session values recur.
    half = len(sub) // 2
    tr, te = sub[:half], sub[half:]
    fingerprints = []
    for j in range(d):
        if stds[j] == 0:
            continue
        table: dict[float, int] = {}
        vals_tr = Xs[tr][:, j]
        for v in np.unique(vals_tr):
            lbl = ys[tr][vals_tr == v]
            table[float(v)] = int(np.bincount(lbl).argmax())
        vals_te = Xs[te][:, j]
        seen = np.array([float(v) in table for v in vals_te])
        if seen.sum() < 100:
            continue
        pred = np.array([table[float(v)] for v in vals_te[seen]])
        acc = float((pred == ys[te][seen]).mean())
        coverage = float(seen.mean())
        if coverage > 0.5 and acc >= FINGERPRINT_ACC:
            fingerprints.append((data.feature_names[j], round(coverage, 3), round(acc, 3),
                                 int(len(table))))
    fingerprints.sort(key=lambda t: -t[2])
    print(f"majority-baseline acc={base:.3f}")
    print("single-feature LEAKAGE suspects, (name, depth-1, depth-8) held-out, "
          f"flagged above base+{LEAK_MARGIN} = {base + LEAK_MARGIN:.3f}: {leaks[:12]}")
    # Two different verdicts, and conflating them is how a real feature gets thrown away.
    # A row index is never a flow feature: drop it. A strongly predictive genuine feature --
    # a flood's byte count, say -- is signal, and belongs in the model.
    print("session-fingerprint suspects, (name, value-coverage, exact-lookup acc, cardinality): "
          f"{fingerprints[:8]}")
    if fingerprints:
        print("  ACTION (drop): an exact-value lookup that generalises to held-out rows means "
              "the feature identifies the capture session, not the flow.")
    if index_like:
        print(f"  ACTION (drop): {index_like} are row counters, not flow features. Put them in "
              "the dataset config's drop_cols and re-run everything computed on this dataset.")
    strong = [f for f, _, _ in leaks if f not in index_like]
    if strong:
        print(f"  ACTION (investigate): {strong} beat the baseline alone. That is leakage only "
              "if the feature could not be observed at prediction time; otherwise it is signal.")
    print("=" * (14 + len(name)))

    out = REPO / "results" / "_logs" / f"audit_{name}.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"dataset": name, "rows": int(n), "features": int(d),
         "stored_rows_pre_dedup": int(n_raw), "unique_rows": int(n_uniq),
         "duplicate_fraction": round(float(dup_frac), 6),
         "class_counts": {str(names[v]): int(c) for v, c in zip(vals, counts)},
         "constant_features": const, "identifier_like": ident,
         "row_index_like": index_like,
         "session_fingerprints": [{"feature": f, "value_coverage": c, "lookup_acc": a,
                                   "cardinality": k} for f, c, a, k in fingerprints],
         "leakage_suspects": [{"feature": f, "depth1_heldout": a1, "depth8_heldout": a8}
                              for f, a1, a8 in leaks],
         "majority_baseline": round(float(base), 4)},
        indent=1))
    print(f"saved {out}")


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="ciciov2024")
    ap.add_argument("--all", action="store_true",
                    help="audit every configured dataset, including the excluded one")
    args = ap.parse_args()
    names = (["fiveg_nidd", "ciciot2023", "ciciomt2024", "ciciov2024"]
             if args.all else [args.dataset])
    for nm in names:
        audit(nm, {} if nm == "synthetic" else load_config(f"datasets/{nm}"))
