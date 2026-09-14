"""Can a classifier tell train rows from test rows? If so, the split leaks.

The column audit finds leakage it can name -- a row counter, a capture clock. It cannot find
leakage it has no test for, and there is always another correlate of capture session waiting.
Adversarial validation closes that gap by attacking the split directly: label every training row
0 and every test row 1, fit a model, and measure how well it separates them on held-out data.

The first measurement corrected the premise, and the correction is the useful part.

On a RANDOM split the adversarial AUC is 0.501 -- and it always will be, because shuffling rows
makes the two sides exchangeable by construction. Adversarial validation therefore cannot detect
this class of leakage on a random split: the exchangeability IS the leakage. Reporting "our
adversarial check passed" over a random split would be a guarantee about nothing, and that is
worth knowing before anyone puts it in a paper.

On a GROUPED split the AUC rises -- 0.677 on 5G-NIDD -- and that is the desired outcome, not a
failure. It says train and test come from genuinely different captures, which is the deployment
condition a NIDS actually faces. A grouped split whose adversarial AUC were 0.5 would mean the
grouping had not separated anything.

So the diagnostic is not the AUC, it is the GAP IN DETECTOR ACCURACY between the two splits:

  random    the split this project used until 2026-08-09 -- rows shuffled, sessions straddling
  grouped   whole capture sessions on one side only (avert.data.grouping)

Detector accuracy 0.917 under the random split and 0.860 under the grouped one, on 5G-NIDD,
means 5.7 accuracy points were session structure rather than detection. That number is what the
paper owes its readers, and reporting both arms is what makes every downstream result honest.
The adversarial AUC is reported alongside as a property of the benchmark -- how far apart the
captures are -- not as a pass mark.

  python scripts/check_split_leakage.py
  python scripts/check_split_leakage.py --datasets fiveg_nidd --rows 120000

Writes results/_logs/split_leakage.json. `make gate` runs it and fails if a grouping turns out
not to separate captures at all, which would mean it is not buying what it claims.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score
from sklearn.model_selection import train_test_split
from sklearn.tree import DecisionTreeClassifier

from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.data.grouping import grouped_split
from avert.eval.runner import RunResult, run_experiment

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]

# A grouped split SHOULD separate: if the adversarial AUC came back at chance, the grouping
# would not have separated captures at all. This floor flags that failure, not its opposite.
MIN_SESSION_SHIFT = 0.55
log = logging.getLogger("split_leakage")


def adversarial_auc(X, train_idx, test_idx, rng, max_rows=60_000, depth=8) -> tuple[float, list]:
    """AUC of a classifier trained to answer 'is this row from the test side?'.

    Scored on a held-out half of the adversarial task itself, never on its own training rows --
    the mistake the original leakage check made, which turned its threshold into a fiction.
    """
    n = min(max_rows // 2, len(train_idx), len(test_idx))
    if n < 200:
        return float("nan"), []
    a = rng.choice(train_idx, n, replace=False)
    b = rng.choice(test_idx, n, replace=False)
    Xa = np.vstack([X[a], X[b]])
    ya = np.r_[np.zeros(n), np.ones(n)]
    Xtr, Xte, ytr, yte = train_test_split(Xa, ya, test_size=0.5, random_state=0, stratify=ya)
    clf = DecisionTreeClassifier(max_depth=depth, random_state=0).fit(Xtr, ytr)
    auc = float(roc_auc_score(yte, clf.predict_proba(Xte)[:, 1]))
    return auc, list(clf.feature_importances_)


def detector_accuracy(X, y, train_idx, test_idx, rng, max_rows=60_000) -> float:
    """Held-out accuracy of the actual detector family, under this split. The number the paper
    would report, so the random-vs-grouped gap is the number the paper was over-reporting by."""
    from avert.detectors.real import XGBoostDetector

    tr = rng.choice(train_idx, min(max_rows, len(train_idx)), replace=False)
    te = rng.choice(test_idx, min(max_rows // 3, len(test_idx)), replace=False)
    det = XGBoostDetector(n_estimators=100).fit(X[tr], y[tr])
    return float((det.predict(X[te]) == y[te]).mean())


def run_dataset(dataset: str, rows: int, seed: int) -> dict:
    data = load_dataset(dataset, load_config(f"datasets/{dataset}"))
    X, y, g = data.X, data.y, data.grouping
    rng = np.random.default_rng(seed)

    if rows and rows < len(X):
        keep = np.sort(rng.choice(len(X), rows, replace=False))
        X, y = X[keep], y[keep]
        from avert.data.grouping import Grouping
        g = Grouping(g.key[keep], g.kind, g.names,
                     None if g.provider_split is None else g.provider_split[keep])

    out = {"dataset": dataset, "rows": int(len(X)), "grouping": g.describe(),
           "grouping_kind": g.kind, "n_groups": g.n_groups, "arms": {}}

    # Random split: what this project did until 2026-08-09.
    perm = rng.permutation(len(X))
    cut = int(0.8 * len(X))
    arms = {"random": (perm[:cut], perm[cut:])}

    gs = grouped_split(g, seed=seed)
    arms["grouped"] = (np.r_[gs.train, gs.calibration], gs.test)
    out["grouped_split_groups"] = gs.n_groups_used

    for arm, (tr, te) in arms.items():
        auc, imp = adversarial_auc(X, tr, te, rng)
        acc = detector_accuracy(X, y, tr, te, rng)
        worst = (sorted(zip(data.feature_names, imp), key=lambda t: -t[1])[:3]
                 if imp else [])
        out["arms"][arm] = {
            "n_train": int(len(tr)), "n_test": int(len(te)),
            "adversarial_auc": round(auc, 4),
            "detector_accuracy": round(acc, 4),
            "most_separating_features": [[n, round(float(v), 4)] for n, v in worst],
        }
        log.info("%-12s %-8s adversarial AUC %.3f | detector acc %.3f | top separators %s",
                 dataset, arm, auc, acc, [n for n, _ in worst])

    r, gr = out["arms"]["random"], out["arms"]["grouped"]
    # The headline: how much of the random-split accuracy was session structure.
    out["accuracy_inflation"] = round(r["detector_accuracy"] - gr["detector_accuracy"], 4)
    out["session_shift_auc"] = gr["adversarial_auc"]
    out["grouping_separated"] = bool(gr["adversarial_auc"] >= MIN_SESSION_SHIFT)
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reports = {}

    def _run(ctx):
        for ds in a.datasets:
            reports[ds] = run_dataset(ds, a.rows, a.seed)
        (ctx.run_dir / "summary" / "split_leakage.json").write_text(
            json.dumps(reports, indent=1))
        return RunResult(summary=reports)

    run_experiment("split_leakage", _run,
                   config={"datasets": a.datasets, "rows": a.rows, "seed": a.seed,
                           "min_session_shift": MIN_SESSION_SHIFT}, seed=a.seed)

    out = REPO / "results" / "_logs" / "split_leakage.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(reports, indent=1))
    log.info("saved %s", out)

    log.info("")
    log.info("%-12s  %-28s  %-28s  %s", "dataset", "detector accuracy", "adversarial AUC",
             "inflation")
    for ds, r in reports.items():
        log.info("%-12s  random %.3f -> grouped %.3f   random %.3f -> grouped %.3f   %+.3f%s",
                 ds, r["arms"]["random"]["detector_accuracy"],
                 r["arms"]["grouped"]["detector_accuracy"],
                 r["arms"]["random"]["adversarial_auc"],
                 r["arms"]["grouped"]["adversarial_auc"],
                 r["accuracy_inflation"],
                 "" if r["grouping_separated"] else "   [grouping did not separate captures]")
    log.info("")
    log.info("Inflation is accuracy the random split was harvesting from capture structure. "
             "The adversarial AUC on the random arm is 0.5 by construction and proves nothing; "
             "on the grouped arm it should exceed %.2f, or the grouping is not doing its job.",
             MIN_SESSION_SHIFT)

    flat = [ds for ds, r in reports.items() if not r["grouping_separated"]]
    if flat:
        log.error("grouped split failed to separate captures on: %s -- the grouping key is too "
                  "coarse to be worth using there", flat)
        return 1
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
