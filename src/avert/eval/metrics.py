"""Evaluation metrics (maps to the Plan evaluation matrix, Section 6)."""
from __future__ import annotations

import numpy as np


def detection_rate(violations: np.ndarray) -> float:
    """Fraction of attacked instances flagged as violations."""
    return float(np.mean(violations))


def empirical_false_alarm(clean_violations: np.ndarray) -> float:
    """Violation rate on CLEAN held-out data; must sit within CI of nominal alpha."""
    return float(np.mean(clean_violations))


def topk_rank_agreement(a: list[int], b: list[int]) -> float:
    """Top-k set overlap (Jaccard). Phase 2 swaps for weighted Kendall on the top-k."""
    sa, sb = set(a), set(b)
    return len(sa & sb) / len(sa | sb) if (sa | sb) else 1.0


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    """AUROC of `scores` separating labels (1=positive). Used for drift-vs-A4."""
    from sklearn.metrics import roc_auc_score

    if len(np.unique(labels)) < 2:
        return float("nan")
    return float(roc_auc_score(labels, scores))


def mean_ci(values: np.ndarray, ci: float = 0.95) -> tuple[float, float]:
    """Mean and half-width of a normal-approx CI over seeds (report as mean +/- hw)."""
    v = np.asarray(values, dtype=float)
    if len(v) < 2:
        return float(v.mean()) if len(v) else float("nan"), 0.0
    from scipy import stats

    sem = stats.sem(v)
    hw = sem * stats.t.ppf(0.5 + ci / 2, len(v) - 1)
    return float(v.mean()), float(hw)


def seed_clustered_ci(values, seeds, ci: float = 0.95) -> tuple[float, float]:
    """Mean and CI half-width when observations are clustered by seed.

    The matrix runs three target classes per seed, and every class in a seed shares one
    bit-identical fitted detector — `_prepare` draws the training split from the seed
    before the target is ever consulted. So nine cells are not nine independent
    observations; they are three, each measured three ways.

    A t-interval over all nine understates the width by up to 2.3x on this data. This
    averages within seed first and puts the interval over the seed means, which is the
    standard cluster-robust move when the number of clusters is too small to bootstrap
    (three to five here). Degrees of freedom are the number of seeds minus one, and that
    honesty is the point: with three seeds the interval is wide because the evidence is
    thin, and printing a narrow one does not make it thicker.

    This is the same error class as the C2 paired interval, which was caught and fixed
    first; it was left live in the matrix for a day.
    """
    v = np.asarray(values, dtype=float)
    s = np.asarray(seeds)
    means = np.array([v[s == k].mean() for k in np.unique(s)])
    if len(means) < 2:
        return float(means.mean()) if len(means) else float("nan"), 0.0
    from scipy import stats

    hw = float(stats.sem(means) * stats.t.ppf(0.5 + ci / 2, len(means) - 1))
    return float(means.mean()), hw


def best_single_auroc(rows: list[dict]) -> dict:
    """Best SINGLE check per (dataset, attack): the check with the highest seed-clustered
    mean AUROC over that group's cells, fused row excluded.

    The check is chosen ONCE per group on its mean, never per cell. A per-cell maximum over
    seven correlated channels reached 0.72-0.73 against cause displacement on the September
    2026 matrix while no single check averaged above 0.67 -- the gap is selection bias, and a
    number no deployable check achieves must not be quoted as detectability. `AttackResult.
    monitor_auroc` (the `auroc` field in raw.json) remains the per-cell maximum including
    FUSED, as the harness has always written it, and is not what the paper quotes.

    Returns {(dataset, attack): {"signal": name, "mean": m, "ci_half": hw,
    "per_cell": [...], "seeds": [...]}} so callers can plot the chosen check's own interval.
    """
    from collections import defaultdict

    grp = defaultdict(lambda: defaultdict(list))
    seeds = defaultdict(list)
    for r in rows:
        k = (r["dataset"], r["attack"])
        seeds[k].append(r["seed"])
        for sig, v in r["per_signal"].items():
            if sig != "FUSED":
                grp[k][sig].append(float(v))
    out = {}
    for k, per in grp.items():
        best = max(per, key=lambda sig: seed_clustered_ci(np.array(per[sig]), np.array(seeds[k]))[0])
        mu, hw = seed_clustered_ci(np.array(per[best]), np.array(seeds[k]))
        out[k] = {"signal": best, "mean": mu, "ci_half": hw, "per_cell": per[best], "seeds": seeds[k]}
    return out
