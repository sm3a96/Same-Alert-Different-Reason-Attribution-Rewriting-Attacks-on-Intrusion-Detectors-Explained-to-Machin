"""Does showing a smoothed attribution blunt the displacement attack, and at what cost?

C3 concludes that no runtime signal can be fielded against A1 displacement, and explains why: the
attacked explanation is faithful, so there is no integrity violation for a monitor to find. That
diagnosis points somewhere specific. If the damage comes from pointwise sensitivity of the
attribution map, the fix is not to detect it but to stop displaying a pointwise quantity.

This measures that, in the unit the harm actually travels through. The C2 analysis
(`scripts/review_2026-08-09/c2_harm_mediator.py`) found that decision harm tracks the CHANGE in
whether the top-ranked feature is causal -- not the level, and not top-k set corruption. So the
defense is scored on exactly that mediator, plus the corruption metric for continuity:

  P(top-1 causal)   does the shown ranking still put a genuinely causal feature first
  corrupt           top-k Jaccard distance, clean vs attacked, under each presentation

Four arms, and the fourth is the one that decides whether this is a defense or a speed bump:

  pointwise / clean        the status quo, unattacked            -- the fidelity reference
  pointwise / attacked     the status quo under attack           -- the harm to be reduced
  smoothed  / clean        the defense, unattacked               -- what the defense COSTS
  smoothed  / attacked     the defense under the same attack     -- what it BUYS
  smoothed  / adaptive x2  the defense under an attacker who optimises against IT, at equal
                           candidate flows AND at equal explainer queries

Reporting the third arm is not optional. A smoothed attribution is less faithful pointwise than
the exact one it replaces, so a defense that removes the attack by making the explanation
uninformative has bought nothing. The pair (cost, benefit) is the result; either number alone is
marketing.

The adaptive arm exists because a defense evaluated only against the attack it was designed to
stop is not evaluated. It runs the same search with the smoothed attributor in the loop, so the
adversary optimises the quantity the defender displays. Expect it to recover some of the damage;
how much is the honest measure of what smoothing is worth.

  python scripts/run_smoothed_defense.py --datasets fiveg_nidd --seeds 0 1 2
  python scripts/run_smoothed_defense.py --mode frequency --smooth-n 30

Writes results/smoothed_defense/{raw,summary}/ and a MANIFEST row.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from avert.attribution.real import TreeSHAPAttributor
from avert.attribution.smoothed import SmoothedAttributor
from avert.benchmark.attacks import DisplacementAttack
from avert.benchmark.ground_truth import causal_features
from avert.benchmark.perturbation import bounds_for
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.data.grouping import grouped_split
from avert.detectors.real import XGBoostDetector
from avert.eval.runner import RunResult, run_experiment

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
log = logging.getLogger("smoothed_defense")


def jaccard_distance(a: set, b: set) -> float:
    return 1.0 - (len(a & b) / len(a | b) if (a | b) else 1.0)


def shown(attributor, detector, x, names, cls, sid, k):
    e = attributor.explain(detector, x, names, cls, sid, top_k=k)
    return set(e.top_features())


def run_dataset(dataset, seeds, n_flows, top_k, n_train, smooth_n, sigma, mode, adaptive_iter):
    data = load_dataset(dataset, load_config(f"datasets/{dataset}"))
    X, y, names = data.X, data.y, data.feature_names
    rows = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Session-aware split, same protocol as the matrix. Using rng.permutation here would
        # have left this experiment on the random split every other result moved off, and the
        # comparison across experiments would silently mix two protocols.
        split = grouped_split(data.grouping, seed=seed)
        tr = (np.sort(rng.choice(split.train, n_train, replace=False))
              if len(split.train) > n_train else split.train)
        eval_pool = split.test
        det = XGBoostDetector(n_estimators=150).fit(X[tr], y[tr])
        exact = TreeSHAPAttributor()
        bounds = bounds_for(X[tr], names, dataset)
        std = X[tr].std(axis=0)

        smooth = SmoothedAttributor(exact, std, n=smooth_n, sigma=sigma, mode=mode,
                                    free_idx=bounds.free_idx, seed=seed)

        bi = data.benign_index()
        benign_ref = X[y == bi].mean(axis=0) if bi is not None else X[tr].mean(axis=0)
        classes, counts = np.unique(y, return_counts=True)
        target = max([c for c in classes if c != bi],
                     key=lambda c: counts[list(classes).index(c)])
        pool = eval_pool[y[eval_pool] == target]      # held-out captures only
        rng.shuffle(pool)
        chosen = [i for i in pool[:4000]
                  if int(det.predict(X[i].reshape(1, -1))[0]) == target][:n_flows]

        # The attack the paper reports: optimises against the exact attributor.
        atk = DisplacementAttack(exact, std, bounds, eps=0.15, n_iter=100, restarts=3,
                              top_k=top_k, seed=seed)
        # The adaptive attack: same search, but the objective is the DEFENDER's statistic.
        #
        # Budget parity matters and the first run did not have it. Each adaptive iteration costs
        # `smooth_n` explainer calls against the plain attack's one, so 12 iterations x 1 restart
        # was 12 candidate flows evaluated against the plain attack's 300. Reporting that the
        # defense "survives an adaptive attacker" from a 25x budget advantage would be worthless.
        # Match on CANDIDATE FLOWS EVALUATED, which is what an adversary actually spends, and
        # record both budgets so the comparison can be audited.
        plain_candidates = 100 * 3
        # Two conventions, because there is no single fair one and the gap between them is the
        # defense's actual contribution.
        #
        #   equal candidates  the adversary evaluates as many flows as against the pointwise
        #                     attributor, and pays smooth_n times the queries to do it. Measures
        #                     what smoothing costs an adversary who does not care about cost.
        #   equal queries     the adversary spends the same number of explainer calls, so
        #                     smoothing buys a smooth_n-fold reduction in search. Measures what
        #                     smoothing is worth against a budgeted adversary, which is the
        #                     realistic one.
        atk_ad_cand = DisplacementAttack(smooth, std, bounds, eps=0.15,
                                      n_iter=plain_candidates // 3, restarts=3,
                                      top_k=top_k, seed=seed)
        atk_ad_query = DisplacementAttack(smooth, std, bounds, eps=0.15,
                                       n_iter=max(1, plain_candidates // (3 * smooth_n)),
                                       restarts=3, top_k=top_k, seed=seed)

        from avert.types import FlowSample
        for i in chosen:
            s = FlowSample(features=X[i].copy(), feature_names=names, dataset=dataset,
                           true_label=int(y[i]), sample_id=f"s{i}")
            # 0.05/3 matches every other result call site. The bare-default call this
            # replaces used 0.1/1, so the smoothed_defense artifact on disk predates this
            # line — re-run before quoting any number from it.
            causal = set(int(j) for j in causal_features(s, det, benign_ref, 0.05, 3)[0])
            x_att = atk.apply(s, det).features
            x_ada_c = atk_ad_cand.apply(s, det).features
            x_ada_q = atk_ad_query.apply(s, det).features

            for arm, attributor, xv in (
                    ("pointwise_clean", exact, X[i]),
                    ("pointwise_attacked", exact, x_att),
                    ("smoothed_clean", smooth, X[i]),
                    ("smoothed_attacked", smooth, x_att),
                    ("adaptive_equal_candidates", smooth, x_ada_c),
                    ("adaptive_equal_queries", smooth, x_ada_q)):
                t = shown(attributor, det, xv, names, target, f"s{i}", top_k)
                ranked = attributor.explain(det, xv, names, target, f"s{i}", top_k=top_k)
                rows.append({
                    "dataset": dataset, "seed": seed, "class": int(target),
                    "sample_id": f"s{i}", "arm": arm,
                    "top1_causal": bool(int(ranked.ranking()[0]) in causal),
                    "any_causal_in_topk": bool(t & causal),
                    "n_causal": len(causal),
                    "attack_candidates": {
                        "adaptive_equal_candidates": atk_ad_cand.n_iter * atk_ad_cand.restarts,
                        "adaptive_equal_queries": atk_ad_query.n_iter * atk_ad_query.restarts,
                    }.get(arm, plain_candidates),
                    "attack_explainer_calls": {
                        "adaptive_equal_candidates":
                            atk_ad_cand.n_iter * atk_ad_cand.restarts * smooth_n,
                        "adaptive_equal_queries":
                            atk_ad_query.n_iter * atk_ad_query.restarts * smooth_n,
                    }.get(arm, plain_candidates),
                    "prediction_preserved": bool(
                        int(det.predict(np.asarray(xv).reshape(1, -1))[0]) == target),
                })
        log.info("%s seed %d: %d flows x 5 arms", dataset, seed, len(chosen))
    return rows


def summarise(rows):
    """Cluster by seed. Flows inside a seed share one detector and one attack instantiation."""
    out = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for arm in ("pointwise_clean", "pointwise_attacked", "smoothed_clean",
                    "smoothed_attacked", "adaptive_equal_candidates",
                    "adaptive_equal_queries"):
            per_seed = []
            for seed in sorted({r["seed"] for r in rows if r["dataset"] == dataset}):
                sel = [r for r in rows if r["dataset"] == dataset and r["arm"] == arm
                       and r["seed"] == seed]
                if sel:
                    per_seed.append(float(np.mean([r["top1_causal"] for r in sel])))
            if not per_seed:
                continue
            n = len(per_seed)
            half = (1.96 * np.std(per_seed, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            out.append({"dataset": dataset, "arm": arm, "seeds": n,
                        "p_top1_causal": round(float(np.mean(per_seed)), 4),
                        "ci_half": round(float(half), 4),
                        "per_seed": [round(v, 4) for v in per_seed]})
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--flows", type=int, default=30)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=20_000)
    ap.add_argument("--smooth-n", type=int, default=30)
    ap.add_argument("--sigma", type=float, default=0.05)
    ap.add_argument("--mode", choices=("mean", "frequency"), default="mean")
    ap.add_argument("--adaptive-iter", type=int, default=15)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    def _run(ctx):
        rows = []
        for ds in a.datasets:
            rows += run_dataset(ds, a.seeds, a.flows, a.top_k, a.n_train,
                                a.smooth_n, a.sigma, a.mode, a.adaptive_iter)
        summary = summarise(rows)
        for s in summary:
            log.info("%-12s %-26s P(top-1 causal) %.3f +/- %.3f   per-seed %s",
                     s["dataset"], s["arm"], s["p_top1_causal"], s["ci_half"], s["per_seed"])
        with open(ctx.run_dir / "raw" / "defense.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (ctx.run_dir / "summary" / "defense.json").write_text(json.dumps(summary, indent=1))
        return RunResult(summary={"by_arm": summary, "n_rows": len(rows)})

    run_experiment("smoothed_defense", _run,
                   config={"datasets": a.datasets, "seeds": a.seeds, "flows": a.flows,
                           "top_k": a.top_k, "n_train": a.n_train, "smooth_n": a.smooth_n,
                           "sigma": a.sigma, "mode": a.mode,
                           "adaptive_iter": a.adaptive_iter, "attack_eps": 0.15},
                   seed=a.seeds[0])
    log.info("saved %s", REPO / "results" / "smoothed_defense")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
