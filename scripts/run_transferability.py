"""Does the displacement attack need query access to the exact deployed model?

The threat model gives A1 query access to the detector `f` and the attributor `g`. A SOC's NIDS
is not a public API, so the honest question a reviewer asks is what the attack is worth without
that access. Nothing in this project had measured it: `transferability` and `adaptive` appear
nowhere in the plan, the notes or the docs.

The experiment is nearly free, because the ingredients already exist. Craft the attack against
one detector and evaluate the resulting FLOW against a different one:

  white-box     attack seed s, score the explanation of seed s          (the paper's setting)
  cross-seed    attack seed s, score the explanation of seed s'         same architecture,
                                                                        different training draw
  cross-arch    attack the XGBoost pipeline, score MLP + integrated gradients

The attacked flow is fixed once and handed to each target unchanged -- that is what an adversary
with a surrogate actually has. Two things are reported per target, and they answer different
questions:

  corrupt          top-k Jaccard distance between the target's clean and attacked explanations.
                   How much of what the TARGET would show has moved.
  valid            whether the TARGET's prediction survives. An attack that flips the transfer
                   target's prediction is caught by ordinary model monitoring and is not an
                   explanation attack against that deployment at all.

Either outcome is publishable and they say opposite things about the threat. High transfer means
the adversary needs only a surrogate, and the attack is a realistic threat to a deployed NIDS.
Low transfer means the attack is white-box in practice, which narrows C1 and C2 to an adversary
with query access -- an insider, or a vendor-distributed model the attacker also holds. The
paper has to say which, and until now it could not.

  python scripts/run_transferability.py --dataset fiveg_nidd --seeds 0 1 2
  python scripts/run_transferability.py --datasets fiveg_nidd ciciot2023 ciciomt2024

Writes results/transferability/{raw,summary}/ and a MANIFEST row.
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from pathlib import Path

import numpy as np

from avert.attribution.real import IntegratedGradientsAttributor, TreeSHAPAttributor
from avert.benchmark.attacks import DisplacementAttack
from avert.benchmark.perturbation import bounds_for
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.data.grouping import grouped_split
from avert.detectors.real import MLPDetector, XGBoostDetector
from avert.eval.runner import RunResult, run_experiment

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
log = logging.getLogger("transferability")


def topk(detector, attributor, x, names, k):
    cls = int(detector.predict(x.reshape(1, -1))[0])
    e = attributor.explain(detector, x, names, cls, "x", top_k=k)
    return cls, set(e.top_features())


def jaccard_distance(a: set, b: set) -> float:
    return 1.0 - (len(a & b) / len(a | b) if (a | b) else 1.0)


def run_dataset(dataset: str, seeds: list[int], n_flows: int, top_k: int,
                n_train: int) -> list[dict]:
    cfg = load_config(f"datasets/{dataset}")
    data = load_dataset(dataset, cfg)
    X, y, names = data.X, data.y, data.feature_names
    rows: list[dict] = []

    for seed in seeds:
        rng = np.random.default_rng(seed)
        # Session-aware split, same protocol as the matrix. Using rng.permutation here would
        # have left this experiment on the random split every other result moved off, and the
        # comparison across experiments would silently mix two protocols.
        split = grouped_split(data.grouping, seed=seed)
        tr = (np.sort(rng.choice(split.train, n_train, replace=False))
              if len(split.train) > n_train else split.train)
        eval_pool = split.test

        # Source pipeline: the one the adversary can query.
        src = XGBoostDetector(n_estimators=150).fit(X[tr], y[tr])
        shap = TreeSHAPAttributor()
        bounds = bounds_for(X[tr], names, dataset)

        # Cross-seed target: same architecture and hyperparameters, different training draw
        # FROM THE SAME GROUPED TRAINING SIDE. Until 2026-09-07 this drew a random permutation
        # of the whole corpus, so the "other seed" trained on rows from the held-out captures
        # the attack is evaluated on, and the cross-seed transfer number was optimistic. The
        # eval pool is untouched either way; only the surrogate's training rows change.
        rng2 = np.random.default_rng(seed + 1000)
        tr2 = (np.sort(rng2.choice(split.train, n_train, replace=False))
               if len(split.train) > n_train else split.train)
        alt = XGBoostDetector(n_estimators=150).fit(X[tr2], y[tr2])

        # Cross-architecture target: differentiable model, gradient attributor.
        mlp = MLPDetector(epochs=12, seed=seed).fit(X[tr], y[tr])
        ig = IntegratedGradientsAttributor()

        # Attack the most common non-benign class the source actually predicts correctly.
        bi = data.benign_index()
        classes, counts = np.unique(y, return_counts=True)
        target = max([c for c in classes if c != bi],
                     key=lambda c: counts[list(classes).index(c)])
        pool = eval_pool[y[eval_pool] == target]      # held-out captures only
        rng.shuffle(pool)
        chosen = [i for i in pool[:4000]
                  if int(src.predict(X[i].reshape(1, -1))[0]) == target][:n_flows]

        atk = DisplacementAttack(shap, X[tr].std(axis=0), bounds, eps=0.15,
                              n_iter=100, restarts=3, top_k=top_k, seed=seed)

        from avert.types import FlowSample
        for i in chosen:
            s = FlowSample(features=X[i].copy(), feature_names=names, dataset=dataset,
                           true_label=int(y[i]), sample_id=f"s{i}")
            xa = atk.apply(s, src).features

            for label, det, attr in (("white_box", src, shap),
                                     ("cross_seed", alt, shap),
                                     ("cross_arch", mlp, ig)):
                c0, t0 = topk(det, attr, X[i], names, top_k)
                c1, t1 = topk(det, attr, xa, names, top_k)
                rows.append({
                    "dataset": dataset, "seed": seed, "class": int(target),
                    "sample_id": f"s{i}", "target": label,
                    "corrupt": jaccard_distance(t0, t1),
                    "valid": bool(c0 == c1),
                    "residual": float(bounds.residual(xa.reshape(1, -1))[0]),
                })
        log.info("%s seed %d: %d flows attacked, 3 targets scored", dataset, seed, len(chosen))
    return rows


def summarise(rows: list[dict]) -> list[dict]:
    """Mean over SEED means. Flows inside a seed share one fitted detector and one attack
    instantiation, so pooling them treats correlated runs as independent."""
    out = []
    for dataset in sorted({r["dataset"] for r in rows}):
        for target in ("white_box", "cross_seed", "cross_arch"):
            per_seed_c, per_seed_v = [], []
            for seed in sorted({r["seed"] for r in rows if r["dataset"] == dataset}):
                sel = [r for r in rows if r["dataset"] == dataset
                       and r["target"] == target and r["seed"] == seed]
                if not sel:
                    continue
                per_seed_c.append(float(np.mean([r["corrupt"] for r in sel])))
                per_seed_v.append(float(np.mean([r["valid"] for r in sel])))
            if not per_seed_c:
                continue
            n = len(per_seed_c)
            half = (1.96 * np.std(per_seed_c, ddof=1) / np.sqrt(n)) if n > 1 else 0.0
            out.append({
                "dataset": dataset, "target": target, "seeds": n,
                "corrupt_mean": round(float(np.mean(per_seed_c)), 4),
                "corrupt_ci_half": round(float(half), 4),
                "corrupt_per_seed": [round(v, 4) for v in per_seed_c],
                "valid_mean": round(float(np.mean(per_seed_v)), 4),
            })
    return out


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--flows", type=int, default=40)
    ap.add_argument("--top-k", type=int, default=5)
    ap.add_argument("--n-train", type=int, default=20_000)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(message)s")

    def _run(ctx):
        rows: list[dict] = []
        for ds in a.datasets:
            rows += run_dataset(ds, a.seeds, a.flows, a.top_k, a.n_train)

        summary = summarise(rows)
        for s in summary:
            log.info("%-12s %-11s corrupt %.3f +/- %.3f  per-seed %s  prediction preserved %.3f",
                     s["dataset"], s["target"], s["corrupt_mean"], s["corrupt_ci_half"],
                     s["corrupt_per_seed"], s["valid_mean"])

        with open(ctx.run_dir / "raw" / "transfer.csv", "w", newline="") as fh:
            w = csv.DictWriter(fh, fieldnames=list(rows[0].keys()))
            w.writeheader()
            w.writerows(rows)
        (ctx.run_dir / "summary" / "transfer.json").write_text(json.dumps(summary, indent=1))
        return RunResult(summary={"by_target": summary, "n_rows": len(rows)})

    run_experiment(
        "transferability", _run,
        config={"datasets": a.datasets, "seeds": a.seeds, "flows": a.flows,
                "top_k": a.top_k, "n_train": a.n_train, "eps": 0.15,
                "attack": "A1_displacement", "source_pipeline": "XGBoost + TreeSHAP"},
        seed=a.seeds[0])
    out = REPO / "results" / "transferability"
    log.info("saved %s", out)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
