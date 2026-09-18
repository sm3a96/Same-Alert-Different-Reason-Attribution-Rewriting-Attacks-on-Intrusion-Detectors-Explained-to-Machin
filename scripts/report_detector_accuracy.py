"""Test accuracy of the tree detector in every matrix cell.

The paper reports harm and detectability on detectors it never scores as classifiers, and a
referee asked the obvious question. This rebuilds the detector of each of the 45 matrix cells
exactly as `XIntBench._prepare` does (same seed, same grouped split, same 60,000-row training
draw, same 150 trees), scores it on the whole test side of that grouped split, and reports
the per-corpus mean over seeds. Classes inside a seed share one fitted detector, so the three
cells of a seed carry the same number; the mean over cells equals the mean over seeds.

Writes results/detector_accuracy/summary/accuracy.json and a MANIFEST row.

  python scripts/report_detector_accuracy.py
"""
from __future__ import annotations

import json
import logging
from collections import defaultdict

import numpy as np

from avert.benchmark.harness import XIntBench, target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.data.grouping import grouped_split, grouping_for
from avert.eval.runner import RunResult, run_experiment

DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
SEEDS = [0, 1, 2, 3, 4]
N_CLASSES, N_CAL, N_TEST = 3, 100, 40
log = logging.getLogger("detector_accuracy")


def run(ctx) -> RunResult:
    rows = []
    for ds in DATASETS:
        data = load_dataset(ds, load_config(f"datasets/{ds}"))
        classes, _ = target_classes(data, N_CLASSES, n_cal=N_CAL, n_test=N_TEST, seed=0,
                                    min_count=2000, seeds=SEEDS)
        grouping = data.grouping or grouping_for(data.name, len(data.X), None)
        for c in classes:
            cname = data.label_names[c] if data.label_names else str(c)
            for seed in SEEDS:
                bench = XIntBench(data, n_cal=N_CAL, n_test=N_TEST, top_k=5, seed=seed)
                st = bench._prepare(target=c)
                test = grouped_split(grouping, seed=seed).test
                pred = st.detector.predict(data.X[test])
                acc = float((pred == data.y[test]).mean())
                rows.append({"dataset": ds, "class": cname, "seed": seed,
                             "n_test_side": int(len(test)), "accuracy": acc})
                log.info("%s class=%s seed=%d test-side rows=%d accuracy=%.4f",
                         ds, cname, seed, len(test), acc)
    per_seed = defaultdict(dict)
    for r in rows:
        per_seed[r["dataset"]][r["seed"]] = r["accuracy"]   # identical across classes of a seed
    summary = {ds: {"mean_over_seeds": float(np.mean(list(v.values()))),
                    "per_seed": {str(k): v[k] for k in sorted(v)}}
               for ds, v in per_seed.items()}
    (ctx.run_dir / "raw" / "cells.json").write_text(json.dumps(rows, indent=1))
    (ctx.run_dir / "summary" / "accuracy.json").write_text(json.dumps(summary, indent=1))
    return RunResult(summary={ds: s["mean_over_seeds"] for ds, s in summary.items()},
                     floats=[{"float_id": "detector_accuracy", "kind": "table", "paper_section": "C1",
                              "path": str(ctx.run_dir / "summary" / "accuracy.json"),
                              "claim": "test accuracy of the tree detector per corpus, mean over seeds"}])


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    run_experiment("detector_accuracy", run, config={"datasets": DATASETS, "seeds": SEEDS,
                                                     "n_classes": N_CLASSES}, seed=0)
