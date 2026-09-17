"""Paper-final EIB matrix: sweep the top attack classes x seeds per dataset and
aggregate each (dataset, attack) cell to mean +/- 95% CI. Saves incrementally to
results/matrix/ so a single failed cell never loses the run.

Routed through `eval.runner.run_experiment`, like every other result in the repo: the matrix
is the central artifact behind Tab5, Fig4 and the corruption numbers, and it used to be the
one thing written with no metadata block, no git commit and no MANIFEST row. `run_experiment`
writes into `results/matrix/` -- the same directory -- so `raw.json` does not move.

  python scripts/run_matrix.py
  python scripts/run_matrix.py --datasets fiveg_nidd --out results/matrix_check --name matrix_check

The `--out`/`--name` pair exists so a determinism re-run writes somewhere else. Re-running
into the same directory overwrites the file you are about to compare against, and then
`check_determinism.py` compares the surviving file with a copy of itself and passes for the
wrong reason -- which happened here on 2 August and is the reason for the flag.
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from avert.benchmark import XIntBench
from avert.benchmark.harness import target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.metrics import best_single_auroc, seed_clustered_ci
from avert.eval.runner import RunResult, run_experiment

log = logging.getLogger("matrix")

DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
SEEDS = [0, 1, 2, 3, 4]      # the plan promises >=5 seeds; it now delivers them
N_CLASSES = 3
OUT = Path("results/matrix")


def top_classes(data, n, min_count=2000, n_cal=100, n_test=40, seed=0, seeds=None):
    """Thin wrapper over the shared rule, kept so the call site reads locally."""
    keep, rejected = target_classes(data, n, n_cal=n_cal, n_test=n_test, seed=seed,
                                    min_count=min_count, seeds=seeds)
    if rejected:
        log.info("%s: %d class(es) excluded -- too rare inside the held-out captures",
                 data.name, len(rejected))
    return keep, rejected


def aggregate(rows):
    agg = defaultdict(lambda: defaultdict(list))
    for r in rows:
        for m in ("valid", "corrupt", "detect", "fa"):
            agg[(r["dataset"], r["attack"])][m].append(r[m])
        agg[(r["dataset"], r["attack"])]["_seed"].append(r["seed"])
    # "best AUROC" is the best single check chosen once per group on its seed-clustered
    # mean, fused row excluded -- see eval.metrics.best_single_auroc for why not per cell.
    for k, b in best_single_auroc(rows).items():
        agg[k]["auroc"] = b["per_cell"]
    lines = ["# EIB paper-final matrix (mean +/- 95% CI over SEED MEANS)", "",
             "Classes within a seed share one bit-identical fitted detector, so the interval is",
             "over seed means rather than over all class x seed cells.", "",
             "| Dataset | Attack | valid | corrupt | best AUROC | detect@a | FA | n |",
             "|---|---|---|---|---|---|---|---|"]
    order = {"A1_misdirection": 0, "A1_displacement": 1, "A3_scaffolding": 2}
    for (ds, atk), mets in sorted(agg.items(), key=lambda kv: (kv[0][0], order.get(kv[0][1], 9))):
        def f(m):
            # over seed means: classes inside a seed share one fitted detector
            mu, hw = seed_clustered_ci(np.array(mets[m]), np.array(mets["_seed"]))
            return f"{mu:.2f}±{hw:.2f}"
        lines.append(f"| {ds} | {atk} | {f('valid')} | {f('corrupt')} | {f('auroc')} | "
                     f"{f('detect')} | {f('fa')} | {len(mets['auroc'])} |")
    return "\n".join(lines)


def sweep(ctx) -> RunResult:
    rows = []
    exclusions = {}
    for ds in ctx.config["datasets"]:
        data = load_dataset(ds, load_config(f"datasets/{ds}"))
        # Every seed the run will use, so a target that dies at seed 2 is never selected.
        classes, excluded = top_classes(data, N_CLASSES, seeds=SEEDS)
        print(f"[{ds}] target classes: {[data.label_names[c] if data.label_names else c for c in classes]}")
        if excluded:
            print(f"[{ds}] excluded (too rare inside the held-out captures): "
                  f"{[(data.label_names[e['class']] if data.label_names else e['class'], e['in_calibration']) for e in excluded[:8]]}")
        exclusions[ds] = excluded
        for c in classes:
            cname = data.label_names[c] if data.label_names else str(c)
            for seed in SEEDS:
                try:
                    bench = XIntBench(data, n_cal=100, n_test=40, top_k=5,
                                      perm_samples=10, stability_n=30, seed=seed)
                    rep = bench.evaluate_pipeline(target=c)
                    for r in rep.results:
                        # per_signal_auroc travels with the row so Tab5 (per signal) and
                        # Fig4 (best signal) read the SAME cells. Two sources would drift.
                        # `auroc` is the best channel INCLUDING the fused row, as the harness
                        # has always written it; the paper's "best single check" is
                        # eval.metrics.best_single_auroc over `per_signal`, computed by every
                        # consumer rather than stored twice.
                        rows.append({"dataset": ds, "class": cname, "seed": seed, "attack": r.attack,
                                     "valid": r.prediction_preserved, "corrupt": r.explanation_corruption,
                                     "auroc": r.monitor_auroc, "detect": r.fused_detection,
                                     "fa": r.false_alarm, "per_signal": r.per_signal_auroc})
                    (OUT / "raw.json").write_text(json.dumps(rows, indent=1))
                    (OUT / "matrix_ci.md").write_text(aggregate(rows))
                    print(f"  done {ds} class={cname} seed={seed}")
                except Exception as e:
                    print(f"  FAIL {ds} class={cname} seed={seed}: {type(e).__name__}: {str(e)[:140]}")
    print("\n" + aggregate(rows))
    print(f"\nsaved: {OUT/'matrix_ci.md'} and {OUT/'raw.json'}")

    cells = {(r["dataset"], r["class"], r["seed"]) for r in rows}
    return RunResult(
        summary={"rows": len(rows), "cells": len(cells), "datasets": DATASETS,
                 "seeds": SEEDS, "classes_per_dataset": N_CLASSES,
                 "attacks": sorted({r["attack"] for r in rows})},
        floats=[
            {"float_id": "Tab5_per_signal_auroc", "kind": "table", "paper_section": "C3",
             "path": str(OUT / "raw.json"),
             "claim": "per-signal detectability across attacks and datasets"},
            {"float_id": "Fig4_efficacy_vs_detectability", "kind": "figure",
             "paper_section": "C3", "path": str(OUT / "raw.json"),
             "claim": "the attacks that do the most damage are the least detectable"},
            {"float_id": "matrix_ci", "kind": "summary", "paper_section": "C1",
             "path": str(OUT / "matrix_ci.md"),
             "claim": "prediction-preserving attacks corrupt the explanation on every dataset"},
        ])


def main():
    global OUT
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--out", default=str(OUT),
                    help="write here instead of results/matrix (use for determinism re-runs)")
    ap.add_argument("--name", default=None,
                    help="experiment name; defaults to the --out directory name so the "
                         "metadata block and the raw file can never end up in different runs")
    a = ap.parse_args()

    OUT = Path(a.out)
    name = a.name or OUT.name
    OUT.mkdir(parents=True, exist_ok=True)
    run_experiment(name, sweep,
                   config={"datasets": a.datasets, "seeds": SEEDS, "n_classes": N_CLASSES,
                           "n_cal": 100, "n_test": 40, "top_k": 5, "perm_samples": 10,
                           "stability_n": 30},
                   seed=SEEDS[0])


if __name__ == "__main__":
    main()
