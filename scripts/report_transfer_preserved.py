"""Transfer corruption on the flows whose predicted class survived.

results/transferability/summary/transfer.json averages the corruption over ALL transferred
flows, class-flipped ones included. A flow whose class flipped on the target detector is not
this paper's threat model -- the whole premise is that the alert stands -- and it is also the
flow most likely to show a large attribution change, so including it flatters the transfer
result. The paper quotes the survivor-only numbers, and until now they were recomputed inside
the claims checker rather than written by a script, which meant the release could not
regenerate them.

Reads results/transferability/raw/transfer.csv, groups by (corpus, seed), takes the mean over
the flows with valid=True, then the seed-clustered mean per corpus and the range over corpora.

  python scripts/report_transfer_preserved.py
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
from collections import defaultdict

import numpy as np

from avert.eval.metrics import seed_clustered_ci
from avert.eval.runner import RESULTS_ROOT, RunResult, run_experiment

log = logging.getLogger("transfer_preserved")

RAW = RESULTS_ROOT / "transferability" / "raw" / "transfer.csv"
OUT = RESULTS_ROOT / "transferability" / "summary" / "preserved.json"
TARGETS = ["white_box", "cross_seed", "cross_arch"]


def sweep(ctx) -> RunResult:
    rows = list(csv.DictReader(open(RAW)))
    log.info("read %d transfer rows from %s", len(rows), RAW)

    out = {}
    for target in TARGETS:
        per_cell = defaultdict(list)
        valid = defaultdict(list)
        for r in rows:
            if r["target"] != target:
                continue
            key = (r["dataset"], int(r["seed"]))
            valid[key].append(r["valid"] == "True")
            if r["valid"] == "True":
                per_cell[key].append(float(r["corrupt"]))
        per_corpus = {}
        for ds in sorted({k[0] for k in valid}):
            cells = [(s, np.mean(per_cell[(ds, s)])) for (d, s) in sorted(valid) if d == ds
                     and per_cell.get((ds, s))]
            if not cells:
                continue
            seeds = np.array([c[0] for c in cells])
            mu, hw = seed_clustered_ci(np.array([c[1] for c in cells]), seeds)
            v = [np.mean(valid[(ds, s)]) for (d, s) in sorted(valid) if d == ds]
            vm, _ = seed_clustered_ci(np.array(v), np.array(sorted({s for (d, s) in valid if d == ds})))
            per_corpus[ds] = {"corrupt_preserved": mu, "ci_lo": mu - hw, "ci_hi": mu + hw,
                              "class_survival": float(vm),
                              "n_preserved": int(sum(len(per_cell[(ds, s)])
                                                     for (d, s) in per_cell if d == ds))}
        vals = [v["corrupt_preserved"] for v in per_corpus.values()]
        surv = [v["class_survival"] for v in per_corpus.values()]
        out[target] = {"per_corpus": per_corpus,
                       "corrupt_preserved_lo": min(vals), "corrupt_preserved_hi": max(vals),
                       "class_survival_lo": min(surv), "class_survival_hi": max(surv)}
        log.info("%s: corruption %.3f-%.3f on preserved flows, class survives %.3f-%.3f",
                 target, min(vals), max(vals), min(surv), max(surv))

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(out, indent=1))
    md = ["# Transfer on prediction-preserved flows", "",
          "| Target | corruption $J$ | class survival |", "|---|---|---|"]
    for t in TARGETS:
        o = out[t]
        md.append(f"| {t} | {o['corrupt_preserved_lo']:.3f}--{o['corrupt_preserved_hi']:.3f} | "
                  f"{o['class_survival_lo']:.3f}--{o['class_survival_hi']:.3f} |")
    (ctx.run_dir / "summary" / "preserved_report.md").write_text("\n".join(md))
    print("\n".join(md))

    return RunResult(
        summary={t: {k: v for k, v in out[t].items() if k != "per_corpus"} for t in TARGETS},
        floats=[{"float_id": "transfer_preserved", "kind": "summary", "paper_section": "benchmark",
                 "path": str(OUT),
                 "claim": "transfer corruption on the flows whose predicted class survived, "
                          "which is the population the threat model defines"}])


def main():
    argparse.ArgumentParser().parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment("transferability", sweep, config={"source": str(RAW)}, seed=0)


if __name__ == "__main__":
    main()
