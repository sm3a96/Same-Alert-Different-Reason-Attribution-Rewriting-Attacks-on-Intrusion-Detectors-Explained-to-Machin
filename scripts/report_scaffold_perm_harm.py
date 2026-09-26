"""Reader harm for explainer scaffolding when the reader is shown the permutation attributor.

Both readers were run on the prediction-preserved case sets of
results/scaffolding_permutation by

    python scripts/run_decision_utility.py --cases results/scaffolding_permutation/raw/cases.json \
        --name decision_utility_scaffold_perm --conditions clean attacked --preserved-only \
        --shuffle-n 0 --sensitivity-n 0

The clean and attacked cases both carry the permutation attributor's list, so unlike the
TreeSHAP arm the clean decisions are new. An attacked decision is rescored against S(x'), the
causal set measured on the component that decided (the surrogate if the router diverted the
flow, the real model otherwise), and also reported against S(x). The arms and the split are
those of scripts/report_split_sample_harm.py: all reader--cells on every prediction-preserved
flow, and competence selected on half A and measured on half B, with cell clusters (both
readers in one cluster) and 8,000 resamples. The halves are drawn over all forty flows of a
cell, as for the TreeSHAP arms, so a cell's halves do not depend on which flows survived.

  python scripts/report_scaffold_perm_harm.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from avert.eval.decision_utility import paired_effect
from avert.eval.runner import RESULTS_ROOT, RunResult, run_experiment

log = logging.getLogger("scaffold_perm_harm")

REPO = Path(__file__).resolve().parents[1]
NAME = "decision_utility_scaffold_perm"
ATTACK = "A3_scaffolding"
CASES = RESULTS_ROOT / "scaffolding_permutation" / "raw" / "cases.json"
DECISIONS = RESULTS_ROOT / NAME / "raw" / "decisions.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sweep(ctx) -> RunResult:
    args = ctx.config["args"]
    sp = _load("report_split_sample_harm")
    cases = json.loads(CASES.read_text())
    by = {(c["dataset"], c["class"], c["seed"], c["sample_id"]): c for c in cases}
    rows = [r for r in json.loads(DECISIONS.read_text())
            if by[(r["dataset"], r["class"], r["seed"], r["sample_id"])]["prediction_preserved"]]

    shown = []
    for r in rows:
        c = by[(r["dataset"], r["class"], r["seed"], r["sample_id"])]
        rr = dict(r)
        if r["condition"] == "attacked":
            rr["det_correct"] = r["det_choice"] in set(c["S_a"])
        shown.append(rr)
    ref_rows = {"vs_S_x": rows, "vs_S_xprime": shown}

    # halves() reads dataset/class/seed/sample_id only, so the case list stands in for the
    # rebuilt instances.
    A, B = sp.halves(cases, seed=args["split_seed"])
    res = {}
    for ref, rr in ref_rows.items():
        acc_A = sp.clean_accuracy_on(rr, A)
        comp = {k for k, v in acc_A.items() if v >= args["floor"]}
        rows_B = [r for r in rr if r["sample_id"] in B.get((r["dataset"], r["class"], r["seed"]), ())]

        def pe(x, cc=None):
            return paired_effect(x, "attacked", attack=ATTACK, competent_cells=cc,
                                 cluster="cell", n_boot=args["n_boot"], seed=args["seed"])

        res[ref] = {"all_cells": pe(rr), "all_cells_on_B": pe(rows_B),
                    "competent_on_A": pe(rows_B, comp),
                    "not_competent_on_A": pe(rows_B, set(acc_A) - comp),
                    "in_sample_competent": None,
                    "per_dataset_all": {d: pe([r for r in rr if r["dataset"] == d])
                                        for d in sorted({r["dataset"] for r in rr})},
                    "n_reader_cells": len(acc_A), "n_competent_on_A": len(comp)}
        # The in-sample floor: clean accuracy on every judged flow of the reader--cell, which
        # here means its prediction-preserved flows, the only ones the readers were run on.
        acc_all = defaultdict(list)
        for r in rr:
            if r["condition"] == "clean":
                acc_all[sp.reader_cell(r)].append(int(r["det_correct"]))
        comp_all = {k for k, v in acc_all.items() if np.mean(v) >= args["floor"]}
        res[ref]["in_sample_competent"] = pe(rr, comp_all)

    kept = [c for c in cases if c["prediction_preserved"]]
    routed = float(np.mean([c["routed_to_surrogate"] for c in kept])) if kept else float("nan")
    clean_acc = float(np.mean([r["det_correct"] for r in rows if r["condition"] == "clean"]))
    p_first = {j: float(np.mean([r["det_choice_idx"] == 0 for r in rows
                                 if r["condition"] == "clean" and r["judge"] == j]))
               for j in sorted({r["judge"] for r in rows})}
    summary = {"effects": res, "case_sets_preserved": len(kept),
               "routed_rate_preserved": routed, "clean_accuracy": clean_acc,
               "p_first_clean": p_first}
    (ctx.run_dir / "summary" / "harm.json").write_text(json.dumps(summary, indent=1))
    L = ["# Scaffolding, permutation attributor shown: reader harm", "",
         f"Prediction-preserved case sets: {len(kept)}; routed to the surrogate: {routed:.3f}; "
         f"clean accuracy {clean_acc:.3f}; clean first-candidate rate {p_first}.", "",
         "| Reference | Population | n | effect | 95% CI | broke/fixed |", "|---|---|---|---|---|---|"]
    for ref, per in res.items():
        for pop in ("all_cells", "all_cells_on_B", "competent_on_A", "not_competent_on_A",
                    "in_sample_competent"):
            e = per[pop]
            if e.get("n"):
                L.append(f"| {ref} | {pop} | {e['n']} | {e['mean_delta']:+.3f} | "
                         f"[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] | {e['broke']}/{e['fixed']} |")
    md = "\n".join(L)
    (ctx.run_dir / "summary" / "harm.md").write_text(md)
    print(md)
    e = res["vs_S_xprime"]["all_cells"]
    return RunResult(
        summary={"case_sets_preserved": len(kept),
                 "all_cells_vs_Sxprime": e.get("mean_delta"),
                 "ci": [e.get("ci_lo"), e.get("ci_hi")]},
        floats=[{"float_id": "Tab_harm_scaffold_permutation", "kind": "table",
                 "paper_section": "harm", "path": str(ctx.run_dir / "summary" / "harm.json"),
                 "claim": "reader harm under scaffolding when the reader sees the permutation "
                          "attributor"}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(NAME, sweep, config={"args": {
        "floor": a.floor, "split_seed": a.split_seed, "n_boot": a.n_boot, "seed": a.seed}},
        seed=a.seed, meta_name="rescore_seed0.json")


if __name__ == "__main__":
    main()
