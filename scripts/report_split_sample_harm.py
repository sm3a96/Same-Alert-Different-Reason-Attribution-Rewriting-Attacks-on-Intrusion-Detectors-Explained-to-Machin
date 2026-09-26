"""Split-sample competence: select the competent reader-cells on one half of each cell's
flows, measure the effect on the other half.

The paper's headline is computed on reader--cells whose clean accuracy clears 0.5. That
floor is read off the same forty flows the effect is then measured on, so the selection and
the measurement share their noise: a reader--cell lands above the floor partly because its
clean decisions happened to go well on those flows, and a paired effect conditioned on an
unusually good clean arm is biased downwards by regression to the mean alone. The control
table's floor sweep does not answer this -- every floor in it is applied to the same flows.

This does answer it. Each cell's forty flows are split once, by a fixed seeded permutation,
into a selection half A and a measurement half B. Competence is clean accuracy on A. The
paired effect is computed on B, where nothing was selected. The split is the same flows for
both readers and all three attacks, so a cell is selected and measured on disjoint evidence
and the two readers of a cell can still be resampled together.

Three arms are reported, each against both references:

  all 90 reader--cells   no competence floor at all, every usable flow. The unselected
                         estimate, and the one the paper should lead with.
  competent on A         the split-sample estimate, on B only.
  not competent on A     the complement, on B only. Reported because the gap between it
                         and the competent arm is the thing the floor was ever about.

Nothing here re-runs a reader or a detector. Every number is a re-scoring of the cached
decisions, using the same rescoring rule as scripts/run_reference_decomposition.py.

  python scripts/report_split_sample_harm.py
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

log = logging.getLogger("split_sample")

REPO = Path(__file__).resolve().parents[1]
DECISIONS = RESULTS_ROOT / "decision_utility" / "raw" / "decisions.json"
PER_CELL = RESULTS_ROOT / "decision_utility" / "summary" / "per_cell.json"
ATTACKED_FLOWS = RESULTS_ROOT / "reference_decomposition" / "raw" / "attacked_flows.json"
# Every rebuild whose effects the paper quotes. Table I's harm column reads across attacks and
# across the scaffold's contamination settings, and those settings live in separate rebuilds; if
# only the main run were split-sampled, that one column would carry two different populations --
# an in-sample estimate for the scaffold and an out-of-sample one for the searches -- which is
# the defect the split sample exists to remove.
RUNS = ["reference_decomposition", "reference_decomposition_a3_c03",
        "reference_decomposition_a3_c01", "reference_decomposition_a3_c005"]
ATTACKS = ["A1_displacement", "A1_misdirection", "A3_scaffolding"]
REFERENCES = ["vs_S_x", "vs_S_xprime"]
NAME = "split_sample_harm"


def _rescored_rows(decisions, inst):
    """Borrow the rescoring rule from the decomposition script rather than restate it.

    Two copies of "which decisions are usable and what counts as correct" would be two
    chances to disagree, and this report has to describe the same instances the paper's
    harm number does.
    """
    spec = importlib.util.spec_from_file_location(
        "run_reference_decomposition", REPO / "scripts" / "run_reference_decomposition.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod.rescored_rows(decisions, inst)


def halves(inst, seed: int = 0):
    """One permutation per cell, over ALL forty of its flows.

    Taken from the rebuilt instances, not from the usable rows, so the split does not depend
    on which flows an attack happened to keep: scaffolding preserves fewer flows than the two
    searches, and a split drawn from the survivors would give each attack a different A half.
    """
    by = defaultdict(set)
    for r in inst:
        by[(r["dataset"], r["class"], r["seed"])].add(r["sample_id"])
    A, B = {}, {}
    for cell, ids in by.items():
        ids = sorted(ids)
        perm = np.random.default_rng(seed).permutation(len(ids))
        cut = len(ids) // 2
        A[cell] = {ids[i] for i in perm[:cut]}
        B[cell] = {ids[i] for i in perm[cut:]}
    return A, B


def reader_cell(r):
    return (r["judge"], r["dataset"], r["class"], r["attack"], r["seed"])


def clean_accuracy_on(rows, ids_by_cell) -> dict:
    """Clean accuracy of each reader--cell on one half. Clean correctness is identical under
    both references -- only attacked decisions are rescored -- so competence is a property of
    the reader and the flows, not of the reference."""
    acc = defaultdict(list)
    for r in rows:
        if r["condition"] != "clean":
            continue
        cell = (r["dataset"], r["class"], r["seed"])
        if r["sample_id"] in ids_by_cell.get(cell, ()):
            acc[reader_cell(r)].append(int(r["det_correct"]))
    return {k: float(np.mean(v)) for k, v in acc.items() if v}


def effects(rows_by_ref, A, B, floor: float, n_boot: int, seed: int) -> dict:
    out = {}
    for ref, rows in rows_by_ref.items():
        acc_A = clean_accuracy_on(rows, A)
        competent = {k for k, v in acc_A.items() if v >= floor}
        rows_B = [r for r in rows
                  if r["sample_id"] in B.get((r["dataset"], r["class"], r["seed"]), ())]
        out[ref] = {}
        for atk in ATTACKS:
            cells_here = {k for k in acc_A if k[3] == atk}
            comp_here = competent & cells_here
            out[ref][atk] = {
                # The unselected estimate on every usable flow: no competence floor, nothing
                # held out, 45 cell clusters. This is the best estimate of the effect because
                # it throws nothing away, and it is what the paper leads with.
                "all_cells": paired_effect(rows, "attacked", attack=atk, cluster="cell",
                                           n_boot=n_boot, seed=seed),
                # The same unselected estimate restricted to half B. Reported because the two
                # selected arms below are on half B, and without this a reader who weights
                # them by cluster count gets a number that does not match "all_cells" -- the
                # arms would be three estimates on two populations presented as one split.
                "all_cells_on_B": paired_effect(rows_B, "attacked", attack=atk, cluster="cell",
                                                n_boot=n_boot, seed=seed),
                "competent_on_A": paired_effect(rows_B, "attacked", attack=atk,
                                                competent_cells=comp_here, cluster="cell",
                                                n_boot=n_boot, seed=seed),
                "not_competent_on_A": paired_effect(rows_B, "attacked", attack=atk,
                                                    competent_cells=cells_here - comp_here,
                                                    cluster="cell", n_boot=n_boot, seed=seed),
                "n_reader_cells": len(cells_here),
                "n_competent_on_A": len(comp_here),
                "mean_clean_acc_on_A_competent": (
                    float(np.mean([acc_A[k] for k in comp_here])) if comp_here else float("nan")),
                "mean_clean_acc_on_A_others": (
                    float(np.mean([acc_A[k] for k in cells_here - comp_here]))
                    if cells_here - comp_here else float("nan")),
            }
    return out


def changed_flow_effect(rows_by_ref, n_boot, seed, attack="A1_misdirection"):
    """The effect on the case sets where the search actually moved the flow.

    The rank-promotion search returns the flow untouched on 758 of 1,800 case sets, and on
    those the explanation, the reader's answer and the paired difference are all exactly what
    they were on the clean flow. Every effect reported for the attack therefore averages a
    real effect over the 1,042 it changed with an exact zero over the 758 it did not. This
    reports the conditional effect on the 1,042, which is what the attack does when it fires.

    Membership is the exported flag `attack_returned_unchanged`, set by comparing x' with x
    coordinate by coordinate -- not inferred from a zero corruption score, which a changed
    flow whose top-5 happens to survive also produces.
    """
    payload = json.loads(ATTACKED_FLOWS.read_text())
    changed, preserved = set(), []
    for r in payload["flows"]:
        if r["attack"] != attack:
            continue
        if not r["attack_returned_unchanged"]:
            changed.add((r["dataset"], r["class"], r["seed"], r["sample_id"]))
            preserved.append(bool(r["prediction_preserved"]))
    out = {"n_case_sets_changed": len(changed),
           "class_survival_on_changed": float(np.mean(preserved)) if preserved else float("nan")}
    for ref, rows in rows_by_ref.items():
        sel = [r for r in rows
               if (r["dataset"], r["class"], r["seed"], r["sample_id"]) in changed]
        out[ref] = paired_effect(sel, "attacked", attack=attack, cluster="cell",
                                 n_boot=n_boot, seed=seed)
    return out


def excluded_cells(attack="A1_displacement", floor=0.5):
    """The reader--cells the competence floor removes, by corpus and class.

    The floor is not a small trim: it removes a third of the reader--cells, and those cells
    are not scattered. Reporting only the count hides which corpora and classes the reader
    cannot read at all, which is the part a reader needs to judge what the remaining
    population represents.
    """
    per_cell = json.loads(PER_CELL.read_text())
    out, by_corpus, by_class = [], defaultdict(int), defaultdict(int)
    for c in per_cell:
        if c.get("attack") != attack or c.get("clean__acc", 0) >= floor:
            continue
        out.append({"judge": c["judge"], "dataset": c["dataset"], "class": c["class"],
                    "seed": c["seed"], "clean_acc": round(float(c["clean__acc"]), 4)})
        by_corpus[c["dataset"]] += 1
        by_class[f"{c['dataset']}/{c['class']}"] += 1
    return {"n_excluded": len(out), "by_corpus": dict(by_corpus), "by_class": dict(by_class),
            "cells": sorted(out, key=lambda r: (r["dataset"], r["class"], r["seed"], r["judge"])),
            "mean_clean_acc": float(np.mean([r["clean_acc"] for r in out])) if out else float("nan")}


def markdown(res, args) -> str:
    L = [f"# Split-sample competence (floor={args['floor']}, split seed={args['split_seed']}, "
         f"{args['n_boot']} resamples, cluster=cell)", "",
         "Competence is clean accuracy on half A of each cell's flows; every effect below is "
         "measured on half B, which nothing selected on. Intervals are a cluster bootstrap "
         "over cells with both readers inside one cluster.", "",
         "| Run | Reference | Attack | arm | reader--cells | n | effect | 95% CI |",
         "|---|---|---|---|---|---|---|---|"]
    for run in args["runs"]:
        L += _rows(res[run], run)
    L += ["", "## Clean accuracy on the selection half (main run)", "",
          "| Attack | competent mean | others mean |", "|---|---|---|"]
    for atk in ATTACKS:
        r = res["reference_decomposition"]["vs_S_xprime"].get(atk)
        if r:
            L.append(f"| {atk} | {r['mean_clean_acc_on_A_competent']:.3f} | "
                     f"{r['mean_clean_acc_on_A_others']:.3f} |")
    return "\n".join(L)


def _rows(res, run) -> list:
    L = []
    for ref in REFERENCES:
        for atk in ATTACKS:
            r = res[ref].get(atk)
            if r is None:
                continue
            for arm, label, n_cells in (
                    ("all_cells", "all reader--cells, all flows", r["n_reader_cells"]),
                    ("all_cells_on_B", "all reader--cells, half B only", r["n_reader_cells"]),
                    ("competent_on_A", "competent on A, measured on B", r["n_competent_on_A"]),
                    ("not_competent_on_A", "not competent on A, measured on B",
                     r["n_reader_cells"] - r["n_competent_on_A"])):
                e = r[arm]
                if not e.get("n"):
                    L.append(f"| {run} | {ref} | {atk} | {label} | {n_cells} | 0 | — | — |")
                    continue
                L.append(f"| {run} | {ref} | {atk} | {label} | {n_cells} | {e['n']} | "
                         f"{e['mean_delta']:+.3f} | [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] |")
    return L


def sweep(ctx) -> RunResult:
    args = ctx.config["args"]
    decisions = json.loads(DECISIONS.read_text())
    res, splits = {}, {}
    for run in args["runs"]:
        inst = json.loads((RESULTS_ROOT / run / "raw" / "instances.json").read_text())
        keep, shown = _rescored_rows(decisions, inst)
        A, B = halves(inst, seed=args["split_seed"])
        sizes = sorted({len(v) for v in A.values()}), sorted({len(v) for v in B.values()})
        log.info("%s: split %d cells into halves of %s (A) and %s (B) flows",
                 run, len(A), sizes[0], sizes[1])
        res[run] = effects({"vs_S_x": keep, "vs_S_xprime": shown}, A, B,
                           args["floor"], args["n_boot"], args["seed"])
        splits[run] = {"A": {"|".join(map(str, k)): sorted(v) for k, v in A.items()},
                       "B": {"|".join(map(str, k)): sorted(v) for k, v in B.items()}}

    inst_main = json.loads((RESULTS_ROOT / "reference_decomposition" / "raw" / "instances.json").read_text())
    keep, shown = _rescored_rows(decisions, inst_main)
    changed = changed_flow_effect({"vs_S_x": keep, "vs_S_xprime": shown},
                                  args["n_boot"], args["seed"])
    excluded = excluded_cells()
    log.info("rank promotion changed the flow on %d case sets, class survives on %.3f of them",
             changed["n_case_sets_changed"], changed["class_survival_on_changed"])
    log.info("competence floor excludes %d reader-cells: %s",
             excluded["n_excluded"], excluded["by_corpus"])

    (ctx.run_dir / "raw" / "split.json").write_text(json.dumps(
        {"split_seed": args["split_seed"], "runs": splits}, indent=1))
    (ctx.run_dir / "raw" / "excluded_cells.json").write_text(json.dumps(excluded, indent=1))
    (ctx.run_dir / "summary" / "changed_flow_effect.json").write_text(json.dumps(changed, indent=1))
    (ctx.run_dir / "summary" / "excluded_cells.json").write_text(json.dumps(
        {k: v for k, v in excluded.items() if k != "cells"}, indent=1))
    (ctx.run_dir / "summary" / "effects.json").write_text(json.dumps(res, indent=1))
    md = markdown(res, args)
    (ctx.run_dir / "summary" / "report.md").write_text(md)
    print(md)

    d = res["reference_decomposition"]["vs_S_xprime"]["A1_displacement"]
    return RunResult(
        summary={"runs": args["runs"],
                 "displacement_all_90_vs_Sxprime": d["all_cells"]["mean_delta"],
                 "displacement_all_90_ci": [d["all_cells"]["ci_lo"], d["all_cells"]["ci_hi"]],
                 "displacement_competent_on_A_vs_Sxprime": d["competent_on_A"]["mean_delta"],
                 "displacement_competent_on_A_ci": [d["competent_on_A"]["ci_lo"],
                                                    d["competent_on_A"]["ci_hi"]],
                 "displacement_competent_count": d["n_competent_on_A"]},
        floats=[{"float_id": "changed_flow_effect", "kind": "summary", "paper_section": "harm",
                 "path": str(ctx.run_dir / "summary" / "changed_flow_effect.json"),
                 "claim": "rank promotion's effect on the case sets where its search moved "
                          "the flow, and the class-survival rate on them"},
                {"float_id": "excluded_reader_cells", "kind": "summary", "paper_section": "harm",
                 "path": str(ctx.run_dir / "summary" / "excluded_cells.json"),
                 "claim": "which reader--cells the competence floor removes, by corpus"},
                {"float_id": "Tab_split_sample_harm", "kind": "table", "paper_section": "harm",
                 "path": str(ctx.run_dir / "summary" / "effects.json"),
                 "claim": "reader harm with competence selected on held-out clean flows"},
                {"float_id": "split_sample_assignment", "kind": "raw", "paper_section": "harm",
                 "path": str(ctx.run_dir / "raw" / "split.json"),
                 "claim": "which flows selected competence and which measured the effect"}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--split-seed", type=int, default=0)
    ap.add_argument("--n-boot", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--runs", nargs="+", default=RUNS,
                    help="reference-decomposition runs to split-sample; each supplies its own "
                         "raw/instances.json and is scored against the same cached decisions")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(NAME, sweep, config={"args": {
        "floor": a.floor, "split_seed": a.split_seed, "n_boot": a.n_boot, "seed": a.seed,
        "runs": a.runs}}, seed=a.seed)


if __name__ == "__main__":
    main()
