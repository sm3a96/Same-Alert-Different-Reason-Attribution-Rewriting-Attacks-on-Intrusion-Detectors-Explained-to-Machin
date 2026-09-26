"""Detectability of explainer scaffolding on the flows whose predicted class survived.

The threat model requires the alert to stand. The two search attacks enforce that by
rejection, so every flow they return is in scope. Scaffolding only measures it: its router
flips the class on some flows, and those flows are not this paper's attack at all. The
September matrix scored every check over all forty test flows of a cell regardless, so the
scaffolding rows of Table IX were computed partly on flows outside the threat model -- and a
class flip is exactly the kind of event a noise-sensitivity check would notice, which is the
direction that flatters the panel.

This recomputes those rows on prediction-preserved flows only. Both arms are restricted to the
same flows: the clean score of a flow whose attacked copy flipped class is dropped with it, so
the comparison stays on one population rather than scoring a clean arm of forty against an
attacked arm of thirty.

It reads results/matrix_a3_survivors/raw/per_flow.json, written by

    python scripts/run_matrix.py --out results/matrix_a3_survivors \
        --name matrix_a3_survivors --per-flow

which re-runs the full matrix with the per-flow check scores kept. All three attacks are
re-run, not just scaffolding: the two certified-stability checks and the permutation attributor
hold persistent RNGs, so scoring scaffolding alone would advance their noise streams differently
and the re-run would not reproduce the published rows. Reproducing them is the point -- the
all-flow arm below must equal results/matrix/raw.json exactly, and if it does not, the
survivor-only difference cannot be attributed to the population.

  python scripts/report_scaffolding_survivors.py
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
from sklearn.metrics import roc_auc_score

from avert.eval.metrics import seed_clustered_ci
from avert.eval.runner import RESULTS_ROOT, RunResult, run_experiment

log = logging.getLogger("survivors")

SOURCE = RESULTS_ROOT / "matrix_a3_survivors" / "raw" / "per_flow.json"
PUBLISHED = RESULTS_ROOT / "matrix" / "raw.json"
NAME = "scaffolding_survivors"
ATTACKS = ["A1_misdirection", "A1_displacement", "A3_scaffolding"]


def _auroc(clean, attacked):
    """AUROC of attacked against clean, ties by average rank, rounded as the matrix rounds."""
    if not clean or not attacked:
        return float("nan")
    y = [0] * len(clean) + [1] * len(attacked)
    return round(float(roc_auc_score(y, list(clean) + list(attacked))), 3)


def per_cell(cells, attack, survivors_only: bool):
    """One row per cell: every check's AUROC on the chosen population."""
    out = []
    for c in cells:
        rows = c["attacks"].get(attack)
        if rows is None:
            continue
        clean = {r["sample_id"]: r for r in c["clean"]}
        keep = [r for r in rows if r["prediction_preserved"] or not survivors_only]
        ids = [r["sample_id"] for r in keep if r["sample_id"] in clean]
        if not ids:
            continue
        signals = sorted(keep[0]["per_signal"]) + ["fused"]

        def col(r, sig):
            return r["fused"] if sig == "fused" else r["per_signal"][sig]

        by_id = {r["sample_id"]: r for r in keep}
        auroc = {sig: _auroc([col(clean[i], sig) for i in ids],
                             [col(by_id[i], sig) for i in ids]) for sig in signals}
        # The matrix names the fused channel FUSED; keep that spelling so a reader can line
        # these rows up against results/matrix/raw.json without a mapping table.
        auroc["FUSED"] = auroc.pop("fused")
        out.append({"dataset": c["dataset"], "class": c["class"], "seed": c["seed"],
                    "n_flows": len(ids), "n_all": len(rows),
                    "prediction_preserved": float(np.mean([r["prediction_preserved"] for r in rows])),
                    "per_signal": auroc})
    return out


def aggregate(rows):
    """Seed-clustered mean and interval per (dataset, attack), as the matrix aggregates."""
    by = defaultdict(list)
    for r in rows:
        by[r["dataset"]].append(r)
    out = {}
    for ds, rs in by.items():
        seeds = np.array([r["seed"] for r in rs])
        out[ds] = {}
        for sig in rs[0]["per_signal"]:
            mu, hw = seed_clustered_ci(np.array([r["per_signal"][sig] for r in rs]), seeds)
            out[ds][sig] = {"mean": mu, "ci_lo": mu - hw, "ci_hi": mu + hw}
        out[ds]["_n_cells"] = len(rs)
        out[ds]["_flows_kept"] = int(sum(r["n_flows"] for r in rs))
        out[ds]["_flows_total"] = int(sum(r["n_all"] for r in rs))
    return out


def reproduces_published(all_flow_rows) -> dict:
    """Does the all-flow arm equal the published matrix, cell by cell and check by check?

    This is the control on the whole report. The re-run used the same seeds and the same
    configuration, so every all-flow number must come back identical; a mismatch means the
    survivor-only difference below is contaminated by something other than the population,
    and the report says so instead of quietly reporting the difference anyway.
    """
    pub = {(r["dataset"], r["class"], r["seed"], r["attack"]): r["per_signal"]
           for r in json.loads(PUBLISHED.read_text())}
    checked = mismatched = 0
    worst = []
    for attack, rows in all_flow_rows.items():
        for r in rows:
            p = pub.get((r["dataset"], r["class"], r["seed"], attack))
            if p is None:
                continue
            for sig, v in r["per_signal"].items():
                if sig not in p:
                    continue
                checked += 1
                if abs(float(p[sig]) - float(v)) > 1e-9:
                    mismatched += 1
                    worst.append({"cell": [r["dataset"], r["class"], r["seed"], attack],
                                  "check": sig, "published": p[sig], "rerun": v})
    return {"values_checked": checked, "mismatched": mismatched, "examples": worst[:10],
            "reproduces": mismatched == 0}


def markdown(agg_all, agg_surv, control) -> str:
    L = ["# Scaffolding detectability on prediction-preserved flows only", "",
         f"All-flow arm against the published matrix: {control['values_checked']} values checked, "
         f"{control['mismatched']} mismatched "
         f"({'reproduces exactly' if control['reproduces'] else 'DOES NOT REPRODUCE'}).", ""]
    for attack in ATTACKS:
        if attack not in agg_surv:
            continue
        L += [f"## {attack}", "",
              "| Corpus | Check | all flows | survivors only | change | flows kept |",
              "|---|---|---|---|---|---|"]
        for ds in sorted(agg_surv[attack]):
            a, s = agg_all[attack][ds], agg_surv[attack][ds]
            for sig in [k for k in s if not k.startswith("_")]:
                L.append(f"| {ds} | {sig} | {a[sig]['mean']:.3f} | {s[sig]['mean']:.3f} | "
                         f"{s[sig]['mean'] - a[sig]['mean']:+.3f} | "
                         f"{s['_flows_kept']} of {s['_flows_total']} |")
        L.append("")
    return "\n".join(L)


def sweep(ctx) -> RunResult:
    cells = json.loads(SOURCE.read_text())
    log.info("read %d cells from %s", len(cells), SOURCE)

    rows_all = {a: per_cell(cells, a, survivors_only=False) for a in ATTACKS}
    rows_surv = {a: per_cell(cells, a, survivors_only=True) for a in ATTACKS}
    control = reproduces_published(rows_all)
    if not control["reproduces"]:
        log.error("the all-flow arm does NOT reproduce results/matrix/raw.json on %d of %d "
                  "values; the survivor-only difference cannot be attributed to the population",
                  control["mismatched"], control["values_checked"])

    agg_all = {a: aggregate(rows_all[a]) for a in ATTACKS if rows_all[a]}
    agg_surv = {a: aggregate(rows_surv[a]) for a in ATTACKS if rows_surv[a]}

    (ctx.run_dir / "raw" / "per_cell_all_flows.json").write_text(json.dumps(rows_all, indent=1))
    (ctx.run_dir / "raw" / "per_cell_survivors.json").write_text(json.dumps(rows_surv, indent=1))
    (ctx.run_dir / "summary" / "auroc.json").write_text(json.dumps(
        {"all_flows": agg_all, "survivors_only": agg_surv, "control": control}, indent=1))
    md = markdown(agg_all, agg_surv, control)
    (ctx.run_dir / "summary" / "report.md").write_text(md)
    print(md)

    a3 = agg_surv.get("A3_scaffolding", {})
    return RunResult(
        summary={"cells": len(cells), "reproduces_published_matrix": control["reproduces"],
                 "values_checked": control["values_checked"],
                 "mismatched": control["mismatched"],
                 "scaffolding_flows_kept": {ds: [a3[ds]["_flows_kept"], a3[ds]["_flows_total"]]
                                            for ds in a3}},
        floats=[{"float_id": "Tab9_survivor_only_auroc", "kind": "table", "paper_section": "detect",
                 "path": str(ctx.run_dir / "summary" / "auroc.json"),
                 "claim": "check AUROC against scaffolding on the flows whose predicted class "
                          "survived, beside the all-flow rows the matrix published"}])


def main():
    ap = argparse.ArgumentParser()
    ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    if not SOURCE.exists():
        raise SystemExit(f"{SOURCE} not found -- run scripts/run_matrix.py --per-flow first")
    run_experiment(NAME, sweep, config={"source": str(SOURCE)}, seed=0)


if __name__ == "__main__":
    main()
