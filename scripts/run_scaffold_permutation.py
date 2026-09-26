"""Explainer scaffolding against an attributor it can reach: the permutation attributor as g.

In the matrix the attributor the reader sees is TreeSHAP, which reads the real booster
directly, so the scaffold cannot move it and the shown list is unchanged by construction. The
scaffold's routing targets query-based attributors. This run makes the permutation attributor
the attributor g of Algorithm 1 -- the one the reader is shown and the one every check that
re-explains uses -- and re-runs the scaffolding cells with nothing else changed.

Per cell it writes

  * the seven-check panel on every clean and scaffolded test flow (per-flow scores kept, so
    detectability is computed on the prediction-preserved flows only, as for the TreeSHAP
    rows). Cross-method consensus pairs g with TreeSHAP here, since pairing g with a second
    copy of itself measures nothing;
  * the reader cases for the same flows (clean and attacked shown lists from g), and for each
    flow S(x) on the deployed detector and S(x') on the component that decided: the surrogate
    if the router sent the flow there, the real model otherwise, exactly as
    scripts/run_reference_decomposition.py measures it for the TreeSHAP scaffolding rows.

The permutation attributor draws its background rows from a generator seeded by the input, so
the shown list a check scores and the shown list a reader case carries are the same list. The
script asserts that, flow by flow.

The grid runs one process per corpus, each writing raw/<file>_<corpus>.json; a last call with
--summarize merges them into raw/per_flow.json, raw.json and cases.json and writes the summary:

  python scripts/run_scaffold_permutation.py --datasets fiveg_nidd     # and the other two
  python scripts/run_scaffold_permutation.py --summarize
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from dataclasses import asdict
from pathlib import Path

import numpy as np

from avert.benchmark import XIntBench
from avert.benchmark.ground_truth import causal_features
from avert.benchmark.harness import target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.runner import RunResult, run_experiment

log = logging.getLogger("scaffold_perm")

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
SEEDS = [0, 1, 2, 3, 4]
ATTACK = "A3_scaffolding"
N_CLASSES, N_CAL, N_TEST, TOP_K, N_SHOWN, TAU, MIN_K = 3, 100, 40, 5, 10, 0.05, 3


def _survivors_report():
    spec = importlib.util.spec_from_file_location(
        "report_scaffolding_survivors", REPO / "scripts" / "report_scaffolding_survivors.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def reference_sets(bench, target):
    """S(x) on the deployed detector and S(x') on the component that decided, per test flow."""
    st = bench._prepare(target=target)
    out = {}
    for s in st.test:
        S_x, _ = causal_features(s, st.detector, st.benign_ref, TAU, MIN_K)
        routed = bool(st.scaffold.ood.is_probe(s.features.reshape(1, -1))[0])
        comp = st.scaffold.surrogate if routed else st.scaffold.real
        S_a, d_a = causal_features(s, comp, st.benign_ref, TAU, MIN_K)
        out[s.sample_id] = {"S_x": [st.names[j] for j in S_x], "S_a": [st.names[j] for j in S_a],
                            "routed_to_surrogate": routed,
                            "fallback_a": bool(max(d_a) < TAU)}
    return out


RAW_FILES = ("per_flow", "raw", "cases")


def run_corpus(ctx) -> RunResult:
    """Every scaffolding cell of one corpus: panel scores, reader cases, reference sets."""
    args = ctx.config["args"]
    (ds,) = args["datasets"]
    per_flow, rows, cases = [], [], []
    shown_mismatch = 0
    data = load_dataset(ds, load_config(f"datasets/{ds}"))
    classes, _ = target_classes(data, N_CLASSES, n_cal=N_CAL, n_test=N_TEST, seed=0,
                                min_count=2000, seeds=SEEDS)
    for c in classes:
        cname = data.label_names[c] if data.label_names else str(c)
        for seed in args["seeds"]:
            bench = XIntBench(data, n_cal=N_CAL, n_test=N_TEST, top_k=TOP_K,
                              perm_samples=10, stability_n=30, seed=seed,
                              attributor="permutation")
            # evaluate_pipeline, decision_cases and reference_sets each call _prepare, which
            # refits the detector and the scaffold. The fit is deterministic, so one setup is
            # shared by all three; scaffolding touches no attack generator, and g draws its
            # background rows per input, so sharing it changes no value.
            st = bench._prepare(target=c)
            bench._prepare = lambda detector=None, explainer=None, target=None, _st=st: _st
            rep = bench.evaluate_pipeline(target=c, attacks=(ATTACK,), per_flow=True)
            per_flow.append({"dataset": ds, "class": cname, "seed": seed, **rep.per_flow})
            for r in rep.results:
                rows.append({"dataset": ds, "class": cname, "seed": seed,
                             "attack": r.attack, "valid": r.prediction_preserved,
                             "corrupt": r.explanation_corruption,
                             "per_signal": r.per_signal_auroc})

            items = bench.decision_cases([ATTACK], target=c, n_shown=N_SHOWN)
            refs = reference_sets(bench, c)
            corrupt = {r["sample_id"]: r["corrupt"]
                       for r in rep.per_flow["attacks"][ATTACK]}
            for it in items:
                cs = it["cases"]
                top_clean = set(cs["clean"].candidates[:TOP_K])
                top_atk = set(cs["attacked"].candidates[:TOP_K])
                j = 1.0 - len(top_clean & top_atk) / len(top_clean | top_atk)
                # The reader must be shown the list the checks scored.
                same = abs(j - corrupt[it["sample_id"]]) < 1e-9
                shown_mismatch += not same
                cases.append({"dataset": ds, "class": cname, "seed": seed,
                              "attack": ATTACK, "sample_id": it["sample_id"],
                              "prediction_preserved": it["prediction_preserved"],
                              "shown_matches_checks": same,
                              **refs[it["sample_id"]],
                              "cases": {k: asdict(v) for k, v in cs.items()}})
            for name, obj in zip(RAW_FILES, (per_flow, rows, cases)):
                (ctx.run_dir / "raw" / f"{name}_{ds}.json").write_text(json.dumps(obj))
            log.info("done %s class=%s seed=%s (%d cells, %d shown mismatches)",
                     ds, cname, seed, len(per_flow), shown_mismatch)
    return RunResult(summary={"dataset": ds, "cells": len(per_flow), "case_sets": len(cases),
                              "shown_mismatch": shown_mismatch})


def summarize_all(ctx) -> RunResult:
    """Merge the per-corpus files and write the survivor-only panel and corruption."""
    merged = {}
    for name in RAW_FILES:
        merged[name] = []
        for f in sorted((ctx.run_dir / "raw").glob(f"{name}_*.json")):
            merged[name] += json.loads(f.read_text())
        (ctx.run_dir / "raw" / f"{name}.json").write_text(json.dumps(merged[name]))
    per_flow, cases = merged["per_flow"], merged["cases"]
    shown_mismatch = sum(not c["shown_matches_checks"] for c in cases)
    sv = _survivors_report()
    surv = sv.aggregate(sv.per_cell(per_flow, ATTACK, survivors_only=True))
    allf = sv.aggregate(sv.per_cell(per_flow, ATTACK, survivors_only=False))
    kept = [c for c in cases if c["prediction_preserved"]]
    summary = {"survivors_only": surv, "all_flows": allf,
               "shown_mismatch": shown_mismatch, "case_sets": len(cases),
               "case_sets_preserved": len(kept),
               "routed_rate_preserved": float(np.mean([c["routed_to_surrogate"] for c in kept]))
               if kept else float("nan")}
    (ctx.run_dir / "summary" / "auroc.json").write_text(json.dumps(summary, indent=1))
    L = ["# Explainer scaffolding with the permutation attributor as g", "",
         f"Case sets: {len(cases)}, prediction preserved: {len(kept)}; shown list differs "
         f"between reader case and check score on {shown_mismatch}.", "",
         "| Corpus | Check | survivors only | 95% CI | all flows | flows kept |",
         "|---|---|---|---|---|---|"]
    for ds in sorted(surv):
        s, a = surv[ds], allf[ds]
        for k in ["_corrupt"] + sorted(x for x in s if not x.startswith("_")):
            L.append(f"| {ds} | {k} | {s[k]['mean']:.3f} | [{s[k]['ci_lo']:.3f}, "
                     f"{s[k]['ci_hi']:.3f}] | {a[k]['mean']:.3f} | "
                     f"{s['_flows_kept']} of {s['_flows_total']} |")
    md = "\n".join(L)
    (ctx.run_dir / "summary" / "report.md").write_text(md)
    print(md)
    return RunResult(
        summary={"cells": len(per_flow), "case_sets": len(cases),
                 "case_sets_preserved": len(kept), "shown_mismatch": shown_mismatch},
        floats=[{"float_id": "scaffold_permutation_auroc", "kind": "table",
                 "paper_section": "detect", "path": str(ctx.run_dir / "summary" / "auroc.json"),
                 "claim": "corruption and check AUROC for scaffolding when the reader is shown "
                          "the permutation attributor, prediction-preserved flows"},
                {"float_id": "scaffold_permutation_cases", "kind": "raw", "paper_section": "harm",
                 "path": str(ctx.run_dir / "raw" / "cases.json"),
                 "claim": "reader cases and reference sets for the permutation-attributor "
                          "scaffolding arm"}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--name", default="scaffolding_permutation")
    ap.add_argument("--summarize", action="store_true",
                    help="merge the per-corpus raw files and write the summary")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = {"args": {"datasets": a.datasets, "seeds": a.seeds, "attributor": "permutation",
                       "attack": ATTACK, "n_cal": N_CAL, "n_test": N_TEST, "tau": TAU,
                       "min_k": MIN_K}}
    if a.summarize:
        run_experiment(a.name, summarize_all, config=config, seed=0)
        return
    for ds in a.datasets:
        config["args"]["datasets"] = [ds]
        run_experiment(a.name, run_corpus, config=config, seed=0, meta_name=f"run_{ds}.json")


if __name__ == "__main__":
    main()
