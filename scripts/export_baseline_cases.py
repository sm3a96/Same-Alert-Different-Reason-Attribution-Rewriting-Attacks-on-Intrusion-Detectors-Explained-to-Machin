"""Reader cases for the random-draw control arm of cause displacement.

The random-draw arm (results/baseline_random_draw) returns the first class-preserving
projected draw instead of the draw that most changes the shown set: the same budget, the same
projection, the same random stream. Its decomposition was reported without reader harm,
because the cached reader decisions were made on the attack's flows, not these. This exports
what a reader needs to decide on the baseline flows, so both readers can be run on them.

Each cell is rebuilt with `rebuild_cell` from scripts/run_reference_decomposition.py with the
first-valid-draw switch on, which is how the arm was produced. For every case set it writes

  * the clean case, taken unchanged from the case cache, so the clean arm is the one the
    readers already saw;
  * the attacked case: the top-10 of the TreeSHAP explanation of the baseline flow x', with
    x' values and importances, and the cached S(x) as the case's causal set;
  * S(x') of the baseline flow, for the rescoring against the flow shown;
  * whether this rebuild reproduces results/baseline_random_draw/raw/instances.json on that
    case set (same S(x'), same shown top-5), and whether the rebuilt clean shown case equals
    the cached clean case.

It writes results/baseline_random_draw/raw/cases.json and attacked_flows.json and does not
touch that run's instances.json.

  python scripts/export_baseline_cases.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

from avert.benchmark import XIntBench
from avert.benchmark.harness import target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.decision_utility import CASE_CACHE
from avert.eval.runner import RESULTS_ROOT, RunResult, run_experiment

log = logging.getLogger("baseline_cases")

REPO = Path(__file__).resolve().parents[1]
RUN = RESULTS_ROOT / "baseline_random_draw"
NAME = "baseline_cases_export"
ATTACK = "A1_displacement"


def _refdecomp():
    spec = importlib.util.spec_from_file_location(
        "run_reference_decomposition", REPO / "scripts" / "run_reference_decomposition.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sweep(ctx) -> RunResult:
    args = ctx.config["args"]
    rd = _refdecomp()
    rd.ATTACKS[:] = [ATTACK]
    cache = json.loads(CASE_CACHE.read_text())["cases"]
    published = {(r["dataset"], r["class"], r["seed"], r["sample_id"]): r
                 for r in json.loads((RUN / "raw" / "instances.json").read_text())}

    cases, flows, feature_order = [], [], {}
    n_run_mismatch = n_clean_mismatch = 0
    for ds in args["datasets"]:
        data = load_dataset(ds, load_config(f"datasets/{ds}"))
        feature_order[ds] = list(data.feature_names)
        classes, _ = target_classes(data, rd.N_CLASSES, n_cal=100, n_test=rd.N_TEST, seed=0,
                                    min_count=2000, seeds=rd.SEEDS)
        for c in classes:
            cname = data.label_names[c] if data.label_names else str(c)
            for seed in args["seeds"]:
                cached_cell = {(r["sample_id"], r["attack"]): r for r in cache
                               if r["dataset"] == ds and r["class"] == cname and r["seed"] == seed}
                bench = XIntBench(data, n_cal=100, n_test=rd.N_TEST, top_k=rd.TOP_K,
                                  perm_samples=10, stability_n=30, seed=seed,
                                  displacement_first_valid=True)
                rows = rd.rebuild_cell(bench, data, c, cached_cell, args["tau"], args["min_k"],
                                       "mean", keep_attacked=True)
                for r in rows:
                    key = (ds, cname, seed, r["sample_id"])
                    pub = published.get(key)
                    matches_run = (pub is not None and sorted(pub["S_a"]) == sorted(r["S_a"])
                                   and pub["top5_atk"] == r["top5_atk"])
                    n_run_mismatch += not matches_run
                    cached = cached_cell[(r["sample_id"], ATTACK)]["cases"]
                    clean = cached["clean"]
                    clean_matches = clean["candidates"][:rd.TOP_K] == r["top5_clean"]
                    n_clean_mismatch += not clean_matches
                    attacked = {**clean, "condition": "attacked", "attack": ATTACK,
                                **r["shown_attacked"], "causal": list(clean["causal"])}
                    cases.append({"dataset": ds, "class": cname, "seed": seed, "attack": ATTACK,
                                  "sample_id": r["sample_id"],
                                  "prediction_preserved": r["prediction_preserved"],
                                  "matches_baseline_run": matches_run,
                                  "clean_matches_cache": clean_matches,
                                  "attack_returned_unchanged": r["attack_returned_unchanged"],
                                  "S_x_cached": list(clean["causal"]), "S_x": r["S_x"],
                                  "S_a": r["S_a"],
                                  "cases": {"clean": clean, "attacked": attacked}})
                    flows.append({"dataset": ds, "class": cname, "seed": seed,
                                  "sample_id": r["sample_id"], "x_prime": r["x_prime"]})
                (RUN / "raw" / "cases.json").write_text(json.dumps(cases))
                (RUN / "raw" / "attacked_flows.json").write_text(json.dumps(
                    {"feature_order": feature_order, "flows": flows}))
                log.info("done %s class=%s seed=%s (%d case sets; %d differ from the run, "
                         "%d clean cases differ from the cache)", ds, cname, seed, len(cases),
                         n_run_mismatch, n_clean_mismatch)

    return RunResult(
        summary={"case_sets": len(cases), "differ_from_baseline_run": n_run_mismatch,
                 "clean_differs_from_cache": n_clean_mismatch,
                 "prediction_preserved": int(sum(c["prediction_preserved"] for c in cases)),
                 "returned_unchanged": int(sum(c["attack_returned_unchanged"] for c in cases))},
        floats=[{"float_id": "baseline_reader_cases", "kind": "raw", "paper_section": "harm",
                 "path": str(RUN / "raw" / "cases.json"),
                 "claim": "reader cases for the random-draw control arm of cause displacement"}])


def main():
    rd = _refdecomp()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=rd.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=rd.SEEDS)
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--min-k", type=int, default=3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(NAME, sweep, config={"args": {
        "datasets": a.datasets, "seeds": a.seeds, "tau": a.tau, "min_k": a.min_k,
        "first_valid_draw": True}}, seed=0)


if __name__ == "__main__":
    main()
