"""Export the attacked flow vectors behind every rescored instance, for the release.

The release ships the detectors, the scaffolds and every result file, and until now it did
not ship the one thing an attack result is about: the flow the adversary produced. A reader
could recompute our corruption numbers only by re-running the search, which is the
expensive, seed-sensitive part, and could not check a single reported instance at all.

This rebuilds each cell exactly as scripts/run_reference_decomposition.py does -- the same
`rebuild_cell`, so there is one implementation of what an attacked flow is -- and keeps only
$x'$, with the flags that say whether this rebuild landed on the same shown case the cached
reader decision was made on. Those flags matter: the detector fit is not bit-identical to the
August run, and erasure deltas within rounding of $\\tau$ can move a causal-set member, so a
handful of instances rebuild to a different case. Shipping them unflagged would invite a
reader to check a number against a flow that never produced it.

It writes only results/reference_decomposition/raw/attacked_flows.json. It does not touch
instances.json, and therefore cannot move a published number.

  python scripts/export_attacked_flows.py
  python scripts/export_attacked_flows.py --datasets fiveg_nidd --seeds 0
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

log = logging.getLogger("attacked_flows")

REPO = Path(__file__).resolve().parents[1]
OUT = RESULTS_ROOT / "reference_decomposition" / "raw" / "attacked_flows.json"
NAME = "attacked_flows_export"


def _refdecomp():
    spec = importlib.util.spec_from_file_location(
        "run_reference_decomposition", REPO / "scripts" / "run_reference_decomposition.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sweep(ctx) -> RunResult:
    args = ctx.config["args"]
    rd = _refdecomp()
    rd.ATTACKS[:] = args["attacks"]
    cache = json.loads(CASE_CACHE.read_text())["cases"]

    out, mismatched, feature_order = [], 0, {}
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
                                  scaffold_contamination=args["scaffold_contamination"])
                rows = rd.rebuild_cell(bench, data, c, cached_cell, args["tau"], args["min_k"],
                                       args["reference"], keep_attacked=True)
                for r in rows:
                    mismatched += not r["shown_matches_cache"]
                    out.append({"dataset": ds, "class": cname, "seed": seed,
                                "sample_id": r["sample_id"], "attack": r["attack"],
                                "prediction_preserved": r["prediction_preserved"],
                                "shown_matches_cache": r["shown_matches_cache"],
                                "routed_to_surrogate": r["routed_to_surrogate"],
                                "attack_returned_unchanged": r["attack_returned_unchanged"],
                                "x_prime": r["x_prime"]})
                OUT.parent.mkdir(parents=True, exist_ok=True)
                OUT.write_text(json.dumps(
                    {"note": "Attacked flow vectors in the feature order of "
                             "artifacts/models/manifest.json for this dataset. "
                             "shown_matches_cache=false means this rebuild landed on a "
                             "different shown case than the cached reader decision, so the "
                             "flow is not the one that decision was made on.",
                     "feature_order": feature_order,
                     "flows": out}))
                log.info("done %s class=%s seed=%s (%d flows so far, %d flagged)",
                         ds, cname, seed, len(out), mismatched)

    log.info("wrote %d attacked flows to %s (%d flagged as not matching the cache)",
             len(out), OUT, mismatched)
    unchanged = {atk: int(sum(r["attack_returned_unchanged"] for r in out
                              if r["attack"] == atk)) for atk in args["attacks"]}
    per_attack = len(out) // max(1, len(args["attacks"]))
    log.info("attack returned the flow unchanged on %s of %d case sets per attack",
             unchanged, per_attack)
    return RunResult(
        summary={"flows": len(out), "shown_mismatch": mismatched,
                 "returned_unchanged": unchanged, "case_sets_per_attack": per_attack,
                 "datasets": args["datasets"], "seeds": args["seeds"],
                 "attacks": args["attacks"]},
        floats=[{"float_id": "attacked_flows", "kind": "raw", "paper_section": "release",
                 "path": str(OUT),
                 "claim": "the attacked flow behind every rescored instance, so an attack "
                          "can be checked rather than re-run"}])


def main():
    rd = _refdecomp()
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=rd.DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=rd.SEEDS)
    ap.add_argument("--attacks", nargs="+", default=rd.ALL_ATTACKS, choices=rd.ALL_ATTACKS)
    ap.add_argument("--reference", choices=["mean", "median"], default="mean")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--scaffold-contamination", type=float, default=0.3)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(NAME, sweep, config={"args": {
        "datasets": a.datasets, "seeds": a.seeds, "attacks": a.attacks,
        "reference": a.reference, "tau": a.tau, "min_k": a.min_k,
        "scaffold_contamination": a.scaffold_contamination}}, seed=0)


if __name__ == "__main__":
    main()
