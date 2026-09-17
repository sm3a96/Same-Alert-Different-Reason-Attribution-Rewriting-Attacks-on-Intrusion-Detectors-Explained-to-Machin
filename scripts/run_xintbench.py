"""Run EIB against a pipeline and emit a vulnerability report (the product entry
point). Default pipeline is XGBoost + TreeSHAP; a vendor swaps in their own.

  python scripts/run_xintbench.py --dataset fiveg_nidd
  python scripts/run_xintbench.py --dataset ciciomt2024
"""
from __future__ import annotations

import argparse
import json

from avert.benchmark.harness import XIntBench
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.runner import RunResult, run_experiment


def experiment(ctx) -> RunResult:
    data = load_dataset(ctx.config["dataset"], ctx.config)
    bench = XIntBench(data, alpha=ctx.config.get("alpha", 0.05), seed=ctx.seed)
    report = bench.evaluate_pipeline()                 # default: XGBoost + TreeSHAP
    print("\n" + report.summary() + "\n")

    out = ctx.run_dir / "summary" / "report.json"
    out.write_text(json.dumps(report.to_dict(), indent=2))
    return RunResult(summary=report.to_dict(), floats=[{
        "float_id": f"xintbench_{data.name}", "kind": "table", "paper_section": "Evaluation",
        "path": str(out), "claim": "EIB vulnerability report: attack efficacy + monitor detection per attack"}])


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="fiveg_nidd")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    cfg = {"dataset": args.dataset, "alpha": 0.05}
    if args.dataset != "synthetic":
        cfg.update(load_config(f"datasets/{args.dataset}"))
        cfg["dataset"] = args.dataset
    run_experiment(f"xintbench_{args.dataset}", experiment, cfg, seed=args.seed)
