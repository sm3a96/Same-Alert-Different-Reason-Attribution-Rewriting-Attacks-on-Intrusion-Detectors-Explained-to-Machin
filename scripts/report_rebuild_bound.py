"""How far the S(x) arm moves when S(x) is recomputed at rescoring time.

The reader cases carry the clean causal set S(x) computed when the cases were built. The
rescoring rebuilds every cell, and the rebuilt S(x) differs from the cached one on a few
instances, because erasure deltas within rounding of tau can flip a member. The paper's S(x)
arm uses the cached set, so it scores each decision against the set the case carried. This
reports the bound on that choice: the same decisions scored against the rebuilt S(x) instead,
clean and attacked alike, on the same usable instances, with the same cell-clustered
bootstrap, and the difference on each quoted population.

  python scripts/report_rebuild_bound.py
"""
from __future__ import annotations

import argparse
import importlib.util
import json
import logging
from pathlib import Path

from avert.eval.decision_utility import competent_cell_set, paired_effect, summarize_cells
from avert.eval.runner import RESULTS_ROOT, RunResult, run_experiment

log = logging.getLogger("rebuild_bound")

REPO = Path(__file__).resolve().parents[1]
NAME = "rebuild_bound"
DECISIONS = RESULTS_ROOT / "decision_utility" / "raw" / "decisions.json"
INST = RESULTS_ROOT / "reference_decomposition" / "raw" / "instances.json"
ATTACKS = ["A1_displacement", "A1_misdirection"]


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def sweep(ctx) -> RunResult:
    args = ctx.config["args"]
    rd = _load("run_reference_decomposition")
    decisions = json.loads(DECISIONS.read_text())
    inst = json.loads(INST.read_text())
    keep, _ = rd.rescored_rows(decisions, inst)
    rebuilt = {(r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]): set(r["S_x"])
               for r in inst}
    n_differ = sum(sorted(r["S_x"]) != sorted(r["S_x_cached"]) for r in inst
                   if r["attack"] in ATTACKS)
    alt = []
    for r in keep:
        k = (r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"])
        alt.append(dict(r, det_correct=r["det_choice"] in rebuilt[k]))

    per_cell = summarize_cells([r for r in decisions if r["variant"] == "default"
                                and r["order"] == "ranked"])
    competent = competent_cell_set(per_cell, args["floor"])
    out = {}
    for atk in ATTACKS:
        out[atk] = {}
        for scope, cc in (("all_cells", None), ("competent_in_sample", competent)):
            a = paired_effect(keep, "attacked", attack=atk, competent_cells=cc, cluster="cell",
                              n_boot=args["n_boot"], seed=args["seed"])
            b = paired_effect(alt, "attacked", attack=atk, competent_cells=cc, cluster="cell",
                              n_boot=args["n_boot"], seed=args["seed"])
            out[atk][scope] = {"cached_S_x": a, "rebuilt_S_x": b,
                               "difference": b["mean_delta"] - a["mean_delta"]}
    res = {"instances_S_x_differs": int(n_differ), "effects": out}
    (ctx.run_dir / "summary" / "bound.json").write_text(json.dumps(res, indent=1))
    L = ["# S(x) arm with the rebuilt clean causal set", "",
         f"Instances whose rebuilt S(x) differs from the cached one (two search attacks): {n_differ}", "",
         "| Attack | Scope | cached S(x) | rebuilt S(x) | difference |", "|---|---|---|---|---|"]
    for atk, per in out.items():
        for scope, v in per.items():
            L.append(f"| {atk} | {scope} | {v['cached_S_x']['mean_delta']:+.4f} | "
                     f"{v['rebuilt_S_x']['mean_delta']:+.4f} | {v['difference']:+.4f} |")
    md = "\n".join(L)
    (ctx.run_dir / "summary" / "report.md").write_text(md)
    print(md)
    return RunResult(
        summary={"instances_S_x_differs": int(n_differ),
                 "displacement_all_cells_difference": out["A1_displacement"]["all_cells"]["difference"]},
        floats=[{"float_id": "rebuild_bound", "kind": "summary", "paper_section": "harm",
                 "path": str(ctx.run_dir / "summary" / "bound.json"),
                 "claim": "how far the S(x) headline moves when S(x) is recomputed at rescoring"}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--floor", type=float, default=0.5)
    ap.add_argument("--n-boot", type=int, default=8000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run_experiment(NAME, sweep, config={"args": {"floor": a.floor, "n_boot": a.n_boot,
                                                 "seed": a.seed}}, seed=a.seed)


if __name__ == "__main__":
    main()
