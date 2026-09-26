"""Reader harm on the random-draw arm, beside the attack's, on the same case sets.

Cause displacement moves the causal set, and so does a random realizable perturbation of the
same budget (results/baseline_random_draw). Whether the reader harm is the attack's or any
perturbation's needs the reader run on the random-draw flows. Both readers were run on them by

    python scripts/run_decision_utility.py --cases results/baseline_random_draw/raw/cases.json \
        --name decision_utility_baseline --conditions clean attacked --shuffle-n 0 --sensitivity-n 0

and this rescores those decisions exactly as the attack's are rescored.

Population. A case set counts when it is usable in both arms: the attack's rescoring keeps it
(shown case matches the cache, class preserved; scripts/run_reference_decomposition.py
`rescored_rows`) and the random draw preserved the class. Both arms are then computed on the
same case sets and the same clean decisions -- the cached ones, which the readers already made
-- so the only thing that differs between the arms is the attacked flow.

Arms, each against S(x) (the clean causal set, cached) and S(x') (the causal set of the flow
shown, the attack's or the random draw's own):

  all reader--cells, all flows      no competence floor, every usable case set
  competent on A, measured on B     the split sample of scripts/report_split_sample_harm.py,
                                    same halves, competence from the clean decisions on half A
  not competent on A, on B          the complement
  all reader--cells, half B only    the unselected estimate on the measurement half

The attack-specific harm is the paired difference, attack minus random draw, per reader and
case set: correct(attack) - correct(random draw), the clean term cancelling. Its interval is
a cluster bootstrap over cells (both readers in one cluster), 8,000 resamples, as every harm
interval in the paper.

The clean decisions are re-judged in the same run; how often the re-judged clean choice equals
the cached one is recorded as a check on the readers' determinism.

  python scripts/report_baseline_reader_harm.py
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

log = logging.getLogger("baseline_harm")

REPO = Path(__file__).resolve().parents[1]
NAME = "decision_utility_baseline"
ATTACK = "A1_displacement"
MAIN_DECISIONS = RESULTS_ROOT / "decision_utility" / "raw" / "decisions.json"
MAIN_INST = RESULTS_ROOT / "reference_decomposition" / "raw" / "instances.json"
BASE_CASES = RESULTS_ROOT / "baseline_random_draw" / "raw" / "cases.json"
BASE_DECISIONS = RESULTS_ROOT / NAME / "raw" / "decisions.json"


def _load(name):
    spec = importlib.util.spec_from_file_location(name, REPO / "scripts" / f"{name}.py")
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def flow_key(r):
    return (r["dataset"], r["class"], r["seed"], r["sample_id"])


def cell_key(r):
    return (r["dataset"], r["class"], r["seed"])


def paired_difference(rows_a, rows_b, keep=None, n_boot=8000, seed=0) -> dict:
    """Mean of correct(A) - correct(B) over matched (reader, case set), cluster bootstrap by cell."""
    b = {(r["judge"],) + flow_key(r): int(r["det_correct"]) for r in rows_b
         if r["condition"] == "attacked"}
    clusters = defaultdict(list)
    for r in rows_a:
        if r["condition"] != "attacked":
            continue
        k = (r["judge"],) + flow_key(r)
        if k not in b or (keep is not None and not keep(r)):
            continue
        clusters[cell_key(r)].append(int(r["det_correct"]) - b[k])
    if not clusters:
        return {"n": 0}
    a = np.concatenate([np.asarray(v, float) for v in clusters.values()])
    rng = np.random.default_rng(seed)
    keys = list(clusters)
    means = np.empty(n_boot)
    for i in range(n_boot):
        pick = rng.integers(0, len(keys), len(keys))
        means[i] = np.concatenate([clusters[keys[j]] for j in pick]).mean()
    lo, hi = (float(x) for x in np.percentile(means, [2.5, 97.5]))
    return {"n": int(len(a)), "n_clusters": len(keys), "mean_delta": float(a.mean()),
            "ci_lo": lo, "ci_hi": hi}


def sweep(ctx) -> RunResult:
    args = ctx.config["args"]
    rd = _load("run_reference_decomposition")
    sp = _load("report_split_sample_harm")

    main = json.loads(MAIN_DECISIONS.read_text())
    inst = json.loads(MAIN_INST.read_text())
    keep_x, keep_xp = rd.rescored_rows(main, inst)
    atk_rows = {"vs_S_x": [r for r in keep_x if r["attack"] == ATTACK],
                "vs_S_xprime": [r for r in keep_xp if r["attack"] == ATTACK]}

    cases = {flow_key(c): c for c in json.loads(BASE_CASES.read_text())}
    new = json.loads(BASE_DECISIONS.read_text())

    # Clean arm: the cached decisions. The re-judged clean decisions only check determinism.
    cached_clean = {(r["judge"],) + flow_key(r): r for r in atk_rows["vs_S_x"]
                    if r["condition"] == "clean"}
    rejudged = [r for r in new if r["condition"] == "clean"]
    same = [r["det_choice"] == cached_clean[(r["judge"],) + flow_key(r)]["det_choice"]
            for r in rejudged if (r["judge"],) + flow_key(r) in cached_clean]

    usable_atk = {flow_key(r) for r in atk_rows["vs_S_x"] if r["condition"] == "attacked"}
    usable_base = {k for k, c in cases.items() if c["prediction_preserved"]}
    common = usable_atk & usable_base

    def restrict(rows):
        return [r for r in rows if flow_key(r) in common]

    base_rows = {"vs_S_x": [], "vs_S_xprime": []}
    for r in new:
        if r["condition"] != "attacked" or flow_key(r) not in common:
            continue
        base_rows["vs_S_x"].append(r)
        base_rows["vs_S_xprime"].append(dict(r, det_correct=r["det_choice"]
                                             in set(cases[flow_key(r)]["S_a"])))
    for ref in base_rows:
        clean = [r for r in atk_rows[ref] if r["condition"] == "clean"]
        base_rows[ref] = restrict(clean) + base_rows[ref]
        atk_rows[ref] = restrict(atk_rows[ref])

    A, B = sp.halves(inst, seed=args["split_seed"])
    in_B = lambda r: r["sample_id"] in B.get(cell_key(r), ())   # noqa: E731
    acc_A = sp.clean_accuracy_on(atk_rows["vs_S_x"], A)
    competent = {k for k, v in acc_A.items() if v >= args["floor"]}
    reader_cells = set(acc_A)

    def arms(rows):
        rows_B = [r for r in rows if in_B(r)]
        pe = lambda rr, cc=None: paired_effect(rr, "attacked", attack=ATTACK,   # noqa: E731
                                              competent_cells=cc, cluster="cell",
                                              n_boot=args["n_boot"], seed=args["seed"])
        return {"all_cells": pe(rows), "all_cells_on_B": pe(rows_B),
                "competent_on_A": pe(rows_B, competent),
                "not_competent_on_A": pe(rows_B, reader_cells - competent)}

    res = {}
    for ref in ("vs_S_x", "vs_S_xprime"):
        a, b = atk_rows[ref], base_rows[ref]
        comp = lambda r: sp.reader_cell(r) in competent   # noqa: E731
        res[ref] = {
            "attack": arms(a), "random_draw": arms(b),
            "attack_minus_random": {
                "all_cells": paired_difference(a, b, n_boot=args["n_boot"], seed=args["seed"]),
                "all_cells_on_B": paired_difference(a, b, keep=in_B, n_boot=args["n_boot"],
                                                    seed=args["seed"]),
                "competent_on_A": paired_difference(a, b, keep=lambda r: in_B(r) and comp(r),
                                                    n_boot=args["n_boot"], seed=args["seed"]),
                "not_competent_on_A": paired_difference(
                    a, b, keep=lambda r: in_B(r) and not comp(r), n_boot=args["n_boot"],
                    seed=args["seed"])}}
    summary = {"population": {"case_sets_common": len(common),
                              "attack_usable": len(usable_atk), "random_usable": len(usable_base),
                              "reader_cells": len(reader_cells),
                              "competent_on_A": len(competent & reader_cells)},
               "clean_rejudged_same_choice": float(np.mean(same)) if same else float("nan"),
               "clean_rejudged_n": len(same),
               "effects": res}
    (ctx.run_dir / "summary" / "harm.json").write_text(json.dumps(summary, indent=1))

    L = ["# Reader harm: cause displacement against its random-draw control", "",
         f"Case sets usable in both arms: {len(common)} (attack {len(usable_atk)}, random draw "
         f"{len(usable_base)}). Re-judged clean choice equals the cached one on "
         f"{summary['clean_rejudged_same_choice']:.4f} of {len(same)}.", "",
         "| Reference | Arm | population | n | effect | 95% CI |", "|---|---|---|---|---|---|"]
    for ref, per in res.items():
        for arm, pops in per.items():
            for pop, e in pops.items():
                if e.get("n"):
                    L.append(f"| {ref} | {arm} | {pop} | {e['n']} | {e['mean_delta']:+.3f} | "
                             f"[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] |")
    md = "\n".join(L)
    (ctx.run_dir / "summary" / "harm.md").write_text(md)
    print(md)
    d = res["vs_S_xprime"]["attack_minus_random"]["all_cells"]
    return RunResult(
        summary={"case_sets_common": len(common),
                 "attack_minus_random_all_vs_Sxprime": d.get("mean_delta"),
                 "ci": [d.get("ci_lo"), d.get("ci_hi")]},
        floats=[{"float_id": "Tab_harm_random_draw", "kind": "table", "paper_section": "harm",
                 "path": str(ctx.run_dir / "summary" / "harm.json"),
                 "claim": "reader harm of cause displacement beside the random-draw control on "
                          "the same case sets, and their paired difference"}])


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
