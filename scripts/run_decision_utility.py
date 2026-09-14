"""C2 — does an explanation attack change what an autonomous triage agent does?

The paper's headline experiment, in two phases so nothing expensive runs twice.

  Phase 1 (CPU): build the matched decision cases -- clean / attacked / repaired /
    no-explanation -- for every (dataset, class, seed, attack) cell and cache them.
    Generating the displacement attack is the slow part of the whole project, so it is
    done once and shared by every judge.
  Phase 2 (GPU): each judge scores the cached cases. Adding a judge later re-reads the
    cache instead of re-attacking, and every judge sees byte-identical inputs, which is
    what makes the two-judge comparison a comparison rather than two experiments.

Each case is scored with ONE forward pass, from which both the deterministic decision
(argmax) and the sampled ones are derived. Two sampled passes over the identical clean
case, differing only in seed, give the clean-vs-clean control: the judge's own noise
floor, which the attacked-vs-clean flip rate must clear before the result means
anything.

  python scripts/run_decision_utility.py --judge Qwen/Qwen3-8B --judge microsoft/phi-4
  python scripts/run_decision_utility.py --build-only          # phase 1 alone
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import defaultdict
from dataclasses import asdict

import numpy as np

from avert.benchmark import XIntBench
from avert.benchmark.harness import target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.decision_utility import (
    CASE_CACHE as CACHE,
)
from avert.eval.decision_utility import (
    CONDITIONS,
    DecisionCase,
    LocalLLMJudge,
    cache_fingerprint,
    competent_cell_set,
    decide_from_logits,
    floor_sweep,
    paired_effect,
    summarize_cells,
)
from avert.eval.metrics import mean_ci
from avert.eval.runner import RunResult, run_experiment

ALL_DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
ALL_ATTACKS = ["A1_misdirection", "A1_displacement", "A3_scaffolding"]


def stratified_slice(cached, n, seed=0):
    """A subsample that actually covers every attack class.

    The cache is ordered with attacks repeating on a period of 3, so slicing with a stride
    that is a multiple of 3 -- which len(cached)//n very easily is -- aliases onto a single
    attack class and silently drops the rest. That happened: a stride of 9 gave 4800
    A1_misdirection rows and zero A1_displacement, the attack that carries the entire result,
    and the arms built on it looked perfectly reasonable.

    Sample within each attack class instead, deterministically.
    """
    import random
    by_attack = defaultdict(list)
    for rec in cached:
        by_attack[rec.get("attack", "?")].append(rec)
    per = max(1, n // max(1, len(by_attack)))
    out = []
    for atk in sorted(by_attack):
        group = by_attack[atk]
        rng = random.Random(f"{seed}-{atk}")
        out.extend(group if len(group) <= per else rng.sample(group, per))
    return out


def stable_offset(key: str, mod: int = 10_000) -> int:
    """A per-case shuffle offset that is identical across processes and runs."""
    return int(hashlib.sha256(key.encode()).hexdigest()[:8], 16) % mod





# ---------------------------------------------------------------- phase 1: cases

def build_cases(datasets, seeds, n_classes, n_test, n_shown):
    CACHE.parent.mkdir(parents=True, exist_ok=True)
    fp = cache_fingerprint(datasets, seeds, n_classes, n_test, n_shown)
    out = []
    for ds in datasets:
        data = load_dataset(ds, load_config(f"datasets/{ds}"))
        supported, _excluded = target_classes(data, n_classes, n_test=n_test, seeds=seeds)
        for c in supported:
            cname = data.label_names[c] if data.label_names else str(c)
            for seed in seeds:
                bench = XIntBench(data, n_cal=100, n_test=n_test, top_k=5,
                                  perm_samples=10, stability_n=30, seed=seed)
                try:
                    items = bench.decision_cases(ALL_ATTACKS, target=c, n_shown=n_shown)
                except Exception as e:
                    print(f"  FAIL cases {ds}/{cname}/s{seed}: {type(e).__name__}: {str(e)[:140]}",
                          flush=True)
                    continue
                for it in items:
                    out.append({
                        "dataset": ds, "class": cname, "seed": seed, "attack": it["attack"],
                        "sample_id": it["sample_id"],
                        "prediction_preserved": it["prediction_preserved"],
                        "cases": {k: asdict(v) for k, v in it["cases"].items()},
                    })
                CACHE.write_text(json.dumps({"fingerprint": fp, "cases": out}))
                print(f"  cases {ds}/{cname}/s{seed}: {len(items)} case-sets "
                      f"({len(out)} total)", flush=True)
    return out


# ---------------------------------------------------------------- phase 2: judging

def judge_cases(judge, cached, temperature, variant="default", shuffle_seed=None):
    rows = []
    for i, rec in enumerate(cached):
        for cond in CONDITIONS:
            payload = rec["cases"].get(cond)
            if payload is None:
                continue
            case = DecisionCase(**payload)
            if shuffle_seed is not None:
                # Break the confound between "followed the attribution" and "answered A".
                # Per-case offset uses a STABLE digest, not hash(): Python randomises string
                # hashing per process, so hash() here would give a different permutation on
                # every run and the arm would not be reproducible. set_seed sets
                # PYTHONHASHSEED but only for child processes, not this one.
                case = case.shuffled(shuffle_seed + stable_offset(case.sample_id))
            logits = judge.letter_logits(case, variant)
            det = decide_from_logits(case, logits, None)
            row = {"judge": judge.name, "dataset": rec["dataset"], "class": rec["class"],
                   "seed": rec["seed"], "attack": rec["attack"], "sample_id": rec["sample_id"],
                   "condition": cond, "causal_visible": case.causal_visible,
                   "prediction_preserved": rec["prediction_preserved"],
                   "variant": variant,
                   "order": "shuffled" if shuffle_seed is not None else "ranked",
                   "det_choice": det.choice, "det_correct": det.correct,
                   "det_choice_idx": det.choice_idx,     # for the position-bias statistic
                   "det_confidence": round(max(det.probs), 5)}
            for tag, s in (("a", rec["seed"] * 1000 + 1), ("b", rec["seed"] * 1000 + 2)):
                d = decide_from_logits(case, logits, temperature, s)
                row[f"smp{tag}_choice"], row[f"smp{tag}_correct"] = d.choice, d.correct
            rows.append(row)
        if (i + 1) % 200 == 0:
            print(f"    {judge.name}/{variant}: {i + 1}/{len(cached)} case-sets", flush=True)
    return rows


# ---------------------------------------------------------------- analysis

def aggregate(per_cell, competence_floor):
    """Mean +/- 95% CI per (judge, dataset, attack).

    Reported twice: over all cells, and over cells where the judge clears a
    clean-condition competence floor. The floor was declared in
    the design notes BEFORE this run, because a judge that cannot answer
    from a clean explanation has no decision quality left for an attack to degrade --
    and filtering cells after seeing the data would be exactly the selective reporting
    the plan rules exist to prevent. Both numbers are always shown.
    """
    def block(cells, title):
        agg = defaultdict(lambda: defaultdict(list))
        for c in cells:
            for k, v in c.items():
                if "__" in k:
                    agg[(c["judge"], c["dataset"], c["attack"])][k].append(v)
        lines = [f"### {title}", "",
                 "| Judge | Dataset | Attack | clean acc | attacked acc | repaired acc | "
                 "rank-withheld acc | attacked flip | control flip | judge conf. | cells |",
                 "|---|---|---|---|---|---|---|---|---|---|---|"]
        for (j, ds, atk), m in sorted(agg.items()):
            def f(k):
                v = [x for x in m.get(k, []) if not np.isnan(x)]
                if not v:
                    return "n/a"
                mu, hw = mean_ci(np.array(v))
                return f"{mu:.2f}±{hw:.2f}"
            lines.append(f"| {j} | {ds} | {atk} | {f('clean__acc')} | {f('attacked__acc')} | "
                         f"{f('repaired__acc')} | {f('no_explanation__acc')} | "
                         f"{f('attacked__flip')} | {f('control__flip_smp')} | "
                         f"{f('clean__confidence')} | {len(m.get('clean__acc', []))} |")
        return lines

    competent = [c for c in per_cell if c.get("clean__acc", 0) >= competence_floor]
    lines = ["# C2 — agent decision utility", "",
             "acc = decision accuracy against interventional ground truth. flip = the decision",
             "changed relative to the matched clean decision. `control` is the identical clean",
             "case sampled twice — the judge's own noise floor. A flip rate at or below control",
             "carries no claim.", ""]
    lines += block(per_cell, "All cells")
    lines += ["", f"({len(competent)}/{len(per_cell)} cells clear the pre-declared "
                  f"clean-accuracy floor of {competence_floor})", ""]
    lines += block(competent, f"Cells where the judge is competent (clean acc >= {competence_floor})")
    return "\n".join(lines)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--judge", action="append", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-classes", type=int, default=3)
    ap.add_argument("--n-test", type=int, default=40)
    ap.add_argument("--n-shown", type=int, default=10)
    ap.add_argument("--temperature", type=float, default=0.7)
    ap.add_argument("--competence-floor", type=float, default=0.5)
    ap.add_argument("--shuffle-n", type=int, default=600,
                    help="case-sets re-scored with the candidate order permuted; separates "
                         "attribution-following from answering-A (0 disables)")
    ap.add_argument("--sensitivity-n", type=int, default=300,
                    help="case-sets re-scored under each alternative prompt wording "
                         "(0 disables the prompt-sensitivity arm)")
    ap.add_argument("--rebuild", action="store_true", help="ignore the cached cases")
    ap.add_argument("--build-only", action="store_true")
    ap.add_argument("--name", default="decision_utility")
    args = ap.parse_args()

    datasets = args.dataset or ALL_DATASETS
    judges = args.judge or ["Qwen/Qwen3-8B", "microsoft/phi-4"]

    want = cache_fingerprint(datasets, args.seeds, args.n_classes, args.n_test, args.n_shown)
    cached = None
    if not args.rebuild and CACHE.exists():
        blob = json.loads(CACHE.read_text())
        # A bare list is a pre-fingerprint cache: treat it as unusable rather than assume
        # it was built with these parameters.
        if isinstance(blob, dict) and blob.get("fingerprint") == want:
            cached = blob["cases"]
            print(f"[phase 1] reusing {len(cached)} cached case-sets from {CACHE}", flush=True)
        else:
            print("[phase 1] cached cases do not match these parameters -- rebuilding",
                  flush=True)
    if cached is None:
        print("[phase 1] building decision cases", flush=True)
        cached = build_cases(datasets, args.seeds, args.n_classes, args.n_test, args.n_shown)
    if args.build_only:
        return

    def experiment(ctx):
        rows, sens, shuf = [], [], []
        for judge_id in judges:
            print(f"[phase 2] {judge_id}", flush=True)
            judge = LocalLLMJudge(judge_id, temperature=args.temperature,
                                  max_candidates=args.n_shown)
            rows.extend(judge_cases(judge, cached, args.temperature))
            (ctx.run_dir / "raw" / "decisions.json").write_text(json.dumps(rows))
            # Prompt-sensitivity arm: the same cases, re-worded. If the conclusion moves
            # between phrasings it was never about the attack. A stratified slice keeps
            # this cheap -- it is a robustness check, not a second experiment.
            # Shuffled-order arm. Same cases, same values, same attributions, order
            # permuted. If the effect survives, the agent was reading the explanation.
            if args.shuffle_n:
                sl = stratified_slice(cached, args.shuffle_n, seed=17)
                shuf.extend(judge_cases(judge, sl, args.temperature, shuffle_seed=17))
                (ctx.run_dir / "raw" / "shuffled.json").write_text(json.dumps(shuf))
            if args.sensitivity_n:
                slice_ = stratified_slice(cached, args.sensitivity_n, seed=23)
                for variant in ("terse", "analyst"):
                    sens.extend(judge_cases(judge, slice_, args.temperature, variant))
                (ctx.run_dir / "raw" / "sensitivity.json").write_text(json.dumps(sens))
            del judge
            import gc
            import torch
            gc.collect()
            torch.cuda.empty_cache()

        per_cell = summarize_cells(rows)
        table = aggregate(per_cell, args.competence_floor)

        # The pre-registered effect size: paired per instance, cluster-bootstrapped by
        # cell. Flows inside a cell share a detector and one attack instantiation, so a
        # paired t-CI over all of them is roughly 2.5x too narrow on this data.
        competent = competent_cell_set(per_cell, args.competence_floor)
        eff_lines = ["", "### Paired effect on the decision", "",
                     "Each matched instance contributes correct(condition) - correct(clean).",
                     "`broke` counts decisions the condition turned from right to wrong, `fixed`",
                     "the reverse. Intervals are cluster bootstraps over cells, not paired t-CIs:",
                     "flows within a cell share a fitted detector and one attack instantiation, so",
                     "treating them as independent understates the interval by about 2.5x.", "",
                     "| Scope | Judge | Condition | n | cells | mean delta | 95% CI | broke | fixed | harms? |",
                     "|---|---|---|---|---|---|---|---|---|---|"]
        for scope, cset in (("all", None), (f"clean>={args.competence_floor}", competent)):
            for j in [None] + sorted({r["judge"] for r in rows}):
                for cond in ("attacked", "repaired", "no_explanation"):
                    e = paired_effect(rows, cond, judge=j, attack="A1_displacement" if cond != "no_explanation" else None,
                                      competent_cells=cset)
                    if not e.get("n"):
                        continue
                    eff_lines.append(
                        f"| {scope} | {j or 'both'} | {cond} | {e['n']} | {e['n_clusters']} | "
                        f"{e['mean_delta']:+.3f} | [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] | "
                        f"{e['broke']} | {e['fixed']} | {'yes' if e['harms'] else 'no'} |")

        # Position bias, and whether the effect survives breaking it.
        from avert.eval.decision_utility import position_bias
        eff_lines += ["", "### Is the agent reading the attribution, or answering A?", "",
                      "Options are listed in attribution order, and the attack re-orders them, so",
                      "position is confounded with the treatment. The shuffled arm permutes the",
                      "candidate order while each feature keeps its own value and attribution.", "",
                      "| Judge | Order | P(picks first option) | attacked effect | 95% CI |",
                      "|---|---|---|---|---|"]
        for j in sorted({r["judge"] for r in rows}):
            for label, rr in (("ranked", rows), ("shuffled", shuf)):
                if not rr:
                    continue
                pb = position_bias(rr, judge=j)
                e = paired_effect(rr, "attacked", judge=j, attack="A1_displacement")
                if not e.get("n"):
                    continue
                eff_lines.append(f"| {j} | {label} | {pb.get('p_first_overall', float('nan')):.3f} | "
                                 f"{e['mean_delta']:+.3f} | [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] |")

        # Robustness of the floor itself. It was fixed before the confirmatory run but
        # after a pilot exposed the problem, so a reviewer is entitled to ask whether the
        # value was chosen to produce the answer. Sweeping it replies with a measurement.
        eff_lines += ["", "### Does the finding depend on where the floor was set?", "",
                      "| floor | cells | n | mean delta | 95% CI | harms? |", "|---|---|---|---|---|---|"]
        for r_ in floor_sweep(rows, per_cell):
            eff_lines.append(f"| {r_['floor']:.1f} | {r_['n_clusters']} | {r_['n']} | "
                             f"{r_['mean_delta']:+.3f} | [{r_['ci_lo']:+.3f}, {r_['ci_hi']:+.3f}] | "
                             f"{'yes' if r_['harms'] else 'no'} |")
        table += "\n".join(eff_lines)
        if sens:
            base = {(r["judge"], r["sample_id"], r["attack"], r["condition"]): r["det_correct"]
                    for r in rows}
            lines = ["", "### Prompt sensitivity", "",
                     "The same decisions re-scored under two alternative wordings of the identical",
                     "question. `agreement` is how often the re-worded prompt yields the same",
                     "correctness as the default; `acc` is accuracy under that wording.", "",
                     "| Judge | Wording | n | agreement with default | acc |", "|---|---|---|---|---|"]
            byv = defaultdict(list)
            for r in sens:
                byv[(r["judge"], r["variant"])].append(r)
            for (j, v), rs in sorted(byv.items()):
                agree = [r["det_correct"] == base.get((j, r["sample_id"], r["attack"], r["condition"]))
                         for r in rs if (j, r["sample_id"], r["attack"], r["condition"]) in base]
                lines.append(f"| {j} | {v} | {len(rs)} | {np.mean(agree):.3f} | "
                             f"{np.mean([r['det_correct'] for r in rs]):.3f} |")
            table += "\n".join(lines)
        (ctx.run_dir / "summary" / "per_cell.json").write_text(json.dumps(per_cell, indent=1))
        (ctx.run_dir / "summary" / "decision_utility.md").write_text(table)
        print("\n" + table)
        return RunResult(
            summary={"n_decisions": len(rows), "n_sensitivity": len(sens),
                     "n_shuffled": len(shuf),
                     "n_cells": len(per_cell), "judges": judges,
                     "competence_floor": args.competence_floor},
            floats=[{"float_id": "Fig3_Tab4_decision_utility", "kind": "figure+table",
                     "paper_section": "C2",
                     "path": str(ctx.run_dir / "summary" / "decision_utility.md"),
                     "claim": "prediction-preserving explanation attacks change the triage agent's decision",
                     "status": "done"}],
        )

    run_experiment(args.name, experiment,
                   config={"datasets": datasets, "judges": judges, "seeds": args.seeds,
                           "n_test": args.n_test, "n_shown": args.n_shown,
                           "temperature": args.temperature,
                           "competence_floor": args.competence_floor},
                   seed=args.seeds[0])


if __name__ == "__main__":
    main()
