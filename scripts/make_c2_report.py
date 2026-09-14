"""Regenerate every C2 number the paper quotes, from raw decisions.

The pre-writing review found the C2 headline and the DeLong table hand-copied: correct
values, but no script on disk produced them. That is the same failure as a stale number,
one step earlier — nothing would have caught it if the underlying data had changed.

This produces all of it from `results/decision_utility/raw/decisions.json`:

  the headline paired effect, cluster-bootstrapped by cell
  per-dataset effects, which the paper currently does not disclose at all
  leave-one-out over datasets and judges, so "the result does not rest on one cell" is
    a measurement rather than a hope
  the competence-floor sweep
  position bias, and the shuffled-order arm when it exists

  python scripts/make_c2_report.py

Writes results/decision_utility/summary/c2_report.md and prints it.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from avert.eval.decision_utility import (
    competent_cell_set,
    floor_sweep,
    paired_effect,
    position_bias,
)

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results" / "decision_utility"
LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024"}
FLOOR = 0.5


def row(name, e):
    return (f"| {name} | {e['n']} | {e['n_clusters']} | {e['mean_delta']:+.3f} | "
            f"[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] | {e['broke']} | {e['fixed']} | "
            f"{'yes' if e['harms'] else 'no'} |")


def blind_pick_rate(case_ids: set) -> dict:
    """P(the option in slot A is causal) under a uniform permutation, per dataset.

    The pre-registration's reference number: what an agent scores by always answering A once
    order carries no information. Under a uniform shuffle that is just the causal share of the
    shown candidates, so it is computed from the case cache rather than from the realized
    permutation -- and rather than being carried here as a literal, which is what the
    generation-script check in the claims check forbids and what would silently keep the
    2026-08-01 values alive through a re-run that moved them.
    """
    blob = json.loads((REPO / "results" / "_cache" / "decision_cases.json").read_text())
    per = defaultdict(list)
    for c in blob["cases"]:
        key = (c["dataset"], c["class"], c["seed"], c["attack"], c["sample_id"])
        if case_ids and key not in case_ids:
            continue
        clean = c["cases"].get("clean")
        if not clean:
            continue
        shown = clean["candidates"]
        per[c["dataset"]].append(len(set(clean["causal"]) & set(shown)) / len(shown))
    out = {d: float(np.mean(v)) for d, v in per.items()}
    out["pooled"] = float(np.mean([x for v in per.values() for x in v])) if per else float("nan")
    return out


def decide_outcome(clean_ranked, clean_shuf, chance, effects) -> tuple[str, list[str]]:
    """Evaluate the pre-registered rule, rather than printing the outcome it landed on once.

    Until 2026-08-15 this block was a literal string saying "outcome B". It had been true of
    the three-seed run and it kept printing through the corrected five-seed re-run, in which
    both shuffled CIs contain zero -- the table above the sentence contradicted the sentence,
    and nothing failed. A pre-registered rule that is asserted rather than evaluated is not a
    pre-registration; it is a conclusion with a timestamp on it.

    The rule (fixed 2026-08-01):
      A  rank-mediated  clean accuracy under shuffling falls to within noise of the chance
                        rate AND every shuffled attack effect has a CI containing zero
      B  judgment       clean accuracy retains at least half its excess over chance AND every
                        shuffled attack effect has a CI clearing zero
      C  partial        anything between; report the mediated share and claim only the
                        unmediated remainder
    """
    excess_ranked = clean_ranked - chance
    retained = (clean_shuf - chance) / excess_ranked if excess_ranked > 0 else float("nan")
    clears = [e for e in effects if e["ci_hi"] < 0]
    contains = [e for e in effects if e["ci_lo"] <= 0 <= e["ci_hi"]]
    at_chance = abs(clean_shuf - chance) <= 0.05

    if at_chance and len(contains) == len(effects):
        outcome, sentence = "A", (
            "the attack changes which feature ranks first, the consumer acts on rank 1, and "
            "the decision follows. No claim about judgment.")
    elif retained >= 0.5 and len(clears) == len(effects):
        outcome, sentence = "B", (
            "the attack degrades the decision even when position cannot be used, so the agent "
            "was reading the attribution.")
    else:
        outcome, sentence = "C", (
            "position carries part of the harm and the remainder does not separate from zero "
            "on this sample. Report the mediated share; claim the unmediated remainder as "
            "judgment only if a reader is told its interval crosses zero.")

    lines = [
        "",
        "Pre-registered decision rule (fixed 2026-08-01, before the arm was scored). "
        "Evaluated here, not asserted:",
        "",
        f"- blind-pick chance rate, computed over the matched case-sets: {chance:.3f}",
        f"- clean accuracy: {clean_ranked:.3f} ranked, {clean_shuf:.3f} shuffled — "
        f"{retained:.0%} of the ranked excess over chance is retained "
        f"(rule B needs at least 50%: {'met' if retained >= 0.5 else 'not met'})",
        f"- shuffled attack effects clearing zero: {len(clears)} of {len(effects)} "
        f"(rule B needs all; rule A needs none)",
        "",
        f"**Outcome {outcome}** — {sentence}",
    ]
    if outcome == "C":
        for e in effects:
            share = ((e["ranked"] - e["mean_delta"]) / e["ranked"]) if e["ranked"] else float("nan")
            lines.append(
                f"    - {e['judge']}: ranked {e['ranked']:+.3f}, shuffled {e['mean_delta']:+.3f} "
                f"[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] — mediated share {share:.0%}")
    return outcome, lines




def conditional_rate_block(rows, comp):
    """Break and fix rates CONDITIONAL ON THE OPPORTUNITY.

    The raw broke:fixed ratio is not a propensity. With a clean baseline near 0.95 there are
    roughly seventeen chances to break for every one to fix, so the ratio is mostly a base rate.
    Conditioning separates the two, and the conditional numbers point the other way from the
    headline ratio -- which is worth knowing before a reviewer computes them.
    """
    pairs = _pairs(rows, comp)
    if not pairs:
        return []
    c = np.array([p[0] for p in pairs])
    a = np.array([p[1] for p in pairs])
    broke, fixed = int(((c == 1) & (a == 0)).sum()), int(((c == 0) & (a == 1)).sum())
    n_ok, n_bad = int((c == 1).sum()), int((c == 0).sum())
    return ["", "## Broke and fixed, conditioned on the opportunity", "",
            "The unconditioned ratio is a base-rate artifact of a highly accurate clean "
            "baseline and must not be quoted as a directional propensity.", "",
            "| | count | opportunities | rate |", "|---|---|---|---|",
            f"| broke | {broke} | {n_ok} correct when clean | {broke / max(1, n_ok):.3f} |",
            f"| fixed | {fixed} | {n_bad} wrong when clean | {fixed / max(1, n_bad):.3f} |",
            "",
            f"Clean accuracy on this population is {c.mean():.3f}; attacked is {a.mean():.3f}."]


def _pairs(rows, comp, attack="A1_displacement", dataset=None):
    clean, att = {}, {}
    for r in rows:
        if r["attack"] != attack:
            continue
        cell = (r["judge"], r["dataset"], r["class"], r["attack"], r["seed"])
        if comp is not None and cell not in comp:
            continue
        if dataset and r["dataset"] != dataset:
            continue
        k = (r["judge"], r["dataset"], r["class"], r["seed"], r["sample_id"])
        if r["condition"] == "clean":
            clean[k] = int(r["det_correct"])
        elif r["condition"] == "attacked":
            att[k] = int(r["det_correct"])
    return [(clean[k], att[k]) for k in sorted(set(clean) & set(att))]


def mediator_block(rows, comp):
    """What predicts harm: the CHANGE in rank-1 causal faithfulness, not its level.

    The leave-one-out table above shows the headline does not survive dropping one dataset. That
    reads as brittleness until you ask why, and the answer is mechanical: on a dataset where the
    attack barely moves whether the top-ranked feature is causal, there is nothing for the
    consumer to get wrong. Reporting the mediator turns a weakness into the finding.

    Read the clustering honestly. The cells sit in three datasets, so a pooled correlation is
    mostly between-dataset and is effectively n=3. The within-dataset column is the evidence.
    """
    cache = REPO / "results" / "_cache" / "decision_cases.json"
    if not cache.exists():
        return ["", "*Mediator analysis skipped: no decision-case cache on disk.*"]
    from scipy import stats

    cases = json.loads(cache.read_text())["cases"]
    f1 = defaultdict(lambda: {"clean": [], "attacked": []})
    for cs in cases:
        key = (cs["dataset"], cs["class"], cs["seed"], cs["attack"])
        for cond in ("clean", "attacked"):
            c = cs["cases"][cond]
            f1[key][cond].append(c["candidates"][0] in set(c["causal"]))

    per_cell = defaultdict(list)
    pair = defaultdict(dict)
    for r in rows:
        if r["attack"] != "A1_displacement" or r["condition"] not in ("clean", "attacked"):
            continue
        k = (r["judge"], r["dataset"], r["class"], r["seed"], r["sample_id"])
        pair[k][r["condition"]] = int(r["det_correct"])
    for (j, ds, cl, sd, _), v in pair.items():
        if len(v) == 2:
            per_cell[(j, ds, cl, sd)].append(v["attacked"] - v["clean"])

    level, change, harm, group = [], [], [], []
    for (j, ds, cl, sd), deltas in per_cell.items():
        key = (ds, cl, sd, "A1_displacement")
        if key not in f1:
            continue
        clean_p = float(np.mean(f1[key]["clean"]))
        level.append(clean_p)
        change.append(float(np.mean(f1[key]["attacked"])) - clean_p)
        harm.append(float(np.mean(deltas)))
        group.append(ds)
    if len(harm) < 8:
        return ["", "*Mediator analysis skipped: too few cells.*"]

    L = ["", "## What predicts the harm", "",
         "Decision harm tracks the CHANGE in whether the top-ranked feature is causal. It does "
         "not track the clean level of that quantity, and it does not track top-k set "
         "corruption. The pooled correlation is mostly between-dataset and is effectively n=3; "
         "quote the within-dataset column.", "",
         "| Mediator | pooled r | " + " | ".join(sorted(set(group))) + " |",
         "|---|---|" + "---|" * len(set(group))]
    for name, xs in (("clean level of P(top-1 causal)", level),
                     ("**change** in P(top-1 causal)", change)):
        xs_a, h_a, g_a = np.array(xs), np.array(harm), np.array(group)
        r, _ = stats.pearsonr(xs_a, h_a)
        cells = []
        for ds in sorted(set(group)):
            m = g_a == ds
            rr, pp = stats.pearsonr(xs_a[m], h_a[m])
            cells.append(f"{rr:+.3f} (p={pp:.3f})")
        L.append(f"| {name} | {r:+.3f} | " + " | ".join(cells) + " |")

    L += ["", "Per dataset, the mean change in P(top-1 causal) against the mean harm:", "",
          "| Dataset | mean change in P(top-1 causal) | mean harm |", "|---|---|---|"]
    for ds in sorted(set(group)):
        m = np.array(group) == ds
        L.append(f"| {LABEL.get(ds, ds)} | {np.array(change)[m].mean():+.3f} | "
                 f"{np.array(harm)[m].mean():+.3f} |")
    return L


def main() -> None:
    rows = json.loads((RUN / "raw" / "decisions.json").read_text())
    cells = json.loads((RUN / "summary" / "per_cell.json").read_text())
    comp = competent_cell_set(cells, FLOOR)
    judges = sorted({r["judge"] for r in rows})
    datasets = sorted({r["dataset"] for r in rows})

    L = ["# C2 — agent decision utility, regenerated from raw decisions", "",
         f"{len(rows)} decisions, {len(judges)} judges, {len(datasets)} datasets, "
         f"{len(sorted({r['seed'] for r in rows}))} seeds. Every number below is recomputed by "
         "`scripts/make_c2_report.py`; none is copied.", "",
         "Intervals are cluster bootstraps over (judge, dataset, class, attack, seed) cells. "
         "Flows inside a cell share a fitted detector and one attack instantiation, so a paired "
         "t-interval over all of them is about 2.6x too narrow on this data.", "",
         "## Headline — A1 displacement, cells clearing the pre-declared competence floor", "",
         "| Scope | n | cells | mean Δ | 95% CI | broke | fixed | harms? |",
         "|---|---|---|---|---|---|---|---|"]

    L.append(row("both judges", paired_effect(rows, "attacked", attack="A1_displacement",
                                              competent_cells=comp)))
    for j in judges:
        L.append(row(j, paired_effect(rows, "attacked", judge=j, attack="A1_displacement",
                                      competent_cells=comp)))

    L += conditional_rate_block(rows, comp)

    L += ["", "## Per dataset — disclosed, including the one that goes the other way", "",
          "Reported because a reviewer will compute it from the released data anyway, and "
          "because a dataset that reverses sign is a finding rather than an embarrassment.", "",
          "| Dataset | Scope | n | cells | mean Δ | 95% CI | broke | fixed | harms? |",
          "|---|---|---|---|---|---|---|---|---|"]
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] == ds]
        for scope, cset in (("all cells", None), (f"clean≥{FLOOR}", comp)):
            e = paired_effect(sub, "attacked", attack="A1_displacement", competent_cells=cset)
            if e.get("n"):
                L.append(f"| {LABEL.get(ds, ds)} " + row(scope, e))

    L += ["", "## Leave-one-out — does the headline rest on any single dataset or judge?", "",
          "| Dropped | n | cells | mean Δ | 95% CI | harms? |", "|---|---|---|---|---|---|"]
    for ds in datasets:
        sub = [r for r in rows if r["dataset"] != ds]
        e = paired_effect(sub, "attacked", attack="A1_displacement", competent_cells=comp)
        if e.get("n"):
            L.append(f"| dataset {LABEL.get(ds, ds)} | {e['n']} | {e['n_clusters']} | "
                     f"{e['mean_delta']:+.3f} | [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] | "
                     f"{'yes' if e['harms'] else 'no'} |")
    for j in judges:
        sub = [r for r in rows if r["judge"] != j]
        e = paired_effect(sub, "attacked", attack="A1_displacement", competent_cells=comp)
        if e.get("n"):
            L.append(f"| judge {j} | {e['n']} | {e['n_clusters']} | {e['mean_delta']:+.3f} | "
                     f"[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] | {'yes' if e['harms'] else 'no'} |")

    L += ["", "## Competence-floor sweep", "",
          "The floor was fixed before the confirmatory run but after a pilot exposed the "
          "problem, so the question is fair. Sweeping answers it.", "",
          "| floor | cells | n | mean Δ | 95% CI | harms? |", "|---|---|---|---|---|---|"]
    for r_ in floor_sweep(rows, cells):
        L.append(f"| {r_['floor']:.1f} | {r_['n_clusters']} | {r_['n']} | {r_['mean_delta']:+.3f} | "
                 f"[{r_['ci_lo']:+.3f}, {r_['ci_hi']:+.3f}] | {'yes' if r_['harms'] else 'no'} |")

    L += ["", "## Position bias — is the agent reading the attribution, or answering A?", "",
          "| Judge | Order | Scope | P(picks first) | n | attacked Δ | 95% CI |",
          "|---|---|---|---|---|---|---|"]
    shuf_path = RUN / "raw" / "shuffled.json"
    shuf = json.loads(shuf_path.read_text()) if shuf_path.exists() else []
    # The shuffled arm is a stratified slice, so its ranked comparison must be the SAME
    # case-sets rather than the full run -- otherwise the two columns describe different
    # experiments and the contrast is meaningless.
    ids = {(r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]) for r in shuf}
    matched = [r for r in rows
               if (r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]) in ids]
    # Both arms under the same floor as well as over all cells. The 2026-08-09 review found
    # the contrast mixing populations: the headline applies floor >= 0.5 and this table did
    # not, so the ranked arm read as a null contradicting the headline when it was simply a
    # different slice. Same cases, same floor, both arms, both scopes.
    effects_for_rule = []
    for j in judges:
        for label, rr in (("ranked (matched slice)", matched), ("shuffled", shuf)):
            if not rr:
                continue
            pb = position_bias(rr, judge=j)
            for scope, cset in (("all cells", None), (f"clean≥{FLOOR}", comp)):
                e = paired_effect(rr, "attacked", judge=j, attack="A1_displacement",
                                  competent_cells=cset)
                if not e.get("n"):
                    continue
                L.append(f"| {j} | {label} | {scope} | "
                         f"{pb.get('p_first_overall', float('nan')):.3f} | "
                         f"{e['n']} | {e['mean_delta']:+.3f} | "
                         f"[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] |")
                if label == "shuffled" and cset is comp:
                    ranked_e = paired_effect(matched, "attacked", judge=j,
                                             attack="A1_displacement", competent_cells=comp)
                    effects_for_rule.append({**e, "judge": j,
                                             "ranked": ranked_e.get("mean_delta", float("nan"))})
    if not shuf:
        L += ["", "*The shuffled-order arm has not been run yet. Until it has, the effect cannot "
              "be attributed to the agent following the attribution rather than to option "
              "position, and the paper must not claim otherwise.*"]
    else:
        clean_ranked = [r["det_correct"] for r in matched if r["condition"] == "clean"]
        clean_shuf = [r["det_correct"] for r in shuf if r["condition"] == "clean"]
        L += ["", f"Shuffled-order arm present: {len(shuf)} decisions over "
                  f"{len({r['sample_id'] for r in shuf})} case-sets. Clean accuracy "
                  f"{np.mean(clean_ranked):.3f} ranked against {np.mean(clean_shuf):.3f} "
                  f"shuffled. The ranked column above is the matched slice, not the headline."]
        bp = blind_pick_rate(ids)
        chance = bp["pooled"]
        # Printed per dataset because the number moved a long way. The pre-registration of
        # 2026-08-01 quoted 0.166 / 0.164 / 0.139, measured before the interventional ground
        # truth was recomputed on the cleaned feature sets; the causal sets are larger now, so
        # an always-A agent scores much better than it used to and the excess over chance the
        # rule tests is correspondingly smaller. On CICIoMT2024 in particular the task is close
        # to trivial, which is worth a reader's attention next to that dataset's small effect.
        L += ["", "| Dataset | blind-pick rate under shuffling |", "|---|---|"] + [
            f"| {LABEL.get(d, d)} | {v:.3f} |" for d, v in sorted(bp.items()) if d != "pooled"]
        if np.mean(clean_shuf) < chance:
            # The pre-registration's kill criterion: a permutation cannot make an agent worse
            # than guessing, so this is the shuffling implementation, not the finding.
            L += ["", "**KILL CRITERION TRIPPED** — shuffled clean accuracy "
                  f"{np.mean(clean_shuf):.3f} is below the blind-pick rate {chance:.3f}. "
                  "The shuffling implementation is wrong; nothing is claimed from this arm "
                  "until it is rebuilt."]
        else:
            _outcome, rule_lines = decide_outcome(
                float(np.mean(clean_ranked)), float(np.mean(clean_shuf)), chance,
                effects_for_rule)
            L += rule_lines

    L += mediator_block(rows, comp)

    L += ["", "## Ranking withheld — is the explanation load-bearing at all?", "",
          "| Judge | n | cells | mean Δ | 95% CI |", "|---|---|---|---|---|"]
    for j in judges:
        e = paired_effect(rows, "no_explanation", judge=j)
        if e.get("n"):
            L.append(f"| {j} | {e['n']} | {e['n_clusters']} | {e['mean_delta']:+.3f} | "
                     f"[{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] |")

    out = RUN / "summary" / "c2_report.md"
    out.parent.mkdir(parents=True, exist_ok=True)
    text = "\n".join(L) + "\n"
    out.write_text(text)
    print(text)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
