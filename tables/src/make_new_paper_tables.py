#!/usr/bin/env python3
"""Generated table bodies for the IEEEtran manuscript.

Every number in the Results section's tables is written here from a saved artifact. The
section files `\\input` these bodies and never carry a measured value by hand. Terminology
follows the macros in the manuscript's Main.tex (\\attackA, \\attackB, \\attackC, ...), so a rename
there propagates through a re-run of this script.

Outputs (tables/out/):
  tab_harm.tex      paired reader harm per attack, against S(x) and S(x'), plus the
                    scaffolding contamination sweep with the measured routing rate
  tab_decomp.tex    top-1 decomposition: paper / infidelity / reliance, J(S(x),S(x')),
                    causal-set sizes, pooled and per corpus
  tab_controls.tex  the controls on the harm measurement: competence-floor sweep,
                    shuffled order, ranking withheld, leave-one-out
  tab_checks.tex    AUROC of the seven runtime integrity checks and the fused score,
                    per attack and corpus, seed-clustered intervals
  tab_generality.tex certified stability across pipelines with per-seed values

Run:  python tables/src/make_new_paper_tables.py
"""
from __future__ import annotations

import csv
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "src"))
sys.path.insert(0, str(REPO / "scripts"))
from avert.eval.decision_utility import (  # noqa: E402
    competent_cell_set, floor_sweep, paired_effect, position_bias)
from avert.eval.metrics import seed_clustered_ci  # noqa: E402

OUT = REPO / "tables" / "out"
RD = REPO / "results" / "reference_decomposition"
RD_A3 = {0.3: RD, 0.1: REPO / "results" / "reference_decomposition_a3_c01",
         0.05: REPO / "results" / "reference_decomposition_a3_c005"}
DU = REPO / "results" / "decision_utility"
MATRIX = REPO / "results" / "matrix" / "raw.json"

DATASETS = ["fiveg_nidd", "ciciomt2024", "ciciot2023"]
DATASET_LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024"}
ATTACKS = ["A1_displacement", "A1_misdirection", "A3_scaffolding"]
ATTACK_MACRO = {"A1_displacement": r"\AttackB{}", "A1_misdirection": r"\AttackA{}",
                "A3_scaffolding": r"\AttackC{}"}
SIGNAL_ORDER = ["feature_consistency", "cross_method_consensus", "certified_stability",
                "certified_stability_free", "pasa_range", "pasa_std", "erasure_faithfulness",
                "FUSED"]
SIGNAL_LABEL = {"feature_consistency": "Feature consistency",
                "cross_method_consensus": "Cross-method consensus",
                "certified_stability": "Certified stability",
                "certified_stability_free": "Certified stability, free coord.",
                "pasa_range": "PASA, range spread",
                "pasa_std": r"PASA, $0.1\sigma_{\mathrm{cal}}$ spread",
                "erasure_faithfulness": r"\Erasecheck{}",
                "FUSED": r"Conformal fusion of all seven"}
FLOOR = 0.5


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)
    print(f"  wrote {OUT / name}")


def ci(e, key="mean_delta", lo="ci_lo", hi="ci_hi", bold=False) -> str:
    # The interval is set in math mode so its minus signs print as minus, not hyphen.
    s = f"${e[key]:+.3f}$ $[{e[lo]:+.3f}, {e[hi]:+.3f}]$"
    return r"{\bfseries\boldmath " + s + "}" if bold else s


def ci_m(m) -> str:
    return f"${m['mean']:+.3f}$ $[{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}]$"


# ------------------------------------------------------------------ harm
def group(label, ncols):
    return r"\multicolumn{" + str(ncols) + r"}{@{}l}{\emph{" + label + r"}} \\"


SPLIT = REPO / "results" / "split_sample_harm" / "summary" / "effects.json"
BASE_HARM = REPO / "results" / "decision_utility_baseline" / "summary" / "harm.json"
PERM = REPO / "results" / "scaffolding_permutation" / "summary" / "auroc.json"
PERM_HARM = REPO / "results" / "decision_utility_scaffold_perm" / "summary" / "harm.json"
RUN_OF_KAPPA = {0.3: "reference_decomposition", 0.1: "reference_decomposition_a3_c01",
                0.05: "reference_decomposition_a3_c005"}
# The four populations every harm row states. "all" is the unselected headline.
POP = {"all_cells": "all", "competent_on_A": r"A$\to$B",
       "not_competent_on_A": r"rest", "in_sample": r"same"}


def tab_harm() -> None:
    """Table VI: every harm arm, each row naming its population.

    all              all 90 reader--cells, every usable (prediction-preserved) flow
    comp. A->B       reader--cells competent on half A of each cell's flows, measured on half B
    rest A->B        the other reader--cells, measured on half B
    comp. in sample  competence and effect on the same forty flows (the earlier headline)

    Intervals resample cells with both readers inside one cluster (effects_cell.json and the
    split-sample report use the same rule), so one point estimate carries one interval.
    """
    split = json.loads(SPLIT.read_text())
    eff = json.loads((RD / "summary" / "effects_cell.json").read_text())
    body = [r"\begin{tabular}[t]{@{}llrrr@{}}", r"\toprule",
            r"\textbf{Scope} & \textbf{Pop.} & $S(x)$ & "
            r"$S(x')$, 95\% CI & \textbf{b/f} \\",
            r"\midrule"]

    def row(label, pop, ex, exp, bold=False):
        body.append(f"{label} & {POP[pop]} & ${ex['mean_delta']:+.3f}$ & "
                    f"{ci(exp, bold=bold)} & {exp['broke']}/{exp['fixed']} \\\\")

    def split_rows(run, atk, label):
        sx, sxp = split[run]["vs_S_x"][atk], split[run]["vs_S_xprime"][atk]
        for pop in ("all_cells", "competent_on_A", "not_competent_on_A"):
            row(label if pop == "all_cells" else "", pop, sx[pop], sxp[pop],
                bold=pop == "all_cells")

    for atk in ("A1_displacement", "A1_misdirection"):
        body.append(group(ATTACK_MACRO[atk], 5))
        split_rows("reference_decomposition", atk, "pooled")
        a, b = eff["vs_S_x"][atk], eff["vs_S_xprime"][atk]
        row("", "in_sample", a["pooled"], b["pooled"])
        for ds in DATASETS:
            row(f"\\quad {DATASET_LABEL[ds]}", "in_sample", a["per_dataset"][ds],
                b["per_dataset"][ds])
        body.append(r"\addlinespace")
        if atk == "A1_displacement" and BASE_HARM.exists():
            # The random-draw control on the same case sets, and the attack minus it.
            h = json.loads(BASE_HARM.read_text())["effects"]
            body.append(group(r"random draw at the \attackB{} budget, no search", 5))
            for pop in ("all_cells", "competent_on_A", "not_competent_on_A"):
                row("pooled" if pop == "all_cells" else "", pop,
                    h["vs_S_x"]["random_draw"][pop], h["vs_S_xprime"]["random_draw"][pop],
                    bold=pop == "all_cells")
            body.append(group(r"\attackB{} minus random draw, paired", 5))
            for pop in ("all_cells", "competent_on_A", "not_competent_on_A"):
                dx, dxp = (h["vs_S_x"]["attack_minus_random"][pop],
                           h["vs_S_xprime"]["attack_minus_random"][pop])
                body.append(f"{'pooled' if pop == 'all_cells' else ''} & {POP[pop]} & "
                            f"${dx['mean_delta']:+.3f}$ & {ci(dxp, bold=pop == 'all_cells')} & "
                            r"-- \\")
            body.append(r"\addlinespace")

    # Scaffolding with TreeSHAP shown: one block per contamination, the routing rate measured on
    # the prediction-preserved population the harm is scored on. It opens the right-hand panel;
    # the two panels sit side by side in one full-width float so the table stays at 7 pt.
    left, body = body, list(body[:4])
    body.append(group(ATTACK_MACRO["A3_scaffolding"] + r", $g$ = TreeSHAP, contamination $\kappa$", 5))
    for c, run in sorted(RD_A3.items(), reverse=True):
        e = json.loads((run / "summary" / "effects_cell.json").read_text())
        d = json.loads((run / "summary" / "decomposition.json").read_text())
        routed = d["A3_scaffolding"]["pooled"]["routed_rate"]["mean"]
        label = f"$\\kappa = {c:g}$ ({routed:.3f})"
        sx = split[RUN_OF_KAPPA[c]]["vs_S_x"]["A3_scaffolding"]
        sxp = split[RUN_OF_KAPPA[c]]["vs_S_xprime"]["A3_scaffolding"]
        row(label, "all_cells", sx["all_cells"], sxp["all_cells"], bold=True)
        row("", "competent_on_A", sx["competent_on_A"], sxp["competent_on_A"])
        a, b = e["vs_S_x"]["A3_scaffolding"], e["vs_S_xprime"]["A3_scaffolding"]
        row("", "in_sample", a["pooled"], b["pooled"])
        if c == 0.3:
            for ds in DATASETS:
                rd = d["A3_scaffolding"][ds]["routed_rate"]["mean"]
                row(f"{DATASET_LABEL[ds]} ({rd:.3f})", "in_sample",
                    a["per_dataset"][ds], b["per_dataset"][ds])
    if PERM_HARM.exists():
        body.append(r"\addlinespace")
        h = json.loads(PERM_HARM.read_text())
        routed = h["routed_rate_preserved"]
        body.append(group(ATTACK_MACRO["A3_scaffolding"] + r", $g$ = permutation, $\kappa = 0.3$", 5))
        ex, exp = h["effects"]["vs_S_x"], h["effects"]["vs_S_xprime"]
        for pop in ("all_cells", "competent_on_A", "not_competent_on_A"):
            row(f"pooled ({routed:.3f})" if pop == "all_cells" else "", pop, ex[pop], exp[pop],
                bold=pop == "all_cells")
        row("", "in_sample", ex["in_sample_competent"], exp["in_sample_competent"])
    if left[-1] == r"\addlinespace":
        left.pop()
    left += [r"\bottomrule", r"\end{tabular}"]
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_harm.tex", "\n".join(left) + "\n\\hfill\n" + "\n".join(body) + "\n")


# ------------------------------------------------------------------ decomposition
def tab_decomp() -> None:
    dec = json.loads((RD / "summary" / "decomposition.json").read_text())
    inst = json.loads((RD / "raw" / "instances.json").read_text())
    # Causal-set sizes on the decomposition population (prediction preserved), seed-clustered
    # means so they match the terms beside them.
    sizes = defaultdict(list)
    for r in inst:
        if not r["prediction_preserved"]:
            continue
        sizes[(r["attack"], r["dataset"], r["seed"])].append((len(r["S_x"]), len(r["S_a"])))

    def size_cell(atk, scope):
        vals, seeds = [], []
        for (a, ds, s), v in sizes.items():
            if a == atk and (scope == "pooled" or ds == scope):
                arr = np.array(v, dtype=float)
                vals.append(arr.mean(axis=0))
                seeds.append(s)
        vals = np.array(vals)
        mx, _ = seed_clustered_ci(vals[:, 0], np.array(seeds))
        ma, _ = seed_clustered_ci(vals[:, 1], np.array(seeds))
        return f"{mx:.2f} $\\to$ {ma:.2f}"

    body = [r"\begin{tabular}{@{}lrrrrc@{}}", r"\toprule",
            r"\textbf{Scope} & $\Delta_{\text{paper}}$ & "
            r"$\Delta_{\text{inf}}$ & $\Delta_{\text{rel}}$ & $J_S$ & "
            r"$|S(x)| \to |S(x')|$ \\",
            r"\midrule"]
    for atk in ATTACKS:
        body.append(group(ATTACK_MACRO[atk] + (r", $g$ = TreeSHAP" if atk == "A3_scaffolding" else ""), 6))
        for scope in ["pooled"] + DATASETS:
            m = dec[atk][scope]
            body.append(f"{'pooled' if scope == 'pooled' else chr(92) + 'quad ' + DATASET_LABEL[scope]} & "
                        f"{ci_m(m['paper'])} & {ci_m(m['infidelity'])} & {ci_m(m['reliance'])} & "
                        f"{m['reliance_jaccard']['mean']:.3f} & {size_cell(atk, scope)} \\\\")
        body.append(r"\addlinespace")

    # The control arm. Same 160 draws, same projection, same RNG stream; the search objective
    # removed, so the attack returns the FIRST class-preserving draw instead of the best one.
    # It separates what the adversary's optimisation adds from what any realizable perturbation
    # of this size does, and the answer is that it adds the infidelity term and almost none of
    # the reliance shift.
    base_path = REPO / "results" / "baseline_random_draw"
    if base_path.exists():
        base = json.loads((base_path / "summary" / "decomposition.json").read_text())
        binst = json.loads((base_path / "raw" / "instances.json").read_text())
        bsizes = defaultdict(list)
        for r in binst:
            if r["prediction_preserved"]:
                bsizes[(r["dataset"], r["seed"])].append((len(r["S_x"]), len(r["S_a"])))

        def bsize_cell(scope):
            vals, seeds = [], []
            for (ds, sd), v in bsizes.items():
                if scope == "pooled" or ds == scope:
                    arr = np.array(v, dtype=float)
                    vals.append(arr.mean(axis=0)); seeds.append(sd)
            vals = np.array(vals)
            mx, _ = seed_clustered_ci(vals[:, 0], np.array(seeds))
            ma, _ = seed_clustered_ci(vals[:, 1], np.array(seeds))
            return f"{mx:.2f} $\\to$ {ma:.2f}"

        body.append(group(r"random draw (same budget, no search)", 6))
        for scope in ["pooled"] + DATASETS:
            m = base["A1_displacement"][scope]
            body.append(f"{'pooled' if scope == 'pooled' else chr(92) + 'quad ' + DATASET_LABEL[scope]} & "
                        f"{ci_m(m['paper'])} & {ci_m(m['infidelity'])} & {ci_m(m['reliance'])} & "
                        f"{m['reliance_jaccard']['mean']:.3f} & {bsize_cell(scope)} \\\\")
        body.append(r"\addlinespace")

    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_decomp.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ controls
def _rescored_vs_sxprime(rows):
    """The same decisions scored against S(x'): an attacked decision is correct when the chosen
    feature is in the rebuilt causal set of the flow shown (instances.json); clean and withheld
    decisions are unchanged. Attacked decisions whose shown case differs from the cache or whose
    class flipped are dropped, as in the rescoring."""
    inst = json.loads((RD / "raw" / "instances.json").read_text())
    sa = {(r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]): set(r["S_a"])
          for r in inst if r["shown_matches_cache"] and r["prediction_preserved"]}
    out = []
    for r in rows:
        k = (r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"])
        if r["condition"] == "attacked":
            if k not in sa:
                continue
            r = dict(r, det_correct=r["det_choice"] in sa[k])
        out.append(r)
    return out


def tab_controls() -> None:
    rows = json.loads((DU / "raw" / "decisions.json").read_text())
    cells = json.loads((DU / "summary" / "per_cell.json").read_text())
    comp = competent_cell_set(cells, FLOOR)
    judges = sorted({r["judge"] for r in rows})
    shuf = json.loads((DU / "raw" / "shuffled.json").read_text())
    # Every control under both references: the run was scored against S(x); S(x') is the
    # paper's reference, so the second column rescores the same decisions.
    rows2, shuf2 = _rescored_vs_sxprime(rows), _rescored_vs_sxprime(shuf)
    # Every row resamples cells with both readers inside one cluster, as Table VI does. The two
    # readers pick the same letter on almost every case, so a reader--cell cluster treats one
    # measurement as two and narrows the interval by about root 2.
    CL = "cell"

    def cic(e):
        # Compact interval for the four-column table: a positive bound carries no sign.
        return f"${e['mean_delta']:+.3f}$ $[{e['ci_lo']:.3f}, {e['ci_hi']:.3f}]$"

    def both(e1, e2):
        return f"{e1['n_clusters']} & {cic(e1)} & {cic(e2)} \\\\"

    def pe(rr, **kw):
        return paired_effect(rr, "attacked", attack="A1_displacement", competent_cells=comp,
                             cluster=CL, **kw)

    body = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
            r"\textbf{Setting} & \textbf{cells} & $\delta$ \textbf{vs} $S(x)$, 95\% CI & $\delta$ \textbf{vs} $S(x')$, 95\% CI \\",
            r"\midrule", group("competence floor (clean accuracy)", 4)]
    for r1, r2 in zip(floor_sweep(rows, cells, floors=(0.0, 0.3, 0.5, 0.8), cluster=CL),
                      floor_sweep(rows2, cells, floors=(0.0, 0.3, 0.5, 0.8), cluster=CL)):
        body.append(f"$\\geq {r1['floor']:.1f}$ & {both(r1, r2)}")
    body.append(group("leave one out (dropped corpus or reader)", 4))
    for ds in DATASETS:
        body.append(f"{DATASET_LABEL[ds]} & "
                    f"{both(pe([r for r in rows if r['dataset'] != ds]), pe([r for r in rows2 if r['dataset'] != ds]))}")
    for j in judges:
        body.append(f"{j} & {both(pe([r for r in rows if r['judge'] != j]), pe([r for r in rows2 if r['judge'] != j]))}")
    body.append(group("shuffled order, matched slice", 4))
    for j in judges:
        body.append(f"{j} & {both(pe([r for r in shuf if r['judge'] == j]), pe([r for r in shuf2 if r['judge'] == j]))}")
    # The withheld case is identical under every attack (one list, no scores), so it is
    # scored once, on the cause-displacement copy, or the three copies would be counted as
    # independent clusters and the interval would be about root-3 too narrow. It shows a
    # clean flow, so both references score it against the same set and both columns carry
    # the same value.
    body.append(group("ranking withheld, competent reader--cells", 4))
    for j in judges:
        e1 = paired_effect(rows, "no_explanation", judge=j, attack="A1_displacement",
                           competent_cells=comp, cluster=CL)
        body.append(f"{j} & {e1['n_clusters']} & {cic(e1)} & {cic(e1)} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_controls.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ checks
def fmt(v, seeds) -> str:
    mu, hw = seed_clustered_ci(np.asarray(v, dtype=float), np.asarray(seeds))
    return f"{mu:.2f} $\\pm$ {hw:.2f}"


def tab_checks() -> None:
    raw = json.loads(MATRIX.read_text())
    # Scaffolding is scored on the flows whose predicted class survived the attack, and on
    # those flows alone. The two searches enforce class survival by rejection, so their rows
    # are unaffected; the scaffold's router flips the class on some flows, and a flipped flow
    # is not this threat model's attack -- scoring the panel on it credits the checks with
    # seeing an event the threat model excludes. Those per-cell rows come from
    # results/scaffolding_survivors, which also asserts that its all-flow arm reproduces
    # results/matrix/raw.json exactly, so the only difference here is the population.
    surv_path = REPO / "results" / "scaffolding_survivors" / "raw" / "per_cell_survivors.json"
    survivors = json.loads(surv_path.read_text()).get("A3_scaffolding", [])
    if not survivors:
        raise SystemExit(f"{surv_path} has no A3_scaffolding rows -- run "
                         "scripts/report_scaffolding_survivors.py first")

    per, seedmap = defaultdict(list), defaultdict(list)
    for r in raw:
        if r["attack"] == "A3_scaffolding":
            continue
        for sig, auroc in (r.get("per_signal") or {}).items():
            per[(r["dataset"], r["attack"], sig)].append(float(auroc))
            seedmap[(r["dataset"], r["attack"], sig)].append(r["seed"])
    for r in survivors:
        for sig, auroc in r["per_signal"].items():
            per[(r["dataset"], "A3_scaffolding", sig)].append(float(auroc))
            seedmap[(r["dataset"], "A3_scaffolding", sig)].append(r["seed"])
    present = {k[2] for k in per}
    signals = [s for s in SIGNAL_ORDER if s in present] + sorted(present - set(SIGNAL_ORDER))
    body = [r"\begin{tabular}{@{}l" + "r" * len(DATASETS) + "@{}}", r"\toprule",
            r"\textbf{Check} & "
            + " & ".join(f"\\textbf{{{DATASET_LABEL[d]}}}" for d in DATASETS) + r" \\",
            r"\midrule"]
    # Best single check per (corpus, attack) on the seed-clustered mean, fused excluded,
    # the same rule the results map and Fig. 4 use (eval.metrics.best_single_auroc).
    best = {}
    for ds in DATASETS:
        for atk in ATTACKS:
            cands = [(seed_clustered_ci(np.array(per[(ds, atk, s)]), np.array(seedmap[(ds, atk, s)]))[0], s)
                     for s in signals if s != "FUSED" and (ds, atk, s) in per]
            best[(ds, atk)] = max(cands)[1]
    for atk in ATTACKS:
        label = ATTACK_MACRO[atk] + (r", $g$ = TreeSHAP" if atk == "A3_scaffolding" else "")
        body.append(group(label, 1 + len(DATASETS)))
        for sig in signals:
            cells = []
            for d in DATASETS:
                c = fmt(per[(d, atk, sig)], seedmap[(d, atk, sig)])
                cells.append(r"\textbf{" + c + "}" if best[(d, atk)] == sig else c)
            if sig == "FUSED":
                body.append(r"\cmidrule(lr){1-" + str(1 + len(DATASETS)) + "}")
            body.append(f"{SIGNAL_LABEL[sig]} & {' & '.join(cells)} \\\\")
        body.append(r"\addlinespace")
    # Scaffolding with the permutation attributor as g: the attributor the scaffold can reach.
    # Survivor-only, seed-clustered, from results/scaffolding_permutation. Consensus there pairs
    # g with TreeSHAP.
    if PERM.exists():
        agg = json.loads(PERM.read_text())["survivors_only"]
        body.append(group(ATTACK_MACRO["A3_scaffolding"] + r", $g$ = permutation", 1 + len(DATASETS)))
        best_p = {ds: max((agg[ds][sg]["mean"], sg) for sg in signals
                          if sg != "FUSED" and sg in agg[ds])[1] for ds in DATASETS}
        for sig in signals:
            cells = []
            for ds in DATASETS:
                m = agg[ds][sig]
                c = f"{m['mean']:.2f} $\\pm$ {(m['ci_hi'] - m['ci_lo']) / 2:.2f}"
                cells.append(r"\textbf{" + c + "}" if best_p[ds] == sig else c)
            if sig == "FUSED":
                body.append(r"\cmidrule(lr){1-" + str(1 + len(DATASETS)) + "}")
            body.append(f"{SIGNAL_LABEL[sig]} & {' & '.join(cells)} \\\\")
        body.append(r"\addlinespace")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_checks.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ generality
def tab_generality() -> None:
    tree_path = REPO / "results" / "_logs" / "signal1_cross_dataset.json"
    tree = {(r["dataset"], r["condition"]): r for r in json.loads(tree_path.read_text())}
    mlp = {}
    for run in sorted(REPO.glob("results/generality_mlp_ig_*")):
        ds = run.name.replace("generality_mlp_ig_", "")
        per = defaultdict(list)
        for row in csv.DictReader(open(run / "raw" / "auroc.csv")):
            per[row["attack"]].append(float(row["auroc"]))
        mlp[ds] = per
    body = [r"\begin{tabular}{@{}lcc@{}}", r"\toprule",
            r"\textbf{Corpus} & \textbf{Trees + TreeSHAP} & "
            r"\textbf{MLP + IG} \\", r"\midrule"]

    def cell(mu, seeds):
        inner = ",".join(f"{v:.2f}" for v in seeds)
        return f"{mu:.2f} {{\\tiny({inner})}}"

    allv = []
    for atk in ("A1_displacement", "A1_misdirection"):
        body.append(group(ATTACK_MACRO[atk], 3))
        for ds in DATASETS:
            trow = tree[(ds, atk)]
            m = mlp[ds][atk]
            allv += list(trow["auroc_per_seed"]) + list(m)
            body.append(f"{DATASET_LABEL[ds]} & "
                        f"{cell(float(np.mean(trow['auroc_per_seed'])), trow['auroc_per_seed'])} & "
                        f"{cell(float(np.mean(m)), m)} \\\\")
        body.append(r"\addlinespace")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_generality.tex", "\n".join(body) + "\n")
    print(f"  certified-stability grid: {min(allv):.3f} to {max(allv):.3f}")


# ------------------------------------------------------------------ certified stability at scale
CERT_SCALE = REPO / "results" / "certified_scale" / "summary" / "auroc.json"


def tab_certified_scale() -> None:
    """Table XI: the certified-stability check with noise in sigma_train units.

    One row per (attack, noise scale, variant): the seed-clustered AUROC per corpus and the
    share of flows the certificate certifies (R > 0), clean and attacked, pooled over the
    three corpora. The ceiling is the largest radius the check can return at N draws.
    """
    if not CERT_SCALE.exists():
        return
    d = json.loads(CERT_SCALE.read_text())
    auroc, ceil = d["auroc"], d["ceiling_sigma_train"]
    body = [r"\begin{tabular}{@{}llrrrr@{}}", r"\toprule",
            r"$\sigma$ (ceil.) & \textbf{Coord.} & " +
            " & ".join(f"\\textbf{{{DATASET_LABEL[x]}}}" for x in DATASETS) +
            r" & \textbf{Cert.} \\", r"\midrule"]
    sigmas = sorted({k.split("_")[0][5:] for k in auroc["A1_displacement"][DATASETS[0]]},
                    key=float)
    for atk in ("A1_displacement", "A1_misdirection"):
        body.append(group(ATTACK_MACRO[atk], 6))
        for sg in sigmas:
            for var, lab in (("all", "all"), ("free", "free")):
                k = f"sigma{sg}_{var}"
                cells = []
                for ds in DATASETS:
                    m = auroc[atk][ds][k]
                    cells.append(f"{m['mean']:.2f} $\\pm$ {(m['ci_hi'] - m['ci_lo']) / 2:.2f}")
                cert_c = np.mean([auroc[atk][ds][k]["certified_clean"] for ds in DATASETS])
                cert_a = np.mean([auroc[atk][ds][k]["certified_attacked"] for ds in DATASETS])
                head = (f"{sg} ({ceil[sg]:.3f})"
                        if var == "all" else "")
                body.append(f"{head} & {lab} & {' & '.join(cells)} & "
                            f"{cert_c:.2f}/{cert_a:.2f} \\\\")
        body.append(r"\addlinespace")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_certified_scale.tex", "\n".join(body) + "\n")


def main() -> None:
    print("paper tables")
    tab_harm()
    tab_decomp()
    tab_controls()
    tab_checks()
    tab_generality()
    tab_certified_scale()


if __name__ == "__main__":
    main()
