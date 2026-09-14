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
                "pasa_range": "PASA range",
                "pasa_std": "PASA std",
                "erasure_faithfulness": r"\Erasecheck{}",
                "FUSED": r"Conformal fusion of all seven"}
FLOOR = 0.5


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)
    print(f"  wrote {OUT / name}")


def ci(e, key="mean_delta", lo="ci_lo", hi="ci_hi", bold=False) -> str:
    s = f"${e[key]:+.3f}$ [{e[lo]:+.3f}, {e[hi]:+.3f}]"
    return r"{\bfseries\boldmath " + s + "}" if bold else s


def ci_m(m) -> str:
    return f"${m['mean']:+.3f}$ [{m['ci_lo']:+.3f}, {m['ci_hi']:+.3f}]"


# ------------------------------------------------------------------ harm
def group(label, ncols):
    return r"\multicolumn{" + str(ncols) + r"}{@{}l}{\emph{" + label + r"}} \\"


def tab_harm() -> None:
    eff = json.loads((RD / "summary" / "effects.json").read_text())
    body = [r"\begin{tabular}{@{}lrrr@{}}", r"\toprule",
            r"\textbf{Scope} & $\Delta$ \textbf{vs} $S(x)$ & "
            r"$\Delta$ \textbf{vs} $S(x')$, 95\% CI & \textbf{broke/fixed} \\",
            r"\midrule"]
    for atk in ("A1_displacement", "A1_misdirection"):
        a, b = eff["vs_S_x"][atk], eff["vs_S_xprime"][atk]
        body.append(group(ATTACK_MACRO[atk], 4))
        body.append(f"pooled & ${a['pooled']['mean_delta']:+.3f}$ & "
                    f"{ci(b['pooled'], bold=True)} & {b['pooled']['broke']} / {b['pooled']['fixed']} \\\\")
        for ds in DATASETS:
            ad, bd = a["per_dataset"][ds], b["per_dataset"][ds]
            body.append(f"\\quad {DATASET_LABEL[ds]} & ${ad['mean_delta']:+.3f}$ & {ci(bd)} & "
                        f"{bd['broke']} / {bd['fixed']} \\\\")
        body.append(r"\addlinespace")
    # Scaffolding: one block per contamination setting, the routing rate measured on the
    # same prediction-preserved population the harm is scored on.
    body.append(group(ATTACK_MACRO["A3_scaffolding"] + ", contamination $c$", 4))
    for c, run in sorted(RD_A3.items(), reverse=True):
        e = json.loads((run / "summary" / "effects.json").read_text())
        d = json.loads((run / "summary" / "decomposition.json").read_text())
        a, b = e["vs_S_x"]["A3_scaffolding"], e["vs_S_xprime"]["A3_scaffolding"]
        routed = d["A3_scaffolding"]["pooled"]["routed_rate"]["mean"]
        body.append(f"$c = {c:g}$ ({routed:.3f}) & ${a['pooled']['mean_delta']:+.3f}$ & "
                    f"{ci(b['pooled'], bold=True)} & {b['pooled']['broke']} / {b['pooled']['fixed']} \\\\")
        if c == 0.3:
            for ds in DATASETS:
                ad, bd = a["per_dataset"][ds], b["per_dataset"][ds]
                rd = d["A3_scaffolding"][ds]["routed_rate"]["mean"]
                body.append(f"\\quad {DATASET_LABEL[ds]} ({rd:.3f}) & ${ad['mean_delta']:+.3f}$ & "
                            f"{ci(bd)} & {bd['broke']} / {bd['fixed']} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_harm.tex", "\n".join(body) + "\n")


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
            r"$\Delta_{\text{inf}}$ & $\Delta_{\text{rel}}$ & $J(S(x),S(x'))$ & "
            r"$|S(x)| \to |S(x')|$ \\",
            r"\midrule"]
    for atk in ATTACKS:
        body.append(group(ATTACK_MACRO[atk], 6))
        for scope in ["pooled"] + DATASETS:
            m = dec[atk][scope]
            body.append(f"{'pooled' if scope == 'pooled' else chr(92) + 'quad ' + DATASET_LABEL[scope]} & "
                        f"{ci_m(m['paper'])} & {ci_m(m['infidelity'])} & {ci_m(m['reliance'])} & "
                        f"{m['reliance_jaccard']['mean']:.3f} & {size_cell(atk, scope)} \\\\")
        body.append(r"\addlinespace")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_decomp.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ controls
def tab_controls() -> None:
    rows = json.loads((DU / "raw" / "decisions.json").read_text())
    cells = json.loads((DU / "summary" / "per_cell.json").read_text())
    comp = competent_cell_set(cells, FLOOR)
    judges = sorted({r["judge"] for r in rows})
    shuf = json.loads((DU / "raw" / "shuffled.json").read_text())
    ids = {(r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]) for r in shuf}
    matched = [r for r in rows
               if (r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]) in ids]

    body = [r"\begin{tabular}{@{}lrr@{}}", r"\toprule",
            r"\textbf{Setting} & \textbf{clusters} & $\Delta$ \textbf{vs} $S(x)$, 95\% CI \\",
            r"\midrule", group("competence floor (clean accuracy)", 3)]
    for r_ in floor_sweep(rows, cells, floors=(0.0, 0.3, 0.5, 0.8)):
        body.append(f"$\\geq {r_['floor']:.1f}$ & {r_['n_clusters']} & {ci(r_)} \\\\")
    body.append(group("leave one out", 3))
    for ds in DATASETS:
        e = paired_effect([r for r in rows if r["dataset"] != ds], "attacked",
                          attack="A1_displacement", competent_cells=comp)
        body.append(f"drop {DATASET_LABEL[ds]} & {e['n_clusters']} & {ci(e)} \\\\")
    for j in judges:
        e = paired_effect([r for r in rows if r["judge"] != j], "attacked",
                          attack="A1_displacement", competent_cells=comp)
        body.append(f"drop {j} & {e['n_clusters']} & {ci(e)} \\\\")
    body.append(group("shuffled order (picks first on clean: ranked $\\to$ shuffled)", 3))
    for j in judges:
        pr = position_bias([r for r in matched if r["condition"] == "clean"], judge=j)["p_first_overall"]
        ps = position_bias([r for r in shuf if r["condition"] == "clean"], judge=j)["p_first_overall"]
        e = paired_effect([r for r in shuf if r["judge"] == j], "attacked",
                          attack="A1_displacement", competent_cells=comp)
        body.append(f"{j}, {pr:.2f} $\\to$ {ps:.2f} & {e['n_clusters']} & {ci(e)} \\\\")
    # The withheld case is identical under every attack (one list, no scores), so it is
    # scored once, on the cause-displacement copy, or the three copies would be counted as
    # independent clusters and the interval would be about root-3 too narrow.
    body.append(group("ranking withheld, competent reader--cells", 3))
    for j in judges:
        e = paired_effect(rows, "no_explanation", judge=j, attack="A1_displacement",
                          competent_cells=comp)
        body.append(f"{j} & {e['n_clusters']} & {ci(e)} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("tab_controls.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ checks
def fmt(v, seeds) -> str:
    mu, hw = seed_clustered_ci(np.asarray(v, dtype=float), np.asarray(seeds))
    return f"{mu:.2f} $\\pm$ {hw:.2f}"


def tab_checks() -> None:
    raw = json.loads(MATRIX.read_text())
    per, seedmap = defaultdict(list), defaultdict(list)
    for r in raw:
        for sig, auroc in (r.get("per_signal") or {}).items():
            per[(r["dataset"], r["attack"], sig)].append(float(auroc))
            seedmap[(r["dataset"], r["attack"], sig)].append(r["seed"])
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
        body.append(group(ATTACK_MACRO[atk], 1 + len(DATASETS)))
        for sig in signals:
            cells = []
            for d in DATASETS:
                c = fmt(per[(d, atk, sig)], seedmap[(d, atk, sig)])
                cells.append(r"\textbf{" + c + "}" if best[(d, atk)] == sig else c)
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


def main() -> None:
    print("paper tables")
    tab_harm()
    tab_decomp()
    tab_controls()
    tab_checks()
    tab_generality()


if __name__ == "__main__":
    main()
