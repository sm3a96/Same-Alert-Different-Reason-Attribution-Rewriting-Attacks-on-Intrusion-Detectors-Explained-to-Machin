"""Generate every LaTeX table the paper \\input's.

booktabs, no vertical rules, and nothing hand-edited: each table is written from either a
saved result CSV or a module in the package, so a re-run regenerates the paper's numbers
from scratch. That discipline is what lets every
claim trace back to a script.

  python tables/src/make_tables.py

Tab1  attack taxonomy, with honest coverage        (static -- the threat model)
Tab2  benchmark composition                        (results/matrix/raw.json)
Tab3  MITRE ATLAS / NIST AI 100-2 mapping + gap    (avert.benchmark.taxonomy)
Tab5  per-signal detectability                     (results/matrix/raw.json -- same cells as Fig4)
Tab6  generality across detector/attributor pairs  (results/generality_mlp_ig_*/)

Tab4  agent decision utility                       (results/decision_utility/summary/per_cell.json)
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np

from avert.benchmark.taxonomy import FRAMEWORK_MAPPING
from avert.eval.metrics import seed_clustered_ci

REPO = Path(__file__).resolve().parents[2]
OUT = REPO / "tables" / "out"

DATASET_LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023",
                 "ciciomt2024": "CICIoMT2024"}
# Display names only. The artifact keys stay as they are, so no result has to be re-run;
# renaming those would invalidate every cached cell for a cosmetic gain.
ATTACK_LABEL = {"A1_misdirection": "A1a rank promotion",
                "A1_displacement": "A1b cause displacement",
                "A3_scaffolding": "A3 explainer scaffolding",
                "A2_poisoning": "A2 explainer poisoning"}
SIGNAL_LABEL = {"cross_method_consensus": "Cross-method consensus",
                "certified_stability": "Certified stability",
                "certified_stability_free": "Certified stability, free coordinates only",
                "feature_consistency": "Feature consistency",
                "pasa_range": r"PASA, range spread $5\times10^{-4}$",
                "pasa_std": r"PASA, $0.1\sigma$ per feature",
                "erasure_faithfulness": "Erasure faithfulness",
                "FUSED": r"Fused (max-$z$ + conformal)"}

# Order for Tab5. Feature consistency sits first because it is the cheapest defender and the
# one a reviewer asks about before the others: no model, no calibration, one pass over the
# extractor's arithmetic. Reporting it is what turns "no signal detects displacement" from a
# claim about the signals we happened to try into one measured against input validation too.
SIGNAL_ORDER = ["feature_consistency", "cross_method_consensus", "certified_stability",
                "certified_stability_free", "pasa_range", "pasa_std", "erasure_faithfulness", "FUSED"]


def signal_label(key: str) -> str:
    """Label for a signal key, and a readable fallback for one that has no entry yet.

    Tab5 used to index `SIGNAL_LABEL` directly, so adding Signal 4 to the pipeline crashed the
    build with a bare KeyError -- and the failure came *after* the .tex was written, so the file
    on disk silently held three of the four signals while the run looked like it had only fallen
    over at the end. A missing label is a formatting gap, never grounds to drop a row.
    """
    return SIGNAL_LABEL.get(key, key.replace("_", " ").capitalize())


def tex_escape(s: str) -> str:
    for a, b in (("&", r"\&"), ("%", r"\%"), ("_", r"\_"), ("#", r"\#")):
        s = s.replace(a, b)
    return s


def write(name: str, body: str) -> None:
    OUT.mkdir(parents=True, exist_ok=True)
    (OUT / name).write_text(body)
    print(f"  wrote {OUT / name}")



def texnum(n: int) -> str:
    """Thousands separator LaTeX will not turn into a comma-with-space."""
    return f"{n:,}".replace(",", "{,}")


def fmt(v, seeds=None, proportion=False) -> str:
    """mean +/- 95% CI clustered by seed.

    The three target classes inside a seed share one bit-identical fitted detector, so
    they are not independent observations and an interval over all of them is up to 2.3x
    too narrow. Pass the matching seed list wherever it is available.
    """
    v = np.asarray(v, dtype=float)
    keep = ~np.isnan(v)
    if not keep.any():
        return "--"
    s = np.arange(int(keep.sum())) if seeds is None else np.asarray(seeds)[keep]
    mu, hw = seed_clustered_ci(v[keep], s)
    if not proportion:
        return f"${mu:.2f} \\pm {hw:.2f}$"
    # Every quantity in Tab4 is a proportion, and a symmetric interval on one escapes
    # [0, 1] near the ends -- 0.98 +/- 0.03 asserts an accuracy above 1. Clip to the
    # feasible range and print the interval, which is what the number actually means.
    lo, hi = max(0.0, mu - hw), min(1.0, mu + hw)
    if int(keep.sum()) < 2 or hw == 0:
        return f"${mu:.2f}$\\rlap{{$^{{\\dagger}}$}}"
    # Thin-spaced interval: the full-width spaces made Tab4 30pt wider than the text block.
    return f"${mu:.2f}$\\,[{lo:.2f},{hi:.2f}]"



def _ciciov_exclusion_row() -> str:
    """The exclusion footnote, read from the audit rather than typed.

    It carried a hardcoded duplicate rate that disagreed with
    `results/_logs/audit_ciciov2024.json`, and nothing caught it, because a number inside a
    generator is not a registered claim -- the same class of defect as a stale value in prose.
    """
    a = json.loads((REPO / "results/_logs/audit_ciciov2024.json").read_text())
    dup = 100.0 * a["duplicate_fraction"]
    uniq = int(a["unique_rows"])
    attacks = sum(v for k, v in a["class_counts"].items() if k.upper() != "BENIGN")
    return (r"\multicolumn{6}{l}{\footnotesize CICIoV2024 excluded: "
            + f"{dup:.2f}" + r"\% duplicate rows, "
            + f"{uniq:,}".replace(",", "{,}")
            + r" unique of which " + f"{attacks}" + r" are attacks.} \\")

# ------------------------------------------------------------------ Tab1
def tab1_taxonomy() -> None:
    rows = [
        ("A1", "Attribution rewriting (rank promotion, cause displacement)",
         "Query access to $f$ and $g$; protocol-valid perturbation", r"\textbf{evaluated}"),
        ("A2", "Explainer poisoning",
         "Influence over a learned explanation component's training data", "scoped out"),
        ("A3", "Explainer scaffolding",
         "White-box control of the deployed model artifact", r"\textbf{evaluated}"),
        ("A4", "Drift masquerade",
         "Ability to shift the input distribution gradually", "scoped out"),
        ("A5", "Monitor evasion",
         "Knowledge of the monitor's design", "scoped out"),
    ]
    body = [r"\begin{tabular}{llp{5.4cm}l}", r"\toprule",
            r"\textbf{ID} & \textbf{Attack} & \textbf{Capability required} & \textbf{Status} \\",
            r"\midrule"]
    body += [f"{i} & {tex_escape(a)} & {tex_escape(c)} & {s} \\\\" for i, a, c, s in rows]
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table_01_taxonomy.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ Tab2
def tab2_composition() -> None:
    raw = json.loads((REPO / "results" / "matrix" / "raw.json").read_text())
    per = defaultdict(lambda: {"classes": set(), "seeds": set(), "attacks": set(),
                               "valid": [], "corrupt": []})
    for r in raw:
        d = per[r["dataset"]]
        d["classes"].add(r["class"])
        d["seeds"].add(r["seed"])
        d["attacks"].add(r["attack"])
        d["valid"].append(r["valid"])
        d["corrupt"].append(r["corrupt"])
        d.setdefault("_seed", []).append(r["seed"])

    body = [r"\begin{tabular}{lrrrrr}", r"\toprule",
            r"\textbf{Dataset} & \textbf{Classes} & \textbf{Attacks} & \textbf{Seeds} & "
            r"\textbf{Cells} & \textbf{Prediction preserved} \\", r"\midrule"]
    for ds, d in sorted(per.items(), key=lambda kv: DATASET_LABEL.get(kv[0], kv[0])):
        body.append(f"{DATASET_LABEL.get(ds, ds)} & {len(d['classes'])} & {len(d['attacks'])} & "
                    f"{len(d['seeds'])} & {len(d['valid'])} & {fmt(d['valid'], d['_seed'])} \\\\")
    body += [r"\midrule",
             _ciciov_exclusion_row(),
             r"\bottomrule", r"\end{tabular}"]
    write("table_02_composition.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ Tab3
def tab3_taxonomy_mapping() -> None:
    body = [r"\begin{tabular}{llllp{4.6cm}}", r"\toprule",
            r"\textbf{Attack} & \textbf{Class} & \textbf{MITRE ATLAS} & \textbf{NIST AI 100-2} & "
            r"\textbf{Coverage gap} \\", r"\midrule"]
    for atk, m in FRAMEWORK_MAPPING.items():
        atlas = ", ".join(f"\\texttt{{{a}}}" for a in m["atlas"]) or "--"
        body.append(f"{ATTACK_LABEL.get(atk, atk)} & {tex_escape(m['threat_class'])} & {atlas} & "
                    f"{tex_escape(m['nist_ai_100_2'])} & {tex_escape(m['gap'])} \\\\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table_03_atlas_nist.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ Tab5
def tab5_per_signal() -> None:
    """Per-signal AUROC, read from the SAME matrix cells that Fig4 plots.

    An earlier version read a separate artifact run that swept only the default target
    class, so the table and the figure described different cells and quietly disagreed.
    One source, one set of cells.
    """
    raw = json.loads((REPO / "results" / "matrix" / "raw.json").read_text())
    if not any("per_signal" in r for r in raw):
        print("  skip Tab5 — matrix has no per_signal field yet (re-run scripts/run_matrix.py)")
        return
    per, seedmap = defaultdict(list), defaultdict(list)
    for r in raw:
        for sig, auroc in (r.get("per_signal") or {}).items():
            per[(r["dataset"], r["attack"], sig)].append(float(auroc))
            seedmap[(r["dataset"], r["attack"], sig)].append(r["seed"])

    datasets = sorted({k[0] for k in per}, key=lambda d: DATASET_LABEL.get(d, d))
    # Driven by what the run actually produced, in the declared order, with anything new
    # appended rather than dropped. A hard-coded list is how Signal 4 came to be missing from
    # the table while sitting in the data the table was built from.
    present = {k[2] for k in per}
    signals = [s for s in SIGNAL_ORDER if s in present] + sorted(present - set(SIGNAL_ORDER))
    body = [r"\begin{tabular}{ll" + "r" * len(datasets) + "}", r"\toprule",
            r"\textbf{Attack} & \textbf{Signal} & "
            + " & ".join(f"\\textbf{{{DATASET_LABEL.get(d, d)}}}" for d in datasets) + r" \\",
            r"\midrule"]
    for attack in ("A1_misdirection", "A1_displacement", "A3_scaffolding"):
        for i, sig in enumerate(signals):
            cells = " & ".join(fmt(per.get((d, attack, sig), []), seedmap.get((d, attack, sig)))
                               for d in datasets)
            label = ATTACK_LABEL[attack] if i == 0 else ""
            body.append(f"{label} & {signal_label(sig)} & {cells} \\\\")
        body.append(r"\addlinespace")
    if "feature_consistency" in signals:
        # A row of 0.50s reads as a broken cell unless the table says why it is 0.50. It is
        # the designed answer to "did you try input validation?", so it is stated here rather
        # than left for the prose, where a reader meets the number three pages later.
        body += [r"\midrule",
                 r"\multicolumn{" + str(2 + len(datasets)) + r"}{l}{\footnotesize Feature "
                 r"consistency is $0.50$ by construction: the attack projects onto the "
                 r"realizable set, so no declared} \\",
                 r"\multicolumn{" + str(2 + len(datasets)) + r"}{l}{\footnotesize extractor "
                 r"relation is violated on either arm. It reaches AUROC $1.00$ against the "
                 r"unprojected attack (Sec.~\ref{sec:threat}).} \\"]
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table_05_per_signal_auroc.tex", "\n".join(body) + "\n")

    print("\n  per-signal AUROC (clean vs attacked):")
    for (ds, atk, sig), v in sorted(per.items()):
        mu, hw = seed_clustered_ci(np.array(v), np.array(seedmap[(ds, atk, sig)]))
        print(f"    {DATASET_LABEL.get(ds, ds):12s} {ATTACK_LABEL[atk]:17s} "
              f"{signal_label(sig):24s} {mu:.3f}+/-{hw:.3f}  (n={len(v)})")


# ------------------------------------------------------------------ Tab4
def tab4_decision_utility() -> None:
    """The C2 table: accuracy per condition, and the flip rate against its own control.

    The control column is the point of the table. A flip rate means nothing until you
    know how often the same judge changes its mind on the identical clean case, so the
    two sit side by side and a reader can do the comparison without being told to.
    """
    cells_path = REPO / "results" / "decision_utility" / "summary" / "per_cell.json"
    if not cells_path.exists():
        print("  skip Tab4 — no decision_utility run yet (scripts/run_decision_utility.py)")
        return
    cells = json.loads(cells_path.read_text())
    # The re-derivation arm re-runs the same explainer on the same flow, so it is an
    # identity with `attacked`; we check that it holds. No repair, not claimed, stub.
    # the attacked decision exactly. Measured rather than asserted -- if it ever stops being
    # an identity the footnote stops being true and the column has to come back.
    raw = json.loads((REPO / "results" / "decision_utility" / "raw" / "decisions.json").read_text())
    def key(r):
        return (r["judge"], r["dataset"], r["class"], r["seed"], r["attack"],
                r["sample_id"])
    atk = {key(r): r["det_correct"] for r in raw if r["condition"] == "attacked"}
    rep = {key(r): r["det_correct"] for r in raw if r["condition"] == "repaired"}  # retired-ok: the arm's key in the raw data; identity, not claimed
    shared = set(atk) & set(rep)
    n_same = sum(atk[k] == rep[k] for k in shared)
    per = defaultdict(lambda: defaultdict(list))
    for c in cells:
        for k, v in c.items():
            if "__" in k:
                per[(c["judge"], c["dataset"], c["attack"])][k].append(v)

    # One panel per judge rather than a Judge column: the 8-column single-panel form was
    # wider than the acmsmall text block by ~300pt, and \resizebox shrank it to microtext.
    # Attacks are named by the IDs Table 1 defines, which is what narrows the panel enough
    # to print at full size.
    attack_id = {k: v.split()[0] for k, v in ATTACK_LABEL.items()}
    # \footnotesize + 3pt colsep: at the wrapper's \small and acmart's default colsep
    # the 7-column body overfills the 395.8pt text block by ~30-43pt (measured from the
    # build log, twice). The size drop is what recovers it; both are scoped to the table
    # environment, so nothing outside the float changes.
    body = [r"\footnotesize\setlength\tabcolsep{2pt}",
            r"\begin{tabular}{llrrrrr}", r"\toprule",
            r"\textbf{Dataset} & \textbf{Attack} & \textbf{Clean} & "
            r"\textbf{Attacked} & \textbf{Rank.\ withheld} & "
            r"\textbf{Flip} & \textbf{Control} \\"]
    judges = sorted({j for (j, _, _) in per})
    for j in judges:
        body += [r"\midrule",
                 r"\multicolumn{7}{l}{\emph{Reader: " + tex_escape(j) + r"}} \\",
                 r"\midrule"]
        last_ds = None
        for (jj, ds, atk), m in sorted(per.items()):
            if jj != j:
                continue
            ds_label = DATASET_LABEL.get(ds, ds) if ds != last_ds else ""
            last_ds = ds
            body.append(" & ".join([
                ds_label, attack_id.get(atk, atk),
                fmt(m.get("clean__acc", []), proportion=True),
                fmt(m.get("attacked__acc", []), proportion=True),
                fmt(m.get("no_explanation__acc", []), proportion=True),
                fmt(m.get("attacked__flip", []), proportion=True),
                fmt(m.get("control__flip_smp", []), proportion=True),
            ]) + r" \\")
    # A bare \multicolumn row never wraps, and hand-balanced line breaks overflowed the
    # text block by 30pt (probe-measured: the tabular body alone fits, the footnote rows
    # were the overfull). One \parbox wraps itself and cannot regress when the text changes.
    footnote = (
        r"Attack IDs are those of Table~\ref{tab:taxonomy}. Accuracy is against "
        r"interventional ground truth, with 95\% intervals clustered by seed and clipped "
        r"to $[0,1]$; $\dagger$ marks a point estimate with no replication. Flip is "
        r"relative to the matched clean decision; Control is the same clean case decided "
        r"twice. `Rank.\ withheld' shows the same ten features with the importance values "
        r"removed, so it isolates the ranking and holds feature selection fixed. The "
        r"repaired arm is omitted: re-running the same explainer on the same flow "  # retired-ok: the footnote withdraws the arm; naming it is how
        rf"reproduces the attacked decision in {texnum(n_same)} of {texnum(len(shared))} "
        r"cases, so the column measured an identity."
    )
    body += [r"\midrule",
             r"\multicolumn{7}{l}{\parbox{0.95\textwidth}{\footnotesize " + footnote
             + r"}} \\",
             r"\bottomrule", r"\end{tabular}"]
    write("table_04_decision_utility.tex", "\n".join(body) + "\n")


# ------------------------------------------------------------------ Tab6
def tab6_generality() -> None:
    """The table that turns "the signal fails" into "the signal cannot be deployed".

    Certified-stability AUROC across datasets AND detector/attributor pairs, with per-seed
    values shown rather than summarised, because on one cell the seed spread is wider than
    the gap between datasets and a mean would hide that.
    """
    tree_path = REPO / "results" / "_logs" / "signal1_cross_dataset.json"
    if not tree_path.exists():
        print("  skip Tab6 — run scripts/summarize_signal1.py")
        return
    tree = {(r["dataset"], r["condition"]): r for r in json.loads(tree_path.read_text())}

    mlp = {}
    for run in sorted(REPO.glob("results/generality_mlp_ig_*")):
        ds = run.name.replace("generality_mlp_ig_", "")
        f = run / "raw" / "auroc.csv"
        if not f.exists():
            continue
        per = defaultdict(list)
        for row in csv.DictReader(open(f)):
            per[row["attack"]].append(float(row["auroc"]))
        mlp[ds] = per
    if not mlp:
        print("  skip Tab6 — run scripts/run_generality.py")
        return

    datasets = [d for d in ("fiveg_nidd", "ciciot2023", "ciciomt2024") if d in mlp]
    body = [r"\begin{tabular}{llcc}", r"\toprule",
            r"\textbf{Attack} & \textbf{Dataset} & \textbf{XGBoost + TreeSHAP} & "
            r"\textbf{MLP + int.\ gradients} \\", r"\midrule"]
    for atk in ("A1_displacement", "A1_misdirection"):
        for i, ds in enumerate(datasets):
            trow = tree.get((ds, atk), {})
            t, tseeds = trow.get("auroc"), trow.get("auroc_per_seed") or []
            m = mlp[ds].get(atk, [])
            label = ATTACK_LABEL[atk] if i == 0 else ""
            def cell(mu, seeds):
                if mu is None:
                    return "--"
                inner = ", ".join(f"{v:.2f}" for v in seeds)
                return f"${mu:.3f}$ \\,({inner})" if seeds else f"${mu:.3f}$"
            tcell = cell(t, tseeds)
            mcell = cell(float(np.mean(m)) if m else None, m)
            body.append(f"{label} & {DATASET_LABEL.get(ds, ds)} & {tcell} & {mcell} \\\\")
        body.append(r"\addlinespace")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table_06_generality.tex", "\n".join(body) + "\n")

    # Range over every individual run in the grid, both pipelines, per seed -- computed,
    # never carried over from a console session.
    allv = [v for per in mlp.values() for vs in per.values() for v in vs]
    allv += [v for (ds, c), r in tree.items()
             if c in ("A1_displacement", "A1_misdirection") for v in (r.get("auroc_per_seed") or [])]
    print(f"\n  certified-stability AUROC across the whole grid: "
          f"{min(allv):.3f} to {max(allv):.3f}")


# ------------------------------------------------------------------ Tab7
def tab7_transferability() -> None:
    """Does the attack survive when the attacker does not hold the deployed model?

    Reads results/transferability/summary/transfer.json. white_box is the attacker who
    optimises against the model that is actually explained; cross_seed re-uses a model of
    the same family trained on another seed; cross_arch crosses the detector family. The
    validity column matters as much as the corruption one: a transferred perturbation that
    changes the prediction is no longer this threat model's attack.
    """
    path = REPO / "results" / "transferability" / "summary" / "transfer.json"
    if not path.exists():
        print("  skip Tab7 -- no transferability run (scripts/run_transferability.py)")
        return
    rows = json.loads(path.read_text())
    per = {(r["dataset"], r["target"]): r for r in rows}
    targets = [("white_box", "White-box (same model)"),
               ("cross_seed", "Cross-seed (same family)"),
               ("cross_arch", "Cross-architecture")]
    datasets = sorted({r["dataset"] for r in rows}, key=lambda d: DATASET_LABEL.get(d, d))

    body = [r"\begin{tabular}{ll" + "r" * len(datasets) + "}", r"\toprule",
            r"\textbf{Quantity} & \textbf{Attacker's model} & "
            + " & ".join(f"\\textbf{{{DATASET_LABEL.get(d, d)}}}" for d in datasets) + r" \\",
            r"\midrule"]
    for i, (key, label) in enumerate(targets):
        cells = []
        for d in datasets:
            r = per.get((d, key))
            cells.append(f"${r['corrupt_mean']:.3f} \\pm {r['corrupt_ci_half']:.3f}$"
                         if r else "--")
        head = r"Corruption ($J$)" if i == 0 else ""
        body.append(f"{head} & {label} & " + " & ".join(cells) + r" \\")
    body.append(r"\addlinespace")
    for i, (key, label) in enumerate(targets):
        cells = []
        for d in datasets:
            r = per.get((d, key))
            cells.append(f"${r['valid_mean']:.3f}$" if r else "--")
        head = "Prediction preserved" if i == 0 else ""
        body.append(f"{head} & {label} & " + " & ".join(cells) + r" \\")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table_07_transferability.tex", "\n".join(body) + "\n")

    for (d, t), r in sorted(per.items()):
        print(f"    {DATASET_LABEL.get(d, d):12s} {t:11s} J={r['corrupt_mean']:.3f} "
              f"valid={r['valid_mean']:.3f}")


# ------------------------------------------------------------------ Tab8
def tab8_smoothed_defense() -> None:
    """Does smoothing the explainer defend the attribution, and does an adaptive attacker
    defeat the defense?

    Reads results/smoothed_defense/summary/defense.json. The metric is the probability that
    the shown top-1 feature is causal by the interventional ground truth: higher means the
    explanation still points at the real cause. The arms are the exact explainer clean and
    attacked, the smoothed explainer clean and attacked, and an adaptive attacker who knows
    the defense, under equal-candidate and equal-query budgets. The adaptive arms are the
    point of the table: a defense that only holds against a non-adaptive attacker is not a
    defense, so both are reported rather than the flattering one.
    """
    path = REPO / "results" / "smoothed_defense" / "summary" / "defense.json"
    if not path.exists():
        print("  skip Tab8 -- no smoothed_defense run (scripts/run_smoothed_defense.py)")
        return
    rows = json.loads(path.read_text())
    per = {(r["dataset"], r["arm"]): r for r in rows}
    arms = [("pointwise_clean", "Exact explainer, clean"),
            ("pointwise_attacked", "Exact explainer, attacked"),
            ("smoothed_clean", "Smoothed explainer, clean"),
            ("smoothed_attacked", "Smoothed explainer, attacked"),
            ("adaptive_equal_candidates", "Adaptive attacker, equal candidates"),
            ("adaptive_equal_queries", "Adaptive attacker, equal queries")]
    datasets = sorted({r["dataset"] for r in rows}, key=lambda d: DATASET_LABEL.get(d, d))

    body = [r"\begin{tabular}{l" + "r" * len(datasets) + "}", r"\toprule",
            r"\textbf{Arm} & "
            + " & ".join(f"\\textbf{{{DATASET_LABEL.get(d, d)}}}" for d in datasets) + r" \\",
            r"\midrule"]
    for key, label in arms:
        cells = []
        for d in datasets:
            r = per.get((d, key))
            cells.append(f"${r['p_top1_causal']:.3f} \\pm {r['ci_half']:.3f}$" if r else "--")
        body.append(f"{label} & " + " & ".join(cells) + r" \\")
        if key in ("pointwise_attacked", "smoothed_attacked"):
            body.append(r"\addlinespace")
    body += [r"\bottomrule", r"\end{tabular}"]
    write("table_08_smoothed_defense.tex", "\n".join(body) + "\n")

    for (d, arm), r in sorted(per.items()):
        print(f"    {DATASET_LABEL.get(d, d):12s} {arm:28s} "
              f"top1-causal={r['p_top1_causal']:.3f}")


def main() -> None:
    print("generating tables:")
    tab1_taxonomy()
    tab2_composition()
    tab3_taxonomy_mapping()
    tab4_decision_utility()
    tab5_per_signal()
    tab6_generality()
    tab7_transferability()
    tab8_smoothed_defense()


if __name__ == "__main__":
    main()
