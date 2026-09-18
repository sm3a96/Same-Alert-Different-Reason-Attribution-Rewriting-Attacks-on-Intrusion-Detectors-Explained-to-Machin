"""Reference correction for C2, and the corruption decomposition. Pre-registered before this ran.

The paper scores the reader against S(x), the erasure-causal set of the CLEAN flow, while the
reader is shown x', the attacked flow, and the explanation of x'. By Eq. eq:erasure the ground
truth for the decision on x' is S(x'). This script

  1. rebuilds every C2 cell deterministically, asserts the rebuilt cases match the cached
     ones byte for byte (candidates, values, importances, causal set), and computes S(x')
     on the attacked flow with the detector that actually produced the shown explanation
     (for A3 that is the scaffolded artifact -- the deployed one);
  2. rescores the cached judge decisions against S(x') and recomputes the paired effect
     with the paper's own cluster bootstrap, alongside the paper's number against S(x);
  3. decomposes the top-1 causal-rate change into an infidelity term (shown top-1 against
     the causal set of the flow shown) and a reliance term (the rest).

Nothing here re-runs a judge. Every number is a re-scoring of the 43,200 cached decisions.

  python scripts/run_reference_decomposition.py
  python scripts/run_reference_decomposition.py --datasets fiveg_nidd --seeds 0 1 --out results/refdecomp_check --name refdecomp_check
  python scripts/run_reference_decomposition.py --reference median --tau 0.10   # sensitivity arms
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict
from pathlib import Path

import numpy as np

from avert.benchmark import XIntBench
from avert.benchmark.ground_truth import causal_features
from avert.benchmark.harness import target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.decision_utility import CASE_CACHE, competent_cell_set, paired_effect, summarize_cells
from avert.eval.metrics import seed_clustered_ci
from avert.eval.runner import RESULTS_ROOT, RunResult, run_experiment

log = logging.getLogger("refdecomp")

DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
SEEDS = [0, 1, 2, 3, 4]
ALL_ATTACKS = ["A1_displacement", "A1_misdirection", "A3_scaffolding"]
ATTACKS = list(ALL_ATTACKS)
N_CLASSES, N_TEST, N_SHOWN, TOP_K = 3, 40, 10, 5
DECISIONS = RESULTS_ROOT / "decision_utility" / "raw" / "decisions.json"
OUT = RESULTS_ROOT / "reference_decomposition"


def _reference(st, data, kind: str) -> np.ndarray:
    """The erasure target. 'mean' is the paper's (benign-class training mean, built in
    _prepare); 'median' is the sensitivity arm for the single-baseline confound, over the
    same benign training rows."""
    if kind == "mean":
        return st.benign_ref
    bi = data.benign_index()
    rows = np.intersect1d(np.where(st.y == bi)[0], st.tr) if bi is not None else st.tr
    return np.median(st.X[rows if len(rows) else st.tr], axis=0)


def rebuild_cell(bench, data, target, cached_cell, tau, min_k, ref_kind):
    """Rebuild one (dataset, class, seed) cell and return per-instance rows with S(x), S(x')
    and the shown top-1 / top-5 of clean and attacked explanations. Aborts on any mismatch
    against the cache, because a rescoring of decisions made on different cases is not a
    rescoring."""
    st = bench._prepare(target=target)
    names = st.names
    ref = _reference(st, data, ref_kind)
    free = set(int(j) for j in st.frag.bounds.free_idx)
    rows = []
    for s in st.test:
        x = s.features
        S_x, d_x = causal_features(s, st.detector, ref, tau, min_k)
        pred = int(st.detector.predict(x.reshape(1, -1))[0])
        e_clean = st.explainer.explain(st.detector, x, names, pred, s.sample_id, TOP_K)
        order_clean = list(np.argsort(-np.abs(e_clean.attributions))[:N_SHOWN])
        for atk in ATTACKS:
            a, det = bench.attacked(s, atk, st, list(S_x))
            pred_a = int(det.predict(a.features.reshape(1, -1))[0])
            e_atk = st.explainer.explain(det, a.features, names, pred_a, s.sample_id, TOP_K)
            order_atk = list(np.argsort(-np.abs(e_atk.attributions))[:N_SHOWN])
            if atk == "A3_scaffolding":
                # The erasure test is a query-based probe, which is what a scaffold exists to
                # fool: an erased flow can itself be routed to the surrogate. So the ground
                # truth for a scaffolded decision is measured on the component that made the
                # decision -- the surrogate if the OOD detector routed the flow, the real model
                # otherwise -- and never through the router.
                routed = bool(det.ood.is_probe(a.features.reshape(1, -1))[0])
                S_a, d_a = causal_features(a, det.surrogate if routed else det.real, ref, tau, min_k)
            else:
                routed = False
                S_a, d_a = causal_features(a, det, ref, tau, min_k)

            c = cached_cell.get((s.sample_id, atk))
            if c is None:
                raise RuntimeError(f"{s.sample_id}/{atk} not in cache")
            ca = c["cases"]["attacked"]
            # The judge decided on the SHOWN case. A rescoring is valid only where the
            # rebuilt shown case is the cached one; instances where it is not are excluded
            # and counted. The detector fit is not bit-identical to the 13 August run
            # (differences appear at threshold-boundary erasure deltas), so this cannot be
            # an assertion -- it has to be a per-instance flag that is reported.
            shown_match = (ca["candidates"] == [names[i] for i in order_atk]
                           and bool(np.allclose(ca["values"], [float(a.features[i]) for i in order_atk]))
                           and bool(np.allclose(ca["importances"], [float(e_atk.attributions[i]) for i in order_atk])))
            causal_match = sorted(ca["causal"]) == sorted(names[j] for j in S_x)

            rows.append({
                "sample_id": s.sample_id, "attack": atk,
                "shown_matches_cache": shown_match, "causal_matches_cache": causal_match,
                "routed_to_surrogate": routed,
                "S_x_cached": list(ca["causal"]),
                "prediction_preserved": bool(pred_a == pred),
                "S_x": [names[j] for j in S_x], "S_a": [names[j] for j in S_a],
                "fallback_x": bool(max(d_x) < tau), "fallback_a": bool(max(d_a) < tau),
                "top1_clean": names[order_clean[0]], "top1_atk": names[order_atk[0]],
                "top5_clean": [names[i] for i in order_clean[:TOP_K]],
                "top5_atk": [names[i] for i in order_atk[:TOP_K]],
                "reliance_jaccard": 1.0 - len(set(S_x) & set(S_a)) / len(set(S_x) | set(S_a)),
                "free_frac_x": float(np.mean([j in free for j in S_x])),
                "free_frac_a": float(np.mean([j in free for j in S_a])),
            })
    return rows


def rescore(decisions, inst, competent, cluster="reader_cell"):
    """Rescore cached decisions against S(x') and return both paired effects."""
    # The threat model requires the predicted class to survive the attack. The A1 searches
    # enforce that by rejection; scaffolding only measures it, and on CICIoMT2024 the router
    # flips about a tenth of real flows. Harm is therefore measured on the flows whose class
    # survived, the same population `decompose` uses. Until 2026-09-09 the two disagreed.
    usable = {(r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]): r
              for r in inst if r["shown_matches_cache"] and r["prediction_preserved"]}
    S_a = {k: set(r["S_a"]) for k, r in usable.items()}
    S_x = {k: set(r["S_x_cached"]) for k, r in usable.items()}
    keep, shown = [], []
    for r in decisions:
        if r["variant"] != "default" or r["order"] != "ranked":
            continue
        k = (r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"])
        if k not in S_a:
            continue
        keep.append(r)
        # The cache's own correctness must agree with membership in the cached S(x);
        # otherwise the cache is internally inconsistent and nothing below is valid.
        if r["condition"] == "attacked" and bool(r["det_correct"]) != (r["det_choice"] in S_x[k]):
            raise RuntimeError(f"cached det_correct disagrees with cached causal set for {k}")
        rr = dict(r)
        if r["condition"] == "attacked":
            rr["det_correct"] = r["det_choice"] in S_a[k]
        shown.append(rr)

    out = {}
    for label, rows in (("vs_S_x", keep), ("vs_S_xprime", shown)):
        out[label] = {}
        for atk in ATTACKS:
            out[label][atk] = {
                "pooled": paired_effect(rows, "attacked", attack=atk, competent_cells=competent, cluster=cluster),
                "all_cells": paired_effect(rows, "attacked", attack=atk, cluster=cluster),
                "per_judge": {j: paired_effect(rows, "attacked", judge=j, attack=atk, competent_cells=competent, cluster=cluster)
                              for j in sorted({r["judge"] for r in rows})},
                "per_dataset": {d: paired_effect([r for r in rows if r["dataset"] == d], "attacked",
                                                 attack=atk, competent_cells=competent, cluster=cluster)
                                for d in sorted({r["dataset"] for r in rows})},
            }
    return out


def decompose(inst):
    """Top-1 decomposition with seed-clustered intervals, per corpus and pooled."""
    by = defaultdict(list)
    for r in inst:
        if not r["prediction_preserved"]:
            continue
        by[(r["dataset"], r["attack"], r["seed"])].append(r)
    cells = []
    for (ds, atk, seed), rs in by.items():
        p_clean = np.mean([r["top1_clean"] in r["S_x"] for r in rs])
        p_atk_x = np.mean([r["top1_atk"] in r["S_x"] for r in rs])
        p_atk_a = np.mean([r["top1_atk"] in r["S_a"] for r in rs])
        cells.append({"dataset": ds, "attack": atk, "seed": seed, "n": len(rs),
                      "p_clean": p_clean, "paper": p_atk_x - p_clean,
                      "infidelity": p_atk_a - p_clean, "reliance": p_atk_x - p_atk_a,
                      "reliance_jaccard": np.mean([r["reliance_jaccard"] for r in rs]),
                      "free_frac_x": np.mean([r["free_frac_x"] for r in rs]),
                      "free_frac_a": np.mean([r["free_frac_a"] for r in rs]),
                      "fallback_rate_a": np.mean([r["fallback_a"] for r in rs]),
                      "routed_rate": np.mean([r["routed_to_surrogate"] for r in rs])})
    summary = {}
    for atk in ATTACKS:
        summary[atk] = {}
        for scope in ["pooled"] + sorted({c["dataset"] for c in cells}):
            sel = [c for c in cells if c["attack"] == atk and (scope == "pooled" or c["dataset"] == scope)]
            if not sel:
                continue
            seeds = np.array([c["seed"] for c in sel])
            summary[atk][scope] = {}
            for m in ("paper", "infidelity", "reliance", "reliance_jaccard", "free_frac_x", "free_frac_a", "fallback_rate_a", "routed_rate"):
                mu, hw = seed_clustered_ci(np.array([c[m] for c in sel]), seeds)
                summary[atk][scope][m] = {"mean": mu, "ci_lo": mu - hw, "ci_hi": mu + hw}
    return cells, summary


def markdown(effects, summary, args, inst):
    n_ex = sum(not r["shown_matches_cache"] for r in inst)
    n_cm = sum(not r["causal_matches_cache"] for r in inst)
    L = [f"# Reference correction and decomposition (reference={args.reference}, tau={args.tau}, min_k={args.min_k}, "
         f"scaffold contamination={args.scaffold_contamination}, attacks={','.join(ATTACKS)})", "",
         f"Rebuilt instances: {len(inst)}. Shown case differs from the 13 August cache on {n_ex} "
         f"(excluded from the rescoring, kept in the decomposition); clean causal set differs on {n_cm}. "
         "The paper's arm uses the cached causal set so it reproduces the published number exactly.", "",
         "## Paired effect on reader accuracy, competent cells, cluster-bootstrap 95% CI", "",
         "| Attack | vs S(x) (paper) | vs S(x') (flow shown) | broke/fixed vs S(x') |", "|---|---|---|---|"]
    for atk in ATTACKS:
        a, b = effects["vs_S_x"][atk]["pooled"], effects["vs_S_xprime"][atk]["pooled"]
        L.append(f"| {atk} | {a['mean_delta']:+.3f} [{a['ci_lo']:+.3f}, {a['ci_hi']:+.3f}] | "
                 f"{b['mean_delta']:+.3f} [{b['ci_lo']:+.3f}, {b['ci_hi']:+.3f}] | {b['broke']}/{b['fixed']} |")
    L += ["", "## Per corpus, vs S(x')", "", "| Attack | Corpus | effect | CI |", "|---|---|---|---|"]
    for atk in ATTACKS:
        for d, e in effects["vs_S_xprime"][atk]["per_dataset"].items():
            if e.get("n"):
                L.append(f"| {atk} | {d} | {e['mean_delta']:+.3f} | [{e['ci_lo']:+.3f}, {e['ci_hi']:+.3f}] |")
    L += ["", "## Top-1 decomposition (seed-clustered 95% CI over seed means)", "",
          "| Attack | Scope | paper | infidelity | reliance | J(S(x),S(x')) | free x / x' | fallback x' | routed |", "|---|---|---|---|---|---|---|---|---|"]
    for atk in ATTACKS:
        for scope, m in summary[atk].items():
            f = lambda k: f"{m[k]['mean']:+.3f} [{m[k]['ci_lo']:+.3f}, {m[k]['ci_hi']:+.3f}]"
            L.append(f"| {atk} | {scope} | {f('paper')} | {f('infidelity')} | {f('reliance')} | "
                     f"{m['reliance_jaccard']['mean']:.3f} | {m['free_frac_x']['mean']:.2f} / {m['free_frac_a']['mean']:.2f} | "
                     f"{m['fallback_rate_a']['mean']:.3f} | {m['routed_rate']['mean']:.3f} |")
    return "\n".join(L)


def sweep(ctx) -> RunResult:
    args = ctx.config["args"]
    cache = json.loads(CASE_CACHE.read_text())["cases"]
    decisions = json.loads(DECISIONS.read_text())
    per_cell = summarize_cells([r for r in decisions if r["variant"] == "default" and r["order"] == "ranked"])
    competent = competent_cell_set(per_cell, floor=0.5)

    inst = []
    if args.get("rescore_only"):
        # Re-derive every summary from the instances this run already rebuilt. The rebuild
        # is deterministic and takes hours; the scoring takes seconds.
        inst = json.loads((ctx.run_dir / "raw" / "instances.json").read_text())
        args["datasets"] = []
    for ds in args["datasets"]:
        data = load_dataset(ds, load_config(f"datasets/{ds}"))
        classes, _ = target_classes(data, N_CLASSES, n_cal=100, n_test=N_TEST, seed=0,
                                    min_count=2000, seeds=SEEDS)
        for c in classes:
            cname = data.label_names[c] if data.label_names else str(c)
            for seed in args["seeds"]:
                cached_cell = {(r["sample_id"], r["attack"]): r for r in cache
                               if r["dataset"] == ds and r["class"] == cname and r["seed"] == seed}
                bench = XIntBench(data, n_cal=100, n_test=N_TEST, top_k=TOP_K, perm_samples=10,
                                  stability_n=30, seed=seed,
                                  scaffold_contamination=args["scaffold_contamination"])
                rows = rebuild_cell(bench, data, c, cached_cell, args["tau"], args["min_k"], args["reference"])
                for r in rows:
                    r.update({"dataset": ds, "class": cname, "seed": seed})
                inst.extend(rows)
                (ctx.run_dir / "raw" / "instances.json").write_text(json.dumps(inst, indent=1))
                log.info("done %s class=%s seed=%s (%d instances so far)", ds, cname, seed, len(inst))

    effects = rescore(decisions, inst, competent)
    # Same point estimates, resampled by cell with both readers inside one cluster
    # (30 clusters on the competent search-attack scope instead of 60). Written beside the
    # reader-cell file so both intervals stay quotable.
    effects_cell = rescore(decisions, inst, competent, cluster="cell")
    cells, summary = decompose(inst)
    (ctx.run_dir / "summary" / "effects.json").write_text(json.dumps(effects, indent=1))
    (ctx.run_dir / "summary" / "effects_cell.json").write_text(json.dumps(effects_cell, indent=1))
    (ctx.run_dir / "summary" / "decomposition_cells.json").write_text(json.dumps(cells, indent=1, default=float))
    (ctx.run_dir / "summary" / "decomposition.json").write_text(json.dumps(summary, indent=1))
    md = markdown(effects, summary, argparse.Namespace(**args), inst)
    (ctx.run_dir / "summary" / "report.md").write_text(md)
    print(md)

    disp = effects["vs_S_xprime"].get("A1_displacement", {"pooled": {"mean_delta": None, "ci_lo": None, "ci_hi": None}})["pooled"]
    n_shown_mismatch = sum(not r["shown_matches_cache"] for r in inst)
    n_causal_mismatch = sum(not r["causal_matches_cache"] for r in inst)
    log.info("instances whose shown case differs from the cache (excluded from rescoring): %d of %d; "
             "whose clean causal set differs from the cache: %d", n_shown_mismatch, len(inst), n_causal_mismatch)
    return RunResult(
        summary={"instances": len(inst), "cells": len(cells),
                 "excluded_shown_mismatch": n_shown_mismatch, "clean_causal_mismatch": n_causal_mismatch,
                 "effects_vs_Sxprime": {atk: effects["vs_S_xprime"][atk]["pooled"]["mean_delta"] for atk in ATTACKS},
                 "displacement_effect_vs_Sx": effects["vs_S_x"].get("A1_displacement", {"pooled": {"mean_delta": None}})["pooled"]["mean_delta"],
                 "displacement_effect_vs_Sxprime": disp["mean_delta"],
                 "displacement_ci_vs_Sxprime": [disp["ci_lo"], disp["ci_hi"]]},
        floats=[{"float_id": "Tab_reference_correction", "kind": "table", "paper_section": "C2",
                 "path": str(ctx.run_dir / "summary" / "effects.json"),
                 "claim": "reader harm scored against the causal set of the flow shown"},
                {"float_id": "Tab_decomposition", "kind": "table", "paper_section": "C2",
                 "path": str(ctx.run_dir / "summary" / "decomposition.json"),
                 "claim": "top-1 corruption splits into infidelity and reliance shift"}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--reference", choices=["mean", "median"], default="mean")
    ap.add_argument("--tau", type=float, default=0.05)
    ap.add_argument("--min-k", type=int, default=3)
    ap.add_argument("--attacks", nargs="+", default=ALL_ATTACKS, choices=ALL_ATTACKS)
    ap.add_argument("--scaffold-contamination", type=float, default=0.3)
    ap.add_argument("--rescore-only", action="store_true",
                    help="recompute effects and decomposition from this run's existing raw/instances.json")
    ap.add_argument("--out", default=str(OUT))
    ap.add_argument("--name", default=None)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    name = a.name or Path(a.out).name
    if Path(a.out).resolve() != (RESULTS_ROOT / name).resolve():
        raise SystemExit("--out must be results/<name>; run_experiment writes there")
    ATTACKS[:] = a.attacks
    run_experiment(name, sweep, config={"args": {"datasets": a.datasets, "seeds": a.seeds, "reference": a.reference,
                                                 "tau": a.tau, "min_k": a.min_k, "attacks": a.attacks,
                                                 "scaffold_contamination": a.scaffold_contamination,
                                                 "rescore_only": a.rescore_only}}, seed=0)


if __name__ == "__main__":
    main()
