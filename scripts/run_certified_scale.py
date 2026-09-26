"""Certified stability at the attack's scale: the check alone, noise in sigma_train units.

The matrix instantiates the certified-stability check with sigma = 0.1 in units of the
calibration flows' spread, and a 1e-8 floor on coordinates that are constant on those flows.
The median free coordinate has sigma_cal / sigma_train = 0.03, so the certificate's largest
radius sits about two orders of magnitude below a perturbation specified at 0.15 sigma_train
per coordinate. A verdict of "not deployable" drawn at that scale is a verdict on the scale.

This runs the same two certified-stability checks as the matrix (every coordinate, and the
attacker-settable coordinates only) with the noise on coordinate j equal to
sigma * sigma_train[j], no floor, at sigma = 0.05, 0.15 and 0.5, against cause displacement
and rank promotion on the 45 matrix cells. Nothing else in the panel is run.

The attacked flows are the matrix's own. Each cell is built by `XIntBench._prepare` with the
matrix's arguments, and the two searches are applied in the matrix's order (rank promotion on
every test flow, then cause displacement on every test flow), so each search's generator is
consumed exactly as in the matrix. The per-flow shown-set corruption is compared with
results/matrix_a3_survivors/raw/per_flow.json, which the matrix wrote with the same code, and
the number of flows that differ is recorded.

The AUROC of a check on a cell is attacked test flows against clean test flows, score -R, as
the matrix computes it. Per corpus it is the seed-clustered mean over the cell AUROCs with the
interval over the five seed means. The ceiling is the largest radius the certificate can
return at N draws: sigma * Phi^-1(p_max) with p_max = conf^(1/N), the Clopper-Pearson lower
bound when every draw agrees.

The grid runs one process per corpus, each writing raw/per_flow_<corpus>.json, and a last
call with --summarize reads every per-corpus file present and writes the summary:

  python scripts/run_certified_scale.py --datasets fiveg_nidd     # and the other two
  python scripts/run_certified_scale.py --summarize
"""
from __future__ import annotations

import argparse
import json
import logging
from collections import defaultdict

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from avert.benchmark import XIntBench
from avert.benchmark.harness import target_classes
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.metrics import seed_clustered_ci
from avert.eval.runner import RESULTS_ROOT, RunResult, run_experiment
from avert.signals.certified_stability import CertifiedStabilitySignal, clopper_pearson_lower

log = logging.getLogger("certified_scale")

DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
SEEDS = [0, 1, 2, 3, 4]
SIGMAS = [0.05, 0.15, 0.5]
ATTACKS = ["A1_misdirection", "A1_displacement"]      # the matrix's order
N_CLASSES, N_CAL, N_TEST, TOP_K, N_DRAWS, CONF = 3, 100, 40, 5, 30, 0.05
MATRIX_FLOWS = RESULTS_ROOT / "matrix_a3_survivors" / "raw" / "per_flow.json"


def ceiling(sigma: float, n: int = N_DRAWS, conf: float = CONF) -> float:
    """Largest radius the certificate can return: every one of the n draws agrees."""
    return float(sigma * norm.ppf(clopper_pearson_lower(n, n, conf)))


def _jaccard_distance(a: set, b: set) -> float:
    return 1.0 - (len(a & b) / len(a | b)) if (a | b) else 0.0


def run_cell(data, target, seed, sigmas):
    """Score clean and attacked test flows of one cell with every certified variant."""
    bench = XIntBench(data, n_cal=N_CAL, n_test=N_TEST, top_k=TOP_K, perm_samples=10,
                      stability_n=N_DRAWS, seed=seed)
    st = bench._prepare(target=target)
    names, det, g = st.names, st.detector, st.explainer
    train_std = st.X[st.tr].std(axis=0)
    checks = {}
    for s in sigmas:
        for variant, bounds in (("all", None), ("free", st.frag.bounds)):
            sig = CertifiedStabilitySignal(g, n=N_DRAWS, sigma=s, top_k=TOP_K, seed=seed,
                                           bounds=bounds, train_std=train_std)
            sig.calibrate(st.cal, None, det)
            checks[f"sigma{s}_{variant}"] = sig

    def score(sample):
        return {k: sig.certified_radius(sample, det) for k, sig in checks.items()}

    clean_topk, clean_rows = {}, []
    for s in st.test:
        pred = int(det.predict(s.features.reshape(1, -1))[0])
        clean_topk[s.sample_id] = set(g.explain(det, s.features, names, pred, s.sample_id,
                                                TOP_K).top_features())
        r = score(s)
        clean_rows.append({"sample_id": s.sample_id,
                           "radius": {k: v[0] for k, v in r.items()},
                           "p_lo": {k: v[1] for k, v in r.items()}})

    from avert.benchmark.ground_truth import causal_features
    attacked = {}
    for atk in ATTACKS:
        rows = []
        for s in st.test:
            causal, _ = causal_features(s, det, st.benign_ref, 0.05, 3)
            a, d = bench.attacked(s, atk, st, list(causal))
            c0 = int(det.predict(s.features.reshape(1, -1))[0])
            pred_a = int(d.predict(a.features.reshape(1, -1))[0])
            e = g.explain(d, a.features, names, pred_a, s.sample_id, TOP_K)
            r = score(a)
            rows.append({"sample_id": s.sample_id,
                         "prediction_preserved": bool(pred_a == c0),
                         "corrupt": _jaccard_distance(clean_topk[s.sample_id],
                                                      set(e.top_features())),
                         "radius": {k: v[0] for k, v in r.items()},
                         "p_lo": {k: v[1] for k, v in r.items()}})
        attacked[atk] = rows
    return {"clean": clean_rows, "attacks": attacked, "checks": sorted(checks),
            "sigma_train_median_free": float(np.median(train_std[np.asarray(
                st.frag.bounds.free_idx, dtype=int)]))}


def cell_auroc(cell, atk, key):
    clean = [-r["radius"][key] for r in cell["clean"]]
    att = [-r["radius"][key] for r in cell["attacks"][atk]]
    return float(roc_auc_score([0] * len(clean) + [1] * len(att), clean + att))


def reproduces_matrix(cells) -> dict:
    """Per-flow corruption against the matrix's own per-flow file, flow by flow."""
    if not MATRIX_FLOWS.exists():
        return {"checked": 0, "mismatched": None}
    ref = {}
    for c in json.loads(MATRIX_FLOWS.read_text()):
        for atk, rows in c["attacks"].items():
            for r in rows:
                ref[(c["dataset"], c["class"], c["seed"], atk, r["sample_id"])] = r["corrupt"]
    checked = mism = 0
    for c in cells:
        for atk, rows in c["attacks"].items():
            for r in rows:
                k = (c["dataset"], c["class"], c["seed"], atk, r["sample_id"])
                if k in ref:
                    checked += 1
                    mism += abs(ref[k] - r["corrupt"]) > 1e-9
    return {"checked": checked, "mismatched": int(mism)}


def summarize(cells, sigmas):
    out = defaultdict(dict)
    keys = [f"sigma{s}_{v}" for s in sigmas for v in ("all", "free")]
    for atk in ATTACKS:
        for ds in sorted({c["dataset"] for c in cells}):
            sel = [c for c in cells if c["dataset"] == ds]
            seeds = np.array([c["seed"] for c in sel])
            out[atk][ds] = {}
            for k in keys:
                vals = np.array([cell_auroc(c, atk, k) for c in sel])
                mu, hw = seed_clustered_ci(vals, seeds)
                clean_r = [r["radius"][k] for c in sel for r in c["clean"]]
                att_r = [r["radius"][k] for c in sel for r in c["attacks"][atk]]
                out[atk][ds][k] = {"mean": mu, "ci_lo": mu - hw, "ci_hi": mu + hw,
                                   "n_cells": len(sel),
                                   "certified_clean": float(np.mean(np.array(clean_r) > 0)),
                                   "certified_attacked": float(np.mean(np.array(att_r) > 0)),
                                   "median_radius_clean": float(np.median(clean_r)),
                                   "median_radius_attacked": float(np.median(att_r))}
    return out


def markdown(summary, ceilings, control, sigmas) -> str:
    L = ["# Certified stability with noise in sigma_train units (N = 30, conf = 0.05)", "",
         f"Per-flow corruption against the matrix: {control['checked']} flows checked, "
         f"{control['mismatched']} differ.", "",
         "Ceiling (largest certifiable radius, sigma_train units): "
         + ", ".join(f"sigma={s}: {ceilings[str(s)]:.3f}" for s in sigmas), "",
         "| Attack | Corpus | Check | AUROC | 95% CI | certified clean | certified attacked |",
         "|---|---|---|---|---|---|---|"]
    for atk, per in summary.items():
        for ds, per_k in per.items():
            for k, m in per_k.items():
                L.append(f"| {atk} | {ds} | {k} | {m['mean']:.3f} | "
                         f"[{m['ci_lo']:.3f}, {m['ci_hi']:.3f}] | {m['certified_clean']:.3f} | "
                         f"{m['certified_attacked']:.3f} |")
    return "\n".join(L)


def run_corpus(ctx) -> RunResult:
    """Score every cell of one corpus and write raw/per_flow_<corpus>.json."""
    args = ctx.config["args"]
    sigmas = args["sigmas"]
    (ds,) = args["datasets"]
    cells = []
    raw = ctx.run_dir / "raw" / f"per_flow_{ds}.json"
    data = load_dataset(ds, load_config(f"datasets/{ds}"))
    classes, _ = target_classes(data, N_CLASSES, n_cal=N_CAL, n_test=N_TEST, seed=0,
                                min_count=2000, seeds=SEEDS)
    for c in classes:
        cname = data.label_names[c] if data.label_names else str(c)
        for seed in args["seeds"]:
            cell = run_cell(data, c, seed, sigmas)
            cells.append({"dataset": ds, "class": cname, "seed": seed, **cell})
            raw.write_text(json.dumps(cells))
            log.info("done %s class=%s seed=%s (%d cells)", ds, cname, seed, len(cells))
    return RunResult(summary={"dataset": ds, "cells": len(cells)},
                     floats=[{"float_id": f"certified_scale_per_flow_{ds}", "kind": "raw",
                              "paper_section": "detect", "path": str(raw),
                              "claim": "per-flow certified radius and lower bound for every "
                                       "variant, one corpus"}])


def summarize_all(ctx) -> RunResult:
    """Read every per-corpus file and write the per-corpus AUROCs and the ceilings."""
    sigmas = ctx.config["args"]["sigmas"]
    cells = []
    for f in sorted((ctx.run_dir / "raw").glob("per_flow_*.json")):
        cells += json.loads(f.read_text())
    control = reproduces_matrix(cells)
    ceilings = {str(s): ceiling(s) for s in sigmas}
    summary = summarize(cells, sigmas)
    (ctx.run_dir / "summary" / "auroc.json").write_text(json.dumps(
        {"auroc": summary, "ceiling_sigma_train": ceilings, "n_draws": N_DRAWS, "conf": CONF,
         "reproduces_matrix_corruption": control}, indent=1))
    md = markdown(summary, ceilings, control, sigmas)
    (ctx.run_dir / "summary" / "report.md").write_text(md)
    print(md)
    return RunResult(
        summary={"cells": len(cells), "sigmas": sigmas, "ceilings": ceilings,
                 "corruption_mismatch": control},
        floats=[{"float_id": "Tab_certified_scale", "kind": "table", "paper_section": "detect",
                 "path": str(ctx.run_dir / "summary" / "auroc.json"),
                 "claim": "certified-stability AUROC with the noise on the attack's scale"}])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--seeds", nargs="+", type=int, default=SEEDS)
    ap.add_argument("--sigmas", nargs="+", type=float, default=SIGMAS)
    ap.add_argument("--out-name", default="certified_scale")
    ap.add_argument("--summarize", action="store_true",
                    help="aggregate every raw/per_flow_<corpus>.json into the summary")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    config = {"args": {"datasets": a.datasets, "seeds": a.seeds, "sigmas": a.sigmas,
                       "n_draws": N_DRAWS, "conf": CONF, "sigma_unit": "sigma_train"}}
    if a.summarize:
        run_experiment(a.out_name, summarize_all, config=config, seed=0)
        return
    for ds in a.datasets:
        config["args"]["datasets"] = [ds]
        run_experiment(a.out_name, run_corpus, config=config, seed=0,
                       meta_name=f"run_{ds}.json")


if __name__ == "__main__":
    main()
