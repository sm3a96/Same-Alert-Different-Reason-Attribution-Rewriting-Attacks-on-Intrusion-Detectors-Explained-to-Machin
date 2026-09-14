"""C3 — save the per-sample artifacts the paper's figures read.

The June runs recorded only scalars (one false-alarm number per dataset), so the
calibration figure, the certified-radius figure and the interventional-verification
figure had nothing to plot from. This regenerates all of them as per-sample rows, so
every C3 float comes out of a saved artifact rather than out of a summary line.

Per dataset it writes to results/c3_artifacts_<dataset>/raw/:

  conformal.csv    one clean held-out flow per row: fused score and conformal
                   p-value. Sweeping alpha over these gives the calibration line
                   (Fig6) instead of a single point.
  radii.csv        per-sample certified radius on CLEAN flows (Fig5). The negative
                   result is that most of these are zero.
  ablation.csv     detection-probability drop as a feature is ablated toward benign,
                   for ground-truth-causal vs non-causal features, swept over the
                   ablation fraction (Fig2). This is what makes the ground truth
                   checkable rather than asserted.
  signal_auroc.csv per-signal AUROC, clean vs attacked, per attack class (Tab5).
                   run_matrix.py kept only the best signal; the table needs all of them.

  python scripts/run_c3_artifacts.py                 # all three datasets
  python scripts/run_c3_artifacts.py --dataset fiveg_nidd --seeds 0
"""
from __future__ import annotations

import argparse
import csv

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from avert.benchmark import XIntBench
from avert.benchmark.ground_truth import causal_features
from avert.config import load_config
from avert.data.datasets import load_dataset
from avert.eval.runner import RunResult, run_experiment
from avert.fusion.conformal import ConformalFuser, SignalAggregator
from avert.signals.certified_stability import CertifiedStabilitySignal, clopper_pearson_lower
from avert.signals.cross_method_consensus import CrossMethodConsensusSignal
from avert.attribution.real import PermutationAttributor

ALL_DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
ATTACKS = ["A1_misdirection", "A1_displacement", "A3_scaffolding"]
ABLATION_FRACS = [0.0, 0.25, 0.5, 0.75, 1.0]


def write_csv(path, rows):
    if not rows:
        return
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)


def ablation_curve(sample, detector, benign_ref, causal_idx, n_noncausal, rng):
    """Detection probability as a feature is moved toward its benign value.

    A ground-truth feature must drag the detection down; a non-causal one must not.
    Sweeping the fraction rather than testing full ablation only shows the whole
    curve, which is harder to fake and easier for a reviewer to check.
    """
    x = sample.features
    pred = int(detector.predict(x.reshape(1, -1))[0])
    base = float(detector.predict_proba(x.reshape(1, -1))[0, pred])
    noncausal = [j for j in range(len(x)) if j not in set(causal_idx)]
    rng.shuffle(noncausal)
    rows = []
    for kind, idxs in (("causal", list(causal_idx)), ("non_causal", noncausal[:n_noncausal])):
        for j in idxs:
            for frac in ABLATION_FRACS:
                xj = x.copy()
                xj[j] = (1 - frac) * x[j] + frac * benign_ref[j]
                p = float(detector.predict_proba(xj.reshape(1, -1))[0, pred])
                rows.append({"sample_id": sample.sample_id, "feature_kind": kind,
                             "feature_idx": int(j), "ablation_frac": frac,
                             "detect_prob": round(p, 5),
                             "detect_drop": round(base - p, 5)})
    return rows


def run_dataset(ds: str, seeds: list[int], n_test: int, n_ablation: int,
                stab_n: int, stab_sigma: float):
    data = load_dataset(ds, load_config(f"datasets/{ds}"))

    def experiment(ctx):
        conformal_rows, radii_rows, ablation_rows, auroc_rows = [], [], [], []
        for seed in seeds:
            bench = XIntBench(data, n_cal=150, n_test=n_test, top_k=5,
                              perm_samples=10, stability_n=stab_n, seed=seed)
            st = bench._prepare()
            det, expl, names = st.detector, st.explainer, st.names
            rng = np.random.default_rng(seed)

            permutation = PermutationAttributor(st.X[st.tr], n_samples=10, seed=seed)
            consensus = CrossMethodConsensusSignal([expl, permutation], top_k=5)
            stability = CertifiedStabilitySignal(expl, n=stab_n, sigma=stab_sigma, top_k=5, seed=seed)
            # The certified radius cannot exceed this, however stable the attribution is:
            # p_lo is a Clopper-Pearson bound, so it saturates at conf^(1/n). Recording it is
            # what lets the figure show saturation rather than assert it.
            ceiling = stab_sigma * float(norm.ppf(clopper_pearson_lower(stab_n, stab_n, 0.05)))
            signals = [consensus, stability]

            cal_e = [expl.explain(det, s.features, names, st.target, s.sample_id, 5) for s in st.cal]
            for sig in signals:
                sig.calibrate(st.cal, cal_e, det)
            clean_cal = {sig.name: np.array([sig.score(s, e, det).score
                                             for s, e in zip(st.cal, cal_e)]) for sig in signals}
            agg = SignalAggregator().fit(clean_cal)
            fuser = ConformalFuser()
            fuser.calibrate(np.array([agg.aggregate({sig.name: sig.score(s, e, det) for sig in signals})
                                      for s, e in zip(st.cal, cal_e)]))

            def score_all(sample, d):
                e = expl.explain(d, sample.features, names,
                                 int(d.predict(sample.features.reshape(1, -1))[0]), sample.sample_id, 5)
                sc = {sig.name: sig.score(sample, e, d) for sig in signals}
                return sc, agg.aggregate(sc)

            # --- clean held-out: conformal p-values (Fig6) + certified radii (Fig5)
            clean_per_signal, gt = [], []
            for s in st.test:
                sc, fused = score_all(s, det)
                p = fuser.p_value(fused)
                conformal_rows.append({"dataset": ds, "seed": seed, "sample_id": s.sample_id,
                                       "fused_score": round(fused, 6), "p_value": round(p, 6)})
                r = float(sc[stability.name].evidence.get("certified_radius", 0.0))
                radii_rows.append({"dataset": ds, "seed": seed, "sample_id": s.sample_id,
                                   "condition": "clean", "certified_radius": round(r, 6),
                                   "certifiable": int(r > 0),
                                   "at_ceiling": int(abs(r - ceiling) < 1e-3 * max(ceiling, 1e-9))})
                clean_per_signal.append({k: v.score for k, v in sc.items()})
                causal, _ = causal_features(s, det, st.benign_ref, 0.05, 3)
                gt.append(causal)

            # --- interventional verification (Fig2): causal vs non-causal ablation
            for s, causal in list(zip(st.test, gt))[:n_ablation]:
                for r in ablation_curve(s, det, st.benign_ref, causal, 3, rng):
                    r.update({"dataset": ds, "seed": seed})
                    ablation_rows.append(r)

            # --- per-signal AUROC, clean vs attacked (Tab5) + attacked radii (Fig5)
            for attack in ATTACKS:
                atk_per_signal = []
                for s, causal in zip(st.test, gt):
                    a, d = bench.attacked(s, attack, st, list(causal))
                    sc, _ = score_all(a, d)
                    atk_per_signal.append({k: v.score for k, v in sc.items()})
                    r = float(sc[stability.name].evidence.get("certified_radius", 0.0))
                    radii_rows.append({"dataset": ds, "seed": seed, "sample_id": s.sample_id,
                                       "condition": attack, "certified_radius": round(r, 6),
                                       "certifiable": int(r > 0),
                                       "at_ceiling": int(abs(r - ceiling) < 1e-3 * max(ceiling, 1e-9))})
                for sig in signals:
                    c = [r[sig.name] for r in clean_per_signal]
                    k = [r[sig.name] for r in atk_per_signal]
                    auroc_rows.append({
                        "dataset": ds, "seed": seed, "attack": attack,
                        "signal": sig.name.value,
                        "auroc": round(float(roc_auc_score([0] * len(c) + [1] * len(k), c + k)), 4),
                    })
            print(f"  [{ds}] seed {seed} done", flush=True)

        raw = ctx.run_dir / "raw"
        write_csv(raw / "conformal.csv", conformal_rows)
        write_csv(raw / "radii.csv", radii_rows)
        write_csv(raw / "ablation.csv", ablation_rows)
        write_csv(raw / "signal_auroc.csv", auroc_rows)

        certifiable = (float(np.mean([r["certifiable"] for r in radii_rows if r["condition"] == "clean"]))
                       if radii_rows else float("nan"))
        clean_radii = [r for r in radii_rows if r["condition"] == "clean"]
        summary = {"dataset": ds, "seeds": seeds, "n_clean": len(conformal_rows),
                   "smoothing_n": stab_n, "sigma": stab_sigma, "radius_ceiling": round(ceiling, 6),
                   "clean_frac_at_ceiling": round(float(np.mean(
                       [r["at_ceiling"] for r in clean_radii])), 4) if clean_radii else None,
                   "clean_certifiable_frac": round(certifiable, 4),
                   "empirical_fa_at_0.05": round(float(np.mean(
                       [r["p_value"] < 0.05 for r in conformal_rows])), 4)}
        print(f"  [{ds}] {summary}", flush=True)
        return RunResult(
            summary=summary,
            floats=[{"float_id": fid, "kind": kind, "paper_section": "C3",
                     "path": str(raw / fn), "claim": claim, "status": "done"}
                    for fid, kind, fn, claim in [
                        ("Fig6_conformal_calibration", "figure", "conformal.csv",
                         "empirical false-alarm rate stays within nominal alpha on clean held-out data"),
                        ("Fig5_certified_radius", "figure", "radii.csv",
                         "most clean samples are not certifiable: the stability premise fails on tabular NIDS"),
                        ("Fig2_interventional", "figure", "ablation.csv",
                         "ablating a ground-truth causal feature moves detection; a non-causal one does not"),
                        ("Tab5_per_signal_auroc", "table", "signal_auroc.csv",
                         "per-signal detectability per attack class, including where signals fail"),
                    ]],
        )

    return run_experiment(f"c3_artifacts_{ds}", experiment,
                          config={"dataset": ds, "seeds": seeds, "n_test": n_test,
                                  "stab_n": stab_n, "stab_sigma": stab_sigma}, seed=seeds[0])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", action="append", default=None)
    ap.add_argument("--seeds", nargs="*", type=int, default=[0, 1, 2, 3, 4])
    ap.add_argument("--n-test", type=int, default=80)
    ap.add_argument("--n-ablation", type=int, default=20)
    ap.add_argument("--stab-n", type=int, default=120, help="smoothing draws; the radius ceiling grows with it")
    ap.add_argument("--stab-sigma", type=float, default=0.05)
    args = ap.parse_args()
    for ds in (args.dataset or ALL_DATASETS):
        print(f"[{ds}] starting", flush=True)
        run_dataset(ds, args.seeds, args.n_test, args.n_ablation, args.stab_n, args.stab_sigma)


if __name__ == "__main__":
    main()
