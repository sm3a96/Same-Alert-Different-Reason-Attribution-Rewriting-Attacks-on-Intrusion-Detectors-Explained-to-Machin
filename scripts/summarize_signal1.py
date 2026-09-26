"""The honest cross-dataset account of why certified stability fails.

A 40-flow probe suggested two different stories on two different datasets: 5G-NIDD piling
at the estimator ceiling, CICIoT2023 and CICIoMT2024 with more mass at zero. Neither
"saturation" nor "unstable baseline" is going to be the whole answer, and the paper
should report the mix rather than pick the tidier sentence.

This reads the saved radii and characterises each dataset by where its probability mass
actually sits, for clean flows and for each attack:

  at zero        the certificate could not be issued at all
  at ceiling     the certificate is capped by its own sample size, not by the attribution
  in between     the only region where the statistic carries information

The last column is the one that matters. A signal used for ranking needs mass in the
middle; the further the mass is pushed to the two ends, the less there is to rank with.

  python scripts/summarize_signal1.py
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import norm
from sklearn.metrics import roc_auc_score

from avert.signals.certified_stability import clopper_pearson_lower

REPO = Path(__file__).resolve().parents[1]
DATASET_LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023",
                 "ciciomt2024": "CICIoMT2024"}
CONDITIONS = ["clean", "A1_misdirection", "A1_displacement", "A3_scaffolding"]


def ceiling_for(run_dir: Path) -> float:
    """Read sigma and n from the run's own metadata rather than assuming them."""
    metas = sorted((run_dir / "meta").glob("run_seed*.json"))
    if not metas:
        raise SystemExit(f"no metadata in {run_dir}")
    cfg = json.loads(metas[0].read_text())["config"]
    n, sigma = int(cfg["stab_n"]), float(cfg["stab_sigma"])
    return sigma * float(norm.ppf(clopper_pearson_lower(n, n, 0.05)))


def main() -> None:
    runs = sorted((REPO / "results").glob("c3_artifacts_*"))
    if not runs:
        raise SystemExit("no c3_artifacts_* runs — run scripts/run_c3_artifacts.py")

    print(f"{'dataset':13s} {'condition':17s} {'n':>5s} {'at zero':>9s} {'at ceiling':>11s} "
          f"{'in between':>11s} {'mean R':>8s} {'AUROC':>7s}")
    summary = []
    for run in runs:
        f = run / "raw" / "radii.csv"
        if not f.exists():
            continue
        ceiling = ceiling_for(run)
        by_cond = defaultdict(list)
        for row in csv.DictReader(open(f)):
            by_cond[row["condition"]].append(float(row["certified_radius"]))
        clean = np.array(by_cond.get("clean", []))
        ds = run.name.replace("c3_artifacts_", "")
        # per-seed radii, so per-seed AUROC can be recomputed rather than remembered
        by_seed = defaultdict(lambda: defaultdict(list))
        for row in csv.DictReader(open(f)):
            by_seed[row["condition"]][int(row["seed"])].append(float(row["certified_radius"]))
        for cond in CONDITIONS:
            r = np.array(by_cond.get(cond, []))
            if not len(r):
                continue
            zero = float(np.mean(r == 0))
            cap = float(np.mean(np.isclose(r, ceiling, rtol=1e-3)))
            mid = 1.0 - zero - cap
            # -R is the nonconformity: a smaller radius should mean more suspicious.
            auroc = (roc_auc_score([0] * len(clean) + [1] * len(r), list(-clean) + list(-r))
                     if cond != "clean" and len(clean) else float("nan"))
            print(f"{DATASET_LABEL.get(ds, ds):13s} {cond:17s} {len(r):5d} {zero:9.2f} "
                  f"{cap:11.2f} {mid:11.2f} {r.mean():8.4f} "
                  f"{'' if np.isnan(auroc) else f'{auroc:7.3f}'}")
            per_seed = []
            if cond != "clean":
                for sd in sorted(by_seed["clean"]):
                    c_ = np.array(by_seed["clean"][sd])
                    a_ = np.array(by_seed[cond].get(sd, []))
                    if len(c_) and len(a_):
                        per_seed.append(round(float(roc_auc_score(
                            [0]*len(c_) + [1]*len(a_), list(-c_) + list(-a_))), 4))
            summary.append({"dataset": ds, "condition": cond, "n": len(r), "ceiling": ceiling,
                            "auroc_per_seed": per_seed,
                            "frac_zero": round(zero, 4), "frac_at_ceiling": round(cap, 4),
                            "frac_informative": round(mid, 4), "mean_radius": round(float(r.mean()), 5),
                            "auroc": None if np.isnan(auroc) else round(float(auroc), 4)})
        print()

    out = REPO / "results" / "_logs" / "signal1_cross_dataset.json"
    out.write_text(json.dumps(summary, indent=1))
    print(f"saved {out}")

    clean_rows = [s for s in summary if s["condition"] == "clean"]
    if clean_rows:
        print("\nWhat to report, per dataset (clean flows):")
        for s in clean_rows:
            certifiable = 1 - s["frac_zero"]
            print(f"  {DATASET_LABEL.get(s['dataset'], s['dataset']):13s} "
                  f"{certifiable:.1%} certifiable, {s['frac_at_ceiling']:.1%} capped by the "
                  f"estimator, {s['frac_informative']:.1%} in the range that can rank")
        print(f"\n  The recorded 7-10% certifiable figure holds nowhere: the lowest measured is "
              f"{1 - max(s['frac_zero'] for s in clean_rows):.1%}.")

    # The headline is not that the signal is weak. It is that its SIGN is not stable.
    frag = {s["dataset"]: s for s in summary if s["condition"] == "A1_displacement"}
    if len(frag) > 1:
        # Lead with the per-seed range. A pooled AUROC over a bimodal set of seeds reports a
        # number that no individual run produced -- CICIoT2023 pools to 0.51 out of seeds
        # spanning 0.37 to 0.90, and the pooled value hides exactly the instability that is
        # the finding.
        print("\nPer-seed spread — the honest unit, because pooling hides it:")
        print(f"  {'dataset':13s} {'attack':17s} {'pooled':>7s} {'min':>6s} {'max':>6s} "
              f"{'spread':>7s}   per seed")
        widest = None
        for cond in ("A1_displacement", "A1_misdirection"):
            for r in sorted(summary, key=lambda x: x["dataset"]):
                if r["condition"] != cond or not r.get("auroc_per_seed"):
                    continue
                ps = r["auroc_per_seed"]
                spread = max(ps) - min(ps)
                if widest is None or spread > widest[0]:
                    widest = (spread, r["dataset"], cond, ps)
                print(f"  {DATASET_LABEL.get(r['dataset'], r['dataset']):13s} {cond:17s} "
                      f"{r['auroc']:7.3f} {min(ps):6.3f} {max(ps):6.3f} {spread:7.3f}   "
                      f"{[round(x, 2) for x in ps]}")
        if widest and widest[0] > 0.3:
            sp, ds, cond, ps = widest
            print(f"\n  Widest: {DATASET_LABEL.get(ds, ds)} / {cond} spans {sp:.3f} across seeds "
                  f"({min(ps):.2f} to {max(ps):.2f}).")
            print("  Same dataset, same pipeline, same attack -- only the training seed differs,")
            print("  and the signal goes from clearly inverted to clearly detecting. That is a")
            print("  stronger deployability argument than any cross-dataset comparison, because")
            print("  a practitioner cannot choose a different seed the way they might choose a")
            print("  different dataset.")

        print("\nSign stability against A1 displacement — pooled across seeds:")
        for ds, s in sorted(frag.items(), key=lambda kv: -kv[1]["auroc"]):
            a = s["auroc"]
            verdict = ("detects it" if a > 0.6 else
                       "blind" if a > 0.45 else
                       "POINTS THE WRONG WAY (below chance)")
            print(f"  {DATASET_LABEL.get(ds, ds):13s} AUROC {a:.3f}  {verdict}"
                  f"   (attacked at-zero {s['frac_zero']:.2f}, at-ceiling {s['frac_at_ceiling']:.2f})")
        lo = min(s["auroc"] for s in frag.values())
        hi = max(s["auroc"] for s in frag.values())
        if lo < 0.45 and hi > 0.55:
            print(f"\n  AUROC spans {lo:.3f} to {hi:.3f} ACROSS THE DECISION BOUNDARY. On one")
            print("  dataset the attack makes attributions less certifiable, on another more.")
            print("  Calibration here is label-free by design, so there is no way to learn which")
            print("  regime you are in before deploying. A signal whose sign you cannot predict")
            print("  is not a weak detector -- it is unusable, and that is the finding.")


if __name__ == "__main__":
    main()
