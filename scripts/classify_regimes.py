"""Classify every (dataset, pipeline) cell into a certified-stability regime, from raw radii.

The regime grid is the C3 section's lead claim, so it cannot be a table someone typed. The
criteria were fixed before the second pipeline was looked at (pre-specified as
"the three-regime reading"):

  detects   every per-seed DeLong interval lies entirely above 0.5
  lottery   between/within variance ratio > 2 AND the extreme seeds' intervals do not overlap
  blind     variance ratio <= 2 AND the pooled interval contains 0.5

Anything else is reported `unclassified` rather than rounded into the nearest box.

This applies them mechanically and, importantly, reports *every* criterion a cell satisfies
rather than the first one that matches. The pre-registration did not specify precedence, and a
cell that meets two definitions is a hole in the criteria that a reviewer will find faster than
I will. Naming it is cheap; being caught having silently taken the flattering label is not.

  python scripts/classify_regimes.py

Writes results/_logs/regime_grid.json.
"""
from __future__ import annotations

import csv
import json
from collections import defaultdict
from pathlib import Path

import numpy as np
from sklearn.metrics import roc_auc_score

REPO = Path(__file__).resolve().parents[1]
LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024"}
DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
PIPELINES = [("XGBoost + TreeSHAP", "c3_artifacts_{ds}"),
             ("MLP + integrated gradients", "generality_mlp_ig_{ds}")]
ATTACK = "A1_displacement"
Z = 1.96


def delong_var(y, s) -> float:
    y = np.asarray(y)
    s = np.asarray(s, dtype=float)
    pos, neg = s[y == 1], s[y == 0]
    m, n = len(pos), len(neg)
    if m < 2 or n < 2:
        return float("nan")
    v10 = np.array([(np.sum(neg < p) + 0.5 * np.sum(neg == p)) / n for p in pos])
    v01 = np.array([(np.sum(pos > q) + 0.5 * np.sum(pos == q)) / m for q in neg])
    return float(v10.var(ddof=1) / m + v01.var(ddof=1) / n)


def auroc_ci(clean, attacked):
    """AUROC of 'smaller radius means attacked', with its DeLong interval."""
    y = np.r_[np.zeros(len(clean)), np.ones(len(attacked))]
    s = np.r_[-np.asarray(clean, float), -np.asarray(attacked, float)]
    a = float(roc_auc_score(y, s))
    v = delong_var(y, s)
    se = float(np.sqrt(v)) if v == v else float("nan")
    return a, v, a - Z * se, a + Z * se


def read_radii(run: Path):
    """Per-seed radii by condition. The two runners write different column names."""
    f = run / "raw" / "radii.csv"
    if not f.exists():
        return None
    by = defaultdict(lambda: defaultdict(list))
    with open(f) as fh:
        for r in csv.DictReader(fh):
            val = r.get("certified_radius", r.get("radius"))
            by[r["condition"]][int(r["seed"])].append(float(val))
    return by


def classify(by) -> dict:
    seeds = sorted(s for s in by.get("clean", {}) if s in by.get(ATTACK, {}))
    per = []
    for sd in seeds:
        c, a = by["clean"][sd], by[ATTACK][sd]
        if len(c) < 2 or len(a) < 2:
            continue
        auc, var, lo, hi = auroc_ci(c, a)
        per.append({"seed": sd, "auroc": auc, "var": var, "lo": lo, "hi": hi,
                    "n_clean": len(c), "n_attacked": len(a)})
    if len(per) < 3:
        return {"regime": "insufficient", "per_seed": per}

    est = np.array([p["auroc"] for p in per])
    var = np.array([p["var"] for p in per])
    between, within = float(est.var(ddof=1)), float(np.nanmean(var))
    ratio = between / within if within > 0 else float("inf")

    hi_i, lo_i = int(np.argmax(est)), int(np.argmin(est))
    separated = bool(per[hi_i]["lo"] > per[lo_i]["hi"])

    # Pooled across seeds, which is the level `blind` is defined at.
    pc = [v for sd in (p["seed"] for p in per) for v in by["clean"][sd]]
    pa = [v for sd in (p["seed"] for p in per) for v in by[ATTACK][sd]]
    p_auc, _, p_lo, p_hi = auroc_ci(pc, pa)

    met = []
    if all(p["lo"] > 0.5 for p in per):
        met.append("detects")
    if ratio > 2 and separated:
        met.append("lottery")
    if ratio <= 2 and p_lo <= 0.5 <= p_hi:
        met.append("blind")

    return {"regime": met[0] if met else "unclassified", "meets": met,
            "ambiguous": len(met) > 1, "per_seed": per,
            "between": between, "within": within, "ratio": ratio,
            "separated": separated, "extremes": [per[lo_i]["auroc"], per[hi_i]["auroc"]],
            "pooled": {"auroc": p_auc, "lo": p_lo, "hi": p_hi}}


def main() -> None:
    grid, out = {}, []
    for ds in DATASETS:
        grid[ds] = {}
        for pname, pat in PIPELINES:
            by = read_radii(REPO / "results" / pat.format(ds=ds))
            grid[ds][pname] = classify(by) if by else {"regime": "missing", "per_seed": []}

    out.append(f"Certified-stability regime against {ATTACK}, criteria fixed in advance.\n")
    w = max(len(p) for p, _ in PIPELINES)
    out.append(f"  {'dataset':13s} " + " ".join(f"{p:>{w}s}" for p, _ in PIPELINES))
    for ds in DATASETS:
        cells = [grid[ds][p]["regime"] for p, _ in PIPELINES]
        out.append(f"  {LABEL[ds]:13s} " + " ".join(f"{c:>{w}s}" for c in cells))
    out.append("")
    for ds in DATASETS:
        for pname, _ in PIPELINES:
            g = grid[ds][pname]
            if not g["per_seed"]:
                continue
            seq = ", ".join(f"{p['auroc']:.2f}" for p in g["per_seed"])
            out.append(f"  {LABEL[ds]:13s} {pname:30s} {g['regime']:>13s}   {seq}")

    kept = [ds for ds in DATASETS
            if len({grid[ds][p]["regime"] for p, _ in PIPELINES}) == 1
            and grid[ds][PIPELINES[0][0]]["regime"] not in ("missing", "insufficient")]
    out.append(f"\n  {len(kept)} of {len(DATASETS)} datasets keep their regime across pipelines.")
    if not kept:
        out.append("  The regime is a joint property of dataset, detector and attributor -- the")
        out.append("  outcome pre-specified as strictly worse for deployability.")

    # The separation that stops a reviewer calling the reshuffling noise.
    out.append("\n  Lottery cells -- are the extremes actually separated?")
    for ds in DATASETS:
        for pname, _ in PIPELINES:
            g = grid[ds][pname]
            if g["regime"] != "lottery":
                continue
            per = g["per_seed"]
            hi = max(per, key=lambda p: p["auroc"])
            lo = min(per, key=lambda p: p["auroc"])
            out.append(f"  {LABEL[ds]:13s} {pname:30s} "
                       f"{hi['auroc']:.3f} [{hi['lo']:.3f}, {hi['hi']:.3f}] vs "
                       f"{lo['auroc']:.3f} [{lo['lo']:.3f}, {lo['hi']:.3f}]  "
                       f"ratio {g['ratio']:.1f}x")

    amb = [(ds, p) for ds in DATASETS for p, _ in PIPELINES if grid[ds][p].get("ambiguous")]
    if amb:
        out.append("\n  DISCLOSED -- cells meeting more than one criterion. The pre-registration")
        out.append("  set no precedence, so the label below is the table order and the fact that")
        out.append("  the cell is double-classified is reported, not resolved after the fact:")
        for ds, p in amb:
            g = grid[ds][p]
            out.append(f"  {LABEL[ds]:13s} {p:30s} meets {' and '.join(g['meets'])} "
                       f"(ratio {g['ratio']:.1f}x)")

    unc = [(ds, p) for ds in DATASETS for p, _ in PIPELINES
           if grid[ds][p]["regime"] == "unclassified"]
    if unc:
        out.append("\n  Unclassified cells, left unclassified:")
        for ds, p in unc:
            g = grid[ds][p]
            out.append(f"  {LABEL[ds]:13s} {p:30s} pooled {g['pooled']['auroc']:.3f} "
                       f"[{g['pooled']['lo']:.3f}, {g['pooled']['hi']:.3f}], "
                       f"ratio {g['ratio']:.1f}x")

    text = "\n".join(out)
    print(text)
    dest = REPO / "results" / "_logs" / "regime_grid.json"
    dest.parent.mkdir(parents=True, exist_ok=True)
    dest.write_text(json.dumps(
        {"attack": ATTACK, "kept_regime": len(kept), "n_datasets": len(DATASETS),
         "grid": {LABEL[ds]: {p: grid[ds][p] for p, _ in PIPELINES} for ds in DATASETS}},
        indent=1))
    print(f"\nsaved {dest}")


if __name__ == "__main__":
    main()
