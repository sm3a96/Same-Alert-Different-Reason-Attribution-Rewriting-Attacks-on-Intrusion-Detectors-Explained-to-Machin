"""Is the competence floor filtering on judge ability, or on explainer faithfulness?

The floor drops cells where the judge scores below 0.5 from a clean explanation, on the
reading that it cannot degrade a decision the agent could not make. But on CICIoT2023 both
judges score 0.069 — the same number, to three decimals, from two different model families.
Two unrelated models failing identically is not a story about either model.

The alternative explanation is that the explanation itself is wrong there: if TreeSHAP's
top-ranked feature is not one of the interventionally causal ones, an agent that faithfully
follows the ranking is *supposed* to be wrong, and the floor is quietly filtering on
explainer faithfulness while the paper calls it judge competence.

This measures that directly and costs nothing: the cached decision cases already carry the
attribution-ranked candidate list and the causal set.

  python scripts/check_explainer_faithfulness.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024"}


def main() -> None:
    cache_path = REPO / "results" / "_cache" / "decision_cases.json"
    blob = json.loads(cache_path.read_text())
    cached = blob["cases"] if isinstance(blob, dict) else blob

    # Clean AND attacked. If judge accuracy tracks P(top-1 causal) in both arms, mediation
    # through rank 1 is shown observationally and the shuffled arm becomes confirmation
    # rather than the whole case.
    per = defaultdict(lambda: defaultdict(lambda: {"top1": [], "top3": [], "visible": [], "nc": []}))
    for rec in cached:
        for arm in ("clean", "attacked"):
            c = rec["cases"].get(arm)
            if c is None:
                continue
            if arm == "attacked" and rec.get("attack") != "A1_displacement":
                continue          # the attack that carries the result
            causal = set(c["causal"])
            cand = c["candidates"]
            d = per[rec["dataset"]][arm]
            d["top1"].append(cand[0] in causal)
            d["top3"].append(bool(set(cand[:3]) & causal))
            d["visible"].append(bool(set(cand) & causal))
            d["nc"].append(len(causal))

    rows = json.loads((REPO / "results" / "decision_utility" / "raw" / "decisions.json").read_text())
    obs = defaultdict(list)
    for r in rows:
        if r["condition"] == "clean":
            obs[(r["dataset"], r["judge"], "clean")].append(r["det_correct"])
        elif r["condition"] == "attacked" and r.get("attack") == "A1_displacement":
            obs[(r["dataset"], r["judge"], "attacked")].append(r["det_correct"])

    print("Does the explanation put a causal feature first, and does accuracy track it?\n")
    print(f"  {'dataset':13s} {'arm':9s} {'P(top-1 causal)':>16s} {'top-3':>7s} "
          f"{'judge accuracy':>26s} {'gap':>7s}")
    for ds in sorted(per, key=lambda k: -np.mean(per[k]["clean"]["top1"])):
        for arm in ("clean", "attacked"):
            d = per[ds].get(arm)
            if not d or not d["top1"]:
                continue
            t1 = float(np.mean(d["top1"]))
            accs = {j: float(np.mean(v)) for (dd, j, aa), v in obs.items()
                    if dd == ds and aa == arm}
            if not accs:
                continue
            mean_acc = float(np.mean(list(accs.values())))
            txt = " ".join(f"{j.split('/')[-1]}={a:.3f}" for j, a in sorted(accs.items()))
            print(f"  {LABEL.get(ds, ds):13s} {arm:9s} {t1:16.3f} {np.mean(d['top3']):7.3f} "
                  f"{txt:>26s} {mean_acc - t1:+7.3f}")

    # The mediation statement: does accuracy move with top-1 faithfulness across BOTH arms?
    pts = []
    for ds in per:
        for arm in ("clean", "attacked"):
            d = per[ds].get(arm)
            if not d or not d["top1"]:
                continue
            accs = [float(np.mean(v)) for (dd, _, aa), v in obs.items() if dd == ds and aa == arm]
            if accs:
                pts.append((float(np.mean(d["top1"])), float(np.mean(accs))))
    mediation = {}
    if len(pts) >= 3:
        x = np.array([p[0] for p in pts])
        y = np.array([p[1] for p in pts])
        r = float(np.corrcoef(x, y)[0, 1])
        mad = float(np.mean(np.abs(y - x)))
        # Persisted, not just printed. The plan quoted this gap as 0.023 while the run said
        # 0.008, and no check caught it because the number lived only in this script's stdout.
        # A value the paper asserts has to survive in a file, or nothing downstream can see it.
        mediation = {"corr_top1_vs_accuracy": round(r, 4),
                     "mean_abs_gap": round(mad, 4),
                     "n_points": len(pts)}
        print(f"\nMediation check across {len(pts)} (dataset, arm) points:")
        print(f"  corr(P(top-1 causal), judge accuracy) = {r:+.3f}")
        print(f"  mean |accuracy - P(top-1 causal)|      = {mad:.3f}")
        if r > 0.9 and mad < 0.12:
            print("  Accuracy tracks top-1 faithfulness in BOTH arms. Mediation through rank 1")
            print("  is shown observationally; the shuffled arm is now confirmation, not the")
            print("  whole case.")

    print("\nReading:")
    ds = min(per, key=lambda k: np.mean(per[k]["clean"]["top1"]))
    d = per[ds]["clean"]
    t1 = np.mean(d["top1"])
    accs = [np.mean(v) for (dd, _, aa), v in obs.items() if dd == ds and aa == "clean"]
    print(f"  On {LABEL.get(ds, ds)} the attribution's top feature is causal {t1:.1%} of the time.")
    if accs:
        print(f"  Both judges score {min(accs):.3f}-{max(accs):.3f} there.")
    if t1 < 0.2 and accs and max(accs) < 0.2:
        print("  The judges are tracking the explanation closely; the explanation is the thing")
        print("  that is wrong. The competence floor is therefore filtering on EXPLAINER")
        print("  FAITHFULNESS on this dataset, not on judge ability, and the paper must say so:")
        print("  an agent that faithfully follows an unfaithful ranking is supposed to be wrong.")
    else:
        print("  Judge accuracy and explainer top-1 faithfulness diverge here, so the floor is")
        print("  not simply reading explainer quality back to us.")

    out = REPO / "results" / "_logs" / "explainer_faithfulness.json"
    out.write_text(json.dumps(
        {"_mediation": mediation,
         **{LABEL.get(k, k): {arm: {"top1_causal": round(float(np.mean(vv["top1"])), 4),
                                 "top3_causal": round(float(np.mean(vv["top3"])), 4),
                                 "any_causal_visible": round(float(np.mean(vv["visible"])), 4),
                                 "mean_causal_set_size": round(float(np.mean(vv["nc"])), 3)}
                           for arm, vv in v.items() if vv["top1"]}
            for k, v in per.items()}}, indent=1))
    print(f"\nsaved {out}")


if __name__ == "__main__":
    main()
