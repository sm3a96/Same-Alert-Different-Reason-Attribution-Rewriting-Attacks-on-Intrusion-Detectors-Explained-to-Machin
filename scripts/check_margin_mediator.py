"""Does the attack flatten the top-1 vs top-2 attribution margin, and does departure
from rank-1 following track that?

The residual needs content. One judge departs from rank-1 following only when the
explanation is attacked, and "it reads values" is a story rather than a finding until
something predicts *when* it departs. The obvious mediator is the margin between the
first- and second-ranked attributions: displacement noise flattens rankings, and a model that
actually reads the numbers would deliberate exactly where the top two are close and defer to
position where they are not.

This is in two halves because only one is answerable yet.

  Half 1 (now, from the cached cases): does the attack flatten the margin at all? If it does
  not, the mediator hypothesis is dead before the shuffle and the residual needs a different
  explanation.

  Half 2 (once `det_choice_idx` is recorded by the rebuilt run): does P(departure from slot
  A) rise as the margin shrinks, and does it do so for Qwen3 and not for phi-4?

Note what this cannot do. Conditioning departure on margin is observational: margin is one
of many things the attack changes at once. It can make the value-reading clause specific and
falsifiable, and it can kill the clause outright, but the causal split between
value-reading and every other attack covariate still belongs to the shuffled-order arm.

  python scripts/check_margin_mediator.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

import numpy as np

REPO = Path(__file__).resolve().parents[1]
LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024"}


def margin(case) -> float:
    """|attribution| gap between the first- and second-ranked options, as shown."""
    imp = case.get("importances")
    if not imp or len(imp) < 2:
        return float("nan")
    a = np.abs(np.asarray(imp, dtype=float))
    return float(a[0] - a[1])


def _qwen_attacked_residual(cached) -> dict:
    """P(top-1 causal) - accuracy on the attacked arm, for the departing judge, per dataset."""
    rows_path = REPO / "results" / "decision_utility" / "raw" / "decisions.json"
    if not rows_path.exists():
        return {}
    rows = json.loads(rows_path.read_text())
    judges = sorted({r["judge"] for r in rows})
    depart = next((j for j in judges if "Qwen" in j), judges[0] if judges else None)
    if depart is None:
        return {}
    t1 = defaultdict(list)
    for rec in cached:
        if rec.get("attack") != "A1_displacement":
            continue
        c = rec["cases"].get("attacked")
        if c:
            t1[rec["dataset"]].append(c["candidates"][0] in set(c["causal"]))
    acc = defaultdict(list)
    for r in rows:
        if (r["condition"] == "attacked" and r.get("attack") == "A1_displacement"
                and r["judge"] == depart):
            acc[r["dataset"]].append(r["det_correct"])
    return {ds: float(np.mean(t1[ds])) - float(np.mean(acc[ds]))
            for ds in t1 if ds in acc and t1[ds] and acc[ds]}


def main() -> None:
    blob = json.loads((REPO / "results" / "_cache" / "decision_cases.json").read_text())
    cached = blob["cases"] if isinstance(blob, dict) else blob

    # ---- half 1: does the attack flatten the margin?
    per = defaultdict(lambda: defaultdict(list))
    by_id = {}
    for rec in cached:
        if rec.get("attack") != "A1_displacement":
            continue
        key = (rec["dataset"], rec["class"], rec["seed"], rec["sample_id"])
        for arm in ("clean", "attacked"):
            c = rec["cases"].get(arm)
            if c is None:
                continue
            m = margin(c)
            if not np.isnan(m):
                per[rec["dataset"]][arm].append(m)
                by_id[(key, arm)] = m

    print("Half 1 — does the displacement attack flatten the top-1 vs top-2 margin?\n")
    print(f"  {'dataset':13s} {'clean median':>13s} {'attacked median':>16s} "
          f"{'paired median Δ':>16s} {'% flattened':>12s}")
    any_flat = False
    for ds in sorted(per):
        cl = np.array(per[ds]["clean"])
        at = np.array(per[ds]["attacked"])
        pairs = [(by_id[(k, "clean")], by_id[(k, "attacked")])
                 for (k, a) in by_id if a == "clean" and (k, "attacked") in by_id
                 and k[0] == ds]
        if not len(cl) or not len(at) or not pairs:
            continue
        d = np.array([b - a for a, b in pairs])
        frac = float(np.mean(d < 0))
        any_flat |= bool(np.median(d) < 0)
        print(f"  {LABEL.get(ds, ds):13s} {np.median(cl):13.4f} {np.median(at):16.4f} "
              f"{np.median(d):+16.4f} {frac:11.1%}")

    # Flattening somewhere is not the test. The test is whether flattening lines up with
    # where the departure actually happens -- checking only the former would let a
    # generous verdict string stand in for a result.
    # Recomputed, never written in. A hardcoded measurement here would be flagged by
    # the claims check, and rightly.
    residual = _qwen_attacked_residual(cached)
    print()
    print("  Does flattening line up with where Qwen3 departs from rank-1 following?\n")
    print(f"  {'dataset':13s} {'median margin Δ':>16s} {'% flattened':>12s} "
          f"{'Qwen3 attacked residual':>24s}")
    order = []
    for ds in sorted(per, key=lambda d: -residual.get(d, 0)):
        pairs = [(by_id[(k, "clean")], by_id[(k, "attacked")])
                 for (k, a) in by_id if a == "clean" and (k, "attacked") in by_id and k[0] == ds]
        if not pairs:
            continue
        d = np.array([b - a for a, b in pairs])
        order.append((residual.get(ds, float("nan")), float(np.mean(d < 0))))
        print(f"  {LABEL.get(ds, ds):13s} {np.median(d):+16.4f} {np.mean(d < 0):11.1%} "
              f"{residual.get(ds, float('nan')):24.3f}")
    if len(order) >= 3:
        r = float(np.corrcoef([o[0] for o in order], [o[1] for o in order])[0, 1])
        print(f"\n  corr(residual, fraction flattened) = {r:+.3f}   (descriptive; 3 points)")
        if r < 0:
            print("\n  ANTI-CORRELATED. Where margins flatten most the departure is smallest,")
            print("  and where they widen it is largest. The simple margin mediator is NOT")
            print("  supported at the dataset level and must not be written as the clause.")
            print("  Per-sample conditioning (half 2) can still rescue it -- an aggregate")
            print("  anti-correlation does not preclude a within-dataset effect -- but the")
            print("  hypothesis now has to earn that, and if half 2 is flat it is dead.")
        else:
            print("\n  Aligned at the dataset level. Half 2 decides whether it holds per sample.")

    # ---- half 2: departure vs margin, when the positions have been recorded
    rows_path = REPO / "results" / "decision_utility" / "raw" / "decisions.json"
    rows = json.loads(rows_path.read_text()) if rows_path.exists() else []
    has_idx = any("det_choice_idx" in r for r in rows)
    print("\nHalf 2 — does departure from slot A track the margin?\n")
    if not has_idx:
        print("  Not answerable on these decisions: they predate `det_choice_idx` being")
        print("  recorded. The rebuilt run captures it. Re-run this then.")
        return

    buckets = defaultdict(lambda: defaultdict(list))
    for r in rows:
        if r["condition"] != "attacked" or r.get("attack") != "A1_displacement":
            continue
        k = ((r["dataset"], r["class"], r["seed"], r["sample_id"]), "attacked")
        m = by_id.get(k)
        if m is None:
            continue
        buckets[r["judge"]][m].append(int(r["det_choice_idx"] != 0))

    print(f"  {'judge':12s} {'margin quartile':>16s} {'n':>6s} {'P(departs slot A)':>19s}")
    for j, d in sorted(buckets.items()):
        ms = np.array(sorted(d))
        if len(ms) < 8:
            continue
        qs = np.quantile(ms, [0.25, 0.5, 0.75])
        for lo, hi, name in [(-np.inf, qs[0], "Q1 (flattest)"), (qs[0], qs[1], "Q2"),
                             (qs[1], qs[2], "Q3"), (qs[2], np.inf, "Q4 (widest)")]:
            vals = [v for m, vs in d.items() if lo < m <= hi for v in vs]
            if vals:
                print(f"  {j:12s} {name:>16s} {len(vals):6d} {np.mean(vals):19.3f}")
    print("\n  A value-reading consumer departs most where the margin is smallest. A")
    print("  rank-follower departs at the same rate everywhere.")


if __name__ == "__main__":
    main()
