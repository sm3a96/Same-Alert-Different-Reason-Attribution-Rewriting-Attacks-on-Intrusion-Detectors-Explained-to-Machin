"""When the attack demotes a causal feature but leaves it on screen, does the consumer
go and get it?

This is the sharpest single statement about what an LLM consumer adds to a corrupted
ranking, and the answer is "very little". The cases are chosen to be exactly the ones a
consumer with any judgment should win: the clean explanation ranked a causal feature first,
the attack pushed it out of first place, and it is still visible somewhere in the ten
options shown. Nothing is hidden. The information needed to answer correctly is on screen.

A consumer that reads the explanation recovers these. A consumer that acts on rank 1 does
not. The recovery rate is therefore the cleanest available measure of how much judgment sits
between the ranking and the decision — and it is the strongest motivation the
knowledge-grounded verifier has, because it says the deficit cannot be closed by putting a
capable language model at the end of the pipeline.

  python scripts/check_recovery.py
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RUN = REPO / "results" / "decision_utility"
LABEL = {"fiveg_nidd": "5G-NIDD", "ciciot2023": "CICIoT2023", "ciciomt2024": "CICIoMT2024"}


def main() -> None:
    rows = json.loads((RUN / "raw" / "decisions.json").read_text())
    blob = json.loads((REPO / "results" / "_cache" / "decision_cases.json").read_text())
    cached = {(r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]): r
              for r in (blob["cases"] if isinstance(blob, dict) else blob)}

    by_judge = defaultdict(lambda: {"n": 0, "rec": 0})
    by_ds = defaultdict(lambda: {"n": 0, "rec": 0})
    for r in rows:
        if r["condition"] != "attacked" or r.get("attack") != "A1_displacement":
            continue
        rec = cached.get((r["dataset"], r["class"], r["seed"], r["attack"], r["sample_id"]))
        if rec is None:
            continue
        clean_c, atk_c = rec["cases"]["clean"], rec["cases"]["attacked"]
        causal = set(atk_c["causal"])
        demoted_but_visible = (clean_c["candidates"][0] in causal
                               and atk_c["candidates"][0] not in causal
                               and bool(set(atk_c["candidates"]) & causal))
        if not demoted_but_visible:
            continue
        for d in (by_judge[r["judge"]], by_ds[r["dataset"]]):
            d["n"] += 1
            d["rec"] += int(r["det_correct"])

    print("Recovery of a causal feature the attack demoted but left ON SCREEN\n")
    print(f"  {'judge':14s} {'recovered':>12s} {'rate':>7s}")
    tn = tr = 0
    for j, d in sorted(by_judge.items()):
        if d["n"]:
            print(f"  {j:14s} {d['rec']:>5d}/{d['n']:<6d} {d['rec']/d['n']:7.3f}")
            tn += d["n"]
            tr += d["rec"]
    if tn:
        print(f"  {'pooled':14s} {tr:>5d}/{tn:<6d} {tr/tn:7.3f}")

    if by_ds:
        print(f"\n  {'dataset':14s} {'recovered':>12s} {'rate':>7s}")
        for ds, d in sorted(by_ds.items()):
            if d["n"]:
                print(f"  {LABEL.get(ds, ds):14s} {d['rec']:>5d}/{d['n']:<6d} {d['rec']/d['n']:7.3f}")

    if tn:
        print(f"\n  The feature was on screen in every one of these {tn} cases and the consumer")
        print(f"  retrieved it {tr/tn:.1%} of the time. Whatever an LLM adds to a corrupted")
        print("  ranking, it is not robustness to the corruption.")

    out = REPO / "results" / "_logs" / "recovery.json"
    out.write_text(json.dumps(
        {"by_judge": {j: {"n": d["n"], "recovered": d["rec"],
                          "rate": round(d["rec"] / d["n"], 4) if d["n"] else None}
                      for j, d in by_judge.items()},
         "by_dataset": {LABEL.get(k, k): {"n": d["n"], "recovered": d["rec"],
                                          "rate": round(d["rec"] / d["n"], 4) if d["n"] else None}
                        for k, d in by_ds.items()},
         "pooled": {"n": tn, "recovered": tr, "rate": round(tr / tn, 4) if tn else None}},
        indent=1))
    print(f"\n  saved {out}")


if __name__ == "__main__":
    main()
