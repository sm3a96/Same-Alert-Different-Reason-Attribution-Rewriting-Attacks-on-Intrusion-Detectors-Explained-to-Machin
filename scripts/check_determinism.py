"""Does the pipeline reproduce its own numbers, given the seed?

The release plan promises that one command regenerates every number in the paper. That
promise is only worth something if a re-run actually produces the same numbers, and the
way to find out is to keep a previous run and compare, not to assert it.

This compares the current matrix against an archived one cell by cell. It is also the
regression test for refactors: the `_prepare` extraction in July 2026 moved the detector
fit, the splits and the attack construction into shared code, and the thing that proves it
was behaviour-preserving is that every cell still lands on the same number.

  python scripts/check_determinism.py
  python scripts/check_determinism.py --baseline results/matrix/raw_pre_persignal.json

Exit code 0 when every shared cell matches.
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
FIELDS = ["valid", "corrupt", "auroc", "detect", "fa"]


def key(r):
    return (r["dataset"], r["class"], r["seed"], r["attack"])


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--current", default="results/matrix/raw.json")
    # The baseline is `matrix_chk`, an independent re-run of the current commit, NOT the
    # archived `raw_baseline.json`. That file predates the 2026-08-09/10 protocol fixes: it
    # shares only 70 of 135 cells with the current matrix (the target-selection rule changed
    # which cells exist) and every shared cell differs because the pipeline it measured no
    # longer exists. Comparing against it can only ever fail, and it fails for a reason that
    # has nothing to do with determinism -- which is worse than not checking, because the
    # failure looks like the real thing.
    ap.add_argument("--baseline", default="results/matrix_chk/raw.json")
    ap.add_argument("--expect-changed-attack", default=None,
                    help="an attack whose cells are EXPECTED to differ because its code "
                         "changed deliberately; every other cell must still reproduce")
    ap.add_argument("--tol", type=float, default=0.0,
                    help="0 means bit-for-bit; raise only with a reason")
    a = ap.parse_args()

    cur_p, base_p = REPO / a.current, REPO / a.baseline
    if not base_p.exists():
        print(f"no baseline at {base_p} — nothing to compare against")
        return 0
    cur = {key(r): r for r in json.loads(cur_p.read_text())}
    base = {key(r): r for r in json.loads(base_p.read_text())}
    shared = sorted(set(cur) & set(base))
    if not shared:
        print("no overlapping cells")
        return 0

    bad, expected = [], []
    for k in shared:
        diffs = {f: abs(cur[k][f] - base[k][f]) for f in FIELDS}
        if max(diffs.values()) <= a.tol:
            continue
        # A deliberate code change makes its own cells differ. That is not nondeterminism,
        # and conflating the two either hides a real regression or cries wolf after every
        # intended fix. Name the changed attack and the rest must still reproduce exactly.
        (expected if a.expect_changed_attack and k[3] == a.expect_changed_attack
         else bad).append((k, diffs))

    print(f"{len(shared)} cells compared against {base_p.name}")
    if expected:
        print(f"  {len(expected)} cells differ in {a.expect_changed_attack}, as declared — "
              f"its implementation changed deliberately")
    for k, d in bad:
        worst = max(d, key=d.get)
        print(f"  DIFFERS {k}  worst: {worst} by {d[worst]:.6f}")
    if bad:
        print(f"\n{len(bad)} of {len(shared)} cells differ — the pipeline is not "
              f"reproducing itself, and the release claim does not hold until it does.")
        return 1
    n_exact = len(shared) - len(expected)
    print(f"\nall {n_exact} unaffected cells reproduce bit-for-bit. Determinism holds and the "
          f"seed is doing its job.")
    if expected:
        print(f"The {len(expected)} changed cells are confined to {a.expect_changed_attack}, "
              f"which is the surgical result: fixing one attack moved that attack and nothing "
              f"else.")

    # The saved per-signal field must agree with the best-signal number it was derived from.
    inconsistent = [k for k in cur if "per_signal" in cur[k]
                    and abs(max(cur[k]["per_signal"].values()) - cur[k]["auroc"]) > 1e-9]
    if inconsistent:
        print(f"per_signal disagrees with the recorded best AUROC in {len(inconsistent)} cells")
        return 1
    print("per-signal AUROC agrees with the recorded best signal in every cell.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
