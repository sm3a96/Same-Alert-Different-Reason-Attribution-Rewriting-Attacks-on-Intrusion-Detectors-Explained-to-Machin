"""Re-check every declared feature identity against clean traffic, and fail if one has moved.

`avert.benchmark.realizability` declares, per dataset, the arithmetic its flow extractor imposes
on its own output -- `TotPkts = SrcPkts + DstPkts` and thirteen more for 5G-NIDD's Argus schema,
an inequality triple for the CIC corpora. Those declarations are load-bearing twice over: the
attack projects onto them, so a wrong one produces unrealizable flows, and the feature-consistency
signal scores against them, so a wrong one produces a fictitious detector.

A declaration is only worth as much as its last check. Datasets get silently re-released, a
loader change alters which columns survive, and a schema written against one feature list quietly
stops applying to another. This script re-derives every hold rate from clean data and exits
non-zero if any declared relation drops below the threshold.

  python scripts/validate_feature_identities.py
  python scripts/validate_feature_identities.py --datasets fiveg_nidd --rows 50000

Writes results/_logs/feature_identities.json. `make gate` runs it.
"""
from __future__ import annotations

import argparse
import json
import logging
from pathlib import Path

import numpy as np

from avert.benchmark.realizability import HOLD_THRESHOLD, schema_for
from avert.config import load_config
from avert.data.datasets import load_dataset

REPO = Path(__file__).resolve().parents[1]
DATASETS = ["fiveg_nidd", "ciciot2023", "ciciomt2024"]
log = logging.getLogger("validate_identities")


def validate(dataset: str, rows: int, seed: int = 0) -> dict:
    data = load_dataset(dataset, load_config(f"datasets/{dataset}"))
    rng = np.random.default_rng(seed)
    idx = rng.permutation(len(data.X))[:rows]
    bound = schema_for(dataset).bind(data.feature_names)
    report = bound.validate(data.X[idx])
    report["sampled_rows"] = int(len(idx))
    report["total_rows"] = int(len(data.X))
    report["n_features"] = len(data.feature_names)
    report["notes"] = schema_for(dataset).notes

    log.info("%s: %d features, %d derived, %d free",
             dataset, report["n_features"], len(report["derived_features"]), report["n_free"])
    for c in report["constraints"]:
        rate = "n/a" if c["hold_rate"] is None else f"{c['hold_rate'] * 100:8.4f}%"
        log.info("   %-4s %-38s %s  on %d rows",
                 "OK" if c["passed"] else "FAIL", c["expr"], rate, c["defined_rows"])
    if not report["all_passed"]:
        log.error("%s: a declared relation no longer holds. Either the data changed or the "
                  "declaration was wrong -- do not run attacks until this is resolved.", dataset)
    return report


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--datasets", nargs="+", default=DATASETS)
    ap.add_argument("--rows", type=int, default=200_000)
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")

    reports = {ds: validate(ds, a.rows, a.seed) for ds in a.datasets}
    out = REPO / "results" / "_logs" / "feature_identities.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(
        {"threshold": HOLD_THRESHOLD, "datasets": reports}, indent=1))
    log.info("saved %s", out)

    failed = [ds for ds, r in reports.items() if not r["all_passed"]]
    if failed:
        log.error("FAILED: %s", ", ".join(failed))
        return 1
    log.info("all declared feature identities hold")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
