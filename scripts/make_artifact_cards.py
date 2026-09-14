"""Generate the release artifacts: dataset card, model cards, Croissant metadata.

Written by script rather than by hand for the same reason the tables are: a card that
drifts from the code is worse than no card, because it looks authoritative. Everything
here is read from the dataset configs, the taxonomy module and the saved results.

  python scripts/make_artifact_cards.py

Writes to artifacts/cards/ and artifacts/benchmark/.
"""
from __future__ import annotations

import json
from collections import defaultdict
from pathlib import Path

from avert.benchmark.harness import ALL_ATTACKS
from avert.benchmark.taxonomy import FRAMEWORK_MAPPING
from avert.config import load_config

REPO = Path(__file__).resolve().parents[1]
CARDS = REPO / "artifacts" / "cards"
BENCH = REPO / "artifacts" / "benchmark"

DATASETS = {
    "fiveg_nidd": ("5G-NIDD", "5G wireless testbed traffic",
                   "Academic use; see data/PROVENANCE.md for the exact terms"),
    "ciciot2023": ("CICIoT2023", "IoT topology, 105 devices, 33 attacks",
                   "CIC academic licence"),
    "ciciomt2024": ("CICIoMT2024", "Internet of Medical Things, multi-protocol",
                    "CIC academic licence"),
}
def dup_rate(key: str) -> str:
    """Pre-dedup duplicate fraction for an INCLUDED dataset, from its saved audit.

    Disclosed because CICIoMT2024 is 38% duplicates before the loader dedups it, and it is
    the dataset carrying the one cell where the certified-stability signal detects. A reader
    should not have to run the audit to learn that.
    """
    f = REPO / "results" / "_logs" / f"audit_{key}.json"
    if not f.exists():
        return "not audited"
    return f"{json.loads(f.read_text())['duplicate_fraction']:.1%}"


def excluded():
    """The exclusion note, with its numbers read from the saved audit rather than restated.

    This sentence is the public justification for dropping a dataset, so it must not be a
    figure someone typed once -- it was 99.5% here and 99.75% in the audit for months.
    """
    f = REPO / "results" / "_logs" / "audit_ciciov2024.json"
    if not f.exists():
        raise SystemExit("run `python scripts/audit_data.py --dataset ciciov2024` first")
    a = json.loads(f.read_text())
    return {"ciciov2024": ("CICIoV2024", "per-frame CAN bus",
                           f"{a['duplicate_fraction']:.2%} duplicate rows "
                           f"({a['stored_rows_pre_dedup'] - a['unique_rows']:,} of "
                           f"{a['stored_rows_pre_dedup']:,}); dedups to {a['unique_rows']:,} "
                           f"unique of which about 40 are attacks. Excluded from every result.")}


def matrix_stats():
    f = REPO / "results" / "matrix" / "raw.json"
    if not f.exists():
        return {}
    per = defaultdict(lambda: {"classes": set(), "seeds": set(), "cells": 0})
    for r in json.loads(f.read_text()):
        d = per[r["dataset"]]
        d["classes"].add(r["class"])
        d["seeds"].add(r["seed"])
        d["cells"] += 1
    return per


def dataset_card() -> None:
    stats = matrix_stats()
    L = ["# Dataset card — XInt-Bench", "",
         "Explanation-integrity attacks on flow-based network intrusion detection, with",
         "ground-truth feature attributions established interventionally.", "",
         "## What this is", "",
         "Matched pairs of clean and explanation-attacked instances. Ground truth is",
         "**detector-relative and interventional**, not generator-derived: a feature counts as"  "<!-- retired-ok: states what the ground truth is NOT -->",
         "causal when ablating it toward its benign value drops the deployed detector's",
         "predicted-class probability past a threshold. The parameterized traffic generators",
         "described in the design are not implemented and no number here comes from them.",
         "",
         "That makes the labels a statement about the model under test rather than about the",
         "physics of the attack — weaker than 'known by construction', and stronger in the way",
         "that matters, because it is what lets you point this at your own deployed pipeline.",
         "The labelling never consults the attacked attributions, so there is no circularity",
         "between the ground truth and what is being scored.", "",
         "## Source datasets", "",
         "| Dataset | Domain | Target classes | Seeds | Cells | Dup. before dedup | Licence |",
         "|---|---|---|---|---|---|---|"]
    for key, (name, domain, lic) in DATASETS.items():
        try:
            load_config(f"datasets/{key}")
        except Exception:
            pass
        s = stats.get(key)
        cl = len(s["classes"]) if s else "--"
        sd = len(s["seeds"]) if s else "--"
        ce = s["cells"] if s else "--"
        L.append(f"| {name} | {domain} | {cl} | {sd} | {ce} | {dup_rate(key)} | {lic} |")
    L += ["", "### Excluded, and why", ""]
    for _, (name, domain, why) in excluded().items():
        L.append(f"- **{name}** ({domain}) — {why}")
    L += ["", "Every dataset is deduplicated at the full-row level, has constant and identifier",
          "columns dropped, and is checked for per-feature leakage before use",
          "(`scripts/audit_data.py`). The exclusion above is what that audit is for.", "",
          "## Attacks", "",
          "| ID | Attack | Fools |", "|---|---|---|",
          "| A1 misdirection | promotes non-causal features while staying locally faithful | every signal tested |",
          "| A1 displacement | tiny prediction-preserving perturbation that rewrites the ranking | every signal tested |",
          "| A3 scaffolding | model artifact that detects explainer probes | perturbation-based explainers only |",
          "",
          f"Implemented: {', '.join(ALL_ATTACKS)}. A2, A4 and A5 are defined in the threat model",
          "and are not evaluated; the paper says so rather than leaving the gap implicit.", "",
          "## Threat-framework mapping", "",
          "| Attack | MITRE ATLAS | NIST AI 100-2 |", "|---|---|---|"]
    for atk, m in FRAMEWORK_MAPPING.items():
        L.append(f"| {atk} | {', '.join(m['atlas']) or '--'} | {m['nist_ai_100_2']} |")
    L += ["", "Neither framework has a category for corrupting an explanation while preserving",
          "the prediction. That gap is part of the contribution, not an oversight in the mapping.",
          "", "## Intended use and limits", "",
          "Intended for red-teaming the explanation layer of an NIDS pipeline, and for developing",
          "integrity monitors against a benchmark with real ground truth. Not a general-purpose",
          "intrusion-detection dataset: the traffic is inherited from the sources above and the",
          "contribution is the attack layer on top of it.", "",
          "The attacks were built by the same people evaluating defenses against them, which",
          "bounds how adversarial the evaluation is. The benchmark is released so that someone",
          "else can build a stronger attack against it.", "",
          "## Reproduction", "", "```bash", "make paper   # every number, figure and table", "```"]
    (CARDS / "dataset_card_xintbench.md").write_text("\n".join(L) + "\n")
    print(f"  wrote {CARDS / 'dataset_card_xintbench.md'}")


def model_cards() -> None:
    L = ["# Model card — scaffolded detector (A3 artifact)", "",
         "A deliberately compromised model, released so the A3 scaffolding attack reproduces.",
         "", "## What it does", "",
         "Wraps a fitted detector and detects off-manifold queries — the kind a perturbation-based",
         "explainer generates — routing them to an innocuous surrogate while answering genuine",
         "traffic normally. The prediction on real flows is unchanged; only the explanation is",
         "corrupted. That is the whole point of the artifact.", "",
         "## Measured behaviour", "",
         "Corrupts perturbation-based attributions. An exact attributor (TreeSHAP) is left",
         "untouched **by construction**: `scaffold.py` routes off-manifold probe queries to a",
         "surrogate but hands TreeSHAP the real booster, so a near-zero measured corruption",
         "confirms the implementation rather than discovering a property of exact attributors.",
         "The mitigation the paper draws from it — prefer an exact attributor where the model",
         "admits one — is architectural, and should be read as such rather than as evidence",
         "that exact attributors resist scaffolding in general.",
         "", "## Misuse", "",
         "This is an attack artifact. It exists to be detected. Do not deploy it, and do not",
         "ship a model derived from it. It is released under the same terms as the benchmark,",
         "for evaluation of defenses only.", "",
         "## Provenance", "",
         "Built by `avert.benchmark.scaffold.make_scaffolded_detector` from the base XGBoost",
         "detector of the corresponding dataset. Regenerate with `make paper`.", ""]
    rev = REPO / "results" / "_logs" / "judge_revisions.json"
    if rev.exists():
        L += ["", "## Judge models used for C2", "",
              "Pinned by commit, because a Hub repo can be updated under an unchanged name and",
              "the model id alone would not identify what produced these decisions.", "",
              "| Model | Revision |", "|---|---|"]
        for mid, sha in sorted(json.loads(rev.read_text()).items()):
            L.append(f"| `{mid}` | `{sha or 'unresolved'}` |")
    (CARDS / "model_card_scaffold.md").write_text("\n".join(L) + "\n")
    print(f"  wrote {CARDS / 'model_card_scaffold.md'}")


def croissant() -> None:
    """Croissant metadata. Deliberately minimal and honest about what is not filled in."""
    meta = {
        "@context": {"@vocab": "https://schema.org/", "cr": "http://mlcommons.org/croissant/"},
        "@type": "sc:Dataset",
        "name": "XInt-Bench",
        "description": ("Explanation-integrity attacks on flow-based network intrusion "
                        "detection, with ground-truth feature attributions known by "
                        "the detector under test."),
        "conformsTo": "http://mlcommons.org/croissant/1.0",
        "license": "See artifacts/cards/dataset_card_xintbench.md; source datasets carry "
                   "their own academic licences.",
        "citeAs": "TODO: fill in once the paper has a DOI",
        "url": "TODO: fill in at release",
        "keywords": ["network intrusion detection", "explainable AI", "adversarial machine "
                     "learning", "attribution", "benchmark"],
        "recordSet": [
            {"@type": "cr:RecordSet", "name": ds,
             "description": f"{label} — {domain}"}
            for ds, (label, domain, _) in DATASETS.items()
        ],
    }
    BENCH.mkdir(parents=True, exist_ok=True)
    (BENCH / "croissant.json").write_text(json.dumps(meta, indent=2) + "\n")
    print(f"  wrote {BENCH / 'croissant.json'}  (two TODOs left: citeAs and url, both need "
          f"the Zenodo DOI)")


def main() -> None:
    CARDS.mkdir(parents=True, exist_ok=True)
    print("generating release artifacts:")
    dataset_card()
    model_cards()
    croissant()


if __name__ == "__main__":
    main()
