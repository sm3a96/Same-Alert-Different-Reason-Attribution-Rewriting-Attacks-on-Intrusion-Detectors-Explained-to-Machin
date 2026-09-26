"""Threat-framework mapping for XInt-Bench attacks → MITRE ATLAS (v5.4.0, Feb 2026) and
NIST AI 100-2 E2025, so industrial red teams can ingest the benchmark in the vocabulary
they already use. Verified ATLAS technique IDs only; where existing taxonomies do not
cover explanation-integrity attacks, that gap is recorded explicitly (it is part of the
contribution).
"""
from __future__ import annotations

FRAMEWORK_MAPPING = {
    "A1_misdirection": {
        "threat_class": "attribution rewriting",
        "atlas": ["AML.T0043"],                 # Craft Adversarial Data
        "atlas_adjacent": ["AML.T0015"],        # Evade AI Model (targets the prediction, not the explanation)
        "nist_ai_100_2": "Evasion",
        "gap": "ATLAS 'evade' targets the prediction; no technique covers rewriting the "
               "explanation while the prediction is preserved.",
    },
    "A1_displacement": {
        "threat_class": "attribution rewriting",
        "atlas": ["AML.T0043"],
        "atlas_adjacent": ["AML.T0015"],
        "nist_ai_100_2": "Evasion",
        "gap": "Same gap: explanation-level rewriting is unrepresented in current taxonomies.",
    },
    "A3_scaffolding": {
        "threat_class": "model-artifact integrity",
        "atlas": ["AML.T0040"],                 # Erode ML Model Integrity
        "atlas_adjacent": [],
        "nist_ai_100_2": "Integrity / supply chain",
        "gap": "ATLAS integrity-erosion targets model outputs, not explanation faithfulness.",
    },
    "A2_poisoning": {                            # planned
        "threat_class": "explainer poisoning",
        "atlas": [],
        "atlas_adjacent": [],
        "nist_ai_100_2": "Poisoning",
        "gap": "Poisoning of the explanation component specifically is unrepresented.",
    },
}


def atlas_tags(attack: str) -> list[str]:
    return FRAMEWORK_MAPPING.get(attack, {}).get("atlas", [])
