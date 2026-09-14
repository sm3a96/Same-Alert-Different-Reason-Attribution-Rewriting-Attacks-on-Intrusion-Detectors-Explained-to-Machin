# XInt-Bench — threat-framework mapping

Maps each explanation-integrity attack to **MITRE ATLAS** (v5.4.0, Feb 2026 — 16 tactics,
84 techniques) and **NIST AI 100-2 E2025**, so industrial red teams can place XInt-Bench in
the vocabulary they already use. Only verified ATLAS technique IDs are used.

| XInt attack | Threat class | Closest MITRE ATLAS | NIST AI 100-2 | Coverage gap |
|---|---|---|---|---|
| A1 misdirection | attribution evasion | `AML.T0043` Craft Adversarial Data (adjacent: `AML.T0015` Evade AI Model) | Evasion | ATLAS "evade" targets the **prediction**; nothing covers evading the **explanation** while the prediction holds |
| A1 displacement | attribution evasion | `AML.T0043` Craft Adversarial Data | Evasion | same gap |
| A3 scaffolding | model-artifact integrity | `AML.T0040` Erode ML Model Integrity | Integrity / supply chain | ATLAS integrity-erosion targets model **outputs**, not explanation **faithfulness** |
| A2 explainer poisoning *(planned)* | poisoning | — | Poisoning | poisoning of the explanation component specifically is unrepresented |

## The gap is part of the contribution

Current AI threat taxonomies (MITRE ATLAS, NIST AI 100-2) enumerate attacks on model
**predictions**. Attacks that corrupt the **explanation** while preserving the prediction
are not represented in either framework. XInt-Bench is, to our knowledge, the first to
operationalize this class for NIDS — and these mappings show precisely where it **extends**
the existing frameworks rather than duplicating them. A red team can run XInt-Bench and
report results under the ATLAS techniques above, with the explanation-integrity dimension
flagged as the novel axis.

Machine-readable form: `avert.benchmark.taxonomy.FRAMEWORK_MAPPING` (the harness can emit
ATLAS tags per attack).

Sources: [MITRE ATLAS — Evade AI Model (AML.T0015)](https://atlas.mitre.org/techniques/AML.T0015),
[Craft Adversarial Data (AML.T0043)](https://atlas.mitre.org/techniques/AML.T0043),
[Erode ML Model Integrity (AML.T0040)](https://atlas.mitre.org/techniques/AML.T0040).
