# Dataset card — XInt-Bench

Explanation-integrity attacks on flow-based network intrusion detection, with
ground-truth feature attributions established interventionally.

## What this is

Matched pairs of clean and explanation-attacked instances. Ground truth is
**detector-relative and interventional**, not generator-derived: a feature counts as<!-- retired-ok: states what the ground truth is NOT -->
causal when ablating it toward its benign value drops the deployed detector's
predicted-class probability past a threshold. The parameterized traffic generators
described in the design are not implemented and no number here comes from them.

That makes the labels a statement about the model under test rather than about the
physics of the attack — weaker than 'known by construction', and stronger in the way
that matters, because it is what lets you point this at your own deployed pipeline.
The labelling never consults the attacked attributions, so there is no circularity
between the ground truth and what is being scored.

## Source datasets

| Dataset | Domain | Target classes | Seeds | Cells | Dup. before dedup | Licence |
|---|---|---|---|---|---|---|
| 5G-NIDD | 5G wireless testbed traffic | 3 | 5 | 45 | 0.0% | Academic use; see data/PROVENANCE.md for the exact terms |
| CICIoT2023 | IoT topology, 105 devices, 33 attacks | 3 | 5 | 45 | 0.0% | CIC academic licence |
| CICIoMT2024 | Internet of Medical Things, multi-protocol | 3 | 5 | 45 | 38.4% | CIC academic licence |

### Excluded, and why

- **CICIoV2024** (per-frame CAN bus) — 99.75% duplicate rows (1,404,631 of 1,408,219); dedups to 3,588 unique of which about 40 are attacks. Excluded from every result.

Every dataset is deduplicated at the full-row level, has constant and identifier
columns dropped, and is checked for per-feature leakage before use
(`scripts/audit_data.py`). The exclusion above is what that audit is for.

## Attacks

| ID | Attack | Fools |
|---|---|---|
| A1 misdirection | promotes non-causal features while staying locally faithful | every signal tested |
| A1 displacement | tiny prediction-preserving perturbation that rewrites the ranking | every signal tested |
| A3 scaffolding | model artifact that detects explainer probes | perturbation-based explainers only |

Implemented: A1_misdirection, A1_displacement, A3_scaffolding. A2, A4 and A5 are defined in the threat model
and are not evaluated; the paper says so rather than leaving the gap implicit.

## Threat-framework mapping

| Attack | MITRE ATLAS | NIST AI 100-2 |
|---|---|---|
| A1_misdirection | AML.T0043 | Evasion |
| A1_displacement | AML.T0043 | Evasion |
| A3_scaffolding | AML.T0040 | Integrity / supply chain |
| A2_poisoning | -- | Poisoning |

Neither framework has a category for corrupting an explanation while preserving
the prediction. That gap is part of the contribution, not an oversight in the mapping.

## Intended use and limits

Intended for red-teaming the explanation layer of an NIDS pipeline, and for developing
integrity monitors against a benchmark with real ground truth. Not a general-purpose
intrusion-detection dataset: the traffic is inherited from the sources above and the
contribution is the attack layer on top of it.

The attacks were built by the same people evaluating defenses against them, which
bounds how adversarial the evaluation is. The benchmark is released so that someone
else can build a stronger attack against it.

## Reproduction

```bash
make paper   # every number, figure and table
```
