# C2 — agent decision utility, regenerated from raw decisions

43200 decisions, 2 judges, 3 datasets, 5 seeds. Every number below is recomputed by `scripts/make_c2_report.py`; none is copied.

Intervals are cluster bootstraps over (judge, dataset, class, attack, seed) cells. Flows inside a cell share a fitted detector and one attack instantiation, so a paired t-interval over all of them is about 2.6x too narrow on this data.

## Headline — A1 displacement, cells clearing the pre-declared competence floor

| Scope | n | cells | mean Δ | 95% CI | broke | fixed | harms? |
|---|---|---|---|---|---|---|---|
| both judges | 2400 | 60 | -0.175 | [-0.230, -0.125] | 454 | 33 | yes |
| Qwen3-8B | 1200 | 30 | -0.177 | [-0.255, -0.107] | 227 | 14 | yes |
| phi-4 | 1200 | 30 | -0.173 | [-0.250, -0.104] | 227 | 19 | yes |

## Broke and fixed, conditioned on the opportunity

The unconditioned ratio is a base-rate artifact of a highly accurate clean baseline and must not be quoted as a directional propensity.

| | count | opportunities | rate |
|---|---|---|---|
| broke | 454 | 2195 correct when clean | 0.207 |
| fixed | 33 | 205 wrong when clean | 0.161 |

Clean accuracy on this population is 0.915; attacked is 0.739.

## Per dataset — disclosed, including the one that goes the other way

Reported because a reviewer will compute it from the released data anyway, and because a dataset that reverses sign is a finding rather than an embarrassment.

| Dataset | Scope | n | cells | mean Δ | 95% CI | broke | fixed | harms? |
|---|---|---|---|---|---|---|---|---|
| CICIoMT2024 | all cells | 1200 | 30 | -0.044 | [-0.071, -0.022] | 56 | 3 | yes |
| CICIoMT2024 | clean≥0.5 | 960 | 24 | -0.054 | [-0.085, -0.028] | 55 | 3 | yes |
| CICIoT2023 | all cells | 1200 | 30 | -0.088 | [-0.162, -0.023] | 140 | 34 | yes |
| CICIoT2023 | clean≥0.5 | 800 | 20 | -0.147 | [-0.245, -0.064] | 124 | 6 | yes |
| 5G-NIDD | all cells | 1200 | 30 | -0.202 | [-0.292, -0.117] | 303 | 61 | yes |
| 5G-NIDD | clean≥0.5 | 640 | 16 | -0.392 | [-0.473, -0.309] | 275 | 24 | yes |

## Leave-one-out — does the headline rest on any single dataset or judge?

| Dropped | n | cells | mean Δ | 95% CI | harms? |
|---|---|---|---|---|---|
| dataset CICIoMT2024 | 1440 | 36 | -0.256 | [-0.331, -0.183] | yes |
| dataset CICIoT2023 | 1600 | 40 | -0.189 | [-0.254, -0.128] | yes |
| dataset 5G-NIDD | 1760 | 44 | -0.097 | [-0.147, -0.055] | yes |
| judge Qwen3-8B | 1200 | 30 | -0.173 | [-0.250, -0.104] | yes |
| judge phi-4 | 1200 | 30 | -0.177 | [-0.255, -0.107] | yes |

## Competence-floor sweep

The floor was fixed before the confirmatory run but after a pilot exposed the problem, so the question is fair. Sweeping answers it.

| floor | cells | n | mean Δ | 95% CI | harms? |
|---|---|---|---|---|---|
| 0.0 | 90 | 3600 | -0.111 | [-0.152, -0.073] | yes |
| 0.2 | 74 | 2960 | -0.144 | [-0.191, -0.101] | yes |
| 0.3 | 70 | 2800 | -0.153 | [-0.201, -0.108] | yes |
| 0.4 | 68 | 2720 | -0.157 | [-0.208, -0.112] | yes |
| 0.5 | 60 | 2400 | -0.175 | [-0.230, -0.125] | yes |
| 0.6 | 60 | 2400 | -0.175 | [-0.230, -0.125] | yes |
| 0.7 | 56 | 2240 | -0.175 | [-0.232, -0.123] | yes |
| 0.8 | 46 | 1840 | -0.180 | [-0.247, -0.122] | yes |

## Position bias — is the agent reading the attribution, or answering A?

| Judge | Order | Scope | P(picks first) | n | attacked Δ | 95% CI |
|---|---|---|---|---|---|---|
| Qwen3-8B | ranked (matched slice) | all cells | 0.718 | 200 | -0.110 | [-0.177, -0.052] |
| Qwen3-8B | ranked (matched slice) | clean≥0.5 | 0.718 | 141 | -0.149 | [-0.243, -0.069] |
| Qwen3-8B | shuffled | all cells | 0.095 | 200 | -0.070 | [-0.154, +0.016] |
| Qwen3-8B | shuffled | clean≥0.5 | 0.095 | 141 | -0.135 | [-0.231, -0.045] |
| phi-4 | ranked (matched slice) | all cells | 0.760 | 200 | -0.080 | [-0.152, -0.016] |
| phi-4 | ranked (matched slice) | clean≥0.5 | 0.760 | 141 | -0.128 | [-0.221, -0.046] |
| phi-4 | shuffled | all cells | 0.134 | 200 | -0.035 | [-0.109, +0.043] |
| phi-4 | shuffled | clean≥0.5 | 0.134 | 141 | -0.099 | [-0.176, -0.030] |

Shuffled-order arm present: 4800 decisions over 546 case-sets. Clean accuracy 0.705 ranked against 0.658 shuffled. The ranked column above is the matched slice, not the headline.

| Dataset | blind-pick rate under shuffling |
|---|---|
| CICIoMT2024 | 0.461 |
| CICIoT2023 | 0.379 |
| 5G-NIDD | 0.197 |

Pre-registered decision rule (fixed 2026-08-01, before the arm was scored). Evaluated here, not asserted:

- blind-pick chance rate, computed over the matched case-sets: 0.349
- clean accuracy: 0.705 ranked, 0.658 shuffled — 87% of the ranked excess over chance is retained (rule B needs at least 50%: met)
- shuffled attack effects clearing zero: 2 of 2 (rule B needs all; rule A needs none)

**Outcome B** — the attack degrades the decision even when position cannot be used, so the agent was reading the attribution.

## What predicts the harm

Decision harm tracks the CHANGE in whether the top-ranked feature is causal. It does not track the clean level of that quantity, and it does not track top-k set corruption. The pooled correlation is mostly between-dataset and is effectively n=3; quote the within-dataset column.

| Mediator | pooled r | ciciomt2024 | ciciot2023 | fiveg_nidd |
|---|---|---|---|---|
| clean level of P(top-1 causal) | -0.425 | -0.328 (p=0.077) | -0.509 (p=0.004) | -0.850 (p=0.000) |
| **change** in P(top-1 causal) | +0.840 | +0.341 (p=0.065) | +0.906 (p=0.000) | +0.927 (p=0.000) |

Per dataset, the mean change in P(top-1 causal) against the mean harm:

| Dataset | mean change in P(top-1 causal) | mean harm |
|---|---|---|
| CICIoMT2024 | -0.080 | -0.044 |
| CICIoT2023 | -0.090 | -0.088 |
| 5G-NIDD | -0.202 | -0.202 |

## Ranking withheld — is the explanation load-bearing at all?

| Judge | n | cells | mean Δ | 95% CI |
|---|---|---|---|---|
| Qwen3-8B | 5400 | 135 | -0.332 | [-0.398, -0.265] |
| phi-4 | 5400 | 135 | -0.293 | [-0.354, -0.231] |
