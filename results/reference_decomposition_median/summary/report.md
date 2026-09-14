# Reference correction and decomposition (reference=median, tau=0.05, min_k=3, scaffold contamination=0.3, attacks=A1_displacement,A1_misdirection,A3_scaffolding)

Rebuilt instances: 5400. Shown case differs from the 13 August cache on 1041 (excluded from the rescoring, kept in the decomposition); clean causal set differs on 4464. The paper's arm uses the cached causal set so it reproduces the published number exactly.

## Paired effect on reader accuracy, competent cells, cluster-bootstrap 95% CI

| Attack | vs S(x) (paper) | vs S(x') (flow shown) | broke/fixed vs S(x') |
|---|---|---|---|
| A1_displacement | -0.175 [-0.230, -0.125] | -0.195 [-0.273, -0.120] | 623/154 |
| A1_misdirection | -0.016 [-0.032, -0.004] | -0.239 [-0.335, -0.149] | 229/20 |
| A3_scaffolding | +0.000 [+0.000, +0.000] | -0.428 [-0.536, -0.321] | 1041/52 |

## Per corpus, vs S(x')

| Attack | Corpus | effect | CI |
|---|---|---|---|
| A1_displacement | ciciomt2024 | -0.135 | [-0.233, -0.044] |
| A1_displacement | ciciot2023 | -0.297 | [-0.464, -0.126] |
| A1_displacement | fiveg_nidd | -0.158 | [-0.261, -0.058] |
| A1_misdirection | ciciomt2024 | -0.075 | [-0.109, -0.031] |
| A1_misdirection | ciciot2023 | -0.119 | [-0.320, +0.000] |
| A1_misdirection | fiveg_nidd | -0.363 | [-0.448, -0.266] |
| A3_scaffolding | ciciomt2024 | -0.429 | [-0.606, -0.257] |
| A3_scaffolding | ciciot2023 | -0.461 | [-0.684, -0.228] |
| A3_scaffolding | fiveg_nidd | -0.386 | [-0.500, -0.263] |

## Top-1 decomposition (seed-clustered 95% CI over seed means)

| Attack | Scope | paper | infidelity | reliance | J(S(x),S(x')) | free x / x' | fallback x' | routed |
|---|---|---|---|---|---|---|---|---|
| A1_displacement | pooled | -0.067 [-0.116, -0.018] | +0.023 [-0.037, +0.083] | -0.089 [-0.148, -0.031] | 0.603 | 0.91 / 0.86 | 0.000 | 0.000 |
| A1_displacement | ciciomt2024 | -0.100 [-0.231, +0.031] | -0.072 [-0.154, +0.010] | -0.028 [-0.219, +0.162] | 0.533 | 0.90 / 0.91 | 0.000 | 0.000 |
| A1_displacement | ciciot2023 | +0.057 [-0.117, +0.230] | +0.125 [-0.141, +0.391] | -0.068 [-0.270, +0.133] | 0.597 | 0.94 / 0.95 | 0.000 | 0.000 |
| A1_displacement | fiveg_nidd | -0.157 [-0.281, -0.032] | +0.015 [-0.153, +0.183] | -0.172 [-0.267, -0.076] | 0.680 | 0.89 / 0.73 | 0.000 | 0.000 |
| A1_misdirection | pooled | -0.089 [-0.140, -0.038] | +0.004 [-0.057, +0.065] | -0.093 [-0.134, -0.052] | 0.289 | 0.91 / 0.92 | 0.025 | 0.000 |
| A1_misdirection | ciciomt2024 | -0.178 [-0.265, -0.092] | +0.030 [-0.068, +0.128] | -0.208 [-0.338, -0.079] | 0.458 | 0.90 / 0.92 | 0.002 | 0.000 |
| A1_misdirection | ciciot2023 | -0.032 [-0.047, -0.016] | -0.020 [-0.039, -0.001] | -0.012 [-0.017, -0.006] | 0.257 | 0.94 / 0.95 | 0.030 | 0.000 |
| A1_misdirection | fiveg_nidd | -0.057 [-0.125, +0.012] | +0.002 [-0.105, +0.108] | -0.058 [-0.100, -0.016] | 0.153 | 0.89 / 0.88 | 0.043 | 0.000 |
| A3_scaffolding | pooled | +0.000 [+0.000, +0.000] | -0.169 [-0.231, -0.107] | +0.169 [+0.107, +0.231] | 0.217 | 0.91 / 0.89 | 0.027 | 0.276 |
| A3_scaffolding | ciciomt2024 | +0.000 [+0.000, +0.000] | -0.360 [-0.543, -0.178] | +0.360 [+0.178, +0.543] | 0.389 | 0.89 / 0.93 | 0.000 | 0.552 |
| A3_scaffolding | ciciot2023 | +0.000 [+0.000, +0.000] | -0.023 [-0.056, +0.009] | +0.023 [-0.009, +0.056] | 0.054 | 0.94 / 0.94 | 0.030 | 0.067 |
| A3_scaffolding | fiveg_nidd | +0.000 [+0.000, +0.000] | -0.123 [-0.183, -0.064] | +0.123 [+0.064, +0.183] | 0.207 | 0.89 / 0.79 | 0.050 | 0.208 |