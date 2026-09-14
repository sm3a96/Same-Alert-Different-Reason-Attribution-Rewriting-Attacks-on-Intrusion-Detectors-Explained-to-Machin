# Reference correction and decomposition (reference=mean, tau=0.1, min_k=3, scaffold contamination=0.3, attacks=A1_displacement,A1_misdirection,A3_scaffolding)

Rebuilt instances: 5400. Shown case differs from the 13 August cache on 829 (excluded from the rescoring, kept in the decomposition); clean causal set differs on 3333. The paper's arm uses the cached causal set so it reproduces the published number exactly.

## Paired effect on reader accuracy, competent cells, cluster-bootstrap 95% CI

| Attack | vs S(x) (paper) | vs S(x') (flow shown) | broke/fixed vs S(x') |
|---|---|---|---|
| A1_displacement | -0.175 [-0.230, -0.125] | -0.183 [-0.240, -0.128] | 581/141 |
| A1_misdirection | -0.091 [-0.131, -0.057] | -0.084 [-0.130, -0.047] | 117/18 |
| A3_scaffolding | +0.000 [+0.000, +0.000] | -0.248 [-0.333, -0.176] | 573/0 |

## Per corpus, vs S(x')

| Attack | Corpus | effect | CI |
|---|---|---|---|
| A1_displacement | ciciomt2024 | -0.174 | [-0.278, -0.078] |
| A1_displacement | ciciot2023 | -0.205 | [-0.314, -0.106] |
| A1_displacement | fiveg_nidd | -0.170 | [-0.245, -0.098] |
| A1_misdirection | ciciomt2024 | -0.070 | [-0.166, +0.017] |
| A1_misdirection | ciciot2023 | -0.128 | [-0.222, -0.063] |
| A1_misdirection | fiveg_nidd | -0.057 | [-0.122, -0.008] |
| A3_scaffolding | ciciomt2024 | -0.463 | [-0.632, -0.301] |
| A3_scaffolding | ciciot2023 | -0.090 | [-0.123, -0.059] |
| A3_scaffolding | fiveg_nidd | -0.152 | [-0.220, -0.087] |

## Top-1 decomposition (seed-clustered 95% CI over seed means)

| Attack | Scope | paper | infidelity | reliance | J(S(x),S(x')) | free x / x' | fallback x' | routed |
|---|---|---|---|---|---|---|---|---|
| A1_displacement | pooled | -0.135 [-0.180, -0.090] | -0.064 [-0.128, -0.001] | -0.071 [-0.131, -0.010] | 0.596 | 0.86 / 0.83 | 0.000 | 0.000 |
| A1_displacement | ciciomt2024 | -0.082 [-0.172, +0.008] | -0.063 [-0.155, +0.028] | -0.018 [-0.154, +0.117] | 0.554 | 0.87 / 0.88 | 0.000 | 0.000 |
| A1_displacement | ciciot2023 | -0.100 [-0.327, +0.127] | -0.048 [-0.190, +0.093] | -0.052 [-0.217, +0.113] | 0.572 | 0.96 / 0.95 | 0.000 | 0.000 |
| A1_displacement | fiveg_nidd | -0.223 [-0.327, -0.119] | -0.082 [-0.197, +0.034] | -0.142 [-0.212, -0.071] | 0.663 | 0.74 / 0.66 | 0.000 | 0.000 |
| A1_misdirection | pooled | -0.119 [-0.133, -0.105] | -0.039 [-0.111, +0.033] | -0.080 [-0.144, -0.016] | 0.275 | 0.86 / 0.87 | 0.030 | 0.000 |
| A1_misdirection | ciciomt2024 | -0.170 [-0.194, -0.146] | +0.013 [-0.117, +0.144] | -0.183 [-0.334, -0.032] | 0.431 | 0.87 / 0.90 | 0.000 | 0.000 |
| A1_misdirection | ciciot2023 | -0.147 [-0.188, -0.105] | -0.118 [-0.197, -0.039] | -0.028 [-0.083, +0.027] | 0.240 | 0.96 / 0.97 | 0.067 | 0.000 |
| A1_misdirection | fiveg_nidd | -0.040 [-0.090, +0.010] | -0.012 [-0.084, +0.061] | -0.028 [-0.061, +0.004] | 0.153 | 0.74 / 0.73 | 0.023 | 0.000 |
| A3_scaffolding | pooled | +0.000 [+0.000, +0.000] | -0.150 [-0.216, -0.083] | +0.150 [+0.083, +0.216] | 0.226 | 0.86 / 0.85 | 0.050 | 0.276 |
| A3_scaffolding | ciciomt2024 | +0.000 [+0.000, +0.000] | -0.349 [-0.553, -0.145] | +0.349 [+0.145, +0.553] | 0.416 | 0.87 / 0.93 | 0.000 | 0.552 |
| A3_scaffolding | ciciot2023 | +0.000 [+0.000, +0.000] | -0.037 [-0.061, -0.013] | +0.037 [+0.013, +0.061] | 0.056 | 0.96 / 0.96 | 0.065 | 0.067 |
| A3_scaffolding | fiveg_nidd | +0.000 [+0.000, +0.000] | -0.063 [-0.123, -0.003] | +0.063 [+0.003, +0.123] | 0.208 | 0.74 / 0.67 | 0.085 | 0.208 |