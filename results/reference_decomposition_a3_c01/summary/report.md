# Reference correction and decomposition (reference=mean, tau=0.05, min_k=3, scaffold contamination=0.1, attacks=A3_scaffolding)

Rebuilt instances: 1800. Shown case differs from the 13 August cache on 49 (excluded from the rescoring, kept in the decomposition); clean causal set differs on 1. The paper's arm uses the cached causal set so it reproduces the published number exactly.

## Paired effect on reader accuracy, competent cells, cluster-bootstrap 95% CI

| Attack | vs S(x) (paper) | vs S(x') (flow shown) | broke/fixed vs S(x') |
|---|---|---|---|
| A3_scaffolding | +0.000 [+0.000, +0.000] | -0.035 [-0.052, -0.020] | 80/0 |

## Per corpus, vs S(x')

| Attack | Corpus | effect | CI |
|---|---|---|---|
| A3_scaffolding | ciciomt2024 | -0.076 | [-0.115, -0.042] |
| A3_scaffolding | ciciot2023 | -0.005 | [-0.010, -0.001] |
| A3_scaffolding | fiveg_nidd | -0.016 | [-0.027, -0.006] |

## Top-1 decomposition (seed-clustered 95% CI over seed means)

| Attack | Scope | paper | infidelity | reliance | J(S(x),S(x')) | free x / x' | fallback x' | routed |
|---|---|---|---|---|---|---|---|---|
| A3_scaffolding | pooled | +0.000 [+0.000, +0.000] | -0.024 [-0.035, -0.012] | +0.024 [+0.012, +0.035] | 0.027 | 0.86 / 0.86 | 0.011 | 0.034 |
| A3_scaffolding | ciciomt2024 | +0.000 [+0.000, +0.000] | -0.060 [-0.100, -0.020] | +0.060 [+0.020, +0.100] | 0.063 | 0.89 / 0.90 | 0.000 | 0.082 |
| A3_scaffolding | ciciot2023 | +0.000 [+0.000, +0.000] | -0.003 [-0.013, +0.006] | +0.003 [-0.006, +0.013] | 0.002 | 0.96 / 0.96 | 0.000 | 0.003 |
| A3_scaffolding | fiveg_nidd | +0.000 [+0.000, +0.000] | -0.008 [-0.019, +0.002] | +0.008 [-0.002, +0.019] | 0.017 | 0.73 / 0.72 | 0.033 | 0.017 |