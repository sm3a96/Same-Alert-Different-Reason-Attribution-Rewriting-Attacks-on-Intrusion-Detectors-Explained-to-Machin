# Reference correction and decomposition (reference=mean, tau=0.05, min_k=3, scaffold contamination=0.3, attacks=A3_scaffolding)

Rebuilt instances: 1800. Shown case differs from the 13 August cache on 54 (excluded from the rescoring, kept in the decomposition); clean causal set differs on 1. The paper's arm uses the cached causal set so it reproduces the published number exactly.

## Paired effect on reader accuracy, competent cells, cluster-bootstrap 95% CI

| Attack | vs S(x) (paper) | vs S(x') (flow shown) | broke/fixed vs S(x') |
|---|---|---|---|
| A3_scaffolding | +0.000 [+0.000, +0.000] | -0.013 [-0.022, -0.006] | 30/0 |

## Per corpus, vs S(x')

| Attack | Corpus | effect | CI |
|---|---|---|---|
| A3_scaffolding | ciciomt2024 | -0.030 | [-0.050, -0.013] |
| A3_scaffolding | ciciot2023 | +0.000 | [+0.000, +0.000] |
| A3_scaffolding | fiveg_nidd | -0.006 | [-0.013, -0.002] |

## Top-1 decomposition (seed-clustered 95% CI over seed means)

| Attack | Scope | paper | infidelity | reliance | J(S(x),S(x')) | free x / x' | fallback x' | routed |
|---|---|---|---|---|---|---|---|---|
| A3_scaffolding | pooled | +0.000 [+0.000, +0.000] | -0.011 [-0.019, -0.003] | +0.011 [+0.003, +0.019] | 0.012 | 0.86 / 0.86 | 0.011 | 0.014 |
| A3_scaffolding | ciciomt2024 | +0.000 [+0.000, +0.000] | -0.030 [-0.059, -0.002] | +0.030 [+0.002, +0.059] | 0.028 | 0.89 / 0.90 | 0.000 | 0.036 |
| A3_scaffolding | ciciot2023 | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | +0.000 [+0.000, +0.000] | 0.000 | 0.96 / 0.96 | 0.000 | 0.000 |
| A3_scaffolding | fiveg_nidd | +0.000 [+0.000, +0.000] | -0.003 [-0.009, +0.002] | +0.003 [-0.002, +0.009] | 0.007 | 0.73 / 0.72 | 0.032 | 0.007 |