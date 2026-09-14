# Reference correction and decomposition (reference=mean, tau=0.05, min_k=3, scaffold contamination=0.3, attacks=A3_scaffolding)

Rebuilt instances: 1800. Shown case differs from the 13 August cache on 0 (excluded from the rescoring, kept in the decomposition); clean causal set differs on 1. The paper's arm uses the cached causal set so it reproduces the published number exactly.

## Paired effect on reader accuracy, competent cells, cluster-bootstrap 95% CI

| Attack | vs S(x) (paper) | vs S(x') (flow shown) | broke/fixed vs S(x') |
|---|---|---|---|
| A3_scaffolding | +0.000 [+0.000, +0.000] | -0.217 [-0.304, -0.143] | 501/0 |

## Per corpus, vs S(x')

| Attack | Corpus | effect | CI |
|---|---|---|---|
| A3_scaffolding | ciciomt2024 | -0.443 | [-0.614, -0.276] |
| A3_scaffolding | ciciot2023 | -0.050 | [-0.074, -0.028] |
| A3_scaffolding | fiveg_nidd | -0.117 | [-0.189, -0.056] |

## Top-1 decomposition (seed-clustered 95% CI over seed means)

| Attack | Scope | paper | infidelity | reliance | J(S(x),S(x')) | free x / x' | fallback x' | routed |
|---|---|---|---|---|---|---|---|---|
| A3_scaffolding | pooled | +0.000 [+0.000, +0.000] | -0.151 [-0.209, -0.092] | +0.151 [+0.092, +0.209] | 0.219 | 0.86 / 0.85 | 0.018 | 0.276 |
| A3_scaffolding | ciciomt2024 | +0.000 [+0.000, +0.000] | -0.366 [-0.553, -0.179] | +0.366 [+0.179, +0.553] | 0.395 | 0.89 / 0.94 | 0.000 | 0.552 |
| A3_scaffolding | ciciot2023 | +0.000 [+0.000, +0.000] | -0.027 [-0.061, +0.007] | +0.027 [-0.007, +0.061] | 0.055 | 0.96 / 0.96 | 0.000 | 0.067 |
| A3_scaffolding | fiveg_nidd | +0.000 [+0.000, +0.000] | -0.060 [-0.117, -0.003] | +0.060 [+0.003, +0.117] | 0.208 | 0.73 / 0.66 | 0.053 | 0.208 |