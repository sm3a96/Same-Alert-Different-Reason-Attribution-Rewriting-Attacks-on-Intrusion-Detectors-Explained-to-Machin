# Reference correction and decomposition (reference=mean, tau=0.05, min_k=3, scaffold contamination=0.3, attacks=A1_displacement,A1_misdirection,A3_scaffolding)

Rebuilt instances: 5400. Shown case differs from the 13 August cache on 2 (excluded from the rescoring, kept in the decomposition); clean causal set differs on 3. The paper's arm uses the cached causal set so it reproduces the published number exactly.

## Paired effect on reader accuracy, competent cells, cluster-bootstrap 95% CI

| Attack | vs S(x) (paper) | vs S(x') (flow shown) | broke/fixed vs S(x') |
|---|---|---|---|
| A1_displacement | -0.175 [-0.230, -0.125] | -0.147 [-0.201, -0.093] | 503/151 |
| A1_misdirection | -0.135 [-0.180, -0.092] | -0.119 [-0.166, -0.073] | 340/55 |
| A3_scaffolding | +0.000 [+0.000, +0.000] | -0.217 [-0.304, -0.143] | 501/0 |

## Per corpus, vs S(x')

| Attack | Corpus | effect | CI |
|---|---|---|---|
| A1_displacement | ciciomt2024 | -0.139 | [-0.240, -0.045] |
| A1_displacement | ciciot2023 | -0.170 | [-0.274, -0.077] |
| A1_displacement | fiveg_nidd | -0.130 | [-0.208, -0.056] |
| A1_misdirection | ciciomt2024 | -0.067 | [-0.131, -0.001] |
| A1_misdirection | ciciot2023 | -0.226 | [-0.316, -0.140] |
| A1_misdirection | fiveg_nidd | -0.064 | [-0.134, -0.013] |
| A3_scaffolding | ciciomt2024 | -0.443 | [-0.614, -0.276] |
| A3_scaffolding | ciciot2023 | -0.050 | [-0.074, -0.028] |
| A3_scaffolding | fiveg_nidd | -0.117 | [-0.189, -0.056] |

## Top-1 decomposition (seed-clustered 95% CI over seed means)

| Attack | Scope | paper | infidelity | reliance | J(S(x),S(x')) | free x / x' | fallback x' | routed |
|---|---|---|---|---|---|---|---|---|
| A1_displacement | pooled | -0.124 [-0.162, -0.086] | -0.043 [-0.106, +0.021] | -0.081 [-0.140, -0.023] | 0.604 | 0.86 / 0.84 | 0.000 | 0.000 |
| A1_displacement | ciciomt2024 | -0.080 [-0.183, +0.023] | -0.067 [-0.154, +0.021] | -0.013 [-0.177, +0.151] | 0.560 | 0.89 / 0.90 | 0.000 | 0.000 |
| A1_displacement | ciciot2023 | -0.090 [-0.308, +0.128] | +0.018 [-0.146, +0.183] | -0.108 [-0.259, +0.043] | 0.578 | 0.96 / 0.95 | 0.000 | 0.000 |
| A1_displacement | fiveg_nidd | -0.202 [-0.306, -0.097] | -0.080 [-0.195, +0.035] | -0.122 [-0.223, -0.021] | 0.673 | 0.73 / 0.66 | 0.000 | 0.000 |
| A1_misdirection | pooled | -0.131 [-0.157, -0.104] | -0.062 [-0.124, +0.001] | -0.069 [-0.139, +0.001] | 0.284 | 0.86 / 0.87 | 0.003 | 0.000 |
| A1_misdirection | ciciomt2024 | -0.172 [-0.218, -0.126] | -0.008 [-0.152, +0.135] | -0.163 [-0.330, +0.003] | 0.444 | 0.89 / 0.91 | 0.000 | 0.000 |
| A1_misdirection | ciciot2023 | -0.172 [-0.203, -0.140] | -0.160 [-0.207, -0.113] | -0.012 [-0.065, +0.041] | 0.234 | 0.96 / 0.97 | 0.000 | 0.000 |
| A1_misdirection | fiveg_nidd | -0.048 [-0.121, +0.024] | -0.017 [-0.104, +0.071] | -0.032 [-0.068, +0.004] | 0.174 | 0.73 / 0.72 | 0.010 | 0.000 |
| A3_scaffolding | pooled | +0.000 [+0.000, +0.000] | -0.151 [-0.209, -0.092] | +0.151 [+0.092, +0.209] | 0.219 | 0.86 / 0.85 | 0.018 | 0.276 |
| A3_scaffolding | ciciomt2024 | +0.000 [+0.000, +0.000] | -0.366 [-0.553, -0.179] | +0.366 [+0.179, +0.553] | 0.395 | 0.89 / 0.94 | 0.000 | 0.552 |
| A3_scaffolding | ciciot2023 | +0.000 [+0.000, +0.000] | -0.027 [-0.061, +0.007] | +0.027 [-0.007, +0.061] | 0.055 | 0.96 / 0.96 | 0.000 | 0.067 |
| A3_scaffolding | fiveg_nidd | +0.000 [+0.000, +0.000] | -0.060 [-0.117, -0.003] | +0.060 [+0.003, +0.117] | 0.208 | 0.73 / 0.66 | 0.053 | 0.208 |