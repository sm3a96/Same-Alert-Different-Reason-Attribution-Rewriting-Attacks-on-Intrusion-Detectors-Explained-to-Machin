# C2 — agent decision utility

acc = decision accuracy against interventional ground truth. flip = the decision
changed relative to the matched clean decision. `control` is the identical clean
case sampled twice — the judge's own noise floor. A flip rate at or below control
carries no claim.

### All cells

| Judge | Dataset | Attack | clean acc | attacked acc | repaired acc | rank-withheld acc | attacked flip | control flip | judge conf. | cells |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | ciciomt2024 | A1_displacement | 0.81±0.19 | 0.76±0.18 | 0.76±0.18 | 0.53±0.24 | 0.20±0.16 | 0.00±0.00 | 1.00±0.00 | 15 |
| Qwen3-8B | ciciomt2024 | A1_misdirection | 0.81±0.19 | 0.73±0.16 | 0.73±0.16 | 0.53±0.24 | 0.14±0.10 | 0.00±0.00 | 1.00±0.00 | 15 |
| Qwen3-8B | ciciomt2024 | A3_scaffolding | 0.81±0.19 | 0.80±0.19 | 0.80±0.19 | 0.53±0.24 | 0.01±0.01 | 0.00±0.00 | 1.00±0.00 | 15 |
| Qwen3-8B | ciciot2023 | A1_displacement | 0.70±0.22 | 0.61±0.19 | 0.61±0.19 | 0.37±0.23 | 0.32±0.15 | 0.01±0.01 | 0.99±0.01 | 15 |
| Qwen3-8B | ciciot2023 | A1_misdirection | 0.70±0.22 | 0.55±0.18 | 0.55±0.18 | 0.37±0.23 | 0.19±0.13 | 0.01±0.01 | 0.99±0.01 | 15 |
| Qwen3-8B | ciciot2023 | A3_scaffolding | 0.70±0.22 | 0.70±0.22 | 0.70±0.22 | 0.37±0.23 | 0.00±0.00 | 0.01±0.01 | 0.99±0.01 | 15 |
| Qwen3-8B | fiveg_nidd | A1_displacement | 0.52±0.19 | 0.30±0.11 | 0.30±0.11 | 0.13±0.08 | 0.52±0.16 | 0.00±0.00 | 1.00±0.00 | 15 |
| Qwen3-8B | fiveg_nidd | A1_misdirection | 0.52±0.19 | 0.48±0.18 | 0.48±0.18 | 0.13±0.08 | 0.08±0.06 | 0.00±0.00 | 1.00±0.00 | 15 |
| Qwen3-8B | fiveg_nidd | A3_scaffolding | 0.52±0.19 | 0.52±0.19 | 0.52±0.19 | 0.13±0.08 | 0.00±0.00 | 0.00±0.00 | 1.00±0.00 | 15 |
| phi-4 | ciciomt2024 | A1_displacement | 0.81±0.19 | 0.77±0.18 | 0.77±0.18 | 0.64±0.19 | 0.21±0.17 | 0.00±0.00 | 1.00±0.00 | 15 |
| phi-4 | ciciomt2024 | A1_misdirection | 0.81±0.19 | 0.74±0.16 | 0.74±0.16 | 0.64±0.19 | 0.13±0.09 | 0.00±0.00 | 1.00±0.00 | 15 |
| phi-4 | ciciomt2024 | A3_scaffolding | 0.81±0.19 | 0.80±0.19 | 0.80±0.19 | 0.64±0.19 | 0.01±0.01 | 0.00±0.00 | 1.00±0.00 | 15 |
| phi-4 | ciciot2023 | A1_displacement | 0.70±0.21 | 0.61±0.18 | 0.61±0.18 | 0.40±0.17 | 0.36±0.17 | 0.00±0.00 | 1.00±0.00 | 15 |
| phi-4 | ciciot2023 | A1_misdirection | 0.70±0.21 | 0.54±0.18 | 0.54±0.18 | 0.40±0.17 | 0.19±0.13 | 0.00±0.00 | 1.00±0.00 | 15 |
| phi-4 | ciciot2023 | A3_scaffolding | 0.70±0.21 | 0.70±0.22 | 0.70±0.22 | 0.40±0.17 | 0.00±0.00 | 0.00±0.00 | 1.00±0.00 | 15 |
| phi-4 | fiveg_nidd | A1_displacement | 0.52±0.19 | 0.33±0.10 | 0.33±0.10 | 0.10±0.08 | 0.55±0.14 | 0.00±0.00 | 0.99±0.01 | 15 |
| phi-4 | fiveg_nidd | A1_misdirection | 0.52±0.19 | 0.47±0.17 | 0.47±0.17 | 0.10±0.08 | 0.09±0.06 | 0.00±0.00 | 0.99±0.01 | 15 |
| phi-4 | fiveg_nidd | A3_scaffolding | 0.52±0.19 | 0.52±0.19 | 0.52±0.19 | 0.10±0.08 | 0.00±0.00 | 0.00±0.00 | 0.99±0.01 | 15 |

(180/270 cells clear the pre-declared clean-accuracy floor of 0.5)

### Cells where the judge is competent (clean acc >= 0.5)

| Judge | Dataset | Attack | clean acc | attacked acc | repaired acc | rank-withheld acc | attacked flip | control flip | judge conf. | cells |
|---|---|---|---|---|---|---|---|---|---|---|
| Qwen3-8B | ciciomt2024 | A1_displacement | 0.96±0.06 | 0.90±0.08 | 0.90±0.08 | 0.62±0.29 | 0.24±0.19 | 0.00±0.00 | 1.00±0.00 | 12 |
| Qwen3-8B | ciciomt2024 | A1_misdirection | 0.96±0.06 | 0.86±0.09 | 0.86±0.09 | 0.62±0.29 | 0.16±0.12 | 0.00±0.00 | 1.00±0.00 | 12 |
| Qwen3-8B | ciciomt2024 | A3_scaffolding | 0.96±0.06 | 0.95±0.06 | 0.95±0.06 | 0.62±0.29 | 0.01±0.01 | 0.00±0.00 | 1.00±0.00 | 12 |
| Qwen3-8B | ciciot2023 | A1_displacement | 0.94±0.09 | 0.81±0.15 | 0.81±0.15 | 0.44±0.28 | 0.38±0.21 | 0.00±0.00 | 1.00±0.01 | 10 |
| Qwen3-8B | ciciot2023 | A1_misdirection | 0.94±0.09 | 0.72±0.15 | 0.72±0.15 | 0.44±0.28 | 0.26±0.19 | 0.00±0.00 | 1.00±0.01 | 10 |
| Qwen3-8B | ciciot2023 | A3_scaffolding | 0.94±0.09 | 0.94±0.09 | 0.94±0.09 | 0.44±0.28 | 0.00±0.01 | 0.00±0.00 | 1.00±0.01 | 10 |
| Qwen3-8B | fiveg_nidd | A1_displacement | 0.81±0.11 | 0.40±0.13 | 0.40±0.13 | 0.21±0.12 | 0.69±0.12 | 0.00±0.00 | 1.00±0.00 | 8 |
| Qwen3-8B | fiveg_nidd | A1_misdirection | 0.81±0.11 | 0.74±0.11 | 0.74±0.11 | 0.21±0.12 | 0.12±0.12 | 0.00±0.00 | 1.00±0.00 | 8 |
| Qwen3-8B | fiveg_nidd | A3_scaffolding | 0.81±0.11 | 0.81±0.11 | 0.81±0.11 | 0.21±0.12 | 0.00±0.00 | 0.00±0.00 | 1.00±0.00 | 8 |
| phi-4 | ciciomt2024 | A1_displacement | 0.96±0.06 | 0.92±0.07 | 0.92±0.07 | 0.75±0.18 | 0.25±0.20 | 0.00±0.00 | 1.00±0.00 | 12 |
| phi-4 | ciciomt2024 | A1_misdirection | 0.96±0.06 | 0.87±0.09 | 0.87±0.09 | 0.75±0.18 | 0.15±0.12 | 0.00±0.00 | 1.00±0.00 | 12 |
| phi-4 | ciciomt2024 | A3_scaffolding | 0.96±0.06 | 0.95±0.06 | 0.95±0.06 | 0.75±0.18 | 0.01±0.01 | 0.00±0.00 | 1.00±0.00 | 12 |
| phi-4 | ciciot2023 | A1_displacement | 0.94±0.09 | 0.78±0.15 | 0.78±0.15 | 0.44±0.19 | 0.44±0.25 | 0.00±0.00 | 1.00±0.00 | 10 |
| phi-4 | ciciot2023 | A1_misdirection | 0.94±0.09 | 0.71±0.15 | 0.71±0.15 | 0.44±0.19 | 0.26±0.19 | 0.00±0.00 | 1.00±0.00 | 10 |
| phi-4 | ciciot2023 | A3_scaffolding | 0.94±0.09 | 0.94±0.09 | 0.94±0.09 | 0.44±0.19 | 0.00±0.01 | 0.00±0.00 | 1.00±0.00 | 10 |
| phi-4 | fiveg_nidd | A1_displacement | 0.81±0.10 | 0.43±0.13 | 0.43±0.13 | 0.13±0.17 | 0.70±0.09 | 0.00±0.01 | 0.99±0.01 | 8 |
| phi-4 | fiveg_nidd | A1_misdirection | 0.81±0.10 | 0.72±0.11 | 0.72±0.11 | 0.13±0.17 | 0.14±0.12 | 0.00±0.01 | 0.99±0.01 | 8 |
| phi-4 | fiveg_nidd | A3_scaffolding | 0.81±0.10 | 0.81±0.10 | 0.81±0.10 | 0.13±0.17 | 0.00±0.00 | 0.00±0.01 | 0.99±0.01 | 8 |
### Paired effect on the decision

Each matched instance contributes correct(condition) - correct(clean).
`broke` counts decisions the condition turned from right to wrong, `fixed`
the reverse. Intervals are cluster bootstraps over cells, not paired t-CIs:
flows within a cell share a fitted detector and one attack instantiation, so
treating them as independent understates the interval by about 2.5x.

| Scope | Judge | Condition | n | cells | mean delta | 95% CI | broke | fixed | harms? |
|---|---|---|---|---|---|---|---|---|---|
| all | both | attacked | 3600 | 90 | -0.111 | [-0.152, -0.073] | 499 | 98 | yes |
| all | both | repaired | 3600 | 90 | -0.111 | [-0.152, -0.073] | 499 | 98 | yes |
| all | both | no_explanation | 10800 | 270 | -0.312 | [-0.357, -0.267] | 3702 | 327 | yes |
| all | Qwen3-8B | attacked | 1800 | 45 | -0.118 | [-0.177, -0.066] | 250 | 38 | yes |
| all | Qwen3-8B | repaired | 1800 | 45 | -0.118 | [-0.177, -0.066] | 250 | 38 | yes |
| all | Qwen3-8B | no_explanation | 5400 | 135 | -0.332 | [-0.398, -0.265] | 1935 | 144 | yes |
| all | phi-4 | attacked | 1800 | 45 | -0.105 | [-0.167, -0.050] | 249 | 60 | yes |
| all | phi-4 | repaired | 1800 | 45 | -0.105 | [-0.167, -0.050] | 249 | 60 | yes |
| all | phi-4 | no_explanation | 5400 | 135 | -0.293 | [-0.354, -0.231] | 1767 | 183 | yes |
| clean>=0.5 | both | attacked | 2400 | 60 | -0.175 | [-0.230, -0.125] | 454 | 33 | yes |
| clean>=0.5 | both | repaired | 2400 | 60 | -0.175 | [-0.230, -0.125] | 454 | 33 | yes |
| clean>=0.5 | both | no_explanation | 7200 | 180 | -0.448 | [-0.496, -0.399] | 3276 | 54 | yes |
| clean>=0.5 | Qwen3-8B | attacked | 1200 | 30 | -0.177 | [-0.255, -0.107] | 227 | 14 | yes |
| clean>=0.5 | Qwen3-8B | repaired | 1200 | 30 | -0.177 | [-0.255, -0.107] | 227 | 14 | yes |
| clean>=0.5 | Qwen3-8B | no_explanation | 3600 | 90 | -0.464 | [-0.538, -0.389] | 1683 | 12 | yes |
| clean>=0.5 | phi-4 | attacked | 1200 | 30 | -0.173 | [-0.250, -0.104] | 227 | 19 | yes |
| clean>=0.5 | phi-4 | repaired | 1200 | 30 | -0.173 | [-0.250, -0.104] | 227 | 19 | yes |
| clean>=0.5 | phi-4 | no_explanation | 3600 | 90 | -0.431 | [-0.493, -0.366] | 1593 | 42 | yes |

### Is the agent reading the attribution, or answering A?

Options are listed in attribution order, and the attack re-orders them, so
position is confounded with the treatment. The shuffled arm permutes the
candidate order while each feature keeps its own value and attribution.

| Judge | Order | P(picks first option) | attacked effect | 95% CI |
|---|---|---|---|---|
| Qwen3-8B | ranked | 0.709 | -0.118 | [-0.177, -0.066] |
| Qwen3-8B | shuffled | 0.095 | -0.070 | [-0.154, +0.016] |
| phi-4 | ranked | 0.755 | -0.105 | [-0.167, -0.050] |
| phi-4 | shuffled | 0.134 | -0.035 | [-0.109, +0.043] |

### Does the finding depend on where the floor was set?

| floor | cells | n | mean delta | 95% CI | harms? |
|---|---|---|---|---|---|
| 0.0 | 90 | 3600 | -0.111 | [-0.152, -0.073] | yes |
| 0.2 | 74 | 2960 | -0.144 | [-0.191, -0.101] | yes |
| 0.3 | 70 | 2800 | -0.153 | [-0.201, -0.108] | yes |
| 0.4 | 68 | 2720 | -0.157 | [-0.208, -0.112] | yes |
| 0.5 | 60 | 2400 | -0.175 | [-0.230, -0.125] | yes |
| 0.6 | 60 | 2400 | -0.175 | [-0.230, -0.125] | yes |
| 0.7 | 56 | 2240 | -0.175 | [-0.232, -0.123] | yes |
| 0.8 | 46 | 1840 | -0.180 | [-0.247, -0.122] | yes |
### Prompt sensitivity

The same decisions re-scored under two alternative wordings of the identical
question. `agreement` is how often the re-worded prompt yields the same
correctness as the default; `acc` is accuracy under that wording.

| Judge | Wording | n | agreement with default | acc |
|---|---|---|---|---|
| Qwen3-8B | analyst | 1200 | 0.873 | 0.552 |
| Qwen3-8B | terse | 1200 | 0.956 | 0.547 |
| phi-4 | analyst | 1200 | 0.938 | 0.607 |
| phi-4 | terse | 1200 | 0.950 | 0.573 |