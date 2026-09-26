# What was fixed before the analysis ran

This file records the decisions the paper calls "pre-registered" or "fixed before", and the
date on which each was fixed. It carries only those commitments. The
planning documents they were extracted from are not part of the release.

The constants are in the code. The dates are from the authors' private development history,
which is not part of the release, so they are stated rather than checkable here.

## Constants fixed before any attack was run

| Commitment | Value | Where it lives | Date fixed |
|---|---|---|---|
| Erasure threshold `tau` | 0.05 | `src/avert/benchmark/ground_truth.py`, and the `--tau` default of `scripts/run_reference_decomposition.py` | fixed 31 July 2026, before every result the paper reports. Re-scored at `tau = 0.10` as a sensitivity arm, reported beside the main value. |
| Competence floor | 0.5 clean accuracy | `competent_cell_set` in `src/avert/eval/decision_utility.py`; `--competence-floor` in `scripts/run_decision_utility.py` | fixed 31 July 2026, after a pilot and before the confirmatory run; swept from 0 to 0.8 afterwards, and the sweep is Table VII. |
| Erasure target | benign class mean | `_prepare` in `src/avert/benchmark/harness.py` | fixed on the same date. Re-scored with the benign median as a sensitivity arm. |
| Conformal level `alpha` | 0.05 | `XIntBench.alpha` | fixed on the same date. Swept to 0.20. |
| Bootstrap resamples | 8,000, seed 0 | `paired_effect` in `src/avert/eval/decision_utility.py` | fixed on the same date. |

## The decomposition analysis, fixed 7 September 2026

The reference correction and the corruption decomposition were specified before the full run,
in a plan dated 7 September 2026. What it fixed:

**Claim 2.** The top-1 corruption decomposes into an infidelity term and a reliance-shift term,
and the reliance term is non-zero on every corpus.

**Measure.** `paper = P[top1(x') in S(x)] - P[top1(x) in S(x)]`,
`infidelity = P[top1(x') in S(x')] - P[top1(x) in S(x)]`, `reliance = paper - infidelity`; each
with a seed-clustered t-interval over seed means (`seed_clustered_ci`), per corpus and pooled.
`alpha = 0.05`; **Holm correction over the three attacks within each claim.**

**Falsifier 2.** *Wrong if the reliance term's seed-clustered 95% confidence interval contains
0 on two or more of three corpora.*

**Falsifier 3.** Wrong if the scaffolding effect against `S(x')` has a confidence interval
containing 0 on the competent cells pooled.

**Kill criterion.** If the pooled effect against `S(x')` has a confidence interval containing 0,
the harm claim for cause displacement is withdrawn from the abstract and the contributions, and
the paper reports the artefact and the decomposition as the finding instead. No second
reference, no other floor, no other judge subset.

## Outcome

Falsifier 2 fired. The reliance term's interval contains zero on two of the three corpora, so
the per-corpus form of Claim 2 is **falsified**, and the paper reports it as falsified. Pooled,
the term survives the Holm correction. The two statements are reported together wherever the
reliance shift is discussed.

The kill criterion did not fire: the pooled competent-cell effect against `S(x')` excludes zero.
The paper nevertheless leads with the unselected estimate over all reader-cells, whose interval
contains zero, because the competence floor is a selection and the reader deserves to see the
number that involves no selection at all.

Reproduce both: `python scripts/run_reference_decomposition.py` writes
`results/reference_decomposition/summary/holm.json`, which carries the per-attack p-values, the
Holm-adjusted values and the per-corpus interval for every term.
