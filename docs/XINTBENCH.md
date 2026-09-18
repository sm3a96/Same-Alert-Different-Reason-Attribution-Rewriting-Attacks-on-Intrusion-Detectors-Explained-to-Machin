# Explanation Integrity Benchmark (EIB) for NIDS

**Test whether your network-intrusion-detection explanations can be attacked.**

ML-based NIDS increasingly show analysts (and, in agentic SOCs, autonomous agents) a
*why* alongside each alert — a feature-attribution explanation. Those explanations are
consumed as trusted ground truth with no integrity checking, yet an adversary can corrupt
them **without changing the prediction**. EIB is the first benchmark that lets you
measure that exposure on your own pipeline, against ground truth established interventionally
against your own detector.

## Quickstart

```python
from avert.data.datasets import load_dataset
from avert.config import load_config
from avert.benchmark import XIntBench

data = load_dataset("fiveg_nidd", load_config("datasets/fiveg_nidd"))
report = XIntBench(data).evaluate_pipeline(detector=my_detector, explainer=my_explainer)
print(report.summary())
```

`detector` implements the `Detector` interface (`predict`, `predict_proba`); `explainer`
implements `Attributor` (`explain`). Defaults are XGBoost + TreeSHAP, so you can run it
with no arguments to see the reference pipeline. CLI:

```bash
python scripts/run_xintbench.py --dataset fiveg_nidd      # or ciciot2023, ciciomt2024
```

## What it does

For a target attack class it (1) establishes **interventional ground truth** — the
features whose ablation toward benign actually moves the detection — (2) runs the attack
suite against your pipeline, and (3) reports, per attack, how badly the explanation is
corrupted while the prediction is preserved, and how well integrity signals catch it.

## Attack suite

| ID | Attack | What it does | Who it fools |
|----|--------|--------------|--------------|
| A1 misdirection | promote non-causal features into the top-k | hides the true cause; explanation stays *locally faithful* | any explainer; only domain knowledge catches it |
| A1 displacement | tiny, prediction-preserving perturbation maximising attribution change | makes the shown explanation hypersensitive | any explainer |
| A3 scaffolding | Slack-style model routing explainer probes to a benign surrogate | corrupts the explanation for **perturbation-based** explainers (LIME/KernelSHAP) | NOT exact TreeSHAP |

Planned: A2 explainer poisoning, A4 drift-masquerade, A5 monitor-targeting.

## Reading the report

| Column | Meaning |
|--------|---------|
| `valid` | fraction with the prediction preserved (attack validity; ~1.0 by construction) |
| `corrupt` | top-k **Jaccard distance** between the clean and attacked explanations. NOT the fraction of shown features replaced: at J the replaced fraction is (k−m)/k with m = 2k(1−J)/(2−J), so J = 0.66–0.81 is 49–68% of the shown top-5, averaged per dataset (per cell, 25%–91%) |
| `AUROC` | best integrity signal separating clean vs attacked explanations |
| `detect` | fused detection rate at the conformal budget alpha |
| `FA` | false-alarm rate on clean traffic (conformal guarantee: ≤ alpha) |

A high `corrupt` with a low `AUROC` is the danger zone: the attacker rewrote your
explanation and no signal noticed.

## Does it matter? Ask the agent, not the top-k

`corrupt` says how much of the shown explanation changed. That is legible inside the
explainability literature and nowhere else — no security team budgets against a rank
overlap. So the benchmark also measures the thing that does matter: what a downstream
autonomous triage agent decides after reading the corrupted explanation.

```bash
python scripts/run_decision_utility.py --judge Qwen/Qwen3-8B --judge microsoft/phi-4
```

The agent is shown the alert and the ranked attribution list and must name the feature the
containment action should target. Because the causal features are established interventionally, the
decision is scored objectively — no human label anywhere. Four conditions: clean, attacked,
ranking-withheld, and a re-derivation arm. The re-derivation arm re-runs the same explainer on
the same attacked flow, so it is an identity by construction (3,600/3,600 paired decisions equal
under each attack) — no repair step is exercised, and no repair result is claimed. <!-- retired-ok: states that no repair is claimed --> Plus two arms that make the result interpretable rather
than merely impressive: the same clean case decided twice, which is the judge's own noise
floor, and the same decisions re-scored under alternative prompt wordings.

## What we have learned 

- **A1 displacement is severe and largely undetectable by statistics**: it rewrites a large
  fraction of the shown explanation while preserving the prediction, at AUROC ≈ 0.5 on
  some datasets. Detectability is dataset-dependent.
- **A3 scaffolding only bites perturbation-based explainers.** If you show exact TreeSHAP,
  scaffolding does not corrupt your explanation. This is the one mitigation here that needs
  no monitor, no calibration and no latency budget — actionable guidance for vendors.
- **Certified attribution stability cannot be deployed, and our first two explanations of
  why were both wrong.** The first said clean attributions are inherently unstable; the second
  overturned it by measuring 86–94% of clean flows certifying <!-- retired-ok: names the withdrawn figure in order to withdraw it -->
  — on corpora still carrying a row counter and a capture clock. On the cleaned feature sets only 1.7–21.0% certify and the mass
  sitting on the estimator ceiling is essentially zero, so the first account pointed the right
  way for a reason nobody had identified. Saturation is not the mechanism either: the radius
  *is* capped by the Clopper–Pearson bound at `sigma * Phi^-1(conf^(1/n))`, and on the leaked
  corpora most clean and attacked samples sat on that cap — but a gradient pipeline on the same
  data does not saturate and the signal still fails there.

  What rules it out is measured across datasets, detector/attributor pairs and seeds: the
  AUROC ranges **0.025 to 0.747** across datasets, pipelines and seeds — and only 1 of 3 datasets keeps its regime across pipelines — sometimes detecting, sometimes blind, and sometimes
  pointing backwards hard enough that inverting it would be the better policy. Calibration is
  label-free by design, so nothing tells you which regime you are in before you deploy. The
  claim is *not deployable*, never *broken* — the difference matters, because "broken" is
  refuted by pointing at the cell where it reaches 0.707.
- **Fusing a dead signal into a working one makes it worse.** Naive max-z aggregation scores
  below the best single signal. Reliability-aware, label-free combination is open.
- **The conformal false-alarm guarantee holds** across datasets, verified with exact
  binomial intervals rather than a normal approximation.

Full cross-dataset numbers come from `make results` and `make floats` and are recorded in the results MANIFEST.
