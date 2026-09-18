# Explanation Integrity Benchmark (EIB) for NIDS

**Test whether your network-intrusion-detection explanations can be attacked.**

ML-based NIDS increasingly show analysts (and, in agentic SOCs, autonomous agents) a
*why* alongside each alert — a feature-attribution explanation. Those explanations are
consumed as trusted ground truth with no integrity checking, yet an adversary can corrupt
them **without changing the prediction**. EIB measures that exposure on your own pipeline, against ground truth established
interventionally against your own detector.

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
| Rank promotion (code: A1_misdirection) | promote non-causal features into the top-k | hides the true cause; explanation stays *locally faithful* | any explainer |
| Cause displacement (code: A1_displacement) | tiny, prediction-preserving perturbation maximising attribution change | makes the shown explanation hypersensitive | any explainer |
| Explainer scaffolding (code: A3_scaffolding) | Slack-style model routing explainer probes to a benign surrogate | corrupts the explanation for **perturbation-based** explainers (LIME/KernelSHAP) | NOT exact TreeSHAP |

## Reading the report

| Column | Meaning |
|--------|---------|
| `valid` | fraction with the prediction preserved (attack validity; ~1.0 by construction) |
| `corrupt` | top-k **Jaccard distance** between the clean and attacked explanations |
| `AUROC` | best integrity signal separating clean vs attacked explanations |
| `detect` | fused detection rate at the conformal budget alpha |
| `FA` | false-alarm rate on clean traffic (nominal level alpha; the paper reports the empirical rate) |

A high `corrupt` with a low `AUROC` is the danger zone: the attacker rewrote your
explanation and no signal noticed.

## Results

The measurements, their populations and intervals are in the paper (Bibers and
Abdallah, under review at IEEE TIFS). `make floats` rebuilds every table in it from
`results/`; `results/MANIFEST.csv` maps each table and figure to the run behind it.
