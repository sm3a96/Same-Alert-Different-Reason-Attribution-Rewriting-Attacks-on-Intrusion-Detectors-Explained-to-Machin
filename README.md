# XInt-Bench: explanation-integrity attacks on network intrusion detection

Code and results for the paper *Same Alert, Different Reason*.

ML-based NIDS attach a *why* to every alert, a feature-attribution explanation. In agentic
security operations that explanation is read and acted on by an automated reader rather than
by a person. An adversary can corrupt it without changing the prediction, so prediction-level
monitoring is blind to the whole attack class.

This repository holds the benchmark, the attacks, the integrity signals, the conformal
calibration, the decision task, and every script that produced a number, figure or table in
the paper. The results directory carries the saved artifacts, so every figure and table
rebuilds without the datasets.

The attack class itself is not new. Prediction-preserving attribution attacks were established
by [Ghorbani et al. 2019](https://arxiv.org/abs/1710.10547) and
[Dombrowski et al. 2019](https://arxiv.org/abs/1906.07983), and
[Senevirathna et al. (CCNC 2024)](https://doi.org/10.1109/CCNC51664.2024.10454633) attacked
NIDS explanations with scaffolding. What is new here is the setting the attacks are scored
in, the labels they are scored against, and the automated reader whose decisions are measured.

## Install

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt -c constraints.txt
make setup
make test
```

Python 3.10. torch is pinned to 2.4.0 by `constraints.txt`; keep the `-c` flag on every
install or a transitive dependency will downgrade it.

## Rebuild the figures and tables (no datasets needed)

```bash
make unpack     # inflate results/_packed/ into the paths the generators read
make floats     # every figure into figures/out/, every table into tables/out/
```

`make verify-clone` does this in a fresh clone and compares each generated table byte for byte
against the committed one.

## Rerun the experiments

Datasets are not redistributed. Each corpus (5G-NIDD, CICIoT2023, CICIoMT2024) carries its own
academic-use terms and is fetched from its original host:

```bash
python scripts/download_data.py
python scripts/audit_data.py --all     # dedup, constant and identifier columns, leakage stumps
make results                           # hours; the decision task needs a GPU and two open-weights judges
```

`scripts/run_matrix.py` is the detectability map, `scripts/run_decision_utility.py` the
automated-reader decision task, and `scripts/run_reference_decomposition.py` the corruption
decomposition. `make determinism` compares two independent matrix runs.

## Test your own pipeline

```python
from avert.benchmark import XIntBench
report = XIntBench(data).evaluate_pipeline(detector=my_detector, explainer=my_explainer)
print(report.summary())
```

See [docs/XINTBENCH.md](docs/XINTBENCH.md) for the benchmark and
[docs/TAXONOMY_MAPPING.md](docs/TAXONOMY_MAPPING.md) for the MITRE ATLAS and NIST AI 100-2 mapping.

## Layout

```
src/avert/
  types.py        shared data contracts (Explanation, SignalScore, IntegrityVerdict, ...)
  data/           dataset loading, deduplication, splits, flow grouping
  detectors/      f: X -> Y  (XGBoost, MLP, FT-Transformer, plus a toy detector)
  attribution/    g(x, f)    (TreeSHAP, integrated gradients, permutation, plus a toy explainer)
  signals/        certified stability, cross-method consensus, feature consistency,
                  erasure faithfulness, PASA; agentic verifier and temporal are stubs
  fusion/         split-conformal calibration and signal aggregation
  repair/         stub, not exercised by any result
  benchmark/      XInt-Bench: attacks, scaffolding, interventional ground truth, harness
  eval/           experiment runner, metrics, decision utility
  pipeline.py     end-to-end orchestration
configs/          declarative experiment configs (one run = config + seed)
scripts/          every experiment, audit and export script
results/          per-experiment summary and meta, MANIFEST.csv, and _packed/ raw artifacts
figures/ tables/  generators under src/, outputs under out/
docs/             benchmark description, threat-framework mapping, cross-dataset results map
artifacts/        Croissant metadata, dataset and model cards
tests/            conformal-coverage gate, pipeline smoke tests, phase gates
```

## Citation and license

MIT license. See `CITATION.cff`; the paper entry is added once it is published.
