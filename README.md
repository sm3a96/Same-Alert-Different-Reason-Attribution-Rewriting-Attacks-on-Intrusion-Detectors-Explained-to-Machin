# Same Alert, Different Reason: Attribution Rewriting Attacks on Intrusion Detectors Explained to Machines

Code and results for the paper. The paper calls this benchmark the Explanation Integrity
Benchmark (EIB); the source keeps the code name XIntBench.

## Install

Python 3.10. The lockfile pins the CUDA 12.4 build of torch 2.4.0, which lives on PyTorch's
own index, so pass that index on the first install and keep `-c constraints.txt` on every
install.

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock.txt -c constraints.txt --extra-index-url https://download.pytorch.org/whl/cu124
make setup
make test
```

## Rebuild the tables

No dataset is needed; every table rebuilds from the committed artifacts under `results/`.

```bash
make floats     # tables/out/tab_*.tex
make verify     # the same in a fresh clone, compared byte for byte
```

## Rerun the experiments

Datasets are not redistributed. Each corpus (5G-NIDD, CICIoT2023, CICIoMT2024) carries its own
academic-use terms; `scripts/download_data.py` fetches them and `scripts/audit_data.py --all`
runs the data audit. The runs behind the paper, with the seeds they were run at:

```bash
python scripts/run_matrix.py                                     # seeds 0-4, 3 classes, 100 calibration and 40 test flows per cell
python scripts/run_c3_artifacts.py --seeds 0 1 2 3 4 --n-test 60 --stab-n 120 --stab-sigma 0.05
python scripts/run_decision_utility.py --seeds 0 1 2 3 4 --n-test 40 --judge Qwen/Qwen3-8B --judge microsoft/phi-4
for d in fiveg_nidd ciciot2023 ciciomt2024; do
    python scripts/run_generality.py --dataset $d --seeds 0 1 2 3 4 --n-test 40 --stab-n 120 --stab-sigma 0.05
done
python scripts/run_reference_decomposition.py --seeds 0 1 2 3 4   # arms: --reference median, --tau 0.10, --scaffold-contamination 0.05 0.1 0.3, each with --name
python scripts/run_transferability.py --seeds 0 1 2 3 4
```

The decision task needs a GPU and the two open-weights judges. Everything else runs on CPU
and takes hours, not minutes.

## Results

One directory per run. `meta/run_seed0.json` records the commit, the configuration and the
wall clock of every run; `MANIFEST.csv` maps each float in the paper to the run behind it.

```
results/
  _cache/                         the cases shown to the judges in the decision task, keyed by flow
  _logs/                          data audits, feature identities, judge prompt revisions, regime grid, signal-1 summary, split-leakage check
  c3_artifacts_<corpus>/          per-flow integrity-signal scores, conformal calibration, certified radii and ablations; tree detector, TreeSHAP, 5 seeds
  decision_utility/               every decision of both judges on clean and attacked explanations, plus the shuffled control
  detector_accuracy/              test accuracy of the tree detector in each of the 45 matrix cells (scripts/report_detector_accuracy.py)
  generality_mlp_ig_<corpus>/     the same signals on the MLP + integrated-gradients pipeline, 5 seeds
  matrix/                         the detectability map: 3 corpora x 3 attacks x 3 classes x 5 seeds, tree detector, TreeSHAP
  motivating_examples/            the three flows of the motivating figure (scripts/export_motivating_examples.py)
  reference_decomposition/        the reader scored against the erasure-causal set of the shown flow, and the corruption decomposition
  reference_decomposition_a3_c005/  scaffold contamination 0.05
  reference_decomposition_a3_c01/   scaffold contamination 0.1
  reference_decomposition_a3_c03/   scaffold contamination 0.3
  reference_decomposition_median/   median instead of mean as the erasure reference
  reference_decomposition_tau010/   causal threshold tau = 0.10 instead of 0.05
  split_leakage/                  can a classifier tell training flows from test flows under the grouped split
  transferability/                the displacement attack crafted on a surrogate and scored on the deployed detector
```

## Citation and license

MIT license. See `CITATION.cff`; the paper entry is added once it is published.
