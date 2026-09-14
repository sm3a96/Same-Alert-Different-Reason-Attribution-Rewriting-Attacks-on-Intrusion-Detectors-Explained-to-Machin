# Model card — scaffolded detector (A3 artifact)

A deliberately compromised model, released so the A3 scaffolding attack reproduces.

## What it does

Wraps a fitted detector and detects off-manifold queries — the kind a perturbation-based
explainer generates — routing them to an innocuous surrogate while answering genuine
traffic normally. The prediction on real flows is unchanged; only the explanation is
corrupted. That is the whole point of the artifact.

## Measured behaviour

Corrupts perturbation-based attributions. An exact attributor (TreeSHAP) is left
untouched **by construction**: `scaffold.py` routes off-manifold probe queries to a
surrogate but hands TreeSHAP the real booster, so a near-zero measured corruption
confirms the implementation rather than discovering a property of exact attributors.
The mitigation the paper draws from it — prefer an exact attributor where the model
admits one — is architectural, and should be read as such rather than as evidence
that exact attributors resist scaffolding in general.

## Misuse

This is an attack artifact. It exists to be detected. Do not deploy it, and do not
ship a model derived from it. It is released under the same terms as the benchmark,
for evaluation of defenses only.

## Provenance

Built by `avert.benchmark.scaffold.make_scaffolded_detector` from the base XGBoost
detector of the corresponding dataset. Regenerate with `make paper`.


## Judge models used for C2

Pinned by commit, because a Hub repo can be updated under an unchanged name and
the model id alone would not identify what produced these decisions.

| Model | Revision |
|---|---|
| `Qwen/Qwen3-8B` | `b968826d9c46dd6066d109eabc6255188de91218` |
| `microsoft/phi-4` | `2db69c1c3e91a05d2c64a3185acfbaf36f744e25` |
