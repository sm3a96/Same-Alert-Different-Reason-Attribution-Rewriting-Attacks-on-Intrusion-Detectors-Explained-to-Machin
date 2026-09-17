.PHONY: setup lock test slice lint clean

# Editable install of the package into the active venv. Keep -c constraints.txt: rtdl and
# tabpfn would otherwise drag torch back to 1.13.
setup:
	python -m pip install -e ".[dev]" -c constraints.txt

lock:
	python -m pip freeze > requirements.lock.txt

# The test suite. The conformal-coverage gate must stay green.
test:
	python -m pytest -q

# Walking skeleton: the whole pipeline end to end on synthetic data.
slice:
	python scripts/run_walking_skeleton.py

lint:
	ruff check src tests scripts tables

clean:
	rm -rf .pytest_cache .ruff_cache **/__pycache__ src/*.egg-info

# ---------------------------------------------------------------- experiments
# Every result in the paper, regenerated from the datasets. Expect hours, not minutes: the
# fragility attack and the smoothing certificate dominate. Each stage writes incrementally,
# so an interrupted run resumes cheaply. The decision-utility run needs a GPU and the two
# open-weights judges.
.PHONY: results
results:
	python scripts/run_matrix.py
	python scripts/run_c3_artifacts.py --seeds 0 1 2 3 4 --n-test 60 --stab-n 120 --stab-sigma 0.05
	python scripts/run_decision_utility.py --seeds 0 1 2 3 4 --n-test 40 \
		--judge Qwen/Qwen3-8B --judge microsoft/phi-4
	for d in fiveg_nidd ciciot2023 ciciomt2024; do \
		python scripts/run_generality.py --dataset $$d --seeds 0 1 2 3 4 --n-test 40 --stab-n 120; \
	done
	python scripts/run_reference_decomposition.py
	python scripts/export_motivating_examples.py

# ---------------------------------------------------------------- figures and tables
# Every table the paper shows, rebuilt from the saved artifacts; figures ship as PDFs. `make unpack` first on a fresh
# clone; no dataset is needed for this step.
.PHONY: floats tables cards
floats: tables

# Dependency order, not alphabetical: summarize_signal1.py writes the summary the later
# generators read.
tables:
	python scripts/summarize_signal1.py
	python scripts/classify_regimes.py
	python scripts/measure_lime_agreement.py
	python scripts/make_c2_report.py
	python scripts/check_explainer_faithfulness.py
	python scripts/check_recovery.py
	python tables/src/make_new_paper_tables.py
	python scripts/make_results_doc.py
	python scripts/check_seed_variance.py

# Release artifacts, generated from the configs and results so they cannot drift.
cards:
	python scripts/export_model_artifacts.py
	python scripts/make_artifact_cards.py

# ---------------------------------------------------------------- checks
.PHONY: determinism identities verify-clone pack unpack
determinism:
	python scripts/check_determinism.py

identities:
	python scripts/validate_feature_identities.py

# Clone HEAD, unpack results/_packed/, rebuild every table from those artifacts alone, and
# compare byte for byte against the working tree.
verify-clone:
	bash scripts/verify_clean_clone.sh

pack:
	python scripts/pack_artifacts.py

unpack:
	python scripts/pack_artifacts.py --unpack
