.PHONY: setup test floats verify

# Editable install into the active venv. Keep -c constraints.txt: a transitive dependency would
# otherwise move torch off 2.4.0.
setup:
	python -m pip install -e . -c constraints.txt

test:
	python -m pytest -q

# Every table the paper shows, rebuilt from the committed artifacts under results/. No dataset
# is needed. summarize_signal1.py runs first because tab_generality reads its output.
floats:
	python scripts/summarize_signal1.py
	python tables/src/make_new_paper_tables.py

# Clone HEAD into a temporary directory, run `make floats` there, and compare every generated
# table byte for byte against the working tree.
verify:
	bash scripts/verify_clean_clone.sh
