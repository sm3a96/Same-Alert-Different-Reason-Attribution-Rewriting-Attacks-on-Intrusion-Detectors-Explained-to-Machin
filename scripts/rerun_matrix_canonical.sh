#!/usr/bin/env bash
# Establish matrix determinism, then re-run the canonical matrix so its recorded commit is true.
#
# Why this exists. `run_experiment` stamps `git_commit` from HEAD, so a run started with
# uncommitted changes records a commit that does not describe the code that produced it. The
# matrix on disk from 2026-08-13 is stamped `7a2aec3` and was produced by the Signal 4 abstain
# change, which was not committed until `a7cd6a0` two days later. The numbers are almost
# certainly right and the provenance is not, which is exactly what a provenance check fails
# it for.
#
# The fix is a re-run at a committed HEAD, and the determinism comparison has to happen BEFORE
# it starts: `run_matrix.py` rewrites raw.json incrementally, so the canonical run would
# overwrite the very file the comparison reads.
#
#   1. matrix_chk (already produced by `run_matrix.py --out results/matrix_chk`) against the
#      matrix on disk. This answers "is the pipeline deterministic across two independent runs
#      of the same code", and its output is kept as the evidence.
#   2. the canonical run into results/matrix at HEAD.
#   3. the canonical run against matrix_chk -- two runs, one commit, and the only comparison
#      that can be quoted as the release claim.
#
#   bash scripts/rerun_matrix_canonical.sh              # run now
#   bash scripts/rerun_matrix_canonical.sh --wait       # wait for matrix_chk to finish first
#
# --wait blocks until matrix_chk is complete and no matrix run is alive, so this can be queued
# behind a run already in flight. It refuses to start on a partial matrix_chk rather than
# comparing against a file that is still being written.
set -euo pipefail
cd "$(dirname "$0")/.."

CHK=results/matrix_chk/raw.json
LOG=results/_logs
EXPECTED_CELLS=135

cells() { python -c "import json,sys; print(len(json.load(open('$CHK'))))" 2>/dev/null || echo 0; }

if [[ "${1:-}" == "--wait" ]]; then
    echo "waiting for matrix_chk to reach $EXPECTED_CELLS cells with no matrix run alive"
    while pgrep -f "run_matrix.py" >/dev/null 2>&1 \
       || pgrep -f "run_generality.py" >/dev/null 2>&1 \
       || [[ ! -f "$CHK" ]] || [[ "$(cells)" -lt "$EXPECTED_CELLS" ]]; do
        sleep 60
    done
fi

if [[ ! -f "$CHK" ]] || [[ "$(cells)" -lt "$EXPECTED_CELLS" ]]; then
    echo "refusing to run: $CHK is missing or partial ($(cells) of $EXPECTED_CELLS cells)." >&2
    echo "A comparison against a partial file is not a determinism check." >&2
    exit 2
fi

echo "=== determinism: matrix_chk against the matrix on disk  $(date) ==="
python scripts/check_determinism.py --current "$CHK" --baseline results/matrix/raw.json \
    | tee "$LOG/determinism_chk_vs_disk.txt"

echo "=== canonical matrix at HEAD  $(date) ==="
python scripts/run_matrix.py

echo "=== determinism: canonical against matrix_chk, one commit, two runs  $(date) ==="
python scripts/check_determinism.py --current results/matrix/raw.json --baseline "$CHK" \
    | tee "$LOG/determinism_canonical_vs_chk.txt"

echo "=== done  $(date) ==="
