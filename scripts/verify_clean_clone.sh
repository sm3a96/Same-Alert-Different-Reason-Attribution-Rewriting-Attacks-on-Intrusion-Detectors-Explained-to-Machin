#!/usr/bin/env bash
# Can somebody with the repo and no datasets rebuild every float, and get OUR numbers?
#
# The release promises one-command reproduction. That promise has two halves and only one of
# them is usually checked: that the floats build, and that what they build equals what the
# paper shows. A clone that regenerates a table from a stale packed artifact builds cleanly
# and disagrees silently, which is the failure worth catching -- the evaluator sees a table,
# not a diff.
#
# So this clones into a temporary directory, unpacks `results/_packed/`, rebuilds the tables
# from those artifacts alone, and compares every generated table byte for byte
# against the working tree. The datasets are never touched: `_packed` is the only input.
#
#   bash scripts/verify_clean_clone.sh          # clone HEAD as committed
#   bash scripts/verify_clean_clone.sh --keep   # leave the clone for inspection
#
# Exit code 0 when every table matches. Figures are checked for existence rather than bytes:
# matplotlib embeds a creation timestamp, so a byte comparison of PDFs fails for reasons that
# have nothing to do with the numbers.
set -euo pipefail
cd "$(dirname "$0")/.."
REPO="$PWD"
KEEP="${1:-}"

TMP="$(mktemp -d)"
cleanup() { [[ "$KEEP" == "--keep" ]] || rm -rf "$TMP"; }
trap cleanup EXIT

echo "cloning HEAD into $TMP"
git clone -q "$REPO" "$TMP/clone"
cd "$TMP/clone"

echo "unpacking release artifacts (no datasets present)"
python scripts/pack_artifacts.py --unpack

echo "rebuilding the tables from the unpacked artifacts alone"
python scripts/summarize_signal1.py >/dev/null
python scripts/classify_regimes.py >/dev/null
python scripts/make_c2_report.py >/dev/null
python tables/src/make_new_paper_tables.py >/dev/null

echo
fail=0
for f in tables/out/*.tex; do
    if cmp -s "$f" "$REPO/$f"; then
        echo "  same   $(basename "$f")"
    else
        echo "  DIFFER $(basename "$f")  -- the clone rebuilt a different table"
        fail=1
    fi
done

echo
if [[ $fail -eq 0 ]]; then
    echo "CLEAN CLONE REPRODUCES: every table byte-identical, no datasets used."
else
    echo "CLEAN CLONE DIVERGES -- run 'make pack' if the working tree is the correct one." >&2
fi
exit $fail
