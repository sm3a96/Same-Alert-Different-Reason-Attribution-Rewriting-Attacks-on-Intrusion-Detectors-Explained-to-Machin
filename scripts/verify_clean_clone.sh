#!/usr/bin/env bash
# Can somebody with the repo and no datasets rebuild every table, and get OUR numbers?
#
# Clones HEAD into a temporary directory, rebuilds the tables from the committed artifacts
# under results/ alone, and compares every generated table byte for byte against the working
# tree. The datasets are never touched.
#
#   bash scripts/verify_clean_clone.sh          # clone HEAD as committed
#   bash scripts/verify_clean_clone.sh --keep   # leave the clone for inspection
#
# Exit code 0 when every table matches.
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

echo "rebuilding the tables from the committed artifacts alone"
make -s floats >/dev/null

echo
fail=0
for f in tables/out/*.tex results/_logs/signal1_cross_dataset.json; do
    if cmp -s "$f" "$REPO/$f"; then
        echo "  same   $f"
    else
        echo "  DIFFER $f  -- the clone rebuilt a different file"
        fail=1
    fi
done

echo
if [[ $fail -eq 0 ]]; then
    echo "CLEAN CLONE REPRODUCES: every table byte-identical, no datasets used."
else
    echo "CLEAN CLONE DIVERGES -- the committed tables were not built from the committed results." >&2
fi
exit $fail
