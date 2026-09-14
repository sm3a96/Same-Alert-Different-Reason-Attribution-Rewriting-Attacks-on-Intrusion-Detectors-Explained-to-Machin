"""Make a fresh clone able to regenerate every figure and table without re-running anything.

The repository ignores `results/*/raw/` because raw per-sample rows are bulky and get rewritten
on every run. The consequence went unnoticed until the artifact was examined as a reviewer would:
`make floats` cannot work on a clean checkout, because Fig2, Fig4, Fig5, Fig6, Tab5 and Tab6 all
read files that git does not carry. An artifact whose own build fails on clone is the first thing
an evaluator finds.

The fix is cheap because the files that matter are small once compressed -- the 19 MB decision
log gzips to 0.3 MB, and the whole set lands under a megabyte. This packs exactly the artifacts
the float scripts read into `results/_packed/`, which IS tracked, and restores them on demand.

  python scripts/pack_artifacts.py            # pack, after a full run
  python scripts/pack_artifacts.py --unpack   # restore, after a clone
  python scripts/pack_artifacts.py --check    # are the packed copies current?

The packed copies are a convenience for reading, never a source of truth. `make paper`
regenerates everything from the datasets; this only means a reader who has not downloaded 7 GB
of network captures can still rebuild every float in the paper and check it against the text.
"""
from __future__ import annotations

import argparse
import gzip
import hashlib
import json
import logging
import shutil
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
PACKED = REPO / "results" / "_packed"
log = logging.getLogger("pack")

# Exactly what the float scripts and the claims registry read. Globs are expanded at pack time,
# so a new dataset is picked up without editing this list.
PATTERNS = [
    "results/matrix/raw.json",
    "results/c3_artifacts_*/raw/*.csv",
    "results/decision_utility/raw/*.json",
    "results/decision_utility/summary/*.json",
    "results/generality_mlp_ig_*/raw/*.csv",
    "results/transferability/raw/*.csv",
    "results/transferability/summary/*.json",
    "results/smoothed_defense/raw/*.csv",
    "results/smoothed_defense/summary/*.json",
    "results/_cache/decision_cases.json",
]


def targets() -> list[Path]:
    out: list[Path] = []
    for pat in PATTERNS:
        out += sorted(REPO.glob(pat))
    return [p for p in out if p.is_file() and PACKED not in p.parents]


def digest(p: Path) -> str:
    h = hashlib.sha256()
    with open(p, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def pack() -> int:
    PACKED.mkdir(parents=True, exist_ok=True)
    index, total_in, total_out = {}, 0, 0
    for src in targets():
        rel = src.relative_to(REPO)
        dst = PACKED / (str(rel).replace("/", "__") + ".gz")
        with open(src, "rb") as fh, gzip.open(dst, "wb", compresslevel=9) as gz:
            shutil.copyfileobj(fh, gz)
        index[str(rel)] = {"packed": dst.name, "sha256": digest(src),
                           "bytes": src.stat().st_size, "packed_bytes": dst.stat().st_size}
        total_in += src.stat().st_size
        total_out += dst.stat().st_size
    (PACKED / "index.json").write_text(json.dumps(index, indent=1, sort_keys=True))
    log.info("packed %d artifacts: %.1f MB -> %.1f MB",
             len(index), total_in / 1e6, total_out / 1e6)
    return 0


def unpack(force: bool) -> int:
    idx_file = PACKED / "index.json"
    if not idx_file.exists():
        log.error("nothing packed at %s", PACKED)
        return 1
    index = json.loads(idx_file.read_text())
    written, skipped = 0, 0
    for rel, meta in sorted(index.items()):
        dst = REPO / rel
        if dst.exists() and not force:
            # Never clobber a fresh run with an older packed copy without being asked.
            if digest(dst) != meta["sha256"]:
                log.warning("%s differs from the packed copy -- leaving it alone (--force to "
                            "overwrite)", rel)
            skipped += 1
            continue
        dst.parent.mkdir(parents=True, exist_ok=True)
        with gzip.open(PACKED / meta["packed"], "rb") as gz, open(dst, "wb") as fh:
            shutil.copyfileobj(gz, fh)
        written += 1
    log.info("restored %d artifacts (%d already present)", written, skipped)
    return 0


def check() -> int:
    idx_file = PACKED / "index.json"
    if not idx_file.exists():
        log.error("nothing packed at %s -- run without --check after a full run", PACKED)
        return 1
    index = json.loads(idx_file.read_text())
    stale, missing = [], []
    for rel, meta in sorted(index.items()):
        p = REPO / rel
        if not p.exists():
            missing.append(rel)
        elif digest(p) != meta["sha256"]:
            stale.append(rel)
    live = {str(p.relative_to(REPO)) for p in targets()}
    unpacked = sorted(live - set(index))
    for rel in stale:
        log.error("STALE  %s has changed since it was packed", rel)
    for rel in unpacked:
        log.error("UNPACKED  %s exists on disk but is not in the packed index", rel)
    for rel in missing:
        log.info("absent  %s is packed but not on disk (fine on a clean clone)", rel)
    if stale or unpacked:
        log.error("run `python scripts/pack_artifacts.py` to refresh")
        return 1
    log.info("packed copies are current (%d artifacts)", len(index))
    return 0


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--unpack", action="store_true")
    ap.add_argument("--check", action="store_true")
    ap.add_argument("--force", action="store_true", help="overwrite files that differ")
    a = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(message)s")
    if a.unpack:
        return unpack(a.force)
    if a.check:
        return check()
    return pack()


if __name__ == "__main__":
    raise SystemExit(main())
