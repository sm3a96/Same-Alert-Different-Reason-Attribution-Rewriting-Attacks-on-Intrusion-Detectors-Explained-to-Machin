"""Dataset acquisition via kagglehub (Plan Phase 1, P1-1).

The new Kaggle API tokens (KGAT_...) are used by `kagglehub`, not the legacy `kaggle`
CLI. Setup once:
  export KAGGLE_API_TOKEN=KGAT_xxx        (or source ~/.config/kaggle/token.sh)

Then:
  python scripts/download_data.py --dataset ciciov2024
  python scripts/download_data.py --dataset all

Downloads cache inside the repo (data/.kagglehub) to keep everything in one folder, and a
symlink data/raw/<name> -> <version dir> gives the loader a stable path. Provenance
(slug, files, sizes) is recorded in data/PROVENANCE.md.
"""
from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime, timezone
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
RAW = REPO / "data" / "raw"
PROVENANCE = REPO / "data" / "PROVENANCE.md"
os.environ.setdefault("KAGGLEHUB_CACHE", str(REPO / "data" / ".kagglehub"))

# name -> confirmed public Kaggle slug.
# NOTE: ciciov2024 (tusharchauhan1898/ciciov2024) EXCLUDED — 99.75% duplicate per-frame CAN
# rows (see audit_data.py). Replaced by ciciomt2024 (medical IoMT, flow-based, clean).
SLUGS = {
    "ciciot2023": "akashdogra/ciciot23csv",              # IoT, combined CSV (label column)
    "ciciomt2024": "limamateus/cic-iomt-2024-wifi-mqtt", # medical IoMT (label col + split-suffix)
    "fiveg_nidd": "humera11/5g-nidd-dataset",            # 5G
}


def download(name: str, slug: str | None = None) -> Path:
    import kagglehub

    slug = slug or SLUGS.get(name)
    if slug is None:
        sys.exit(f"unknown dataset '{name}'; pass --slug")
    if "KAGGLE_API_TOKEN" not in os.environ and "KAGGLE_KEY" not in os.environ:
        sys.exit("Kaggle not authenticated. `source ~/.config/kaggle/token.sh` first.")

    print(f"[{name}] downloading {slug} ...")
    path = Path(kagglehub.dataset_download(slug))
    RAW.mkdir(parents=True, exist_ok=True)
    link = RAW / name
    if link.is_symlink() or link.exists():
        link.unlink() if link.is_symlink() else None
    if not link.exists():
        link.symlink_to(path, target_is_directory=True)
    _record_provenance(name, slug, path)
    print(f"[{name}] ready at {link} -> {path}")
    return link


def _record_provenance(name: str, slug: str, path: Path) -> None:
    files = sorted(p for p in path.rglob("*") if p.is_file())
    total_mb = sum(p.stat().st_size for p in files) / 1e6
    PROVENANCE.parent.mkdir(parents=True, exist_ok=True)
    header = "# Dataset provenance\n\n" if not PROVENANCE.exists() else ""
    with open(PROVENANCE, "a") as f:
        f.write(header)
        f.write(f"## {name}\n")
        f.write(f"- kaggle slug: `{slug}`\n")
        f.write(f"- downloaded: {datetime.now(timezone.utc).isoformat()}\n")
        f.write(f"- size: {total_mb:.1f} MB across {len(files)} files\n")
        f.write(f"- cache: `{path.relative_to(REPO) if path.is_relative_to(REPO) else path}`\n")
        f.write("- license: record exact terms from the dataset's Kaggle page.\n\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True, choices=list(SLUGS) + ["all"])
    ap.add_argument("--slug", default=None)
    args = ap.parse_args()
    targets = list(SLUGS) if args.dataset == "all" else [args.dataset]
    for name in targets:
        download(name, args.slug if args.dataset != "all" else None)


if __name__ == "__main__":
    main()
