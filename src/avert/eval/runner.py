"""Experiment runner — the one harness every experiment uses.

Guarantees the saving discipline automatically: seeds all RNGs, times the run,
writes run_metadata.json (seed, git commit, timings, device), and appends the
experiment's output floats to results/MANIFEST.csv. Nothing in the paper should be
produced outside this harness.
"""
from __future__ import annotations

import csv
import fcntl
import json
import os
import platform
import random
import subprocess
import time
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

import numpy as np

REPO_ROOT = Path(__file__).resolve().parents[3]
RESULTS_ROOT = REPO_ROOT / "results"
MANIFEST = RESULTS_ROOT / "MANIFEST.csv"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    try:  # torch is present but optional for the toy path
        import torch

        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
    except Exception:
        pass


def _git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--short", "HEAD"], cwd=REPO_ROOT, stderr=subprocess.DEVNULL
        ).decode().strip()
    except Exception:
        return "nogit"


@dataclass
class RunResult:
    summary: dict
    floats: list[dict] = field(default_factory=list)  # {float_id, kind, path, claim}


def run_experiment(name: str, fn: Callable[["ExperimentContext"], RunResult], config: dict, seed: int = 0) -> RunResult:
    set_seed(seed)
    run_dir = RESULTS_ROOT / name
    for sub in ("raw", "summary", "meta", "logs"):
        (run_dir / sub).mkdir(parents=True, exist_ok=True)
    ctx = ExperimentContext(name=name, config=config, seed=seed, run_dir=run_dir)

    # Stamp the commit BEFORE the experiment runs. Stamped after, a multi-hour run records
    # whatever HEAD had become by the time it finished, which on 2026-09-08 was three commits
    # past the code that actually produced the numbers.
    commit = _git_commit()
    t0 = time.time()
    result = fn(ctx)
    wall = time.time() - t0

    meta = {
        "experiment": name,
        "seed": seed,
        "git_commit": commit,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "wall_clock_s": round(wall, 3),
        "python": platform.python_version(),
        "platform": platform.platform(),
        "config": config,
        "summary": result.summary,
    }
    with open(run_dir / "meta" / f"run_seed{seed}.json", "w") as f:
        json.dump(meta, f, indent=2, default=str)
    _append_manifest(name, result.floats)
    return result


MANIFEST_HEADER = [
    "float_id", "kind", "paper_section", "generating_experiment",
    "path", "claim_supported", "status",
]


CHECK_PREFIXES = ("matrix_chk", "_chk", "scratch_")


def _append_manifest(experiment: str, floats: list[dict]) -> None:
    """Upsert this experiment's floats into MANIFEST.csv.

    Keyed on (float_id, generating_experiment) so a re-run replaces its own rows
    instead of appending a duplicate. Re-running an experiment is normal; a
    manifest with four copies of the same float is not an evidence index.

    The whole read-modify-write holds an exclusive `flock`. Two experiments running at
    once is the normal way to use this box -- the matrix on one GPU, the generality grid
    on the other -- and without the lock the second one to finish reads the manifest
    before the first one writes it, then overwrites the file with rows that never saw the
    first run. The failure leaves no trace: the artifacts are all on disk and the index
    that points at them is silently short a few rows, which is exactly the state the
    pre-writing gate exists to catch and cannot distinguish from a run that never
    happened.
    """
    # Determinism and scratch runs are not evidence. The manifest is what an evaluator reads
    # to find the artifact behind each float, and a throwaway run has no float behind it.
    if experiment.startswith(CHECK_PREFIXES):
        return
    RESULTS_ROOT.mkdir(exist_ok=True)
    with open(MANIFEST.with_suffix(".lock"), "w") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX)
        try:
            rows: dict[tuple[str, str], list[str]] = {}
            if MANIFEST.exists():
                with open(MANIFEST, newline="") as f:
                    reader = csv.reader(f)
                    next(reader, None)  # header
                    for row in reader:
                        if len(row) == len(MANIFEST_HEADER):
                            rows[(row[0], row[3])] = row
            for fl in floats:
                # Repo-relative. An absolute /home/... path is meaningless to anyone who
                # unpacks the release, and the manifest is the evidence index they read first.
                path = str(fl.get("path", ""))
                if path:
                    try:
                        path = str(Path(path).resolve().relative_to(RESULTS_ROOT.parent))
                    except ValueError:
                        pass
                row = [
                    fl.get("float_id", ""), fl.get("kind", ""), fl.get("paper_section", ""),
                    experiment, path, fl.get("claim", ""), fl.get("status", "done"),
                ]
                rows[(row[0], row[3])] = row
            with open(MANIFEST, "w", newline="") as f:
                writer = csv.writer(f)
                writer.writerow(MANIFEST_HEADER)
                writer.writerows(rows.values())
        finally:
            fcntl.flock(lock, fcntl.LOCK_UN)


@dataclass
class ExperimentContext:
    name: str
    config: dict
    seed: int
    run_dir: Path
