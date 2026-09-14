"""MANIFEST.csv survives two experiments finishing at once.

The manifest is the evidence index: for every float in the paper it names the experiment
that produced it and the artifact on disk. `_append_manifest` rewrites the whole file on
every run, so two experiments running concurrently -- the normal way to use a two-GPU box --
race on a read-modify-write. The loser's rows vanish while its artifacts stay on disk, and
nothing downstream can tell that state apart from an experiment that was never run.

This forks several writers at once and asserts every row is still there. Falsified on
2026-08-15 by disabling `flock` and running the same fixture 20 times: 19 runs lost rows,
up to 35 of 40, and one run happened to lose none. That last one is why the writer count
is 8 rather than 2 -- a two-writer version of this test passes often enough with the lock
removed to be worthless.
"""
from __future__ import annotations

import csv
import multiprocessing as mp

import pytest

from avert.eval import runner

N_WRITERS = 8
N_ROWS_EACH = 5


def _write(experiment: str, manifest_path) -> None:
    runner.MANIFEST = manifest_path
    runner._append_manifest(experiment, [
        {"float_id": f"{experiment}_f{i}", "kind": "table", "paper_section": "V",
         "path": f"results/{experiment}/raw/f{i}.csv", "claim": "concurrency fixture"}
        for i in range(N_ROWS_EACH)
    ])


@pytest.mark.parametrize("start_method", ["fork"])
def test_concurrent_upserts_lose_no_rows(tmp_path, start_method):
    manifest = tmp_path / "MANIFEST.csv"
    runner.MANIFEST = manifest
    runner.RESULTS_ROOT = tmp_path

    ctx = mp.get_context(start_method)
    procs = [ctx.Process(target=_write, args=(f"exp{w}", manifest)) for w in range(N_WRITERS)]
    for p in procs:
        p.start()
    for p in procs:
        p.join(timeout=60)
        assert p.exitcode == 0, "a writer failed outright, which is a different bug"

    with open(manifest, newline="") as f:
        rows = list(csv.reader(f))
    keys = {(r[0], r[3]) for r in rows[1:]}
    expected = {(f"exp{w}_f{i}", f"exp{w}")
                for w in range(N_WRITERS) for i in range(N_ROWS_EACH)}
    missing = expected - keys
    assert not missing, f"{len(missing)} of {len(expected)} rows lost to the race: {sorted(missing)[:5]}"


def test_reruns_replace_their_own_rows_rather_than_duplicating(tmp_path):
    """The upsert key, pinned. A manifest with four copies of a float is not an index."""
    manifest = tmp_path / "MANIFEST.csv"
    runner.MANIFEST = manifest
    runner.RESULTS_ROOT = tmp_path
    for _ in range(3):
        _write("exp_repeat", manifest)
    with open(manifest, newline="") as f:
        rows = list(csv.reader(f))[1:]
    assert len(rows) == N_ROWS_EACH
