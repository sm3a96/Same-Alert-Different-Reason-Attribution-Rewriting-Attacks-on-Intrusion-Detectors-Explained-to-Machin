"""Phase 2 gate. The certified top-k rank-stability radius must
satisfy basic sanity: a certified radius r means NO protocol-valid perturbation of
norm < r changes the top-k set. Validate against brute-force search on small cases
and against known closed-form values from the hybrid-smoothing certificate.

Marked xfail until Signal 1's certificate replaces the empirical fallback.
"""
import pytest


@pytest.mark.xfail(reason="Phase 2 P2-1: certified radius not yet implemented")
def test_certified_radius_is_sound():
    raise NotImplementedError("brute-force check: no perturbation < r flips top-k")


def test_consensus_nan_branch_is_max_suspicion_and_never_fires_on_our_data():
    """Pin the NaN path in cross-method consensus, and record that it is dead in practice.

    `spearmanr` returns NaN when an input is constant, and the signal maps that to
    correlation 0.0 -- maximal disagreement, so maximal suspicion. That is the
    anti-conservative direction for a false-alarm guarantee, which makes it worth an
    explicit test rather than an implicit branch.

    It also never fires: 0 of 16,200 attribution vectors in the cached decision cases are
    constant. The branch cannot be quietly carrying any result in the paper, and this test
    fails the day that stops being true.
    """
    import json
    from pathlib import Path

    import numpy as np

    from avert.signals.cross_method_consensus import _mean_pairwise_rank_distance

    const = np.ones(8)
    varied = np.arange(8, dtype=float)
    assert _mean_pairwise_rank_distance([const, varied]) == pytest.approx(1.0)

    cache = Path(__file__).resolve().parents[1] / "results/_cache/decision_cases.json"
    if not cache.exists():
        pytest.skip("no cached decision cases on this machine")
    blob = json.loads(cache.read_text())
    cases = blob["cases"] if isinstance(blob, dict) else blob
    constant = sum(
        1
        for rec in cases
        for c in rec["cases"].values()
        if c.get("importances") and np.std(np.abs(np.asarray(c["importances"], float))) == 0
    )
    assert constant == 0, f"{constant} constant attribution vectors now reach the NaN branch"
