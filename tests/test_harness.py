"""Smoke test for the XInt-Bench product harness: the full evaluate_pipeline path runs
end to end and returns a well-formed vulnerability report. Small sizes keep it fast.
"""
from avert.benchmark import XIntBench
from avert.data.datasets import SyntheticDataset


def test_xintbench_evaluate_pipeline_runs():
    data = SyntheticDataset(n=700, d=12, n_causal=4, seed=0).load()
    bench = XIntBench(data, n_train=400, n_cal=25, n_test=15, top_k=4,
                      perm_samples=5, stability_n=12, seed=0)
    report = bench.evaluate_pipeline()                 # default XGBoost + TreeSHAP

    assert report.n_features == 12
    assert {r.attack for r in report.results} == {"A1_misdirection", "A1_displacement", "A3_scaffolding"}
    for r in report.results:
        assert r.n == 15
        assert 0.0 <= r.prediction_preserved <= 1.0
        assert 0.0 <= r.monitor_auroc <= 1.0
        assert 0.0 <= r.false_alarm <= 1.0
        assert "FUSED" in r.per_signal_auroc
    assert isinstance(report.summary(), str)
    assert "attack" in report.summary()
