"""The central correctness gate (Plan Section 3.6): the conformal fuser's empirical
false-alarm rate on exchangeable clean data must not exceed the nominal alpha by
more than sampling noise. If this ever fails, the paper's headline guarantee is
broken. Keep it green.
"""
import numpy as np

from avert.fusion.conformal import ConformalFuser


def test_split_conformal_false_alarm_bounded():
    rng = np.random.default_rng(0)
    alpha = 0.1
    false_alarms = []
    for _ in range(300):
        calib = rng.normal(0, 1, size=500)     # clean calibration scores
        test = rng.normal(0, 1, size=1)[0]     # exchangeable clean test score
        fuser = ConformalFuser()
        fuser.calibrate(calib)
        false_alarms.append(fuser.is_violation(test, alpha))
    rate = float(np.mean(false_alarms))
    # Distribution-free guarantee: E[false alarm] <= alpha. Allow Monte-Carlo slack.
    assert rate <= alpha + 0.03, f"false-alarm rate {rate} exceeds alpha={alpha}"


def test_pvalue_in_unit_interval():
    rng = np.random.default_rng(1)
    fuser = ConformalFuser()
    fuser.calibrate(rng.normal(size=200))
    for s in rng.normal(size=50):
        p = fuser.p_value(float(s))
        assert 0.0 < p <= 1.0
