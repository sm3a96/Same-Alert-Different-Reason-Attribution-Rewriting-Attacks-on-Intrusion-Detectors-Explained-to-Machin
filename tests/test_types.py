"""Smoke test for the data contracts."""
import numpy as np

from avert.types import Explanation


def test_explanation_ranking():
    e = Explanation("s", np.array([0.1, -0.9, 0.3, 0.05]), ["a", "b", "c", "d"], "m", 1, top_k=2)
    assert e.top_features() == [1, 2]
