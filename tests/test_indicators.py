"""Sanity tests: the estimators recover known values on synthetic AR(1) data.
Run with:  python -m pytest tests   (or just: python tests/test_indicators.py)
"""
import os
import sys

import numpy as np

sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))
from ews.indicators import rolling_indicators, gaussian_detrend
from ews.models import may_equilibrium, may_drift


def ar1_series(phi, n=4000, seed=0):
    rng = np.random.default_rng(seed)
    x = np.zeros(n)
    for t in range(1, n):
        x[t] = phi * x[t - 1] + rng.standard_normal()
    return x


def test_ar1_and_variance_on_clean_ar1():
    x = ar1_series(0.7)
    ind = rolling_indicators(x, 0.5)
    assert abs(np.nanmean(ind["ar1"]) - 0.7) < 0.03
    assert abs(np.nanmean(ind["variance"]) - 1 / (1 - 0.49)) < 0.15


def test_noise_biases_ar1_but_not_robust_fit():
    rng = np.random.default_rng(1)
    x = ar1_series(0.8) + 2.0 * rng.standard_normal(4000)
    ind = rolling_indicators(x, 0.25)
    assert np.nanmean(ind["ar1"]) < 0.45            # strongly attenuated
    assert abs(np.nanmean(ind["ar1_fit"]) - 0.8) < 0.05


def test_gaps_handled_pairwise():
    rng = np.random.default_rng(2)
    x = ar1_series(0.7)
    x[rng.random(len(x)) < 0.4] = np.nan
    ind = rolling_indicators(x, 0.5)
    assert abs(np.nanmean(ind["ar1"]) - 0.7) < 0.05


def test_detrend_removes_trend():
    t = np.arange(1000)
    x = 0.01 * t + ar1_series(0.5, 1000)
    r = gaussian_detrend(x)
    assert abs(np.polyfit(t, r, 1)[0]) < 1e-3


def test_may_equilibrium_is_fixed_point():
    for c in [1.0, 2.0, 2.5]:
        assert abs(may_drift(may_equilibrium(c), c)) < 1e-8


if __name__ == "__main__":
    for name, f in list(globals().items()):
        if name.startswith("test_"):
            f()
            print("passed", name)
