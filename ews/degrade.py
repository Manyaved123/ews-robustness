"""Ways of making a clean simulated time series look like real field data."""
import numpy as np


def subsample(X, k):
    """Keep every k-th observation (coarser sampling interval: dt_obs * k)."""
    return X[:, ::k]


def add_observation_noise(X, rel_level, ref_sd, rng):
    """Add white Gaussian *measurement* noise with sd = rel_level * ref_sd.

    ref_sd is the standard deviation of the true fluctuations (per replicate),
    so rel_level is a noise-to-signal ratio: 0 = perfect sensor,
    1 = measurement error as large as the real fluctuations.
    """
    if rel_level == 0:
        return X.copy()
    noise = rng.standard_normal(X.shape) * (rel_level * ref_sd)[:, None]
    return X + noise


def add_drifting_noise(X, start_level, end_level, ref_sd, rng):
    """Measurement noise whose sd changes linearly over the record
    (e.g. a sensor that degrades, or one that is upgraded halfway)."""
    n = X.shape[1]
    levels = np.linspace(start_level, end_level, n) * ref_sd
    return X + rng.standard_normal(X.shape) * levels[None, :]


def drop_random(X, frac, rng):
    """Remove a fraction of observations at random (gaps -> NaN)."""
    if frac == 0:
        return X.copy()
    X = X.copy()
    mask = rng.random(X.shape) < frac
    X[mask] = np.nan
    return X


def interpolate_gaps(X):
    """Linear interpolation over interior NaNs (one common, naive fix).
    Trailing NaNs (after a transition) are kept as NaN."""
    X = X.copy()
    t = np.arange(X.shape[1])
    for i in range(X.shape[0]):
        ok = ~np.isnan(X[i])
        if ok.sum() < 2:
            continue
        last = np.where(ok)[0][-1]
        X[i, :last + 1] = np.interp(t[:last + 1], t[ok], X[i, ok])
    return X
