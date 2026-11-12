"""Early-warning indicators computed in a rolling window, NaN-aware.

Pipeline for one series (Dakos et al. 2012):
  1. keep pre-transition data only (trailing NaNs are ignored)
  2. detrend with a Gaussian kernel smoother -> residuals
  3. rolling-window variance, lag-1 autocorrelation (AR1), skewness
  4. Kendall's tau of each indicator against time = strength of the trend

Gaps (NaN) inside the series are handled *pairwise*: a lag-k product is used
only if both points exist. Rolling sums are done with cumulative sums, so the
whole thing is O(n) per series.

Extra indicator (our mitigation): 'ar1_corr' = rho(2) / rho(1).
If the true process is AR(1) with coefficient phi and the sensor adds white
noise, then rho(k) = A * phi^k for k >= 1 (A < 1 is the attenuation caused by
the noise), so rho(2)/rho(1) = phi regardless of the noise level.
In other words, AR(1) + white noise = ARMA(1,1), and this ratio is the
method-of-moments estimate of the AR part.
"""
import numpy as np
from scipy.ndimage import gaussian_filter1d
from scipy.stats import kendalltau

INDICATORS = ["variance", "ar1", "skewness", "ar1_corr", "ar1_fit"]
N_LAGS = 4
_PHI_GRID = np.linspace(0.0, 0.995, 200)


def _fit_ar_part(rhos):
    """Multi-lag noise-robust AR estimate ('ar1_fit').

    Fits rho(k) = A * phi^k for k = 1..K by least squares (A free, so white
    measurement noise -- which only shrinks A -- does not bias phi).
    Using several lags instead of the ratio rho2/rho1 makes it far less
    jumpy. Solved by a grid search over phi with A profiled out.
    rhos: (K, nwin) -> phi: (nwin,)
    """
    K = rhos.shape[0]
    P = _PHI_GRID[:, None] ** np.arange(1, K + 1)[None, :]     # (G, K)
    num = P @ np.nan_to_num(rhos)                               # (G, nwin)
    den = (P**2).sum(1)[:, None]                                # (G, 1)
    sse = -(num**2) / den        # SSE up to a constant, A profiled out
    sse = np.where(num > 0, sse, 0.0)   # require A > 0
    phi = _PHI_GRID[np.argmin(sse, axis=0)]
    return np.where(np.isnan(rhos).any(0), np.nan, phi)


def valid_length(x):
    ok = np.where(~np.isnan(x))[0]
    return 0 if ok.size == 0 else ok[-1] + 1


def gaussian_detrend(x, bw_frac=0.05):
    """NaN-aware Gaussian kernel detrending (normalised convolution).
    Bandwidth (kernel sd) is bw_frac * length of the series."""
    m = ~np.isnan(x)
    xf = np.where(m, x, 0.0)
    s = max(bw_frac * len(x), 1.0)
    num = gaussian_filter1d(xf, s, mode="nearest")
    den = gaussian_filter1d(m.astype(float), s, mode="nearest")
    trend = num / np.maximum(den, 1e-12)
    return np.where(m, x - trend, np.nan)


def _roll_sum(a, w):
    """Sum over windows [s, s+w) for every start s; a has no NaN."""
    c = np.concatenate([[0.0], np.cumsum(a)])
    return c[w:] - c[:-w]


def rolling_indicators(r, win_frac=0.5, min_pts=10):
    """Rolling indicators of residual series r (may contain NaN gaps).
    Returns dict name -> array (value at the END of each window)."""
    n = len(r)
    w = max(int(win_frac * n), min_pts)
    if n < w + 2:
        return {k: np.array([]) for k in INDICATORS}
    m = (~np.isnan(r)).astype(float)
    z = np.where(m > 0, r, 0.0)

    cnt = _roll_sum(m, w)
    s1 = _roll_sum(z, w)
    s2 = _roll_sum(z**2, w)
    s3 = _roll_sum(z**3, w)
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = s1 / cnt
        var = s2 / cnt - mean**2
        sd = np.sqrt(np.maximum(var, 0))
        skew = (s3 / cnt - 3 * mean * s2 / cnt + 2 * mean**3) / sd**3

        def rho(k):
            # pairs (t, t+k) with t in [s, s+w-k)
            pm = m[:-k] * m[k:]
            pz = z[:-k] * z[k:]
            pc = _roll_sum(pm, w - k)[: len(cnt)]
            ps = _roll_sum(pz, w - k)[: len(cnt)]
            return (ps / pc - mean**2) / var

        rhos = np.vstack([rho(k) for k in range(1, N_LAGS + 1)])  # (K, nwin)
        r1, r2 = rhos[0], rhos[1]
        corr = r2 / r1
        corr = np.where(np.abs(r1) < 0.05, np.nan, np.clip(corr, -1, 1))
        fit = _fit_ar_part(rhos)

    bad = cnt < min_pts
    out = dict(variance=var, ar1=r1, skewness=skew, ar1_corr=corr, ar1_fit=fit)
    return {k: np.where(bad, np.nan, v) for k, v in out.items()}


def kendall_trend(series):
    """Kendall's tau of an indicator vs time (NaNs dropped)."""
    ok = ~np.isnan(series)
    if ok.sum() < 5:
        return np.nan
    t = np.arange(len(series))[ok]
    return kendalltau(t, series[ok]).statistic


def ews_taus(x, win_frac=0.5, bw_frac=0.05):
    """Full pipeline for one raw series -> dict of Kendall taus."""
    L = valid_length(x)
    x = x[:L]
    r = gaussian_detrend(x, bw_frac)
    ind = rolling_indicators(r, win_frac)
    return {k: kendall_trend(v) for k, v in ind.items()}


def ews_taus_batch(X, **kw):
    """Apply ews_taus to each row of X; returns dict name -> array of taus."""
    res = {k: np.empty(X.shape[0]) for k in INDICATORS}
    for i in range(X.shape[0]):
        t = ews_taus(X[i], **kw)
        for k in INDICATORS:
            res[k][i] = t[k]
    return res
