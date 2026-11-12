"""Turning Kendall-tau distributions into detection performance."""
import numpy as np
from scipy.stats import mannwhitneyu


def auc(tau_forced, tau_null):
    """ROC area under curve for classifying 'approaching a transition' vs
    'null' using Kendall tau as the score. 1 = perfect, 0.5 = chance.
    Equals the Mann-Whitney U statistic / (n1*n2)."""
    a = tau_forced[~np.isnan(tau_forced)]
    b = tau_null[~np.isnan(tau_null)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    u = mannwhitneyu(a, b, alternative="two-sided").statistic
    return u / (len(a) * len(b))


def detection_rate(tau_forced, tau_null, fpr=0.05):
    """Fraction of forced runs flagged when the alarm threshold is set so that
    only `fpr` of null runs raise a (false) alarm."""
    a = tau_forced[~np.isnan(tau_forced)]
    b = tau_null[~np.isnan(tau_null)]
    if len(a) < 5 or len(b) < 5:
        return np.nan
    thr = np.quantile(b, 1 - fpr)
    return float(np.mean(a > thr))


def roc_curve(tau_forced, tau_null):
    a = tau_forced[~np.isnan(tau_forced)]
    b = tau_null[~np.isnan(tau_null)]
    thr = np.sort(np.concatenate([a, b, [np.inf, -np.inf]]))[::-1]
    tpr = np.array([(a > t).mean() for t in thr])
    fpr = np.array([(b > t).mean() for t in thr])
    return fpr, tpr


def bootstrap_auc_ci(tau_forced, tau_null, n_boot=500, seed=0):
    rng = np.random.default_rng(seed)
    a = tau_forced[~np.isnan(tau_forced)]
    b = tau_null[~np.isnan(tau_null)]
    vals = [auc(rng.choice(a, len(a)), rng.choice(b, len(b))) for _ in range(n_boot)]
    return np.nanpercentile(vals, [2.5, 97.5])
