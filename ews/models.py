"""Stochastic models that approach a critical transition (fold bifurcation),
plus matching null models that never approach one.

All simulators are vectorised over replicates and integrated with the
Euler-Maruyama scheme. Output is sampled every `dt_obs` time units, which is
the "perfect" observation resolution that degrade.py later makes worse.
"""
import numpy as np


def _em_integrate(drift, x0, param_path, sigma, dt, dt_obs, n_reps, rng):
    """Generic Euler-Maruyama integrator.

    drift(x, p) -> dx/dt ; param_path: array of bifurcation-parameter values,
    one per *observation* time. Returns array (n_reps, n_obs).
    """
    n_obs = len(param_path)
    sub = int(round(dt_obs / dt))
    x = np.full(n_reps, x0, dtype=float)
    out = np.empty((n_reps, n_obs))
    sq = sigma * np.sqrt(dt)
    for i in range(n_obs):
        p = param_path[i]
        for _ in range(sub):
            x = x + drift(x, p) * dt + sq * rng.standard_normal(n_reps)
        out[:, i] = x
    return out


# ----------------------------------------------------------------------------
# Model 1: May (1977) harvesting / overgrazing model
#   dx = [ r x (1 - x/K) - c x^2 / (x^2 + h^2) ] dt + sigma dW
# With r=1, K=10, h=1 the upper (vegetated) state disappears in a fold
# at c ~= 2.604. Used throughout the early-warning literature
# (e.g. Dakos et al. 2012, PLoS ONE).
# ----------------------------------------------------------------------------
MAY_PARAMS = dict(r=1.0, K=10.0, h=1.0)
MAY_C_CRIT = 2.604


def may_drift(x, c, r=1.0, K=10.0, h=1.0):
    return r * x * (1 - x / K) - c * x**2 / (x**2 + h**2)


def may_equilibrium(c, r=1.0, K=10.0, h=1.0):
    """Upper stable equilibrium of the May model (largest real root)."""
    # r x (1-x/K)(x^2+h^2) - c x^2 = 0, divide by x:
    # -r/K x^3 + r x^2 - (r h^2/K + c) x + r h^2 = 0
    roots = np.roots([-r / K, r, -(r * h**2 / K + c), r * h**2])
    real = roots[np.abs(roots.imag) < 1e-9].real
    return real.max()


def simulate_may(n_reps=200, T=1000.0, dt=0.01, dt_obs=1.0, sigma=0.15,
                 c_start=1.0, c_end=2.55, forced=True, seed=0):
    """Simulate the May model. forced=True ramps c linearly towards the fold;
    forced=False keeps c fixed at c_start (null model, no transition)."""
    rng = np.random.default_rng(seed)
    n_obs = int(T / dt_obs)
    c_path = np.linspace(c_start, c_end, n_obs) if forced else np.full(n_obs, c_start)
    x0 = may_equilibrium(c_start)
    X = _em_integrate(lambda x, c: may_drift(x, c), x0, c_path, sigma, dt,
                      dt_obs, n_reps, rng)
    return X, c_path


# ----------------------------------------------------------------------------
# Model 2: saddle-node (fold) normal form
#   dx = (mu - x^2) dt + sigma dW,   stable state x* = sqrt(mu),
#   recovery rate lambda = 2 sqrt(mu) -> 0 as mu -> 0.
# ----------------------------------------------------------------------------
def fold_drift(x, mu):
    return mu - x**2


def simulate_fold(n_reps=200, T=1000.0, dt=0.01, dt_obs=1.0, sigma=0.05,
                  mu_start=1.0, mu_end=0.05, forced=True, seed=0):
    rng = np.random.default_rng(seed)
    n_obs = int(T / dt_obs)
    mu_path = np.linspace(mu_start, mu_end, n_obs) if forced else np.full(n_obs, mu_start)
    X = _em_integrate(fold_drift, np.sqrt(mu_start), mu_path, sigma, dt,
                      dt_obs, n_reps, rng)
    return X, mu_path


def truncate_before_transition(X, threshold):
    """Replace everything from the first crossing below `threshold` with NaN,
    so indicators are only computed on pre-transition data (standard practice).
    Returns the masked array and a boolean vector: did the replicate tip?"""
    X = X.copy()
    tipped = np.zeros(X.shape[0], dtype=bool)
    for i in range(X.shape[0]):
        below = np.where(X[i] < threshold)[0]
        if below.size:
            X[i, below[0]:] = np.nan
            tipped[i] = True
    return X, tipped
