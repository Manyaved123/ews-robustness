"""Run all experiments and save results to results/*.csv.

Usage:  python run_experiments.py            (full run, ~5-10 min)
        python run_experiments.py --quick    (fewer replicates, for testing)

Experiments
  E1  sampling interval x observation noise grid      (both models)
  E2  same number of points: sparse-but-long vs dense-but-short
  E3  missing observations: pairwise handling vs linear interpolation
  E4  sensitivity to rolling-window size and detrending bandwidth
  E5  measurement noise that drifts over time (sensor degrading / improving)
"""
import argparse
import os
import time

import numpy as np
import pandas as pd

from ews.models import simulate_may, simulate_fold, truncate_before_transition
from ews.degrade import (subsample, add_observation_noise, add_drifting_noise,
                         drop_random, interpolate_gaps)
from ews.indicators import ews_taus_batch, gaussian_detrend, INDICATORS
from ews.evaluate import auc, detection_rate

OUT = "results"
SUBSAMPLE = [1, 2, 3, 5, 10, 20]          # sampling interval in multiples of dt_obs
NOISE = [0.0, 0.25, 0.5, 1.0, 2.0, 4.0]   # measurement-noise sd / true-fluctuation sd
MISSING = [0.0, 0.1, 0.2, 0.4, 0.6]


def make_data(model, n_reps):
    """Clean forced + null series for one model (pre-transition only)."""
    if model == "may":
        Xf, _ = simulate_may(n_reps=n_reps, forced=True, seed=11)
        Xn, _ = simulate_may(n_reps=n_reps, forced=False, seed=22)
        thr = 4.0
    else:
        Xf, _ = simulate_fold(n_reps=n_reps, forced=True, seed=33)
        Xn, _ = simulate_fold(n_reps=n_reps, forced=False, seed=44)
        thr = -0.1
    Xf, tf = truncate_before_transition(Xf, thr)
    Xn, tn = truncate_before_transition(Xn, thr)
    # reference fluctuation size: sd of detrended null series (pooled), so the
    # SAME absolute sensor noise is added to forced and null runs
    ref = np.nanstd(np.concatenate([gaussian_detrend(x) for x in Xn]))
    print(f"[{model}] forced tipped early: {tf.sum()}/{n_reps}, null tipped: {tn.sum()}, ref sd={ref:.4f}")
    return Xf, Xn, ref


def score(Xf, Xn, **kw):
    tf = ews_taus_batch(Xf, **kw)
    tn = ews_taus_batch(Xn, **kw)
    # composite indicator: variance and AR1 trends must agree
    for t in (tf, tn):
        t["composite"] = t["variance"] + t["ar1"]
    rows = []
    for k in INDICATORS + ["composite"]:
        rows.append(dict(indicator=k, auc=auc(tf[k], tn[k]),
                         det5=detection_rate(tf[k], tn[k], 0.05),
                         tau_forced_med=np.nanmedian(tf[k]),
                         tau_null_med=np.nanmedian(tn[k])))
    return rows, tf, tn


def degrade_pair(Xf, Xn, ref, k=1, noise=0.0, missing=0.0, interp=False, seed=0):
    rng = np.random.default_rng(seed)
    out = []
    for X in (Xf, Xn):
        Y = subsample(X, k)
        Y = add_observation_noise(Y, noise, np.full(Y.shape[0], ref), rng)
        Y = drop_random(Y, missing, rng)
        if interp:
            Y = interpolate_gaps(Y)
        out.append(Y)
    return out


def e1(data):
    rows = []
    taus = {}
    for model, (Xf, Xn, ref) in data.items():
        for k in SUBSAMPLE:
            for nz in NOISE:
                Yf, Yn = degrade_pair(Xf, Xn, ref, k=k, noise=nz, seed=1000 * k + int(nz * 100))
                r, tf, tn = score(Yf, Yn)
                for d in r:
                    rows.append(dict(model=model, subsample=k, n_points=Yf.shape[1], noise=nz, **d))
                if (k, nz) in [(1, 0.0), (1, 1.0), (5, 0.0), (10, 1.0)]:
                    for ind in INDICATORS + ["composite"]:
                        taus[(model, k, nz, ind)] = (tf[ind], tn[ind])
        print(f"  E1 {model} done")
    pd.DataFrame(rows).to_csv(f"{OUT}/e1_interval_noise.csv", index=False)
    # store tau distributions used for ROC / violin figures
    trows = []
    for (model, k, nz, ind), (tf, tn) in taus.items():
        for v in tf:
            trows.append(dict(model=model, subsample=k, noise=nz, indicator=ind, run="forced", tau=v))
        for v in tn:
            trows.append(dict(model=model, subsample=k, noise=nz, indicator=ind, run="null_model", tau=v))
    pd.DataFrame(trows).to_csv(f"{OUT}/e1_tau_samples.csv", index=False)


def e2(data):
    """Fixed budget of n points: spread over the whole record (sparse, long)
    or packed into the final part of the record (dense, short)."""
    rows = []
    Xf, Xn, ref = data["may"]
    N = Xf.shape[1]
    for n in [50, 100, 200, 500]:
        k = N // n
        for strategy in ["sparse_long", "dense_short"]:
            for nz in [0.0, 1.0]:
                if strategy == "sparse_long":
                    Yf, Yn = degrade_pair(Xf, Xn, ref, k=k, noise=nz, seed=n)
                else:
                    Yf, Yn = degrade_pair(Xf[:, -n:], Xn[:, -n:], ref, k=1, noise=nz, seed=n)
                r, _, _ = score(Yf, Yn)
                for d in r:
                    rows.append(dict(model="may", n_points=n, strategy=strategy, noise=nz, **d))
    pd.DataFrame(rows).to_csv(f"{OUT}/e2_budget.csv", index=False)
    print("  E2 done")


def e3(data):
    rows = []
    for model, (Xf, Xn, ref) in data.items():
        for nz in [0.0, 1.0]:
            for p in MISSING:
                for method in ["pairwise", "interpolate"]:
                    Yf, Yn = degrade_pair(Xf, Xn, ref, k=1, noise=nz, missing=p,
                                          interp=(method == "interpolate"), seed=int(p * 100) + 7)
                    r, _, _ = score(Yf, Yn)
                    for d in r:
                        rows.append(dict(model=model, noise=nz, missing=p, method=method, **d))
        print(f"  E3 {model} done")
    pd.DataFrame(rows).to_csv(f"{OUT}/e3_missing.csv", index=False)


def e4(data):
    rows = []
    Xf, Xn, ref = data["may"]
    for cond, (k, nz) in {"clean": (1, 0.0), "degraded": (3, 1.0)}.items():
        Yf, Yn = degrade_pair(Xf, Xn, ref, k=k, noise=nz, seed=5)
        for w in [0.25, 0.5, 0.75]:
            for bw in [0.02, 0.05, 0.1, 0.2]:
                r, _, _ = score(Yf, Yn, win_frac=w, bw_frac=bw)
                for d in r:
                    rows.append(dict(condition=cond, win_frac=w, bw_frac=bw, **d))
    pd.DataFrame(rows).to_csv(f"{OUT}/e4_sensitivity.csv", index=False)
    print("  E4 done")


def e5(data):
    """Non-stationary measurement noise. Same average noise level, but the
    sensor gets worse ('rising') or better ('falling') over the record."""
    rows = []
    scen = {"constant": (1.0, 1.0), "rising": (0.25, 1.75), "falling": (1.75, 0.25)}
    for model, (Xf, Xn, ref) in data.items():
        for k in [1, 2]:
            for name, (a, b) in scen.items():
                rng = np.random.default_rng(99 + k)
                Yf = add_drifting_noise(subsample(Xf, k), a, b, ref, rng)
                Yn = add_drifting_noise(subsample(Xn, k), a, b, ref, rng)
                r, tf, tn = score(Yf, Yn)
                for d in r:
                    ind = d["indicator"]
                    # false-alarm rate: null runs whose tau exceeds the 95th
                    # percentile of null taus under *constant* noise
                    rows.append(dict(model=model, subsample=k, scenario=name, **d,
                                     _tn=tn[ind]))
        print(f"  E5 {model} done")
    df = pd.DataFrame(rows)
    out = []
    for (model, k, ind), g in df.groupby(["model", "subsample", "indicator"], sort=False):
        base = g[g.scenario == "constant"]["_tn"].iloc[0]
        thr = np.nanquantile(base, 0.95)
        for _, row in g.iterrows():
            d = row.drop("_tn").to_dict()
            d["false_alarm"] = float(np.nanmean(row["_tn"] > thr))
            out.append(d)
    pd.DataFrame(out).to_csv(f"{OUT}/e5_drifting_noise.csv", index=False)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--reps", type=int, default=300)
    a = ap.parse_args()
    n_reps = 40 if a.quick else a.reps
    os.makedirs(OUT, exist_ok=True)
    t0 = time.time()
    data = {m: make_data(m, n_reps) for m in ["may", "fold"]}
    e1(data)
    e2(data)
    e3(data)
    e4(data)
    e5(data)
    print(f"all done in {time.time() - t0:.0f}s")
