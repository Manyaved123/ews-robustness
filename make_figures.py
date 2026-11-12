"""Make every figure in figures/ from results/*.csv (+ a few fresh simulations).

Usage:  python make_figures.py
"""
import os

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap

from ews.models import simulate_may, may_equilibrium, may_drift
from ews.degrade import subsample, add_observation_noise
from ews.indicators import gaussian_detrend, rolling_indicators
from ews.evaluate import roc_curve

R, F = "results", "figures"
os.makedirs(F, exist_ok=True)

# palette (validated reference palette, light mode)
BLUE, ORANGE, AQUA, YELLOW, MAGENTA, VIOLET = ("#2a78d6", "#eb6834", "#1baf7a",
                                               "#eda100", "#e87ba4", "#6250d6")
INK, INK2, GRID, SURF = "#0b0b0b", "#52514e", "#e4e3df", "#fcfcfb"
SEQ = LinearSegmentedColormap.from_list("blue", ["#f4f8fd", "#cde2fb", "#86b6ef",
                                                 "#3987e5", "#1c5cab", "#0d366b"])
IND_COLOR = {"variance": BLUE, "ar1": ORANGE, "composite": AQUA,
             "ar1_fit": VIOLET, "ar1_corr": MAGENTA, "skewness": YELLOW}
IND_LABEL = {"variance": "Variance", "ar1": "Lag-1 autocorrelation (AR1)",
             "composite": "Composite (var + AR1)", "ar1_fit": "Noise-robust AR (multi-lag fit)",
             "ar1_corr": "Noise-robust AR (ρ2/ρ1)", "skewness": "Skewness"}

plt.rcParams.update({
    "figure.facecolor": SURF, "axes.facecolor": SURF, "savefig.facecolor": SURF,
    "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
    "ytick.color": INK2, "text.color": INK, "axes.spines.top": False,
    "axes.spines.right": False, "axes.grid": True, "grid.color": GRID,
    "grid.linewidth": 0.8, "font.size": 10, "axes.titlesize": 11,
    "axes.titleweight": "bold", "lines.linewidth": 2, "legend.frameon": False,
})


def save(fig, name):
    fig.savefig(f"{F}/{name}.png", dpi=180, bbox_inches="tight")
    plt.close(fig)
    print("saved", name)


# ---------------------------------------------------------------------------
def fig1_example():
    Xf, c = simulate_may(n_reps=1, seed=3)
    Xn, _ = simulate_may(n_reps=1, forced=False, seed=4)
    rng = np.random.default_rng(0)
    ref = np.nanstd(gaussian_detrend(Xn[0]))
    t = np.arange(Xf.shape[1])
    Yf = add_observation_noise(subsample(Xf, 5), 1.0, np.array([ref]), rng)
    t5 = t[::5]

    fig, ax = plt.subplots(3, 2, figsize=(10, 7.5), sharex=True)
    ax[0, 0].plot(t, Xf[0], color=BLUE, lw=1)
    ax[0, 0].plot(t, [may_equilibrium(ci) for ci in c], color=INK2, lw=1, ls="--")
    ax[0, 0].set_title("A. Clean data: dense (Δt = 1), no sensor noise")
    ax[0, 0].set_ylabel("State x (biomass)")
    ax[0, 1].plot(t5, Yf[0], color=BLUE, lw=1, marker="o", ms=2.5)
    ax[0, 1].set_title("B. Degraded: every 5th point + sensor noise (ratio 1)")
    for j, (tt, x) in enumerate([(t, Xf[0]), (t5, Yf[0])]):
        r = gaussian_detrend(x)
        ind = rolling_indicators(r, 0.5)
        w = len(x) - len(ind["variance"])
        te = tt[w:]
        ax[1, j].plot(te, ind["variance"], color=BLUE)
        ax[2, j].plot(te, ind["ar1"], color=ORANGE)
        ax[1, j].set_ylabel("Rolling variance" if j == 0 else "")
        ax[2, j].set_ylabel("Rolling AR1" if j == 0 else "")
        ax[2, j].set_xlabel("Time")
        for a in (ax[1, j], ax[2, j]):
            a.axvspan(tt[0], tt[w - 1], color=GRID, alpha=0.5, lw=0)
    ax[0, 0].text(30, may_equilibrium(1.0) + 0.35, "equilibrium (moves towards fold)",
                  color=INK2, fontsize=8)
    fig.suptitle("Approaching a fold bifurcation in the May harvesting model "
                 "(shaded = first rolling window)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig1_example_series")


# ---------------------------------------------------------------------------
def heat(ax, M, xt, yt, title, vmin=0.5, vmax=1.0):
    im = ax.imshow(M, cmap=SEQ, vmin=vmin, vmax=vmax, aspect="auto", origin="upper")
    ax.set_xticks(range(len(xt)), [str(x) for x in xt])
    ax.set_yticks(range(len(yt)), [str(y) for y in yt])
    ax.grid(False)
    for s in ax.spines.values():
        s.set_visible(False)
    for i in range(M.shape[0]):
        for j in range(M.shape[1]):
            v = M[i, j]
            ax.text(j, i, f"{v:.2f}", ha="center", va="center", fontsize=8,
                    color="white" if v > 0.82 else INK)
    ax.set_title(title)
    return im


def fig2_heatmaps():
    d = pd.read_csv(f"{R}/e1_interval_noise.csv")
    inds = ["variance", "ar1", "composite"]
    fig, axes = plt.subplots(2, 3, figsize=(12.5, 8))
    for i, model in enumerate(["may", "fold"]):
        for j, ind in enumerate(inds):
            p = d[(d.model == model) & (d.indicator == ind)].pivot(
                index="subsample", columns="noise", values="auc")
            im = heat(axes[i, j], p.values, p.columns, p.index,
                      f"{'May model' if model == 'may' else 'Fold normal form'}\n{IND_LABEL[ind]}")
            if j == 0:
                axes[i, j].set_ylabel("Sampling interval\n(× base Δt)")
            if i == 1:
                axes[i, j].set_xlabel("Measurement-noise / signal ratio")
    cb = fig.colorbar(im, ax=axes, shrink=0.6, pad=0.02)
    cb.set_label("ROC AUC (1 = perfect, 0.5 = chance)")
    fig.suptitle("Where early warnings break down: detection AUC across sampling interval × sensor noise",
                 fontsize=12, x=0.45)
    save(fig, "fig2_auc_heatmaps")


def fig3_detection_rate():
    d = pd.read_csv(f"{R}/e1_interval_noise.csv")
    fig, axes = plt.subplots(1, 2, figsize=(11, 4), sharey=True)
    for a, model in zip(axes, ["may", "fold"]):
        for ind in ["variance", "ar1", "composite", "ar1_fit"]:
            g = d[(d.model == model) & (d.indicator == ind) & (d.noise == 0.5)]
            a.plot(g.subsample, g.det5, marker="o", ms=6, color=IND_COLOR[ind],
                   label=IND_LABEL[ind])
        a.set_xscale("log")
        a.set_xticks([1, 2, 3, 5, 10, 20], ["1", "2", "3", "5", "10", "20"])
        a.axhline(0.05, color=INK2, ls=":", lw=1)
        a.text(1.02, 0.07, "chance (5%)", color=INK2, fontsize=8)
        a.set_xlabel("Sampling interval (× base Δt)")
        a.set_title("May model" if model == "may" else "Fold normal form")
    axes[0].set_ylabel("Detection rate at 5% false alarms")
    axes[1].legend(loc="upper right", fontsize=8)
    fig.suptitle("AR1 collapses with sparser sampling; variance degrades slowly "
                 "(noise ratio 0.5)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig3_detection_vs_interval")


def fig4_roc():
    s = pd.read_csv(f"{R}/e1_tau_samples.csv")
    conds = [(1, 0.0, "Δt=1, no noise"), (1, 1.0, "Δt=1, noise 1"),
             (5, 0.0, "Δt=5, no noise"), (10, 1.0, "Δt=10, noise 1")]
    cols = [BLUE, ORANGE, AQUA, YELLOW]
    fig, axes = plt.subplots(1, 2, figsize=(10, 4.4))
    for a, ind in zip(axes, ["variance", "ar1"]):
        for (k, nz, lab), col in zip(conds, cols):
            g = s[(s.model == "may") & (s.indicator == ind) & (s.subsample == k) & (s.noise == nz)]
            fpr, tpr = roc_curve(g[g.run == "forced"].tau.values, g[g.run == "null_model"].tau.values)
            a.plot(fpr, tpr, color=col, label=lab)
        a.plot([0, 1], [0, 1], color=INK2, ls=":", lw=1)
        a.set_title(IND_LABEL[ind])
        a.set_xlabel("False-alarm rate")
        a.set_aspect("equal")
    axes[0].set_ylabel("Detection rate")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("ROC curves, May model (Kendall τ as alarm score)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig4_roc_curves")


def fig5_budget():
    d = pd.read_csv(f"{R}/e2_budget.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, ind in zip(axes, ["variance", "ar1"]):
        for strat, col, lab in [("sparse_long", BLUE, "Spread over whole record (sparse, long)"),
                                ("dense_short", ORANGE, "Only the most recent points (dense, short)")]:
            g = d[(d.indicator == ind) & (d.strategy == strat) & (d.noise == 0.0)]
            a.plot(g.n_points, g.auc, marker="o", ms=6, color=col, label=lab)
        a.set_xscale("log")
        a.set_xticks([50, 100, 200, 500], ["50", "100", "200", "500"])
        a.axhline(0.5, color=INK2, ls=":", lw=1)
        a.set_xlabel("Number of observations available")
        a.set_title(IND_LABEL[ind])
    axes[0].set_ylabel("ROC AUC")
    axes[0].legend(loc="lower right", fontsize=8)
    fig.suptitle("Same sampling budget: a long, sparse record beats a short, dense one",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig5_sampling_budget")


def fig6_missing():
    d = pd.read_csv(f"{R}/e3_missing.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 4), sharey=True)
    for a, model in zip(axes, ["may", "fold"]):
        for ind in ["variance", "ar1"]:
            for meth, ls in [("pairwise", "-"), ("interpolate", "--")]:
                g = d[(d.model == model) & (d.indicator == ind) & (d.method == meth) & (d.noise == 1.0)]
                a.plot(g.missing * 100, g.auc, ls=ls, marker="o", ms=6,
                       color=IND_COLOR[ind], label=f"{IND_LABEL[ind]} - {meth}")
        a.axhline(0.5, color=INK2, ls=":", lw=1)
        a.set_xlabel("Observations missing at random (%)")
        a.set_title("May model" if model == "may" else "Fold normal form")
    axes[0].set_ylabel("ROC AUC (noise ratio 1)")
    axes[0].legend(loc="lower left", fontsize=8)
    fig.suptitle("Random gaps: variance is robust; AR1 loses power as gaps grow", fontsize=11)
    fig.tight_layout()
    save(fig, "fig6_missing_data")


def fig7_drift():
    d = pd.read_csv(f"{R}/e5_drifting_noise.csv")
    inds = ["variance", "ar1", "composite", "ar1_fit"]
    scen = ["constant", "rising", "falling"]
    fig, axes = plt.subplots(1, 2, figsize=(11, 4.2), sharey=True)
    for a, model in zip(axes, ["may", "fold"]):
        g = d[(d.model == model) & (d.subsample == 1)]
        w = 0.2
        for j, ind in enumerate(inds):
            vals = [g[(g.indicator == ind) & (g.scenario == s)].false_alarm.iloc[0] for s in scen]
            xs = np.arange(3) + (j - 1.5) * (w + 0.02)
            a.bar(xs, vals, width=w, color=IND_COLOR[ind], label=IND_LABEL[ind])
            for x, v in zip(xs, vals):
                if v > 0.12:
                    a.text(x, v + 0.015, f"{v:.0%}", ha="center", fontsize=8, color=INK)
        a.axhline(0.05, color=INK2, ls=":", lw=1)
        a.set_xticks(range(3), ["Constant noise", "Sensor gets worse", "Sensor gets better"])
        a.set_title("May model" if model == "may" else "Fold normal form")
        a.grid(axis="x", visible=False)
    axes[0].set_ylabel("False-alarm rate on null runs\n(threshold set for 5% under constant noise)")
    axes[0].legend(loc="upper left", fontsize=8)
    fig.suptitle("Drifting measurement noise fakes warnings; the noise-robust AR estimate does not",
                 fontsize=11)
    fig.tight_layout()
    save(fig, "fig7_drifting_noise_false_alarms")


def fig8_level():
    """Bias of AR1 under noise vs the multi-lag estimate, against theory."""
    n_reps = 100
    Xf, c = simulate_may(n_reps=n_reps, seed=8)
    Xn, _ = simulate_may(n_reps=n_reps, forced=False, seed=9)
    ref = np.nanstd(np.concatenate([gaussian_detrend(x) for x in Xn]))
    rng = np.random.default_rng(1)
    Y = add_observation_noise(Xf, 1.0, np.full(n_reps, ref), rng)
    A1, Af, A0 = [], [], []
    for i in range(n_reps):
        ind = rolling_indicators(gaussian_detrend(Y[i]), 0.25)
        ind0 = rolling_indicators(gaussian_detrend(Xf[i]), 0.25)
        A1.append(ind["ar1"]); Af.append(ind["ar1_fit"]); A0.append(ind0["ar1"])
    A1, Af, A0 = map(np.array, (A1, Af, A0))
    w = Xf.shape[1] - A1.shape[1]
    t = np.arange(Xf.shape[1])[w:]
    # theory: AR1 = exp(-lambda * dt) at the window midpoint parameter value
    lam = []
    for ci in c:
        xe = may_equilibrium(ci); h = 1e-5
        lam.append(-(may_drift(xe + h, ci) - may_drift(xe - h, ci)) / (2 * h))
    theo = np.exp(-np.array(lam))
    mid = t - w // 2
    fig, ax = plt.subplots(figsize=(8, 4.2))
    ax.plot(t, theo[mid], color=INK2, ls="--", lw=1.5, label="Theory  exp(−λ·Δt)")
    for arr, col, lab in [(A0, BLUE, "AR1, clean data"), (A1, ORANGE, "AR1, noisy data (ratio 1)"),
                          (Af, VIOLET, "Noise-robust AR, noisy data")]:
        med = np.nanmedian(arr, 0)
        lo, hi = np.nanpercentile(arr, [25, 75], axis=0)
        ax.fill_between(t, lo, hi, color=col, alpha=0.15, lw=0)
        ax.plot(t, med, color=col, label=lab)
    ax.set_xlabel("Time (end of rolling window)")
    ax.set_ylabel("Estimated autocorrelation")
    ax.set_ylim(0, 1)
    ax.legend(loc="upper left", fontsize=8)
    ax.set_title("Noise hides the true AR level, but the multi-lag fit recovers it (median, IQR)")
    fig.tight_layout()
    save(fig, "fig8_ar_level_bias")


def fig9_sensitivity():
    d = pd.read_csv(f"{R}/e4_sensitivity.csv")
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.6))
    for a, ind in zip(axes, ["variance", "ar1"]):
        p = d[(d.condition == "degraded") & (d.indicator == ind)].pivot(
            index="win_frac", columns="bw_frac", values="auc")
        heat(a, p.values, p.columns, [f"{int(x*100)}%" for x in p.index],
             IND_LABEL[ind], vmin=0.6, vmax=1.0)
        a.set_xlabel("Detrending bandwidth (fraction of record)")
    axes[0].set_ylabel("Rolling window size")
    fig.suptitle("Sensitivity to analysis choices (May model, Δt = 3, noise ratio 1)", fontsize=11)
    fig.tight_layout()
    save(fig, "fig9_sensitivity")


if __name__ == "__main__":
    fig1_example()
    fig2_heatmaps()
    fig3_detection_rate()
    fig4_roc()
    fig5_budget()
    fig6_missing()
    fig7_drift()
    fig8_level()
    fig9_sensitivity()
