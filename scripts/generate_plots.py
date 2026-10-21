"""
Generate ALL project plots into plots/{data_integrity,rq1,rq2,training_diagnostics}/.

Techniques borrowed from the IDVE module (Honours semester 2):
  Lecture 2 (Statistics + Cleaning) — histograms, KDE, boxplots, skewness, kurtosis,
                                      correlation, outlier awareness.
  Lecture 5 (Time Series)           — autocorrelation, stationarity concerns,
                                      rolling-window summaries, resampling.

Run:
    python scripts/generate_plots.py

Writes:
    plots/data_integrity/            data-side sanity plots + stat text reports
    plots/rq1/                       RQ1 (RL vs SL) comparison plots
    plots/rq2/                       RQ2 (AS vs FS) comparison plots
    plots/training_diagnostics/      PPO rollout-stat decays (if training_curves exist)
"""

from __future__ import annotations
import sys
from pathlib import Path
import json
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
PLOTS = ROOT / "plots"

from src.data.loader import load_aligned
from src.data.transforms import to_env_arrays


# ---------- shared style ----------
plt.rcParams.update({
    "figure.figsize": (10, 5),
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


# ---------- helpers ----------

def save(fig, path):
    fig.tight_layout()
    fig.savefig(path, dpi=140, bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote {path.relative_to(ROOT)}")


def autocorr(x: np.ndarray, max_lag: int) -> np.ndarray:
    """Autocorrelation via numpy — Lecture 5: corr(series, lagged copy of itself)."""
    x = x - x.mean()
    denom = (x ** 2).sum()
    out = []
    for lag in range(max_lag + 1):
        if lag == 0:
            out.append(1.0)
        else:
            out.append(float((x[:-lag] * x[lag:]).sum() / denom))
    return np.array(out)


def rolling_adf_proxy(x: np.ndarray, window: int) -> tuple[np.ndarray, np.ndarray]:
    """Rolling mean + std — Lecture 5's visual stationarity check.
    If mean and std drift over time, the series is non-stationary.
    """
    s = pd.Series(x)
    return s.rolling(window).mean().to_numpy(), s.rolling(window).std().to_numpy()


# ---------- spread: which variation does an error bar show? ----------
#
# Each configuration was run over 7 seeds x 31 walk-forward windows = 217 rows.
# A standard deviation over all 217 rows at once answers no question, because it
# mixes two unrelated sources of variation and treats non-independent rows as
# independent samples (adjacent windows share 4/5 of their training data):
#
#   seed spread    "would I get this again from a different random start?"
#                  collapse each seed's 31 windows to one number, then take the
#                  std over the 7 seeds. n = 7. This is the supervisor's rule.
#
#   window spread  "would I get this again in a different year?"
#                  average the 7 seeds inside each window, then take the std
#                  over the 31 windows. n = 31. This is the unit the paired
#                  AS-vs-FS significance test uses (decisions log D030).
#
# Both are reported. Seed spread draws the error bars, because that is what was
# asked for; window spread is printed in the caption so the larger and more
# honest number is never hidden. On this data window spread is roughly three
# times seed spread, which is the whole argument for showing both.

SEED_RULE = "error bars = ±1 std across the 7 seeds"


def spread_table(df: pd.DataFrame, group_col: str, metric: str = "agent_sharpe") -> pd.DataFrame:
    """Per configuration: the mean, and both standard deviations, of `metric`.

    Returns one row per value of `group_col` with:
        mean          mean over every row of the group
        seed_std      std over the 7 per-seed means          (n_seeds)
        window_std    std over the 31 per-window seed-means  (n_windows)
        pooled_std    std over all 217 rows — kept only so the old number can
                      be quoted when explaining why it was replaced
    """
    per_seed = df.groupby([group_col, "seed"])[metric].mean()
    per_window = df.groupby([group_col, "window_index"])[metric].mean()

    out = pd.DataFrame({
        "mean": df.groupby(group_col)[metric].mean(),
        "seed_std": per_seed.groupby(group_col).std(ddof=1),
        "window_std": per_window.groupby(group_col).std(ddof=1),
        "pooled_std": df.groupby(group_col)[metric].std(ddof=1),
        "n_seeds": df.groupby(group_col)["seed"].nunique(),
        "n_windows": df.groupby(group_col)["window_index"].nunique(),
        "n_rows": df.groupby(group_col)[metric].size(),
    })
    return out.reset_index()


def spread_caption(tbl: pd.DataFrame) -> str:
    """One line naming the unit of analysis and both spreads, for a figure title."""
    n_seeds = int(tbl["n_seeds"].max())
    n_windows = int(tbl["n_windows"].max())
    return (f"±1 std across {n_seeds} seeds (window spread, n={n_windows}, "
            f"is ~{tbl['window_std'].mean() / max(tbl['seed_std'].mean(), 1e-9):.1f}x larger)")


def by_window(df: pd.DataFrame, group_col: str, group_val, metric: str = "agent_sharpe") -> pd.DataFrame:
    """Per walk-forward window: the mean over seeds and the spread across seeds.

    This is the supervisor's std rule applied where it is well defined — at each
    window there really are 7 independent numbers to take a spread over.
    """
    sub = df[df[group_col] == group_val]
    return sub.groupby("window_index").agg(
        mean=(metric, "mean"),
        std=(metric, lambda s: s.std(ddof=1)),
        bh=("bh_sharpe", "mean"),
    )


def equity_band(df: pd.DataFrame, group_col: str, group_val,
                metric: str = "agent_cumret") -> pd.DataFrame:
    """Compounded equity per seed, then the mean and ±1 std across seeds.

    Each window reports a cumulative log-return for its own test year. Chaining
    them means summing the logs and exponentiating once, which is why the curve
    is built with cumsum inside exp rather than by multiplying returns.

    The spread has to be computed on the compounded curves, not on the per-window
    returns, because compounding is what the reader is looking at: two seeds that
    differ slightly in an early year stay apart for every year after it.
    """
    sub = df[df[group_col] == group_val]
    per_seed = (
        sub.groupby(["seed", "window_index"])[metric].mean()
           .unstack("window_index").sort_index(axis=1)
    )
    curves = np.exp(per_seed.cumsum(axis=1))
    return pd.DataFrame({
        "mean": curves.mean(axis=0),
        "lo": curves.mean(axis=0) - curves.std(axis=0, ddof=1),
        "hi": curves.mean(axis=0) + curves.std(axis=0, ddof=1),
    })


# ============================================================
# PART 1 — DATA INTEGRITY (Lecture 2 + 5 techniques)
# ============================================================

def plot_data_integrity(obs: np.ndarray, rets: np.ndarray, dates: pd.DatetimeIndex):
    out = PLOTS / "data_integrity"
    print("[data_integrity]")

    # --- 1. raw price series — supervisor can see the regimes ---
    df = load_aligned()
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    ax1.plot(df.index, df["vix_close"], color="tab:red", lw=0.8)
    ax1.set_title("VIX close — observed series (regime signal)")
    ax1.set_ylabel("VIX")
    ax2.plot(df.index, df["sp_close"], color="tab:blue", lw=0.8)
    ax2.set_title("S&P 500 close — traded underlying")
    ax2.set_ylabel("S&P 500")
    ax2.set_xlabel("date")
    save(fig, out / "01_price_series.png")

    # --- 2. log-return histograms (Lecture 2: distribution shape) ---
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].hist(obs, bins=120, color="tab:red", alpha=0.75, edgecolor="white", lw=0.3)
    axes[0].set_title("VIX log-returns — histogram")
    axes[0].set_xlabel("log-return")
    axes[0].axvline(0, color="k", lw=0.6, alpha=0.4)
    axes[1].hist(rets, bins=120, color="tab:blue", alpha=0.75, edgecolor="white", lw=0.3)
    axes[1].set_title("S&P next-day log-returns — histogram")
    axes[1].set_xlabel("log-return")
    axes[1].axvline(0, color="k", lw=0.6, alpha=0.4)
    save(fig, out / "02_returns_histogram.png")

    # --- 3. KDE overlay (Lecture 2) ---
    def kde(ax, x, label, color):
        from scipy.stats import gaussian_kde
        grid = np.linspace(np.quantile(x, 0.001), np.quantile(x, 0.999), 400)
        dens = gaussian_kde(x)(grid)
        ax.plot(grid, dens, color=color, lw=1.6, label=label)
        ax.fill_between(grid, dens, color=color, alpha=0.15)
    fig, ax = plt.subplots(figsize=(10, 5))
    kde(ax, obs, "VIX log-return", "tab:red")
    kde(ax, rets, "S&P next-day log-return", "tab:blue")
    ax.set_title("Log-return distributions — KDE overlay")
    ax.set_xlabel("log-return")
    ax.set_ylabel("density")
    ax.legend()
    save(fig, out / "03_returns_kde_overlay.png")

    # --- 4. Boxplots by decade (Lecture 2: regime via grouped boxplots) ---
    dec = pd.Series(dates.year // 10 * 10, name="decade")
    df_box = pd.DataFrame({"vix_ret": obs, "sp_ret": rets, "decade": dec.values})
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    df_box.boxplot(column="vix_ret", by="decade", ax=axes[0], showfliers=False)
    axes[0].set_title("VIX log-returns by decade")
    axes[0].set_xlabel("decade")
    df_box.boxplot(column="sp_ret", by="decade", ax=axes[1], showfliers=False)
    axes[1].set_title("S&P next-day log-returns by decade")
    axes[1].set_xlabel("decade")
    plt.suptitle("")  # drop pandas's auto-title
    save(fig, out / "04_boxplots_by_decade.png")

    # --- 5. Autocorrelation (Lecture 5: ACF reveals periodicity / predictability) ---
    MAX_LAG = 40
    acf_vix = autocorr(obs, MAX_LAG)
    acf_sp = autocorr(rets, MAX_LAG)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].bar(range(MAX_LAG + 1), acf_vix, color="tab:red", width=0.8)
    axes[0].axhline(0, color="k", lw=0.6)
    axes[0].set_title("VIX log-returns — autocorrelation")
    axes[0].set_xlabel("lag (trading days)")
    axes[1].bar(range(MAX_LAG + 1), acf_sp, color="tab:blue", width=0.8)
    axes[1].axhline(0, color="k", lw=0.6)
    axes[1].set_title("S&P next-day log-returns — autocorrelation")
    axes[1].set_xlabel("lag (trading days)")
    save(fig, out / "05_autocorrelation.png")

    # --- 6. Rolling mean + std (Lecture 5: visual stationarity check) ---
    WINDOW = 252   # ~1 trading year
    mv_vix, sv_vix = rolling_adf_proxy(obs, WINDOW)
    mv_sp, sv_sp = rolling_adf_proxy(rets, WINDOW)
    fig, axes = plt.subplots(2, 1, figsize=(11, 7), sharex=True)
    axes[0].plot(dates, mv_vix, color="tab:red", label="rolling mean (252d)", lw=1.1)
    axes[0].plot(dates, sv_vix, color="darkorange", label="rolling std (252d)", lw=1.1)
    axes[0].axhline(0, color="k", lw=0.5)
    axes[0].set_title("VIX log-returns — rolling mean & std (stationarity visual)")
    axes[0].legend()
    axes[1].plot(dates, mv_sp, color="tab:blue", label="rolling mean (252d)", lw=1.1)
    axes[1].plot(dates, sv_sp, color="navy", label="rolling std (252d)", lw=1.1)
    axes[1].axhline(0, color="k", lw=0.5)
    axes[1].set_title("S&P next-day log-returns — rolling mean & std")
    axes[1].set_xlabel("date")
    axes[1].legend()
    save(fig, out / "06_rolling_mean_std.png")

    # --- 7. Rolling VIX↔S&P correlation (Lecture 2: scatter → heatmap intuition) ---
    corr = pd.Series(obs).rolling(252).corr(pd.Series(rets))
    fig, ax = plt.subplots(figsize=(11, 4))
    ax.plot(dates, corr, color="tab:purple", lw=1.0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("Rolling 252d correlation — VIX log-returns vs next-day S&P return")
    ax.set_xlabel("date")
    ax.set_ylabel("correlation")
    save(fig, out / "07_rolling_correlation_vix_sp.png")

    # --- 8. Stats text report (skewness, kurtosis, ADF proxy) ---
    from scipy.stats import skew, kurtosis
    report_lines = [
        "DATA INTEGRITY REPORT",
        "=" * 60,
        f"Date span      : {dates[0].date()} → {dates[-1].date()}",
        f"Observations   : {len(obs)} aligned trading days",
        f"NaN in obs     : {int(np.isnan(obs).sum())}",
        f"NaN in rets    : {int(np.isnan(rets).sum())}",
        "",
        "VIX log-returns:",
        f"  mean         : {obs.mean():+.6f}",
        f"  std          : {obs.std():.6f}",
        f"  skewness     : {skew(obs):+.4f}   (>0: right-skewed; VIX spikes expected)",
        f"  kurtosis     : {kurtosis(obs):+.4f}  (fisher: 0 = normal; >0: fat-tailed/leptokurtic)",
        f"  min / max    : {obs.min():+.4f} / {obs.max():+.4f}",
        "",
        "S&P next-day log-returns:",
        f"  mean         : {rets.mean():+.6f}",
        f"  std          : {rets.std():.6f}",
        f"  skewness     : {skew(rets):+.4f}",
        f"  kurtosis     : {kurtosis(rets):+.4f}  (fat tails typical for daily returns)",
        f"  min / max    : {rets.min():+.4f} / {rets.max():+.4f}",
        "",
        f"Autocorrelation lag-1 — VIX  : {acf_vix[1]:+.4f}",
        f"Autocorrelation lag-1 — S&P  : {acf_sp[1]:+.4f}",
        "(Near-zero lag-1 autocorrelation on returns is EXPECTED and consistent",
        " with the efficient-markets intuition — next-day return not trivially",
        " predictable from today's return alone.)",
        "",
        "Rolling mean/std inspection (visual proxy for ADF/KPSS, Lecture 5):",
        "  - If rolling mean/std drift significantly over time, the series is",
        "    non-stationary. Our log-returns rolling mean hovers near zero across",
        "    the whole span; rolling std does drift (volatility clustering —",
        "    higher in 2008, 2020). This is the standard financial time-series",
        "    picture: returns are approximately stationary in mean but not in",
        "    variance (heteroskedasticity). Walk-forward evaluation handles this",
        "    by retraining per window.",
    ]
    (out / "stats_report.txt").write_text("\n".join(report_lines), encoding="utf-8")
    print(f"  wrote {(out / 'stats_report.txt').relative_to(ROOT)}")


# ============================================================
# PART 2 — RQ1 PLOTS (LSTM vs FS, "does RL earn its keep?")
# ============================================================

def plot_rq1():
    out = PLOTS / "rq1"
    print("[rq1]")

    lstm = pd.read_csv(ROOT / "results/rq1_lstm_sweep/windows.csv")
    fs = pd.read_csv(ROOT / "results/rq_rl_fs_sweep/windows.csv")

    # --- LSTM champion identification: bar with error bars ---
    hp_stats = spread_table(lstm, "hp_id")
    shapes = lstm.groupby("hp_id").agg(hidden=("hp_hidden_size", "first"),
                                       layers=("hp_num_layers", "first"),
                                       bh_mean=("bh_sharpe", "mean"))
    hp_stats = hp_stats.join(shapes, on="hp_id")
    hp_stats["label"] = hp_stats.apply(lambda r: f"h{int(r.hidden)}L{int(r.layers)}", axis=1)
    hp_stats = hp_stats.sort_values("mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:green" if i == 0 else "tab:gray" for i in range(len(hp_stats))]
    ax.bar(hp_stats["label"], hp_stats["mean"], yerr=hp_stats["seed_std"],
           capsize=6, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(hp_stats["bh_mean"].mean(), color="red", ls="--", lw=1,
               label=f"B&H (Sharpe {hp_stats['bh_mean'].mean():+.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("RQ1 — LSTM champion picker\n" + spread_caption(hp_stats))
    ax.set_ylabel("Sharpe ratio")
    ax.legend()
    save(fig, out / "01_lstm_champion_bar.png")

    # --- FS champion identification by k ---
    fs_stats = spread_table(fs, "k")
    fs_stats = fs_stats.join(fs.groupby("k").agg(bh_mean=("bh_sharpe", "mean")), on="k")
    fs_stats = fs_stats.sort_values("mean", ascending=False).reset_index(drop=True)

    fig, ax = plt.subplots(figsize=(8, 5))
    colors = ["tab:green" if i == 0 else "tab:gray" for i in range(len(fs_stats))]
    ax.bar([f"k={int(k)}" for k in fs_stats["k"]], fs_stats["mean"],
           yerr=fs_stats["seed_std"], capsize=6, color=colors, edgecolor="black", alpha=0.85)
    ax.axhline(fs_stats["bh_mean"].mean(), color="red", ls="--", lw=1,
               label=f"B&H (Sharpe {fs_stats['bh_mean'].mean():+.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("RQ1 (RL side) — FS champion picker across k\n" + spread_caption(fs_stats))
    ax.set_ylabel("Sharpe ratio")
    ax.legend()
    save(fig, out / "02_fs_champion_bar.png")

    # --- Head-to-head: LSTM champion vs FS champion vs B&H ---
    lstm_champ = hp_stats.iloc[0]
    fs_champ = fs_stats.iloc[0]
    bh = hp_stats["bh_mean"].mean()
    # B&H has no seeds — it is a fixed strategy, so its only spread is across
    # test years. Shown as a window-spread bar, labelled differently, rather
    # than as a zero-height bar that implies certainty.
    fig, ax = plt.subplots(figsize=(8, 5))
    labels = [f"LSTM champ\n{lstm_champ['label']}", f"FS champ\nk={int(fs_champ['k'])}", "Buy & Hold"]
    means = [lstm_champ["mean"], fs_champ["mean"], bh]
    stds = [lstm_champ["seed_std"], fs_champ["seed_std"], 0.0]
    colors = ["tab:orange", "tab:green", "tab:red"]
    ax.bar(labels, means, yerr=stds, capsize=8, color=colors, edgecolor="black", alpha=0.85)
    # window spread drawn as a lighter, wider whisker behind the seed bar
    for i, (m, ws) in enumerate(zip(means, [lstm_champ["window_std"], fs_champ["window_std"],
                                            fs["bh_sharpe"].groupby(fs["window_index"]).mean().std(ddof=1)])):
        ax.plot([i, i], [m - ws, m + ws], color="k", lw=1.0, alpha=0.35, zorder=0)
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("RQ1 head-to-head — champion LSTM vs champion FS vs B&H\n"
                 "thick bar = ±1 std across 7 seeds; faint line = ±1 std across 31 test years")
    ax.set_ylabel("Sharpe ratio")
    for i, (m, s) in enumerate(zip(means, stds)):
        ax.text(i, m + (s + 0.06), f"{m:+.3f}", ha="center", fontsize=11, fontweight="bold")
    save(fig, out / "03_rq1_head_to_head.png")

    # --- Per-window Sharpe timelines (median across seeds) ---
    lstm_champ_id = int(hp_stats.iloc[0]["hp_id"])
    fs_champ_k = int(fs_stats.iloc[0]["k"])
    # mean across the 7 seeds at each window, with ±1 seed std as the band.
    # min-max was replaced: it is an extreme-value range, so it widens with the
    # number of seeds and cannot be compared between configurations.
    lstm_ch = by_window(lstm, "hp_id", lstm_champ_id)
    fs_ch = by_window(fs, "k", fs_champ_k)

    fig, ax = plt.subplots(figsize=(11, 5))
    x = lstm_ch.index
    ax.fill_between(x, lstm_ch["mean"] - lstm_ch["std"], lstm_ch["mean"] + lstm_ch["std"],
                    color="tab:orange", alpha=0.18)
    ax.plot(x, lstm_ch["mean"], color="tab:orange", lw=1.8, label=f"LSTM champ ({hp_stats.iloc[0]['label']})")
    ax.fill_between(fs_ch.index, fs_ch["mean"] - fs_ch["std"], fs_ch["mean"] + fs_ch["std"],
                    color="tab:green", alpha=0.18)
    ax.plot(fs_ch.index, fs_ch["mean"], color="tab:green", lw=1.8, label=f"FS champ (k={fs_champ_k})")
    ax.plot(lstm_ch.index, lstm_ch["bh"], color="tab:red", ls="--", lw=1.2, label="B&H")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("RQ1 — Sharpe across all 31 walk-forward windows\n"
                 "line = mean over 7 seeds, band = ±1 std across those 7 seeds")
    ax.set_xlabel("walk-forward window index (chronological)")
    ax.set_ylabel("Sharpe")
    ax.legend(loc="best")
    save(fig, out / "04_sharpe_timeline_lstm_vs_fs.png")

    # --- Equity curves (cumulative return over the test range) ---
    # These previously showed a single line with no variance at all. Each seed
    # now gets its own compounded curve; the band is the spread across those 7
    # curves at each window, the line is their mean.
    eq_lstm = equity_band(lstm, "hp_id", lstm_champ_id)
    eq_fs = equity_band(fs, "k", fs_champ_k)
    eq_bh = equity_band(lstm, "hp_id", lstm_champ_id, "bh_cumret")
    fig, ax = plt.subplots(figsize=(11, 5))
    for eq, color, label in [
        (eq_lstm, "tab:orange", f"LSTM champ ({hp_stats.iloc[0]['label']})"),
        (eq_fs, "tab:green", f"FS champ (k={fs_champ_k})"),
    ]:
        ax.fill_between(eq.index, eq["lo"], eq["hi"], color=color, alpha=0.18)
        ax.plot(eq.index, eq["mean"], color=color, lw=1.8, label=label)
    ax.plot(eq_bh.index, eq_bh["mean"], color="tab:red", ls="--", lw=1.2, label="B&H")
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_yscale("log")
    ax.set_title("RQ1 — compounded equity across walk-forward windows\n"
                 "line = mean over 7 seeds, band = ±1 std across those 7 seeds (log scale)")
    ax.set_xlabel("walk-forward window index")
    ax.set_ylabel("equity (1.0 = starting capital)")
    ax.legend()
    save(fig, out / "05_equity_curves.png")

    # --- Position distribution comparison ---
    def pos_frac(df_sub, group_col, group_val):
        sub = df_sub[df_sub[group_col] == group_val]
        total = sub[["pos_long", "pos_flat", "pos_short"]].sum().sum()
        return (sub[["pos_long", "pos_flat", "pos_short"]].sum() / total).values
    fs_pos = pos_frac(fs, "k", fs_champ_k)
    lstm_pos = pos_frac(lstm, "hp_id", lstm_champ_id)
    fig, ax = plt.subplots(figsize=(8, 5))
    x = np.arange(3)
    w = 0.35
    ax.bar(x - w/2, lstm_pos, w, label=f"LSTM champ ({hp_stats.iloc[0]['label']})", color="tab:orange")
    ax.bar(x + w/2, fs_pos, w, label=f"FS champ (k={fs_champ_k})", color="tab:green")
    ax.set_xticks(x)
    ax.set_xticklabels(["long", "flat", "short"])
    ax.set_ylabel("fraction of days")
    ax.set_title("RQ1 — position distribution of champions")
    ax.legend()
    save(fig, out / "06_position_distribution.png")

    print(f"  RQ1 champion: LSTM={hp_stats.iloc[0]['label']} ({hp_stats.iloc[0]['mean']:+.3f}) "
          f"vs FS k={fs_champ_k} ({fs_stats.iloc[0]['mean']:+.3f})")


# ============================================================
# PART 3 — RQ2 PLOTS (AS vs FS, primary contribution)
# ============================================================

def plot_rq2():
    out = PLOTS / "rq2"
    print("[rq2]")

    fs = pd.read_csv(ROOT / "results/rq_rl_fs_sweep/windows.csv")
    aslog = pd.read_csv(ROOT / "results/rq_rl_as_sweep/windows.csv")

    fs_k = spread_table(fs, "k").join(
        fs.groupby("k").agg(bh_sharpe=("bh_sharpe", "mean")), on="k")
    as_k = spread_table(aslog, "k").join(
        aslog.groupby("k").agg(bh_sharpe=("bh_sharpe", "mean")), on="k")

    # --- Grouped bars: FS vs AS at each k ---
    fig, ax = plt.subplots(figsize=(10, 5))
    x = np.arange(len(fs_k))
    w = 0.38
    ax.bar(x - w/2, fs_k["mean"], w, yerr=fs_k["seed_std"], capsize=5,
           label="Frame Stacking (FS)", color="tab:blue", edgecolor="black", alpha=0.85)
    ax.bar(x + w/2, as_k["mean"], w, yerr=as_k["seed_std"], capsize=5,
           label="Adaptive Stacking (AS)", color="tab:purple", edgecolor="black", alpha=0.85)
    # faint whiskers carry the window spread so the tighter seed bars are never
    # read as the whole uncertainty
    for offset, tbl in [(-w/2, fs_k), (+w/2, as_k)]:
        ax.errorbar(x + offset, tbl["mean"], yerr=tbl["window_std"],
                    fmt="none", ecolor="k", elinewidth=1.0, capsize=4,
                    capthick=1.0, alpha=0.35, zorder=0)
    ax.axhline(fs_k["bh_sharpe"].mean(), color="red", ls="--", lw=1.2,
               label=f"B&H (Sharpe {fs_k['bh_sharpe'].mean():+.2f})")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={int(k)}" for k in fs_k["k"]])
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("RQ2 — AS vs FS at matched memory budget (per k)\n"
                 "capped bar = ±1 std across 7 seeds; faint line = ±1 std across 31 test years")
    ax.legend()
    save(fig, out / "01_as_vs_fs_by_k_bars.png")

    # --- Line plot: Sharpe vs k ---
    fig, ax = plt.subplots(figsize=(9, 5))
    ax.fill_between(fs_k["k"], fs_k["mean"] - fs_k["seed_std"],
                    fs_k["mean"] + fs_k["seed_std"], color="tab:blue", alpha=0.15)
    ax.plot(fs_k["k"], fs_k["mean"], marker="o", lw=2.0, color="tab:blue", label="FS")
    ax.fill_between(as_k["k"], as_k["mean"] - as_k["seed_std"],
                    as_k["mean"] + as_k["seed_std"], color="tab:purple", alpha=0.15)
    ax.plot(as_k["k"], as_k["mean"], marker="o", lw=2.0, color="tab:purple", label="AS")
    ax.axhline(fs_k["bh_sharpe"].mean(), color="red", ls="--", lw=1.2, label="B&H")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_xscale("log")
    ax.set_xticks(fs_k["k"])
    ax.set_xticklabels([int(k) for k in fs_k["k"]])
    ax.set_xlabel("memory budget k")
    ax.set_ylabel("Sharpe ratio")
    ax.set_title("RQ2 — Sharpe scaling with memory budget\n" + spread_caption(fs_k))
    ax.legend()
    save(fig, out / "02_sharpe_vs_k_lines.png")

    # --- Retention histogram per k (AS only — the proposal's diagnostic) ---
    # Columns are named mem_slot_0 .. mem_slot_19 and padded to the sweep's
    # largest k. They must be selected BY NUMBER: sorting the names as text puts
    # mem_slot_10 directly after mem_slot_1, so a k=3 run used to read
    # [mem_slot_0, mem_slot_1, mem_slot_10] and plot an always-empty third slot.
    # k=20 gets twice the width of the small-k panels: 20 bars in a quarter of a
    # 16-inch figure leaves each bar narrower than its own error bar.
    ks = [int(k) for k in sorted(aslog["k"].unique())]
    fig, axes = plt.subplots(1, len(ks), figsize=(19, 4.2),
                             gridspec_kw={"width_ratios": [1, 1, 1.3, 2.2]})
    for i, k in enumerate(ks):
        ax = axes[i]
        sub = aslog[aslog["k"] == k]
        active_cols = [f"mem_slot_{s}" for s in range(k)]
        counts = sub[active_cols].apply(pd.to_numeric, errors="coerce")

        # Per-run fractions first, so each (seed, window) run contributes once
        # regardless of how many overwrites it made, then a spread across seeds.
        per_run = counts.div(counts.sum(axis=1), axis=0)
        per_run["seed"] = sub["seed"].values
        per_seed = per_run.groupby("seed")[active_cols].mean()
        fracs = per_seed.mean(axis=0).to_numpy()
        errs = per_seed.std(axis=0, ddof=1).to_numpy()

        ax.bar(range(k), fracs, yerr=errs, capsize=3 if k < 20 else 2,
               error_kw={"capthick": 0.8, "elinewidth": 0.8,
                         "ecolor": "0.35", "alpha": 0.75},
               color="tab:purple", edgecolor="black", linewidth=0.6, alpha=0.85)
        ax.axhline(1 / k, color="red", ls="--", lw=1,
                   label=f"uniform (1/k = {1/k:.2f})")
        ax.set_xticks(range(k))
        ax.tick_params(axis="x", labelsize=8 if k == 20 else 10)
        ax.set_xlabel("slot index (0 = oldest survivor)")
        ax.set_ylabel("fraction of overwrites" if i == 0 else "")
        ax.set_title(f"AS retention — k = {k}")
        ax.legend(fontsize=8)
    fig.suptitle("RQ2 — Adaptive Stacking slot-overwrite distribution per k "
                 "(pooled over runs; ±1 std across 7 seeds)", y=1.04)
    save(fig, out / "03_retention_histogram_by_k.png")

    # --- Per-window Sharpe timeline: AS champion vs FS champion ---
    fs_champ_k = int(fs_k.sort_values("mean", ascending=False).iloc[0]["k"])
    as_champ_k = int(as_k.sort_values("mean", ascending=False).iloc[0]["k"])

    fs_t = by_window(fs, "k", fs_champ_k)
    as_t = by_window(aslog, "k", as_champ_k)
    fig, ax = plt.subplots(figsize=(11, 5))
    ax.fill_between(fs_t.index, fs_t["mean"] - fs_t["std"], fs_t["mean"] + fs_t["std"],
                    color="tab:blue", alpha=0.18)
    ax.plot(fs_t.index, fs_t["mean"], color="tab:blue", lw=1.8, label=f"FS k={fs_champ_k}")
    ax.fill_between(as_t.index, as_t["mean"] - as_t["std"], as_t["mean"] + as_t["std"],
                    color="tab:purple", alpha=0.18)
    ax.plot(as_t.index, as_t["mean"], color="tab:purple", lw=1.8, label=f"AS k={as_champ_k}")
    ax.plot(fs_t.index, fs_t["bh"], color="tab:red", ls="--", lw=1.2, label="B&H")
    ax.axhline(0, color="k", lw=0.5)
    ax.set_title("RQ2 — champion Sharpe across all 31 walk-forward windows\n"
                 "line = mean over 7 seeds, band = ±1 std across those 7 seeds")
    ax.set_xlabel("walk-forward window index")
    ax.set_ylabel("Sharpe")
    ax.legend()
    save(fig, out / "04_sharpe_timeline_as_vs_fs.png")

    # --- Equity curves: AS champ vs FS champ ---
    eq_fs = equity_band(fs, "k", fs_champ_k)
    eq_as = equity_band(aslog, "k", as_champ_k)
    eq_bh = equity_band(fs, "k", fs_champ_k, "bh_cumret")
    fig, ax = plt.subplots(figsize=(11, 5))
    for eq, color, label in [(eq_fs, "tab:blue", f"FS k={fs_champ_k}"),
                             (eq_as, "tab:purple", f"AS k={as_champ_k}")]:
        ax.fill_between(eq.index, eq["lo"], eq["hi"], color=color, alpha=0.18)
        ax.plot(eq.index, eq["mean"], color=color, lw=1.8, label=label)
    ax.plot(eq_bh.index, eq_bh["mean"], color="tab:red", ls="--", lw=1.2, label="B&H")
    ax.axhline(1.0, color="k", lw=0.5)
    ax.set_yscale("log")
    ax.set_title("RQ2 — compounded equity of champions\n"
                 "line = mean over 7 seeds, band = ±1 std across those 7 seeds (log scale)")
    ax.set_xlabel("walk-forward window index")
    ax.set_ylabel("equity (1.0 = starting)")
    ax.legend()
    save(fig, out / "05_equity_curves_champions.png")

    # --- Position-flip comparison per k ---
    fs_flips = spread_table(fs, "k", metric="pos_flips")
    as_flips = spread_table(aslog, "k", metric="pos_flips")
    fig, ax = plt.subplots(figsize=(9, 5))
    x = np.arange(len(fs_flips))
    w = 0.38
    ax.bar(x - w/2, fs_flips["mean"], w, yerr=fs_flips["seed_std"], capsize=5,
           label="FS", color="tab:blue", edgecolor="black", alpha=0.85)
    ax.bar(x + w/2, as_flips["mean"], w, yerr=as_flips["seed_std"], capsize=5,
           label="AS", color="tab:purple", edgecolor="black", alpha=0.85)
    ax.set_xticks(x)
    ax.set_xticklabels([f"k={int(k)}" for k in fs_flips["k"]])
    ax.set_ylabel("mean position flips per test window")
    ax.set_title("RQ2 — position decisiveness (fewer flips = more committed trader)\n"
                 + spread_caption(fs_flips))
    ax.legend()
    save(fig, out / "06_position_flips_by_k.png")

    print(f"  RQ2 champion: AS k={as_champ_k} "
          f"({as_k.sort_values('mean', ascending=False).iloc[0]['mean']:+.3f}) "
          f"vs FS k={fs_champ_k} "
          f"({fs_k.sort_values('mean', ascending=False).iloc[0]['mean']:+.3f})")


# ============================================================
# PART 3b — RQ2 PAIRED TEST (the headline AS-vs-FS claim)
# ============================================================
#
# RQ2 asks whether AS beats FS *on non-stationary financial data*, which is a
# claim about different years. So the comparison is made window by window: at
# each walk-forward window both mechanisms saw the same training years and were
# scored on the same test year, with the 7 seeds averaged inside the window to
# remove training-run luck. That leaves 31 paired differences per k.
#
# The interval around their mean is bootstrapped rather than taken from a
# t-test, because 31 per-window Sharpes are neither normal nor independent:
# train_size 1260 with step 252 means five consecutive windows share training
# data. A moving-block bootstrap with block length 5 resamples whole runs of
# consecutive windows instead of individual ones, so that dependence survives
# the resampling. Both the plain and the block version are reported; where they
# disagree, the block version is the one to quote.

N_BOOT = 10000
BLOCK_LEN = 5          # windows sharing training data: train_size / step = 5
BOOT_SEED = 0


def _boot_means(values: np.ndarray, block_len: int, n_boot: int, rng) -> np.ndarray:
    """Bootstrap distribution of the mean. block_len=1 is the ordinary bootstrap."""
    n = len(values)
    if block_len <= 1:
        idx = rng.integers(0, n, size=(n_boot, n))
        return values[idx].mean(axis=1)
    # moving blocks: draw ceil(n/block_len) starts, concatenate, trim to n
    n_blocks = int(np.ceil(n / block_len))
    starts = rng.integers(0, n - block_len + 1, size=(n_boot, n_blocks))
    offsets = np.arange(block_len)
    idx = (starts[:, :, None] + offsets[None, None, :]).reshape(n_boot, -1)[:, :n]
    return values[idx].mean(axis=1)


def _iqm(values: np.ndarray) -> float:
    """Interquartile mean — mean of the middle 50%, robust to the fat tails here."""
    lo, hi = np.percentile(values, [25, 75])
    middle = values[(values >= lo) & (values <= hi)]
    return float(middle.mean()) if middle.size else float(values.mean())


def paired_by_window(fs: pd.DataFrame, aslog: pd.DataFrame, k: int) -> np.ndarray:
    """AS minus FS at matched k, one value per walk-forward window (seeds averaged)."""
    f = fs[fs["k"] == k].groupby("window_index")["agent_sharpe"].mean()
    a = aslog[aslog["k"] == k].groupby("window_index")["agent_sharpe"].mean()
    joined = pd.concat([f.rename("fs"), a.rename("as")], axis=1).dropna()
    return (joined["as"] - joined["fs"]).to_numpy()


def plot_rq2_paired():
    out = PLOTS / "rq2"
    print("[rq2 paired]")

    fs = pd.read_csv(ROOT / "results/rq_rl_fs_sweep/windows.csv")
    aslog = pd.read_csv(ROOT / "results/rq_rl_as_sweep/windows.csv")
    ks = sorted(int(k) for k in fs["k"].unique())
    rng = np.random.default_rng(BOOT_SEED)

    rows = []
    for k in ks:
        d = paired_by_window(fs, aslog, k)
        plain = _boot_means(d, 1, N_BOOT, rng)
        block = _boot_means(d, BLOCK_LEN, N_BOOT, rng)
        # probability of improvement: how often AS wins the year outright
        wins = float((d > 0).mean())
        win_boot = rng.integers(0, len(d), size=(N_BOOT, len(d)))
        win_dist = (d[win_boot] > 0).mean(axis=1)
        rows.append({
            "k": k,
            "n_windows": len(d),
            "mean_diff": float(d.mean()),
            "iqm_diff": _iqm(d),
            "median_diff": float(np.median(d)),
            "ci_lo": float(np.percentile(plain, 2.5)),
            "ci_hi": float(np.percentile(plain, 97.5)),
            "block_ci_lo": float(np.percentile(block, 2.5)),
            "block_ci_hi": float(np.percentile(block, 97.5)),
            "win_rate": wins,
            "win_lo": float(np.percentile(win_dist, 2.5)),
            "win_hi": float(np.percentile(win_dist, 97.5)),
        })
    tbl = pd.DataFrame(rows)

    # --- forest plot: effect size with interval, and win rate with interval ---
    fig, (axl, axr) = plt.subplots(1, 2, figsize=(13, 4.6))
    y = np.arange(len(tbl))[::-1]

    axl.errorbar(tbl["mean_diff"], y,
                 xerr=[tbl["mean_diff"] - tbl["block_ci_lo"], tbl["block_ci_hi"] - tbl["mean_diff"]],
                 fmt="o", color="tab:purple", capsize=5, lw=1.6, markersize=7,
                 label="mean, 95% block-bootstrap CI")
    axl.scatter(tbl["iqm_diff"], y, marker="D", s=34, color="tab:gray", zorder=3,
                label="interquartile mean")
    axl.axvline(0, color="k", lw=1.2)
    axl.set_yticks(y)
    axl.set_yticklabels([f"k={k}" for k in tbl["k"]])
    axl.set_xlabel("AS − FS Sharpe (positive = AS better)")
    axl.set_title("RQ2 effect size, paired by test year\n"
                  f"n=31 windows per k, seeds averaged inside each window")
    axl.legend(fontsize=8, loc="lower right")

    axr.errorbar(tbl["win_rate"], y,
                 xerr=[tbl["win_rate"] - tbl["win_lo"], tbl["win_hi"] - tbl["win_rate"]],
                 fmt="o", color="tab:purple", capsize=5, lw=1.6, markersize=7)
    axr.axvline(0.5, color="k", lw=1.2, ls="--")
    axr.set_xlim(0, 1)
    axr.set_yticks(y)
    axr.set_yticklabels([f"k={k}" for k in tbl["k"]])
    axr.set_xlabel("fraction of test years AS wins")
    axr.set_title("RQ2 probability of improvement\n"
                  "dashed line = coin flip, 95% bootstrap CI")
    save(fig, out / "07_paired_as_minus_fs.png")

    # --- the same numbers as a table, so the figure is auditable ---
    lines = [
        "RQ2 — PAIRED AS vs FS, BY WALK-FORWARD WINDOW",
        "=" * 78,
        "One row per memory budget. Each of the 31 windows contributes one paired",
        "difference: the mean over 7 AS seeds minus the mean over 7 FS seeds, for",
        "the same training years and the same test year.",
        "",
        f"Bootstrap: {N_BOOT} resamples, seed {BOOT_SEED}. 'block' uses moving blocks",
        f"of {BLOCK_LEN} consecutive windows, because windows that are fewer than",
        f"{BLOCK_LEN} apart share training data (train_size 1260 / step 252 = 5).",
        "",
        f"{'k':>3} {'n':>3} {'mean':>8} {'IQM':>8} {'median':>8} "
        f"{'95% CI':>18} {'95% CI (block)':>20} {'win rate':>18}",
    ]
    for _, r in tbl.iterrows():
        lines.append(
            f"{int(r['k']):>3} {int(r['n_windows']):>3} "
            f"{r['mean_diff']:>+8.3f} {r['iqm_diff']:>+8.3f} {r['median_diff']:>+8.3f} "
            f"{f'[{r.ci_lo:+.3f}, {r.ci_hi:+.3f}]':>18} "
            f"{f'[{r.block_ci_lo:+.3f}, {r.block_ci_hi:+.3f}]':>20} "
            f"{f'{r.win_rate:.0%} [{r.win_lo:.0%}, {r.win_hi:.0%}]':>18}"
        )
    lines += [
        "",
        "Reading: an interval containing 0 means the data does not distinguish the",
        "two mechanisms at that memory budget. A win rate whose interval contains",
        "50% means AS is not reliably better year to year.",
    ]
    (out / "07_paired_as_minus_fs.txt").write_text("\n".join(lines), encoding="utf-8")
    tbl.to_csv(out / "07_paired_as_minus_fs.csv", index=False)
    print(f"  wrote {(out / '07_paired_as_minus_fs.txt').relative_to(ROOT)}")
    print(f"  wrote {(out / '07_paired_as_minus_fs.csv').relative_to(ROOT)}")
    for _, r in tbl.iterrows():
        print(f"  k={int(r['k']):>2}  AS-FS {r['mean_diff']:+.3f}  "
              f"block CI [{r['block_ci_lo']:+.3f}, {r['block_ci_hi']:+.3f}]  "
              f"win {r['win_rate']:.0%}")


# ============================================================
# PART 4 — TRAINING DIAGNOSTICS (if progress.csv files exist)
# ============================================================

def plot_training_diagnostics():
    out = PLOTS / "training_diagnostics"
    print("[training_diagnostics]")

    def gather_curves(run_name):
        curves_root = ROOT / f"results/{run_name}/training_curves"
        if not curves_root.exists():
            return []
        rows = []
        for d in curves_root.iterdir():
            p = d / "progress.csv"
            if p.exists():
                try:
                    df = pd.read_csv(p)
                    df["run"] = d.name
                    rows.append(df)
                except Exception:
                    pass
        return rows

    def plot_metric(curves_fs, curves_as, metric, label, filename):
        if not curves_fs and not curves_as:
            print(f"  skip {label}: no training_curves found")
            return
        fig, ax = plt.subplots(figsize=(10, 5))
        for df in curves_fs:
            if metric in df.columns:
                ax.plot(df["time/total_timesteps"] if "time/total_timesteps" in df.columns else df.index,
                        df[metric], color="tab:blue", alpha=0.08, lw=0.9)
        for df in curves_as:
            if metric in df.columns:
                ax.plot(df["time/total_timesteps"] if "time/total_timesteps" in df.columns else df.index,
                        df[metric], color="tab:purple", alpha=0.08, lw=0.9)
        # legend proxies
        ax.plot([], [], color="tab:blue", lw=2, label="FS runs")
        ax.plot([], [], color="tab:purple", lw=2, label="AS runs")
        ax.set_title(f"PPO training curves — {label}")
        ax.set_xlabel("timesteps")
        ax.set_ylabel(label)
        ax.legend()
        save(fig, out / filename)

    curves_fs = gather_curves("rq_rl_fs_sweep")
    curves_as = gather_curves("rq_rl_as_sweep")

    plot_metric(curves_fs, curves_as, "train/entropy_loss", "entropy_loss (lower → more decisive policy)",
                "01_entropy_decay.png")
    plot_metric(curves_fs, curves_as, "train/value_loss", "value_loss (lower → better critic)",
                "02_value_loss.png")
    plot_metric(curves_fs, curves_as, "train/explained_variance", "explained_variance (higher → better critic fit)",
                "03_explained_variance.png")


# ============================================================
# MAIN
# ============================================================

def main():
    print("Loading data...")
    df_full = load_aligned()
    obs, rets, dates = to_env_arrays(df_full, "vix_close", "sp_close")
    print(f"  {len(obs)} aligned trading days ({dates[0].date()} → {dates[-1].date()})\n")

    plot_data_integrity(obs, rets, dates)
    print()
    plot_rq1()
    print()
    plot_rq2()
    print()
    plot_rq2_paired()
    print()
    plot_training_diagnostics()
    print("\nAll plots written to plots/.")


if __name__ == "__main__":
    main()
