# Data Terminal — "Where do I find …?"

> This file is the landing page for anyone (a data analyst, a reviewer, a future-me) who wants to see what was produced from the raw data and what story the plots tell.
>
> Format: every entry is a **reference** — one line that names what you're looking for, one line that tells you where to go, one line that previews what the plot is saying.

**Project headline:** we trained 2,604 fresh models across three paradigms (LSTM supervised baseline, PPO + Frame Stacking, PPO + Adaptive Stacking) using daily VIX → S&P 500 positioning over ~24 years of walk-forward evaluation. Plots below answer the project's two research questions.

**Snapshot:**
- **RQ1 champion:** PPO + Frame Stacking at k=3, mean Sharpe **+0.484 ± 1.194** (beats LSTM champion h128L1 at +0.208).
- **RQ2 champion:** Adaptive Stacking at k=5, mean Sharpe **+0.515 ± 1.188** (beats FS at 3 of 4 matched memory budgets).
- **Buy-and-hold floor:** +0.841 (nothing beats it — expected during a long bull market).

---

## Visualisation techniques borrowed from IDVE module (Honours semester 2)

- **Lecture 2 (Statistics + Cleaning):** histograms, kernel density estimates, boxplots-by-group, skewness + kurtosis, correlation.
- **Lecture 5 (Time Series):** autocorrelation plots, rolling mean/std stationarity visual, rolling correlation, time-series decomposition intuition.

Every plot in `plots/` has at least one technique rooted in those two lectures; see the per-plot "what you're looking at" notes below.

---

# 1. Data Integrity — "Is the raw data clean enough to trust?"

If you are a data analyst looking for whether the input data is sane, free of leaks, and consistent with the Yahoo source — look here first.

### 1.1 Raw price series over time

- **If you want:** to see the regime structure in VIX and the overall growth of S&P 500 across 1990–2026 (incl. the 2008 crash, 2020 spike, 2022 drawdown).
- **File:** [`data_integrity/01_price_series.png`](data_integrity/01_price_series.png)
- **What it shows:** VIX top panel in red (regime signal — spikes = fear), S&P bottom panel in blue (traded underlying, steady long-run growth).

### 1.2 Log-return distributions — histogram

- **If you want:** to confirm log-returns look like a well-behaved distribution before feeding them to any model.
- **File:** [`data_integrity/02_returns_histogram.png`](data_integrity/02_returns_histogram.png)
- **Technique source:** IDVE Lecture 2 — histograms as the first look at a distribution.
- **What it shows:** VIX log-returns are right-skewed with a long positive tail (big spikes dominate). S&P next-day log-returns are symmetric but fat-tailed (leptokurtic).

### 1.3 KDE overlay of VIX and S&P log-returns

- **If you want:** a smoother picture of the distributions, overlaid for comparison.
- **File:** [`data_integrity/03_returns_kde_overlay.png`](data_integrity/03_returns_kde_overlay.png)
- **Technique source:** IDVE Lecture 2 — kernel density estimates as a smoothed histogram.
- **What it shows:** VIX log-return distribution is wider and more right-skewed than S&P's.

### 1.4 Boxplots of returns by decade

- **If you want:** to see whether return distributions themselves evolve across decades (visual regime check).
- **File:** [`data_integrity/04_boxplots_by_decade.png`](data_integrity/04_boxplots_by_decade.png)
- **Technique source:** IDVE Lecture 2 — grouped boxplots for distribution comparison across a categorical axis.
- **What it shows:** median and IQR of returns are reasonably stable decade-over-decade, with 2000s/2020s having wider spreads.

### 1.5 Autocorrelation of returns (lags 0–40)

- **If you want:** to confirm next-day returns are approximately unpredictable from today's return alone.
- **File:** [`data_integrity/05_autocorrelation.png`](data_integrity/05_autocorrelation.png)
- **Technique source:** IDVE Lecture 5 — autocorrelation reveals periodicity or lagged dependence.
- **What it shows:** lag-1 and all other lags near zero on returns. Consistent with the efficient-markets intuition. This is why memory-based agents matter — the signal isn't in direct lag-1 autocorrelation but in longer-horizon regime structure captured through the stacked window.

### 1.6 Rolling mean and standard deviation

- **If you want:** a visual stationarity test (Lecture 5's "rolling-window check" from the Dickey-Fuller worked examples).
- **File:** [`data_integrity/06_rolling_mean_std.png`](data_integrity/06_rolling_mean_std.png)
- **Technique source:** IDVE Lecture 5 — if rolling mean/std drift, the series is non-stationary.
- **What it shows:** rolling mean stays close to zero (mean-stationary), rolling std drifts upward during 2008 and 2020 (variance non-stationary — volatility clustering). Walk-forward evaluation addresses this by retraining per window.

### 1.7 Rolling 252d correlation between VIX returns and S&P returns

- **If you want:** to see whether VIX and S&P move together or against each other, and how that relationship evolves.
- **File:** [`data_integrity/07_rolling_correlation_vix_sp.png`](data_integrity/07_rolling_correlation_vix_sp.png)
- **Technique source:** IDVE Lecture 2 — correlation; Lecture 5 — rolling-window summaries.
- **What it shows:** strongly negative correlation (around −0.7 to −0.8) throughout — exactly why VIX serves as an informative *observation* for a learner trying to position on S&P.

### 1.8 Full stats report (descriptive, text)

- **If you want:** mean, std, skewness, kurtosis, min/max, lag-1 autocorrelation, missing-data audit, all in one scrollable text file.
- **File:** [`data_integrity/stats_report.txt`](data_integrity/stats_report.txt)
- **Technique source:** IDVE Lecture 2 — descriptive statistics as the first analytical pass.
- **What it shows:** everything numerical about the dataset in one place.

### 1.9 Yahoo re-fetch integrity audit (script + report)

- **If you want:** to confirm the local CSVs on disk match what Yahoo currently reports for the same tickers.
- **Script to run:** `python scripts/data_integrity_check.py`
- **Output file (after running):** [`data_integrity/integrity_report.txt`](data_integrity/integrity_report.txt)
- **What it shows:** per-date comparison of local vs fresh close prices, flagging any drift > 0.1%. Addresses "is the raw data still what we trained on?"

### 1.10 Formal data integrity disclaimer (LaTeX)

- **If you want:** the pedantic, academic-grade "this is research only, not for trading" statement.
- **File:** [`data_integrity/disclaimer.tex`](data_integrity/disclaimer.tex)
- **Compile with:** `pdflatex plots/data_integrity/disclaimer.tex`
- **What it covers:** non-advisory nature, data source, deliberate limitations, reproducibility steps, author attestation.

---

# 2. RQ1 — "Does the RL framing earn its keep against a strong LSTM baseline?"

If you are a data analyst looking for the paradigm comparison (supervised LSTM vs sequential-decision PPO + Frame Stacking), look here.

### 2.1 LSTM champion picker

- **If you want:** to see which of the four LSTM HP shapes won, with error bars across seeds.
- **File:** [`rq1/01_lstm_champion_bar.png`](rq1/01_lstm_champion_bar.png)
- **What it shows:** bar chart with error bars, champion highlighted green. Champion: hidden=128, layers=1 (**Sharpe +0.208**).

### 2.2 FS champion picker across k values

- **File:** [`rq1/02_fs_champion_bar.png`](rq1/02_fs_champion_bar.png)
- **What it shows:** which memory budget wins for Frame Stacking. Champion: k=3 (**Sharpe +0.484**).

### 2.3 Head-to-head: LSTM champion vs FS champion vs B&H

- **If you want:** the one plot that answers RQ1 in a single glance.
- **File:** [`rq1/03_rq1_head_to_head.png`](rq1/03_rq1_head_to_head.png)
- **What it shows:** three bars with error bars. **FS wins (+0.484) over LSTM (+0.208). Neither beats B&H (+0.841).**

### 2.4 Per-window Sharpe timeline

- **If you want:** to see how performance varies across walk-forward windows (chronological) — is the champion consistent or driven by a few lucky windows?
- **File:** [`rq1/04_sharpe_timeline_lstm_vs_fs.png`](rq1/04_sharpe_timeline_lstm_vs_fs.png)
- **Technique source:** IDVE Lecture 5 — time-series line plot with uncertainty bands.
- **What it shows:** median across seeds + min-max shaded band for each champion, overlaid with B&H dashed reference.

### 2.5 Equity curves (compounded return across walk-forward windows)

- **If you want:** the "money grew" picture — how 1 unit of capital grew if you compounded this champion's returns window-after-window.
- **File:** [`rq1/05_equity_curves.png`](rq1/05_equity_curves.png)
- **What it shows:** exponentiated cumulative return for LSTM champ, FS champ, and B&H.

### 2.6 Position distribution comparison

- **If you want:** to see how the two champions actually trade — do they go long more often, short more often, or sit flat?
- **File:** [`rq1/06_position_distribution.png`](rq1/06_position_distribution.png)
- **What it shows:** grouped bar chart — fraction of days spent long / flat / short, for each champion.

### 2.7 Raw data for RQ1

- **LSTM sweep rows:** [`../results/rq1_lstm_sweep/windows.csv`](../results/rq1_lstm_sweep/windows.csv) (868 rows)
- **LSTM sweep summary:** [`../results/rq1_lstm_sweep/summary.json`](../results/rq1_lstm_sweep/summary.json)
- **FS sweep rows:** [`../results/rq_rl_fs_sweep/windows.csv`](../results/rq_rl_fs_sweep/windows.csv) (868 rows)
- **FS sweep summary:** [`../results/rq_rl_fs_sweep/summary.json`](../results/rq_rl_fs_sweep/summary.json)
- **Per-step data (LSTM):** [`../results/rq1_lstm_sweep/steps.csv`](../results/rq1_lstm_sweep/steps.csv)
- **Per-step data (FS):** [`../results/rq_rl_fs_sweep/steps.csv`](../results/rq_rl_fs_sweep/steps.csv)

---

# 3. RQ2 — "Does Adaptive Stacking beat Frame Stacking at matched memory budget?" (PRIMARY)

If you are a data analyst looking for the primary research contribution — the AS vs FS comparison — look here.

### 3.1 AS vs FS at each k (grouped bars with error bars)

- **If you want:** the single plot that answers RQ2 at a glance.
- **File:** [`rq2/01_as_vs_fs_by_k_bars.png`](rq2/01_as_vs_fs_by_k_bars.png)
- **What it shows:** at k=3 FS edges AS; at k=5, 10, 20 **AS wins**. The advantage grows with memory budget.

### 3.2 Sharpe vs k — line plot

- **If you want:** the scaling picture — does the gap between AS and FS grow, shrink, or stay flat as k grows?
- **File:** [`rq2/02_sharpe_vs_k_lines.png`](rq2/02_sharpe_vs_k_lines.png)
- **What it shows:** two lines (FS and AS), each with a shaded ±std band. The AS line stays higher at k ≥ 5. Buy-and-hold is the red dashed reference.

### 3.3 Retention histogram per k (AS only)

- **If you want:** to see whether AS actually learned to prefer some memory slots over others (the proposal's qualitative diagnostic).
- **File:** [`rq2/03_retention_histogram_by_k.png`](rq2/03_retention_histogram_by_k.png)
- **Technique source:** IDVE Lecture 2 — distribution-shape visualisation via histogram.
- **What it shows:** fraction of overwrites per slot, with the "uniform (1/k)" reference line. Non-uniform bars = AS learned a selection preference.

### 3.4 Per-window Sharpe timeline — AS champion vs FS champion

- **File:** [`rq2/04_sharpe_timeline_as_vs_fs.png`](rq2/04_sharpe_timeline_as_vs_fs.png)
- **Technique source:** IDVE Lecture 5 — time-series line with uncertainty band.
- **What it shows:** AS k=5 (purple) vs FS k=3 (blue) across 31 walk-forward windows, median + min-max band, B&H dashed.

### 3.5 Equity curves — AS champion vs FS champion

- **File:** [`rq2/05_equity_curves_champions.png`](rq2/05_equity_curves_champions.png)
- **What it shows:** compounded growth. The gap between AS and FS equity curves is the "AS value" over the walk-forward period.

### 3.6 Position-flip comparison (decisiveness)

- **If you want:** to see if AS trades differently than FS — fewer flips means more decisive, committed positions.
- **File:** [`rq2/06_position_flips_by_k.png`](rq2/06_position_flips_by_k.png)
- **What it shows:** grouped bar chart — mean position flips per test window. AS tends to flip less than FS at larger k.

### 3.7 Raw data for RQ2

- **FS sweep rows:** [`../results/rq_rl_fs_sweep/windows.csv`](../results/rq_rl_fs_sweep/windows.csv) (868 rows: 4 k × 7 seeds × 31 windows)
- **AS sweep rows:** [`../results/rq_rl_as_sweep/windows.csv`](../results/rq_rl_as_sweep/windows.csv) (868 rows)
- **FS sweep summary:** [`../results/rq_rl_fs_sweep/summary.json`](../results/rq_rl_fs_sweep/summary.json)
- **AS sweep summary:** [`../results/rq_rl_as_sweep/summary.json`](../results/rq_rl_as_sweep/summary.json)
- **FS per-step data:** [`../results/rq_rl_fs_sweep/steps.csv`](../results/rq_rl_fs_sweep/steps.csv)
- **AS per-step data:** [`../results/rq_rl_as_sweep/steps.csv`](../results/rq_rl_as_sweep/steps.csv)
- **AS per-step memory-action column** inside `steps.csv` carries the slot chosen each day — raw material for the retention diagnostic.

---

# 4. Training Diagnostics — "Did the PPO agents actually learn?"

If you are a data analyst looking at whether training itself was healthy (as opposed to the end result), look here.

### 4.1 Entropy decay over training timesteps

- **If you want:** to see whether the policy became more decisive as training progressed (entropy going down = more decisive).
- **File:** [`training_diagnostics/01_entropy_decay.png`](training_diagnostics/01_entropy_decay.png)
- **What it shows:** entropy_loss over timesteps, one thin line per window. FS runs blue, AS runs purple.

### 4.2 Value loss evolution

- **File:** [`training_diagnostics/02_value_loss.png`](training_diagnostics/02_value_loss.png)
- **What it shows:** critic's prediction error over timesteps. Downward = the critic is learning to estimate state values better.

### 4.3 Explained variance evolution

- **File:** [`training_diagnostics/03_explained_variance.png`](training_diagnostics/03_explained_variance.png)
- **What it shows:** how much of the return variance the critic captures. Rising = critic is learning. Industry rule-of-thumb: >0.5 is healthy, >0.9 is strong.

### 4.4 Raw per-rollout training stats

- **Location:** `../results/rq_rl_{fs,as}_sweep/training_curves/w{window}_k{k}_s{seed}/progress.csv`
- **What they contain:** SB3's per-rollout metric dump — ep_rew_mean, ep_len_mean, entropy_loss, value_loss, approx_kl, clip_fraction, explained_variance, learning_rate, more. Loadable with pandas for custom plots.

---

# 5. How to regenerate everything

```bash
# All plots (data integrity, RQ1, RQ2, training diagnostics)
python scripts/generate_plots.py

# Yahoo refetch audit (requires internet)
python scripts/data_integrity_check.py

# Compile the disclaimer PDF (requires pdflatex)
cd plots/data_integrity
pdflatex disclaimer.tex
```

---

## Appendix — glossary for the data analyst

- **Sharpe ratio:** `mean(returns) / std(returns) × √252`. Higher is better. +1 is good, +2 is exceptional, 0 is break-even. Industry-standard risk-adjusted return.
- **Cumulative return:** total growth; +0.08 = +8% over the period.
- **Max drawdown:** worst peak-to-trough loss in the equity curve, as a positive fraction.
- **Walk-forward evaluation:** rolling train-then-test, slide forward, repeat. 31 windows total in this project (5-year train, 1-year test, non-overlapping).
- **k (memory budget):** how many observations the agent sees stacked in its memory. Frame Stacking uses them FIFO; Adaptive Stacking lets the agent choose which to overwrite.
- **FS, AS, LSTM, PPO:** Frame Stacking (fixed memory wrapper), Adaptive Stacking (learned memory wrapper), Long Short-Term Memory (the supervised baseline), Proximal Policy Optimisation (the RL algorithm).
- **Walk-forward "window":** one slice of training data followed by one slice of test data. 31 windows span 1990 train through 2026 test.
