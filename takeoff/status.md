# Project Status — Snapshot for Supervisor

**Date:** 2026-10-05
**Student:** Banele Mkhize
**Supervisor:** Geraud Nangue Tasse

---

## In One Line

All three training sweeps are **complete** — we have the full dataset needed to answer RQ1 and RQ2. Next phase is analysis, plots, and the write-up (poster + Overleaf report).

---

## What's Done (cumulative)

- **Research scoping:** 2 research questions locked (RQ1 paradigm — RL vs LSTM; RQ2 mechanism — Adaptive Stacking vs Frame Stacking at matched memory budget).
- **Full data pipeline:** VIX + S&P 500 from Yahoo Finance (1990–2026, ~9188 aligned trading days), log-returns, per-window z-scoring, walk-forward splits (31 windows: 5-year train + 1-year test, non-overlapping).
- **Three sweeps complete:**
  - **RQ1 LSTM baseline sweep** — 4 HP combos × 7 seeds × 31 walk-forward windows = 868 trained models. Champion identified by mean Sharpe across seeds.
  - **RQ2 FrameStack sweep** — k ∈ {3, 5, 10, 20} × 7 seeds × 31 windows = 868 trained PPO agents.
  - **RQ2 AdaptiveStack sweep** — κ ∈ {3, 5, 10, 20} × 7 seeds × 31 windows = 868 trained PPO agents.
  - PPO base config held identical between FS and AS so any performance difference is attributable to the memory mechanism alone.
- **Diagnostic infrastructure:** `src/evaluation/retention.py` built, writes memory-slot overwrite histogram showing whether AS learned a non-uniform retention preference.
- **Fairness items addressed:** multi-seed (7 seeds per config), per-window standardisation on train slice only, gradient clipping, weight decay, matched PPO base across FS and AS.

---

## What Was Done This Week

- Finished the two RL sweeps (FS and AS) locally on CPU. Each sweep took ~2 days wall-clock.
- Confirmed all `summary.json`, `windows.csv`, and `steps.csv` artifacts are on disk and complete.
- Rebuilt cluster onboarding path (anaconda module + conda env on Python 3.10, as Wits mscluster default is Python 3.14).

---

## What's Still To Do (ordered by priority)

1. **RQ1 comparison plots** — champion LSTM Sharpe (mean ± std, 7 seeds) vs champion FS Sharpe (mean ± std, 7 seeds).
2. **RQ2 comparison plots** — FS vs AS across k ∈ {3, 5, 10, 20}, mean Sharpe ± std per k, both mechanisms on one plot. Plus buy-and-hold floor.
3. **Retention diagnostic plots** — run `retention.py` on the AS sweep, show slot-overwrite histogram per k.
4. **Training-curves plots** — SB3 per-rollout logs are captured in `training_curves/`; show entropy_loss / value_loss / explained_variance evolving over timesteps.
5. **Poster** (per supervisor's expectation) — one-page visual summary of RQ2 result.
6. **Overleaf report** — the written thesis.

---

## Open Risks

- **Retention diagnostic not yet consumed** — the data is logged, but the summary plot isn't generated yet.
- **Final-holdout rule** (CLAUDE.md §3) — strict version where the last ~2 years are untouched for any sweep still isn't enforced in code. All sweeps use the full walk-forward range. Need to decide whether to re-run with a holdout or document the deviation.
- **Memory interface unification** (CLAUDE.md §7) — never formalised but didn't block results.

---

## Readable Numbers (for supervisor conversation)

Direct from the sweep `summary.json` files:

- Headline question for Tuesday meeting: did AS beat FS at matched k? Which k won? Did the champion AS beat buy-and-hold?
- These are *one `python -m` call away* — not yet in a plot, but the numbers are there.

