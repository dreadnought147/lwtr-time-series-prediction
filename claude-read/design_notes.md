# Design Notes

> Companion to CLAUDE.md and project_overview.md. This file explains **why** the code in `src/memory/`, `src/evaluation/`, `scripts/train.py`, and `configs/` looks the way it does. Read it if the code doesn't explain itself.

---

## 2026-08-20 — Adaptive Stacking, metrics, walk-forward, config-driven training

Everything below was added in one implementation pass. Design decisions listed here so future-me (or supervisor) can trace the *why* without reading every commit.

### 1. Adaptive Stacking wrapper (`src/memory/adaptive_stack.py`)

- **Ported from** `github.com/geraudnt/adaptive-stacking/envs/wrappers.py` (`AdaptiveStack` class, `multi_head=True` branch). Adapted for our `TimeSeriesEnv`.
- **Action space:** `MultiDiscrete([env.action_space.n, num_stack])`. First head = trading action (unchanged from base env), second head = which slot 0..k-1 to overwrite. PPO's MlpPolicy handles MultiDiscrete natively — no algorithm changes.
- **Observation space:** `Box(shape=(k * obs_dim,))` — flat, not 2D. Reference implementation uses 2D `(num_stack, *obs_shape)`. I flatten so `FrameStack` and `AdaptiveStack` produce identical-shape observations for PPO. That way, any RQ2 difference is attributable to the memory mechanism, not to the input layer.
- **Cold-start:** all k slots filled with the initial observation. Reference uses zero-fill for the first k-1 slots; I deviate so FS and AS have *identical* burn-in behaviour. Zero-fill would inject artificial extreme values into log-return space the network sees nowhere else.
- **Buffer type:** plain `list`, not `deque`. Reason: we need index-based `pop(memory_action)`, which `deque` doesn't support. Length is enforced by our own pop-then-append logic.
- **Diagnostic hook:** each `step()` returns `info["pre_frames"]` — a copy of the stack *before* the memory action is applied. This feeds the retention-histogram diagnostic ("which slots does the agent tend to overwrite over training?"). Committed to in the proposal §6 workplan.

### 2. Evaluation metrics (`src/evaluation/metrics.py`)

- **What's here:** `cumulative_return`, `sharpe_ratio`, `max_drawdown`, and a `summary()` that returns all three plus `n_steps`.
- **Sharpe convention:** annualised, zero risk-free rate, sample std (ddof=1). 252 trading days per year. Log-returns as input (a very small bias vs simple returns at daily frequency; ranking of configs is unaffected).
- **Max drawdown convention:** returned as a *positive* number in [0, 1]. `0.15` = "worst peak-to-trough loss was 15%."
- **Degenerate cases handled:** empty array → 0.0; single-element array → 0.0 (std undefined); constant series (std=0) → 0.0.
- **Why these three:** locked in the proposal §6.4 as the primary financial metrics. Cumret is the headline number, Sharpe is risk-adjusted, max DD captures downside pain.

### 3. Walk-forward split generator (`src/evaluation/walk_forward.py`)

- **What it does:** yields `Window` objects — each is a `[train chunk][test chunk]` slice pinned to a starting index. Slides forward by `step` (default = `test_size`, so test slices don't overlap).
- **Sanity band** (from published quant-finance practice, 2026, sourced via WebSearch on this date):
  - train:test ratio 2:1 to 5:1 → my default 1260/252 (5:1) sits at the upper end.
  - minimum 6–8 windows for reliability, ideally 12–20. My default over ~9188 days = ~31 windows. Well above.
  - step = test_size (non-overlapping) = "cleanest setup." Matches default.
- **Sources cited:**
  - [Walk-Forward Optimization: How It Works, Its Limitations, and Backtesting Implementation](https://blog.quantinsti.com/walk-forward-optimization-introduction/)
  - [Walk-Forward Analysis: Strategy Validation Guide](https://stratbase.ai/en/blog/walk-forward-analysis-guide)
- **Hard rules encoded in comments** (from CLAUDE.md §3): most recent slice is touched once at the very end; normalization statistics come from each split's train window only.

### 4. Config-driven training script (`scripts/train.py`)

- **The wrapper dispatcher** (`build_memory_wrapper`) is where memory choice happens — the *single explicit place* for the FS/AS swap. This is the answer to CLAUDE.md §4's open design question about memory-interface unification. Everything upstream (train loop, eval loop) and downstream (PPO, metrics) is memory-agnostic.
- **Matched-budget enforcement** is at the config level: `configs/rq2_fs_k5_seed0.yaml` and `configs/rq2_as_k5_seed0.yaml` are identical except for `memory.type`. The pairing is a convention, not enforced by code — deliberately, so I can also run mismatched-budget experiments if that becomes interesting.
- **One config = one run = one row in the results table.** `results/{run_name}/summary.json` is the JSON dump; multi-seed / multi-k comparisons happen at a higher level (aggregation over multiple JSONs, later).
- **Buy-and-hold benchmark** is computed alongside every window and reported separately. Any agent whose Sharpe doesn't beat B&H doesn't earn a "useful trader" label.
- **`total_timesteps: 50000`** in the shipped configs is a smoke-friendly default. Production runs will bump to 200000 (or higher). Per the WebSearch findings: no literature-hard number for how much PPO needs on non-stationary time-series; PPO is stable under non-stationarity because of its trust-region constraint, but sample efficiency is still open research.
- **Sources cited:**
  - [Overcoming Non-stationary Dynamics with Evidential PPO](https://arxiv.org/pdf/2503.01468)
  - [Generalized Proximal Policy Optimization with Sample Reuse](https://openreview.net/pdf?id=in_RVSTqYxK)

### 5. Config file schema

```yaml
seed: <int>
device: cpu | cuda
data:
  vix_file, sp_file: CSV filenames in data/raw/
  observation_col, asset_col: which columns feed the env
env:
  transaction_cost: float
memory:
  type: none | fs | as
  k: int
ppo:
  total_timesteps, n_steps, batch_size, learning_rate
walk_forward:
  train_size, test_size: ints in days
  max_windows: null | int
```

Extend this schema in-place as new hyperparameters get needed. Never rename existing keys — old configs must still run.

---

## Open design questions this session did NOT resolve

- **Memory interface unification** (CLAUDE.md §7 open item): partially addressed via `build_memory_wrapper`, but there's no shared `MemoryWrapper` base class asserting a common interface. Both classes just happen to inherit from `gym.Wrapper` and expose compatible signatures. Good enough for now; formalise if a third memory mechanism gets added.
- **LSTM baseline fairness** (CLAUDE.md §7): still open. `src/baselines/lstm.py` is a single untuned run. RQ1 needs a small HP sweep + multiple seeds before results are quotable.
- **Retention-histogram diagnostic**: `info["pre_frames"]` is emitted by `AdaptiveStack.step()`, but nothing consumes it yet. A separate `src/evaluation/retention.py` will need to aggregate these across a trained rollout and plot the histogram.
