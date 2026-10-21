# Project Overview — Learning What to Remember for Financial Time-Series

*Written 2026-06-24. A textual "diagram" of the whole system: what it is, how it's shaped, and how the parts fit together. Read top-to-bottom for the full picture; jump to any section for a slice.*

---

## 1. What This Project Is, in One Paragraph

You are testing whether a specific memory mechanism from reinforcement-learning research — **Adaptive Stacking** — transfers from clean, controlled benchmarks (T-Maze, MiniGrid, Rubik's Cube) to a **noisy, non-stationary financial time-series**. The setup: an agent watches a stream of daily VIX values, chooses to be **short, flat, or long the S&P 500**, and gets rewarded by the resulting return. You compare three configurations — a supervised LSTM predictor, an RL agent with fixed-window memory (Frame Stacking), and an RL agent with learned memory (Adaptive Stacking) — and ask which one is best. The primary contribution lives at the Frame-Stacking-vs-Adaptive-Stacking comparison; the LSTM is there to prove the RL framing earns its keep at all.

---

## 2. The Research Shape

Picture two research questions stacked vertically. Each one is answered by comparing two boxes in the same row:

```
              LEFT                       vs                       RIGHT

RQ1     Supervised LSTM                              PPO + Frame Stacking
        (one-shot prediction,                        (sequential decisions,
         implicit LSTM memory)                        fixed FIFO memory)

RQ2     PPO + Frame Stacking                         PPO + Adaptive Stacking
        (fixed sliding window)                       (agent-controlled memory)
```

**RQ1 asks a paradigm question:** does *sequential decision-making with memory* (RL) beat *one-shot forecasting with memory* (supervised LSTM)?

**RQ2 asks a mechanism question:** does *learned memory* beat *fixed memory* when both agents get the same memory budget?

The RQs are chained: RQ1 establishes whether the RL framing is worth anything at all; RQ2 is the primary contribution, sitting on top of that established framing.

---

## 3. The Architecture — Layers of Concern

The codebase is organised **by-layer** — each folder is a horizontal slice of capability, stacked on the ones below it. Higher layers depend only on the layers directly beneath them.

```
┌────────────────────────────────────────────────────────────┐
│  scripts/                                                  │  ← orchestration
│  (train.py, evaluate.py, download_data.py)                 │
│  Reads a config, wires everything together, runs it.       │
├────────────────────────────────────────────────────────────┤
│  src/agents/          src/baselines/                       │  ← learners
│  PPO wrapper          Supervised LSTM forecaster           │
├────────────────────────────────────────────────────────────┤
│  src/memory/                                               │  ← memory mechanisms
│  FrameStack, AdaptiveStack (wrap the env's observations)   │
├────────────────────────────────────────────────────────────┤
│  src/envs/                                                 │  ← environment
│  TimeSeriesEnv — data-agnostic Gymnasium env               │
│  Takes numpy arrays, exposes reset() / step()              │
├────────────────────────────────────────────────────────────┤
│  src/data/                                                 │  ← data plumbing
│  loader.py (reads CSVs, aligns dates)                      │
│  (upcoming: log-return transform, walk-forward splits)     │
├────────────────────────────────────────────────────────────┤
│  data/raw/                                                 │  ← source data
│  vix.csv, sp500.csv (yfinance downloads)                   │
└────────────────────────────────────────────────────────────┘
```

The rule that keeps this clean: **a layer only knows about the layers directly beneath it.** `agents/` uses `memory/` and `envs/`; it does not reach past them into `data/`. If you ever find yourself importing from a distant layer, that's a design smell — the intermediate layer is missing a helper.

**Everything in `src/` is a library** — code that gets *imported*. Nothing in `src/` is run directly except for smoke tests.

**Everything outside `src/` is orchestration or storage:**
- `scripts/` — files you run.
- `configs/` — YAML files, one per experiment, describing which pieces to plug in.
- `results/` — trained model checkpoints, logs, plots.
- `data/{raw,processed}/` — CSVs.

---

## 4. The Data Flow — From Yahoo Finance to a Reward Number

Follow one observation from source to reward:

```
Yahoo Finance
     │
     │  (yfinance.download)
     ▼
data/raw/vix.csv           data/raw/sp500.csv
     │                             │
     └─────────┬───────────────────┘
               │  (loader.load_aligned, inner-join on date)
               ▼
        DataFrame:  date | vix_close | sp_close
               │
               │  (bridge: log-returns transform)
               ▼
     obs = log(vix[t] / vix[t-1])
     ret = log(sp[t+1] / sp[t])
               │
               │  (TimeSeriesEnv wraps these arrays)
               ▼
    ┌──────────────────────────┐
    │ env.step(action)         │
    │  action ∈ {0, 1, 2}      │
    │  new_position = action-1 │
    │  reward = new_position * ret[t] - cost * |new_position - prev|
    └──────────────────────────┘
               │
               ▼
    (obs, reward, terminated, truncated, info)
               │
               │  (PPO consumes the transition, updates policy)
               ▼
      trained policy checkpoint → results/
```

**Two things to notice about this flow:**

1. **The env never touches a CSV.** By the time data reaches `TimeSeriesEnv`, it is two plain numpy arrays. The env doesn't know or care that the underlying source is VIX/S&P. Swap in a different time-series and only `scripts/download_data.py` + `src/data/loader.py` change.
2. **The reward calculation is dead simple.** It's a single line in `env.step`. Everything hard about the project lives *around* that line — the memory mechanism deciding what the agent gets to see, and the PPO update learning from the reward it produced.

---

## 5. The Training Loop — What Happens in One Experiment

A single experiment is a **config file** plus a **training script** that reads it. Think of it as a recipe (config) and a cook (script):

```
configs/rq2_adaptive_k5_seed42.yaml
       │
       │  read by scripts/train.py
       ▼
┌─────────────────────────────────────────────────────┐
│  1. Load config                                     │
│  2. Load raw data (loader.load_aligned)             │
│  3. Compute log-returns → (obs_array, ret_array)    │
│  4. Split into walk-forward windows                 │
│  5. For each window:                                │
│       a. Build TimeSeriesEnv(obs, ret) for train    │
│       b. Wrap with the chosen memory mechanism      │
│          (FrameStack | AdaptiveStack) at size k     │
│       c. Instantiate PPO with the wrapped env       │
│       d. Train for N timesteps                      │
│       e. Save checkpoint → results/{run_name}/     │
│       f. Evaluate on the held-out slice             │
│  6. Aggregate metrics across windows                │
│  7. Write summary JSON + plots → results/{run_name}/│
└─────────────────────────────────────────────────────┘
```

**One experiment produces one row in your final results table.** Different config → different row. The `run_name` is derived from the config filename so results are self-labelling.

**Reproducibility hook:** every config includes a seed. Rerunning the same config file produces the same numbers. This matters because your final claims will read like *"across five seeds, Adaptive Stacking with κ=5 achieved Sharpe X ± Y."* The seed is what makes that claim honest.

---

## 6. The Comparison Structure — How Configs Talk to Each Other

Each config is one point in a grid. Different slices of the grid answer different RQs:

```
                                  k = 3    k = 5    k = 10   k = 20
                                  ─────    ─────    ──────   ──────
Supervised LSTM (fixed hidden)      -        S        -        -      ← baseline row for RQ1
PPO + Frame Stacking                F₃       F₅       F₁₀      F₂₀    ← RL memory row
PPO + Adaptive Stacking             A₃       A₅       A₁₀      A₂₀    ← learned-memory row
```

**RQ1 comparison:** S vs any-of-{F₃, F₅, F₁₀, F₂₀}. Report the best-performing FS as the RL row's representative; compare against the LSTM baseline.

**RQ2 comparison:** Fₖ vs Aₖ *at each k*. Four paired comparisons. Report as a plot (x-axis: k, y-axis: metric, two lines). If AS wins at every k, the claim is "learned memory beats fixed memory across memory budgets." If it only wins at small k, the claim is "learned memory shines when the budget is tight."

**Multiply everything by seeds.** Each cell in that grid is actually five (or three, if compute is tight) runs with different random seeds. Metrics are reported as mean ± std, not single numbers.

---

## 7. The Evaluation Flow — Walk-Forward + Metrics

Financial data is non-stationary — the statistical shape of returns in 2001 is not the same as in 2020. Training on the whole history and testing on the tail hides this. Solution: **walk-forward evaluation**.

```
Full data timeline:
├────────────────────────────────────────────────────────────┤
2000                                                        2024

Window 1:   [ train  ][ test ]
Window 2:            [ train  ][ test ]
Window 3:                     [ train  ][ test ]
...                                    [ train  ][ test ]
```

Each window: retrain (or fine-tune) on its train slice, evaluate on its immediately-following test slice. Metrics are computed **per window** and then aggregated. The claim you make at the end is *"across N walk-forward windows spanning 2000–2024, config X achieved mean Sharpe Y."*

**Metrics computed on each test slice:**

- **Cumulative return** — how much money did the agent make?
- **Sharpe ratio** — return per unit of volatility. The industry-standard risk-adjusted number.
- **Maximum drawdown** — the worst peak-to-trough loss during the window. Captures downside pain.
- **Position-flip rate** — how often the agent changed its mind. Useful for sanity-checking transaction cost sensitivity.

For Adaptive Stacking specifically, one extra diagnostic:
- **Retention histogram** — for each slot in the stack, how old are the observations the agent tends to keep? If AS is doing what it claims, retentions should cluster around volatility spikes and regime transitions, not be uniform across the trailing history.

---

## 8. File Map — Where Concepts Live on Disk

```
Project/
├── proposal.md              ← formal proposal (submitted 2026-05-18)
├── scopedinrqs.md           ← post-scope RQs (2026-06-24)
├── project_overview.md      ← THIS FILE
├── requirements.txt         ← Python 3.10 + AS-repo pinned deps
├── .gitignore               ← excludes .venv, raw data, results
│
├── .venv/                   ← Python 3.10 virtual environment
│
├── src/                     ← THE LIBRARY (code that gets imported)
│   ├── data/
│   │   └── loader.py        ← reads CSVs, inner-joins on date
│   ├── envs/
│   │   └── timeseries_env.py← Gymnasium env, data-agnostic
│   ├── memory/              ← (upcoming) FrameStack, AdaptiveStack wrappers
│   ├── baselines/           ← (upcoming) LSTM supervised forecaster
│   ├── agents/              ← (upcoming) PPO training wrapper
│   └── evaluation/          ← (upcoming) metrics, walk-forward runner
│
├── scripts/                 ← THE ORCHESTRATION (files you run)
│   ├── download_data.py     ← yfinance → data/raw/
│   ├── train.py             ← (upcoming) reads config, runs one experiment
│   └── evaluate.py          ← (upcoming) recomputes metrics from checkpoints
│
├── configs/                 ← THE RECIPES (one YAML per experiment)
│   └── (upcoming) rq1_lstm.yaml, rq2_fs_k5.yaml, rq2_as_k5.yaml, ...
│
├── data/
│   ├── raw/                 ← yfinance CSVs (git-ignored)
│   └── processed/           ← cleaned splits (git-ignored)
│
├── results/                 ← model checkpoints, logs, plots (git-ignored)
├── notebooks/               ← exploration Jupyter notebooks
└── tests/                   ← unit tests for src/
```

---

## 9. Terminology Anchor (Future-You Reference)

- **Observation**: what the agent sees at one timestep. Here, a VIX log-return.
- **Action**: what the agent chooses. Here, `{short, flat, long}` mapped to `{-1, 0, +1}`.
- **Reward**: the scalar signal the agent maximises. Here, `position × next-day S&P log-return − transaction cost × |position change|`.
- **Episode**: a full run through the data from `reset()` until `terminated`.
- **Frame Stacking (FS)**: memory mechanism where the last k observations are concatenated into a fixed-size window. Passive. Oldest drops off automatically.
- **Adaptive Stacking (AS)**: memory mechanism where the agent takes an extra "memory action" each step, choosing which slot in a bounded stack to overwrite. Active. Learned via the same reward signal as the trading action.
- **PPO (Proximal Policy Optimisation)**: the RL algorithm used for all three-agent variants. Chosen to match the Adaptive Stacking reference paper.
- **Walk-forward evaluation**: rolling train/test split scheme that respects the temporal ordering of financial data.
- **Sharpe ratio**: `mean(returns) / std(returns) × sqrt(252)`. Annualised risk-adjusted return.
- **Maximum drawdown**: largest peak-to-trough loss during a test window.
- **κ (kappa)**: Adaptive Stacking's memory-budget parameter — number of slots in the stack.
- **k**: Frame Stacking's window size. Matched to κ in RQ2 comparisons.
- **k***: the true minimum history length the environment needs to be Markovian. Almost never known in practice.

---

## 10. What Is Not in This Diagram (Yet)

The following are known-unknowns — pieces that are on the plan but not yet written:
- The log-return transform (bridge from loader → env).
- Walk-forward split generator.
- Memory wrappers (FrameStack, AdaptiveStack integration).
- The LSTM supervised baseline.
- PPO training runner.
- Metric computation.
- Config-driven training script.
- Retention-histogram diagnostic for AS.

Each of those becomes one file, added to the appropriate layer, with the rest of the system untouched. That is the payoff of by-layer design — new capabilities land in exactly one place.
