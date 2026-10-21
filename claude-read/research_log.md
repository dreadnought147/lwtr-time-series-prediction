# Research Log

> Dated journey entries for this project. Append-only, chronological. Format per CLAUDE.md §11:
> `- [tag, YYYY-MM-DD] whatever happened`
>
> Tags used: `update` (default event), `decision` (a scope/design call with rationale), `ah-ha` (a realisation worth remembering), `milestone` (something end-to-end works), `open` (unresolved concern), `plan` (something scheduled or expected).
>
> Context and decisions belong in `CLAUDE.md`, not here. This file is the *narrative* — what was tried, what worked, what didn't, what got learned along the way, in first-person.

---

## March 2026

- [update, 2026-03] Submitted annotated bibliography — 6 IEEE sources: Sutton & Barto (RL foundations), Mnih et al. (DQN, frame stacking at Atari scale), Hochreiter & Schmidhuber (LSTM), Moody & Saffell (Recurrent Reinforcement Learning for direct trading), Schulman et al. (PPO), Nangue Tasse et al. 2025 (Adaptive Stacking, the paper this project extends). This is the citation spine that flows through the lit review and proposal.

## April 2026

- [update, 2026-04-19] Submitted literature review. Traced the evolution of memory mechanisms in deep RL: fixed frame stacking → LSTM gating → Adaptive Stacking, situated inside financial time-series prediction. Added Demir (2021) as a contrast case — intrinsic-motivation memory, novelty-driven — juxtaposed with AS's reward-driven memory action.

## May 2026

- [update, ~2026-05-04] Bootstrapped the project. Created initial `.venv`, VS Code setup, PyTorch + CUDA on the GTX 1660 Ti. Nothing in `src/` yet — pure scaffolding.
- [update, 2026-05-17] Proposal draft locked. Three research questions and four configurations: supervised LSTM baseline, memoryless PPO, PPO+FrameStacking sweep k ∈ {3, 5, 10, 20}, PPO+AdaptiveStacking sweep κ ∈ {3, 5, 10, 20}. Walk-forward evaluation over 2000–2024 daily VIX + S&P 500. Primary metrics: cumulative return, Sharpe ratio, max drawdown.
- [update, 2026-05-18] Proposal submitted. Produced Week 2 slide deck reflecting proposal state. Central framing crystallised into one sentence: *"does AS transfer from the clean, controlled, episodic benchmarks it was validated on (T-Maze, MiniGrid, Rubik's Cube, FetchReach) to noisy, non-stationary financial time-series?"*

### Late May 2026 — Dataset reasoning

- [ah-ha, ~2026-05-18] Committed to **VIX as observation + S&P as underlying** after considering three candidates:
  - T-bills + S&P (the Moody & Saffell canonical setup): rejected — T-bills don't carry the memory structure the thesis needs.
  - VIX-only, predict next VIX: rejected — no sequential consequence, supervised trivially wins, RL framing earns nothing.
  - **Chosen — VIX observation + S&P position:** VIX literally encodes regime memory in its construction (decay tail of crashes weeks/months back). That is exactly the *sparse-signal-in-noisy-history* pattern AS is built for. The setup is structurally honest about the κ ≪ k* assumption AS depends on.

---

## 2026-06-24 — Scope reduction + scaffolding session

- [decision, 2026-06-24] **Cut scope from 4 configurations / 3 RQs down to 3 configurations / 2 RQs.** Dropped the memoryless PPO baseline and its dedicated research question ("does memory help at all"). Rationale: (a) memoryless-vs-FS is already established in the DQN / POMDP literature; (b) the proposal itself called it "a sanity check more than a contribution"; (c) freeing that PPO sweep buys back August time for the actual Adaptive Stacking integration, which is where the primary contribution lives. Trade-off accepted: can no longer standalone-claim "memory helps in financial RL," only "learned memory beats fixed memory" — which is the claim that actually matters. Requires supervisor sign-off in next meeting. Documented in `scopedinrqs.md`.
- [update, 2026-06-24] Rebuilt `.venv` on Python 3.10 (was 3.13). Motivation: the AS reference repo pins Python 3.10, torch 2.4.1, numpy 1.24.3, gymnasium 1.2.3, sb3 2.7.1. Version drift in August would silently break integration. Matched pins verbatim.
- [decision, 2026-06-24] Adopted **by-layer** architecture over by-experiment: `src/{data, envs, memory, agents, baselines, evaluation}/`. Reasoning: RQs are chained — the "winner" of one is the baseline of the next — so shared components (env, PPO base, metrics) should live in one place, and each experiment becomes a *config file*, not a folder. "Experiment" = a row in a results table, not a directory.
- [update, 2026-06-24] Laid folder structure, `.gitignore` (excludes `.venv`, raw data, results), `git init`, first commit.
- [update, 2026-06-24] Wrote `scripts/download_data.py` — yfinance downloads for `^VIX` and `^GSPC`, saved to `data/raw/`. Deliberately the *only* file allowed to know the source is Yahoo Finance.
- [update, 2026-06-24] Wrote `src/data/loader.py` — reads yfinance CSVs, inner-joins on shared trading dates → clean `date | vix_close | sp_close` DataFrame. Chose inner join over left/right/outer to avoid NaN rewards (any missing return would silently corrupt training).
- [ah-ha, 2026-06-24] yfinance CSVs have a **3-row header** ("Price / Ticker / Date"), not 1. Naive `pd.read_csv(..., index_col=0)` treats "Ticker,^GSPC,..." as the first data row and everything breaks downstream. Fix: `skiprows=3, header=None, names=COLS`.
- [update, 2026-06-24] Wrote `src/data/transforms.py` — the **bridge** from loader DataFrame → env-ready `(obs, returns)` numpy arrays. Computes log-returns of the observation column. Uses `.shift(-1)` on the asset-return series so that index `t` holds the *next-day* return — the reward at day t is scored against the return from t → t+1, never t-1 → t (that would score against a return already knowable when the agent acted). Drops the first + last rows (NaN from the shifts). 9190 → 9188 aligned rows.
- [update, 2026-06-24] Wrote `src/envs/timeseries_env.py` — data-agnostic `TimeSeriesEnv` (Gymnasium). Takes `(observations, returns)` numpy arrays. `Discrete(3)` action → position ∈ {-1, 0, +1}. Reward = `position × returns[t] − cost × |Δposition|`. `terminated=True` when the data runs out. Smoke test on random arrays passed.

---

## 2026-08-05 — PPO handshake

- [milestone, 2026-08-05] **RL pipeline is real end-to-end.** Wrote `scripts/smoke_ppo.py` — plugs `TimeSeriesEnv` into SB3 PPO, trains 2000 steps. SB3's `check_env` passes, PPO trains without errors, one full episode of the trained (nearly-random) policy → `total reward = 0.3965` across 9188 days.
- [observation, 2026-08-05] SB3 warns "PPO with MLP policy prefers CPU over GPU (MLPs don't parallelise well like CNNs)." Cosmetic. Silence with `device='cpu'` in the `PPO(...)` call.
- [note, 2026-08-05] `entropy_loss ≈ −1.1 ≈ ln(3)` throughout — policy is fully random after 2000 steps. Expected at smoke-test scale. Real training will need 100k+ steps to see entropy drift toward less negative values.

---

## 2026-08-09 — FrameStack wrapper

- [update, 2026-08-09] Wrote `src/memory/framestack.py` — `FrameStack(env, k)` Gymnasium wrapper. `collections.deque(maxlen=k)` is the entirety of the memory model. `reset()` fills the buffer with k copies of the initial observation (standard DQN convention — avoids the "cold-start" of zeros polluting early steps). `step()` appends new obs on the right, oldest silently drops off the left. `observation_space` shape changes `(1,)` → `(k,)`; every other env field (reward, terminated, info) passes through unchanged.
- [ah-ha, 2026-08-09] Frame Stacking encodes **zero information about the underlying data type** — pure structural wrapper. Works for any 1-D observation, whether it's a scalar return, an image feature vector, or gridworld cell values. This confirmed the intuition that AS should also be env-agnostic (later verified by the AS README: "compatible with *any* RL algorithm and gymnasium environment").
- [milestone, 2026-08-09] `scripts/smoke_ppo_fs.py` — PPO + FrameStack handshake test. Total reward = 2.987 across 9188 days (vs 0.3965 memoryless — not statistically meaningful at smoke-test scale, but the pipeline is confirmed). **RQ1's RL arm is architecturally complete.**
- [ah-ha, 2026-08-09] Realised the memoryless smoke test I ran on 2026-08-05 was architecturally the same thing as the memoryless RL agent I *dropped* on 2026-06-24. Not a research config now, but the code foundation both FS and AS build on. De-risked the whole RL half of the project in ~8 seconds of training.

---

## 2026-08-16 — LSTM baseline + AS repo inspection

- [update, 2026-08-16] Wrote `src/baselines/lstm.py` — supervised sequential predictor for RQ1. Components:
  - `SequenceDataset` — reshapes (obs, target) arrays into rolling (window, target) supervised pairs.
  - `LSTMForecaster` — `nn.LSTM(input_size=1, hidden_size=32)` + `nn.Linear(hidden_size, 1)` head. Takes only the final timestep's hidden state as the sequence summary.
  - `train_forecaster` — vanilla Adam + MSE training loop.
  - `forecasts_to_positions` — threshold rule: `sign(forecast)` if magnitude exceeds threshold, else 0.
  - `evaluate` — computes total reward using the **exact same formula** as `TimeSeriesEnv`. This is what makes RQ1's comparison fair: RL agents and the LSTM baseline are scored on identical terms.
- [decision, 2026-08-16] **Chose vanilla LSTM over pretrained foundation models** (Chronos, TimeGPT, Moirai). Reasoning: proposal commits to LSTM; my scope is a memory-mechanism paper, not a forecasting paper. The LSTM is the *control* baseline, not the star. A foundation model would be a scope drift and would change the RQ1 claim from "does RL beat a strong supervised sequential baseline" to "does RL beat a general-purpose time-series foundation model."
- [update, 2026-08-16] Cloned Nangue Tasse's AS reference repo to `../adaptive-stacking/` (side-clone, not into project — cleaner git history, easier inspection).
- [update, 2026-08-16] Read AS `README.md`. Confirmed: AS is env-agnostic and algorithm-agnostic, drop-in `AdaptiveStack(env, num_stack=k)` usage. Tested by authors on FetchReach (continuous), TMaze (grid), MiniGrid (grid), Rubik's Cube (discrete). Wrapper lives in `envs/wrappers.py`. Caveat: `--single_head` flag exists for Discrete actions to avoid MultiDiscrete complications.
- [update, 2026-08-16] Introduced `claude-read/` folder inside the project with:
  - `CLAUDE.md` — persistent anchor context, append-only, format `- [tag, YYYY-MM-DD] ...`. Read on explicit session start cue.
  - `project_overview.md` — layered-architecture doc, matches the by-layer decision from June.
  - `research_log.md` — this file. Journey entries, brainstorm-like but formal enough to lift into slides.

---

## Standing Plans / Scheduling

- [plan, 2026-08-16] **Immediate next step:** read `envs/wrappers.py` in the AS repo. Two possible paths after that: (a) copy the AdaptiveStack class into `src/memory/adaptive_stack.py` and adapt for the `TimeSeriesEnv` action space, or (b) add the AS repo as a git submodule and import directly. Decision deferred until I've seen the code.
- [plan, 2026-08-16] **Upcoming training run estimates** (revise once first real run measures actual wall-clock):
  - Single PPO+FS run, k=5, 100k steps: ~10–20 min CPU (smoke test hit 2k steps in ~15 s).
  - Full PPO+FS sweep, k ∈ {3, 5, 10, 20}, 3 seeds: 12 runs × ~15 min ≈ **3 hours**.
  - Full PPO+AS sweep, κ ∈ {3, 5, 10, 20}, 3 seeds: 12 runs × ~15 min ≈ **3 hours** (may be slower — larger action space, wrapper overhead).
  - LSTM baseline, 3 seeds × 10 epochs on 9188 rows: **<10 minutes total.**
- [plan, 2026-08-16] Wits mscluster (SLURM, GPU partitions `stampede` / `bigbatch` / `biggpu`) onboarding completed. Long sweeps migrate off local box once the metrics + walk-forward infrastructure is in.

---

## Open Concerns (mirrored from CLAUDE.md §7 for narrative visibility)

- [open, 2026-08-16] **Dataset span** — ~20 years of daily VIX/S&P may be short by published financial-ML standards. Needs a literature check against comparable RL-for-finance work before making claims about generalisability.
- [open, 2026-08-16] **Memory interface unification** — `FrameStack` and (upcoming) `AdaptiveStack` need a shared interface so both are config-selectable via one YAML key, and the matched-budget constraint (κ = k for RQ2) is enforced in one explicit place, not implicitly across scripts.
- [open, 2026-08-16] **LSTM baseline fairness** — the current LSTM is untuned. For RQ1 to mean anything, it needs its own small hyperparameter sweep (hidden_size, num_layers, sequence length, learning rate), gradient clipping, and multi-seed reporting. A single lazy run answers "does PPO beat a lazy LSTM," not RQ1 as actually written.

---

## 2026-08-20 — Adaptive Stacking, metrics, walk-forward, config-driven training

Big implementation push. Four new files, two configs, one design doc, one running-response log. Design rationale for each lives in `claude-read/design_notes.md`.

- [update, 2026-08-20] Wrote `src/memory/adaptive_stack.py` — the AS wrapper, ported and adapted from `github.com/geraudnt/adaptive-stacking/envs/wrappers.py`. Action space becomes `MultiDiscrete([env.action_space.n, k])`; observation stays flat `(k * obs_dim,)` to match FrameStack so any RQ2 difference is attributable to the memory mechanism, not to the input layer. Cold-start filled with k copies of the initial observation, matching FrameStack burn-in (deviates from reference's zero-fill by design). Emits `info["pre_frames"]` on every step for the retention-histogram diagnostic committed to in the proposal.
- [update, 2026-08-20] Wrote `src/evaluation/metrics.py` — `cumulative_return`, `sharpe_ratio` (annualised, 252d/yr, zero risk-free), `max_drawdown` (returned as positive [0, 1]), and a `summary()` that returns all three plus `n_steps`. Handles degenerate cases (empty, single-element, std=0).
- [update, 2026-08-20] Wrote `src/evaluation/walk_forward.py` — yields `Window` objects. Default step = test_size (non-overlapping test slices). Includes sanity-band comment (2–5:1 train:test ratio, ≥6–8 windows minimum).
- [update, 2026-08-20] Wrote `scripts/train.py` — config-driven training runner. Loads a YAML, walks-forward through the data, trains PPO on each window, evaluates + reports metrics, dumps `results/{run_name}/summary.json`. Buy-and-hold benchmark computed alongside every window. The `build_memory_wrapper()` function is the *single explicit place* where the memory-mechanism swap happens (addresses CLAUDE.md §4 open question, partially).
- [update, 2026-08-20] Wrote `configs/rq2_fs_k5_seed0.yaml` and `configs/rq2_as_k5_seed0.yaml` — matched-budget pair, identical except for `memory.type`. `total_timesteps: 50000` for smoke-friendly first runs; bump to 200000+ for production.
- [update, 2026-08-20] Wrote `claude-read/design_notes.md` — dated design-decision record for this session's work. Explains the *why* behind each file, cites WebSearch sources for the walk-forward sanity band.
- [update, 2026-08-20] Wrote `claude-read/log.txt` — running record of the longer explanatory responses I want to revisit later. Started with the walk-forward + PPO training-length recap.
- [decision, 2026-08-20] AS wrapper uses flat observation `(k,)` not 2D `(k, obs_dim)`. Deviates from reference. Reason: matches our FrameStack shape so the RQ2 comparison is honest at the input layer.
- [decision, 2026-08-20] AS wrapper cold-starts with k copies of the initial observation (not zero-fill). Deviates from reference. Reason: matches our FrameStack burn-in so both wrappers start from identical state.
- [decision, 2026-08-20] Configs default to `total_timesteps: 50000` — smoke-friendly. Production runs will bump to 200000+. No literature-hard number available (verified via WebSearch); PPO's trust region gives it stability under non-stationarity, but sample efficiency remains open research.
- [ah-ha, 2026-08-20] The matched-budget constraint (κ = k) doesn't need to be enforced in code — it's a *convention* honoured by the config filenames and the run scripts. Encoding it in code would prevent me from also running mismatched-budget experiments if that ever became interesting. Comment in `design_notes.md`.
- [update, 2026-08-20] Ran §9 search prompts 9 and 12 (walk-forward window sizing) plus one on PPO sample efficiency under non-stationarity. Confirmed my defaults sit inside published bands. No new items added to CLAUDE.md's "Concerns Bubbled Up" section — sanity band held.
- [plan, 2026-08-20] Next: run `python scripts/train.py --config configs/rq2_fs_k5_seed0.yaml` and its AS twin to get first real end-to-end numbers. Expected wall-clock at 50k timesteps × ~31 windows: ~30–60 minutes each on CPU. Watch for: PPO stability, whether the agent converges away from `entropy_loss ≈ ln(3)`, and whether either beats buy-and-hold at all.
- [open, 2026-08-20] Retention-histogram diagnostic: `info["pre_frames"]` is being emitted by AS but nothing consumes it yet. Needs `src/evaluation/retention.py` before the RQ2 write-up.

### 2026-08-20 (later same day) — smoke configs, richer JSON, CSV outputs, resume, design doc

Filling in what happened after the initial adaptive_stack/metrics/walk_forward/train push, before overnight training on 2026-08-25.

- [update, 2026-08-20] Wrote `configs/smoke_fs.yaml` and `configs/smoke_as.yaml` — tiny-scale configs (`total_timesteps: 2000`, `train_size: 200`, `test_size: 50`, `max_windows: 2`) for proving the pipeline handshake without waiting an hour. Same pattern as the earlier smoke_ppo → smoke_ppo_fs progression: prove plumbing before scale.
- [update, 2026-08-20] Refactored `scripts/train.py`'s `summary.json` format. Now per-window records include real trading-date ranges, sizes, training wall-seconds, agent metrics + position stats (n_long/flat/short/flips) + per-step rewards, buy-and-hold metrics + raw per-day returns, plus (AS only) memory action counts by slot. Rich but structured — descriptive without being fat.
- [update, 2026-08-20] Modified `src/data/transforms.py` — `to_env_arrays` now returns `(obs, rets, dates)` instead of `(obs, rets)`. The date index is needed so per-window records can carry actual trading dates ("train 1990-01-03..1990-10-16" reads better than "train [0:200]"). Downstream: `scripts/train.py` consumes it; the transforms `__main__` smoke test updated to match.
- [decision, 2026-08-20] Emit CSVs alongside JSON — `windows.csv` (one row per window, flat schema) and `steps.csv` (one row per (window, day)). Reason: pandas-friendly, `pd.concat` across many runs = one dataframe for RQ2 aggregation plots. JSON stays for narrative / debugging.
- [decision, 2026-08-20] Cluster-friendly write pattern: CSVs are appended row-by-row and flushed after each window. If the process dies mid-run, the CSV is valid up to the last completed window — never corrupt. `summary.json` written atomically at the end (`.tmp` → rename).
- [update, 2026-08-20] Added resume logic to `scripts/train.py`. On startup, reads existing `windows.csv` to determine which windows were already completed, skips them, and continues. Schema check on resume: if the CSV header doesn't match the current config, fails loudly rather than silently drifting.
- [update, 2026-08-20] Ran the smokes end-to-end. FS smoke: agent traded (18–29 flips per window), lost badly against B&H in an early-90s bull market. AS smoke: **policy fully collapsed** — went short 100/100 days across both windows with zero flips. Traced to the MultiDiscrete([3, 5]) action space needing more exploration than 2000 timesteps provides. Not a bug, expected under-training behaviour. Pipeline verdict: all CSVs, JSON, dates, position stats, memory histograms landed clean.
- [update, 2026-08-20] Wrote `design_doc.md` at project root — draw-along guide, 7 pictures, step-by-step drawing instructions that let a non-technical reader (or a 10-year-old) sketch the entire pipeline: whole-system view, one-loop-step zoom, walk-forward outer loop, PPO head structure, output consumption, aspirational HTTP API wrapper, proposal-objective checklist.
- [update, 2026-08-20] Created `claude-read/log.txt` as a running record of the longer explanatory answers worth revisiting (walk-forward sanity band, new summary format, CSV+resume design, learning-signal explanation). Append-mode; new sections get added when substantive.
- [ah-ha, 2026-08-20] Realised the matched-budget constraint (κ = k) doesn't need code enforcement — it's a *convention* honoured by config filenames and the run script. Encoding it as an assertion would prevent mismatched-budget experiments if those ever became interesting. Documented in `design_notes.md`.
- [reflection, 2026-08-20] Compact bullet on why AS's memory action is *forced* rather than optional: skipping is a trivially safe option → agent would collapse to always-skip → memory head learns nothing. Force overwrite = every choice has a downstream reward consequence = actual learning signal for the memory head. Full write-up lives in `log.txt`.

---

## 2026-08-25 — First real end-to-end training runs (single seed, k=5)

Ran both `rq2_fs_k5_seed0.yaml` and `rq2_as_k5_seed0.yaml` overnight on local box (CPU, `total_timesteps: 100000`, `max_windows: null`). Both finished cleanly — all 31 walk-forward windows completed for each config. This is the **first time actual RL numbers exist for the project.**

**Headline aggregate metrics (single seed each — treat as directional, not conclusive):**

| Metric               | FS (k=5)        | AS (k=5)        | Buy-and-Hold    |
|----------------------|-----------------|-----------------|-----------------|
| Sharpe (mean ± std)  | +0.412 ± 1.02   | **+0.531 ± 1.24** | +0.841 ± 1.12   |
| Cumret               | +0.0472         | +0.0424         | +0.0875         |
| Max drawdown         | 0.164           | 0.155           | 0.150           |
| Positions L/F/S/flips| 5669/247/1896/1793 | 6048/408/1356/**1191** | — |

- [milestone, 2026-08-25] Both overnight runs completed all 31 walk-forward windows without a single crash. CSV + JSON + resume design proven in practice.
- [ah-ha, 2026-08-25] **AS edges out FS on Sharpe** (+0.531 vs +0.412). Direction matches RQ2 hypothesis. Single-seed hint, not statistical evidence — need multi-seed + all k values before making any claim.
- [ah-ha, 2026-08-25] **Retention diagnostic showing real signal.** AS memory slot overwrite counts across all 31 test windows: `[779, 1264, 2141, 1625, 2003]`. Non-uniform — slot 0 (oldest) is preserved (least overwritten), slot 2 is churned hardest. The AS agent has *learned a preference* about which slots to keep and which to churn. This is exactly the retention-clustering behaviour the proposal §4.2 objective 7 predicted for the diagnostic.
- [observation, 2026-08-25] Neither agent beats buy-and-hold on Sharpe. Expected — the 1990–2024 window is a long-run bull market for the S&P; long-only wins by default in that regime. Not a failure of the memory mechanism; a floor that the *relative* AS-vs-FS claim still lives above.
- [observation, 2026-08-25] **AS is more decisive** — fewer flips (1191 vs 1793) and stronger long bias (6048 vs 5669). Consistent with "memory-driven policy takes fewer, more committed positions" — but could equally be under-exploration. Multi-seed will clarify.
- [observation, 2026-08-25] Smoke-scale policy collapse (2000 timesteps → agent picks one action forever) did NOT reproduce at 100k timesteps. Training length matters far more than the smoke suggested.
- [plan, 2026-08-25] Next: draft multi-seed × multi-k sweep configs (k ∈ {3, 5, 10, 20}, seeds ∈ {0, 1, 2, 3, 4} for both FS and AS = 40 configs). Dispatch to Wits mscluster via SLURM. Local box does the design work; cluster does the compute.
- [open, 2026-08-25] `src/evaluation/retention.py` still not built — the per-slot histogram exists in the CSV column but no aggregation/plot layer yet. Needed for the RQ2 figure in the write-up.

---

## 2026-09-03 — RQ1 strong LSTM baseline (sweep-native, multi-seed) built

- [update, 2026-09-03] Built the RQ1 supervised baseline as a proper HP sweep + multi-seed LSTM — matches what the proposal commits to. Files: `scripts/train_lstm.py` (sweep runner), `configs/rq1_lstm_sweep.yaml` (4 HP combos × 3 seeds = 12 runs across 31 walk-forward windows). Extended `src/baselines/lstm.py` with `weight_decay`, `grad_clip`, and per-window standardisation helpers (`fit_standardiser` / `apply_standardiser`) so obs are z-scored using TRAIN-slice stats only. Full design rationale in `claude-read/log.txt` under the 2026-09-03 heading.
- [decision, 2026-09-03] HP grid informed by WebSearch on 2026-LSTM-for-finance practice: hidden ∈ {64, 128}, num_layers ∈ {1, 2}, dropout 0.2, seq_len 20, grad_clip L2-norm 5.0, weight_decay 1e-5, 15 epochs. Champion identified by mean Sharpe across seeds and reported in `summary.json`.
- [update, 2026-09-03] Built `src/evaluation/retention.py` — reads any AS run's `summary.json` (fast) or `windows.csv` (per-window), reports slot-overwrite fractions plus a scalar concentration score (0 = uniform, 1 = extreme skew) with an auto-verdict line. Closes the "no consumer for retention histogram" open item from 2026-08-25.
- [milestone, 2026-09-03] CLAUDE.md §7 LSTM-baseline-fairness open concern closed by the sweep-native build above.
- [plan, 2026-09-03] Next: run `python scripts/train_lstm.py --config configs/rq1_lstm_sweep.yaml` (~3h CPU) → produces first defensible RQ1 champion + spread. Then multi-seed × multi-k sweep configs for the RQ2 grid; cluster dispatch after that.

---

## 2026-09-08 — Fairness principle for cross-paradigm comparison

Sat with the FS-vs-LSTM comparison shape today and one worry surfaced: if I sweep hyperparameters on one side but not the other, I'm boosting a model against an unboosted one — same category of mistake as any unfair benchmark. Wrote out the reasoning so it doesn't drift.

- [reflection, 2026-09-08] The fairness rule I'm going to hold to: **equal-quality champions get compared, not same-knob-count champions.** Each paradigm gets its own tuning treatment appropriate to its family, then champions face off.
- [decision, 2026-09-08] Concretely for my three configurations:
  - **LSTM (SL baseline):** gets its own HP sweep (hidden, num_layers, dropout) × multi-seed. Champion by mean-Sharpe-across-seeds.
  - **FS (RL memory control):** PPO base config held identical to AS. Only memory budget k is swept — over a predefined grid k ∈ {3, 5, 10, 20}. Multi-seed per k. Champion by mean-Sharpe-across-seeds.
  - **AS (primary contribution):** PPO base config held identical to FS. Only κ swept over the same {3, 5, 10, 20}. Multi-seed per κ. Champion by mean-Sharpe-across-seeds.
- [decision, 2026-09-08] What stays fixed across FS and AS: PPO learning rate, batch size, network widths, everything except the memory mechanism. That's what isolates the RQ2 claim — any difference between FS-champion and AS-champion is attributable to the wrapper, not to some HP that only one side got tuned on.
- [decision, 2026-09-08] The FS k-grid is set **before** I look at any results. No "try k=5, if it doesn't beat LSTM try k=10, if that doesn't beat LSTM try k=20" — that's iterative early-stopping on the test set, same family as p-hacking. Predefine {3, 5, 10, 20}, run all four, pick best mean at the end.
- [reflection, 2026-09-08] RQ1 outcome and RQ2 outcome are independent. Even if FS loses to LSTM (RQ1 says "no, RL doesn't beat SL here"), RQ2 (AS vs FS at matched budget) is still the primary contribution — the interesting claim is about memory mechanisms, not paradigms. Don't let RQ1 disappointment collapse the RQ2 experiment.

---

## 2026-09-08 — Design + evidence hydration (thickening earlier decisions with the reasoning behind them)

Filling in the "why" behind decisions that got made quickly in earlier sessions but weren't fully written down in this log. Chronological in-file order preserved; each bullet is tagged with the original date it belongs to for traceability.

### Walk-forward sizing (originally decided 2026-08-20)

- [reasoning, 2026-08-20] The 5-year train / 1-year test choice sits at the *upper end* of the 2-to-5:1 train-to-test ratio range that quant-finance practice treats as reasonable. Chose the conservative end because: (a) financial regimes are long-lived — 5 years is enough to include multiple regimes per train slice; (b) 1-year test is dense enough (252 days) that per-window Sharpe is not dominated by a handful of outlier days; (c) with 9188 days total this still yields ~31 windows, well inside the 6–8 minimum and 12–20 preferred bands that published walk-forward practice recommends.
- [reasoning, 2026-08-20] Non-overlapping test slices (`step = test_size`) chosen because it's the cleanest setup — every test day appears in exactly one test slice, so aggregate metrics don't double-count. Overlapping tests would inflate window count but correlate results, and the statistical power gain would be spurious.
- [decision, 2026-08-20] Sanity band recorded as a comment in `walk_forward.py` so future-me knows what range to stay in without re-deriving. Ranges outside 2:1–5:1 or fewer than ~6 windows need a written reason.

### Why AS's memory action is forced (originally decided 2026-08-20)

- [reasoning, 2026-08-20] The AS wrapper does NOT expose a "skip" slot — every step, the agent must overwrite one slot. Kept this design (matching the reference paper) because: RL agents only learn from reward differences between choices. If skipping were an option, it would be a trivially safe action (nothing lost, nothing risked). The memory head would collapse to always-skip because that choice always looks neutral, and no learning signal would ever reach the memory decision. **The forced-overwrite constraint IS the learning signal** for the memory head.
- [note, 2026-08-20] If a future variant experiment wants to test "does skip help?", the change is small (add a k+1 slot to the memory head) but it's a research question in its own right, not a fix for the base design.

### CSV+JSON output pattern (originally decided 2026-08-20)

- [reasoning, 2026-08-20] Emitting BOTH `windows.csv` and `summary.json` per run is deliberate: JSON is human-readable narrative (config embedded, easy to debug a single run), CSV is tidy analysis dataset (one `pd.concat` across many runs gives one DataFrame for the cross-run comparison plots I'll need for RQ2). Different consumers, different formats — no attempt to force one file to serve both.
- [reasoning, 2026-08-20] Cluster-friendly write pattern (append-and-flush per window, atomic `.tmp` → rename for JSON) chosen because Wits mscluster jobs can be killed by SLURM at any point. Partial CSVs must stay valid up to the last completed window, so a resumed job can pick up cleanly.

### Matched-budget as convention, not code (originally decided 2026-08-20)

- [reasoning, 2026-08-20] The "κ = k" matched-budget constraint for RQ2 is enforced by config filename pairing (`rq2_fs_k5_seed0.yaml` ↔ `rq2_as_k5_seed0.yaml`), not by any code assertion. Deliberate — encoding it as a code invariant would prevent mismatched-budget experiments if those ever become interesting research questions later.

### LSTM HP grid grounding (originally decided 2026-09-03)

- [reasoning, 2026-09-03] WebSearch on 2026-09-03 for "LSTM financial time-series forecasting best practices" surfaced these convergent recommendations: hidden 128–256 for financial forecasting (recommended range), 1–2 layers, dropout ≈ 0.2 as the widely-accepted compromise between overfit and accuracy, gradient clipping L2-norm ≈ 5.0 as literature standard.
- [decision, 2026-09-03] Grid chosen: hidden ∈ {64, 128} spans from budget-conscious to literature-lower-bound; num_layers ∈ {1, 2}; dropout fixed at 0.2 per convention; seq_len fixed at 20 (~1 trading month); grad_clip = 5.0. Deliberately compact — LSTM is a control for RQ1, not the research focus, so a 4-combo grid is proportionate. If the champion sits at the top edge of the grid (128, 2 layers), that's a signal to widen the sweep before final report.

### Seed count sanity check (originally decided 2026-09-08)

- [reasoning, 2026-09-08] WebSearch on 2026-09-08 for random-seed conventions in ML/SL confirmed: 5 seeds is bare minimum for reportable results, 7–10 is safer, 3 is prototyping only. Deep-RL papers using ≤5 seeds have been criticised as underpowered.
- [decision, 2026-09-08] Bumping the LSTM sweep from 3 → 7 seeds sits comfortably inside the defensible 5–10 band. If asked "why 7?" the answer is "5–10 is standard, started at 3 for prototyping, bumped to 7 for the reportable version." Same reasoning will apply to the FS and AS sweeps in RQ2.
- [decision, 2026-09-08] Add-seeds-via-resume approach chosen over fresh-sweep-from-scratch: edit the YAML to `seeds: [0..6]`, rerun the same command, resume logic only trains the new seeds 3–6 and reuses seeds 0–2 already on disk. This is not cheating — the final mean ± std aggregates ALL seven seeds, not a cherry-picked subset. Same statistical result as running seven from scratch, half the wall-clock.

### Plot design (originally decided 2026-09-08)

- [reasoning, 2026-09-08] Two plot families chosen for the LSTM sweep: (1) per-HP-shape metric-timeline with median-across-seeds line + min-max shaded band + buy-and-hold reference — shows both when the model did well and how sensitive it was to seed; (2) aggregate bar-chart with error bars across HP shapes — the between-config champion picker a reviewer expects to see. Deliberately sparse — no ROC curves, no confusion matrices, no threshold analysis. LSTM is a control, not the star.
- [decision, 2026-09-08] Plots for each seed-count snapshot live under `results/{run}/seed_{N}/` so different snapshots (seed_3, seed_7, later seed_10 if needed) coexist as historical checkpoints. No README next to the plots — the sweep YAML in the run's `summary.json` is the config docs; a README would double-document.

### Data leakage boundary (originally discussed 2026-09-08)

- [reasoning, 2026-09-08] Re-verified that the current setup is leakage-clean on all the axes that matter for reviewer scrutiny: walk-forward splits are strictly train-before-test in time; per-window standardisation stats (mean, std) computed from train-slice only; HP champion selected by mean-across-seeds AFTER all training is done, not by peeking at test-slice numbers mid-sweep; multi-seed spread reported honestly without cherry-picking. The only remaining gap is the CLAUDE.md §3 final-2-year holdout — not enforced in code yet, still an open item to add before final reporting runs.
- [reflection, 2026-09-08] The instinct that "sweeping HPs on one side is unfair boosting" is a real one and the fix is not "don't sweep" but "sweep both sides equally." Each paradigm gets its own tuning treatment appropriate to its family (LSTM: hidden/layers/dropout grid; FS: k grid; AS: κ grid, PPO base held identical between FS and AS). Champions of each face off. That is what "equal-quality comparison" means in practice.

---

## 2026-09-08 (later) — Sweep-native RL runner + cluster prep

Bunch of build activity same day as the fairness reflection. Consolidated so RQ2 sweeps can actually run.

- [update, 2026-09-08] Rewrote `scripts/train.py` as a sweep-native runner mirroring `train_lstm.py`'s pattern. Config now takes `memory.k_grid` (list) + `seeds` (list), and the runner iterates every (k, seed, walk-forward window) triplet with a fresh PPO. Adds SB3 training-curve capture per window (`training_curves/w{w}_k{k}_s{seed}/progress.csv`) via `sb3_configure_logger` — enables "did it learn?" diagnostics later. windows.csv gains `k`, `seed`, `memory_type`, `run_id` columns; `mem_slot_0..19` padded to max k for stable schema across the sweep. Resume logic: (k, seed, window) triplets already in windows.csv are skipped.
- [update, 2026-09-08] Wrote `configs/rq_rl_fs_sweep.yaml` and `configs/rq_rl_as_sweep.yaml` — matched-budget pair, identical PPO block, k_grid ∈ {3, 5, 10, 20}, seeds [0..6]. Only `memory.type` differs between the two. Comment blocks in both YAMLs explicitly forbid changing the PPO block in one without matching the other.
- [update, 2026-09-08] Wrote three SLURM batch scripts for Wits mscluster: `scripts/slurm_train_fs.sh`, `slurm_train_as.sh`, `slurm_train_lstm.sh`. All use CPU (MLP+PPO is CPU-optimal per SB3), `--time=48:00:00` for RL sweeps, `--time=12:00:00` for LSTM. Placeholder partition `stampede` — needs adjusting to whatever Wits calls its CPU partition.
- [decision, 2026-09-08] `.gitignore` cleanup for cluster hygiene: docs, notes, slides not needed on the training environment. Added `*.md`, `Slides/`, `claude-read/` — but explicitly NOT `*.txt` because `requirements.txt` is a `.txt` file and the cluster clone needs it. Historical tracked files (proposal.md, claude-read/log.txt) stay tracked unless explicitly `git rm --cached`'d.

## 2026-09-09 — LSTM RQ1 done, cluster onboarding, AS sweep in flight locally

- [milestone, 2026-09-09] **LSTM strong RQ1 baseline COMPLETE at 7 seeds.** `results/rq1_lstm_sweep/` — 868 rows in windows.csv (4 HP × 7 seeds × 31 windows), summary.json emitted, champion identified. seed_7/ plots generated (aggregate bar + 4×2 per-HP timelines). This closes RQ1's supervised-side deliverable — waiting on FS-champion to complete the paradigm comparison.
- [update, 2026-09-09] Cluster onboarding attempted for the RQ2 RL sweeps. Only `anaconda/anaconda-2025-12-2` is available as a module on Wits mscluster (no standalone `python/3.10`). Default Python is 3.14, which is incompatible with our torch 2.4.1 / numpy 1.24.3 pins. Workflow settled on: `module load anaconda/anaconda-2025-12-2` → `conda create -n lwtr python=3.10 -y` → `conda activate lwtr` → pip install requirements. SLURM scripts still say `source .venv/bin/activate`; needs updating to `conda activate lwtr` before cluster runs actually work.
- [update, 2026-09-09] Started `rq_rl_as_sweep` locally on the Windows box as a backup path while cluster env is stabilised. Roughly 17:00 on 2026-09-08 launch; ~46-hour wall-time estimate (868 windows × ~3.7 min/window). Killed at some point overnight, resumed today with the same command — resume logic picked up cleanly at ~27% (237/868 rows in windows.csv, currently on `as_k5_seed0`).
- [reflection, 2026-09-09] Local AS sweep buys us data even if cluster stalls, and cluster wins if it comes online because it can run FS + AS in parallel. Not committed to one path — both are moving.
- [plan, 2026-09-09] Once AS local run completes, kick off FS locally OR — if cluster is up by then — dispatch FS via SLURM there. Whichever finishes first, run the RQ2 comparison plots.

---

## 2026-10-07 — Open follow-up

- [open, 2026-10-07] `plots/data_integrity/disclaimer.tex` title + preamble need a clean pass against `proposal.md` as the source of truth: official working title (likely "Learning What to Remember for Time-Series Prediction Using Reinforcement Learning"), author block, declaration-style framing, and a tightened preamble consistent with how the proposal is set up. Not actioning now — flagged so it doesn't get forgotten.
