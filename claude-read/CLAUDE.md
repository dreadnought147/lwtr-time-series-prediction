# CLAUDE.md — Adaptive Stacking for Non-Stationary Financial Time-Series RL

> **Purpose of this file**
> This is the persistent context file for Claude Code on this project. Claude Code has no memory between sessions — this file is what gets read at the start of every session so the agent isn't starting blind. It must live inside this project directory, not in any global or home-directory config location, so the project's full context stays portable with the repo itself.
>
> **Rule for this file: append-only.** New information gets added as a dated line under the relevant section, format `- [update, YYYY-MM-DD] whatever changed`. Never remove, rewrite, or silently overwrite an existing line. If something here turns out to be wrong, add a correction below it, don't delete the original.

---

## 1. Project Identity

Working title: "Learning What to Remember for Time-Series Prediction Using Reinforcement Learning."

Core question: does explicit, learned memory management (Adaptive Stacking) outperform fixed-window memory (Frame Stacking) and a supervised LSTM baseline, on a non-stationary, noisy financial time-series positioning task?

Student: Benny (Banele Mkhize), Honours CS, University of the Witwatersrand, part-time.
Supervisor: Geraud Nangue Tasse — who is also the author of the Adaptive Stacking paper this project builds on.

## 2. Current Scope (locked 2026-06-24, pending final supervisor sign-off)

Originally three research questions and four configurations. Scoped down to two RQs and three configurations after a deliberate cut:

- **RQ1 (paradigm):** on this task, can PPO + Frame Stacking match or beat a strong, tuned LSTM supervised baseline (with a threshold trading rule)?
- **RQ2 (mechanism — the primary contribution):** at a matched memory budget (κ = k), does Adaptive Stacking outperform Frame Stacking specifically in a non-stationary, noisy financial time-series stream, rather than the stationary, episodic benchmarks (T-Maze, MiniGrid-Memory, Rubik's Cube, FetchReach) it's been validated on so far?

**Dropped:** the original RQ2, "does Frame Stacking beat a memoryless agent," and its dedicated no-memory baseline. Reasoning on record: that comparison is already established in the DQN/POMDP literature, it was framed in the original proposal itself as "a sanity check more than a contribution," and dropping it freed a full PPO sweep's worth of July time for the actual Adaptive Stacking integration in August. Accepted trade-off: can't claim "memory helps" standalone anymore, only "learned memory beats fixed memory" — which is the claim that actually matters here.

**Three configurations:**
1. Supervised LSTM + threshold rule → answers RQ1
2. PPO + Frame Stacking, k ∈ {3, 5, 10, 20} → control for both RQ1 and RQ2
3. PPO + Adaptive Stacking, κ ∈ {3, 5, 10, 20}, matched to (2) → answers RQ2

Buy-and-hold S&P is reported alongside all three as a sanity floor — nothing gets called a "useful trader" if it can't beat that, independent of how the memory comparison lands.

**Metrics:** cumulative return, Sharpe ratio, max drawdown (primary); sample efficiency, training-time memory footprint (secondary); plus a qualitative read of what the Adaptive Stacking policy actually chooses to retain over time — does it cluster around VIX spikes / regime transitions, or is retention roughly uniform across the trailing history?

## 3. Environment & Data Design

- Observation: VIX-derived series (currently `vix_close` → log return). Action: a position in the S&P (long/short/flat, or continuous weight). Reward at step t = position(t) × log-return of the S&P from close(t) to close(t+1).
- The asset return the agent is scored against is deliberately **not** shown to the agent as an observation — it only ever appears as the reward consequence of an action already taken. `shift(-1)` is used to align the reward at day t with the return from t→t+1, not t-1→t, so the agent is never scored against a return that was already knowable when it acted. This was the trickiest piece of the whole pipeline to get right — worth double-checking on every change to `transforms.py`.
- Log returns are used throughout (not simple percent returns): they're additive across time, symmetric around zero, keep modeled prices from going negative, are closer to normally distributed, and match the convention in Moody & Saffell and the broader financial-RL lineage this project sits in.
- Walk-forward evaluation, not a single train/test split. A cutoff date is fixed *before* any development happens; the most recent slice is touched exactly once, at the very end. This is treated as a hard rule, not a guideline — breaking it invisibly invalidates the results in a way nothing downstream would catch.
- Any scaling/normalization statistics must be computed only from each split's *training* window, never from the full dataset — computing them globally leaks future information into the past, the same category of mistake as an early train/test cutoff violation.

## 4. Architecture & Repo Conventions

```
src/{data, envs, memory, agents, baselines, evaluation}
configs/     one YAML per experiment: env, memory type, k/κ, seed
scripts/     train.py, evaluate.py
results/     logs, checkpoints, plots
data/{raw, processed}
```

**Open design question, not yet resolved in code:** Frame Stacking is a stateless observation-space transform — it sits outside the agent, needs no learning, and the agent never "knows" it happened. Adaptive Stacking is learned memory management — the choice of what to retain lives inside the agent's policy. Both currently sit under `src/memory/`, described loosely as "wrappers around env or agent," but that phrase papers over a real architectural difference. Before this gets built out further: what shared interface do both satisfy so they're swappable via one YAML key, and — more importantly — where does "matched memory budget k = κ" actually get enforced so the RQ2 comparison stays honest? This needs an explicit answer, not an implicit one.

## 5. Literature Grounding

Core annotated bibliography (6 sources, IEEE style):
- Nangue Tasse et al. (2025) — Adaptive Stacking, the central paper this project extends
- Sutton & Barto (2018) — RL foundations
- Mnih et al. (2015) — DQN, fixed frame stacking at Atari scale
- Hochreiter & Schmidhuber (1997) — LSTM, implicit gated memory
- Moody & Saffell (2001) — Recurrent Reinforcement Learning for direct trading, Sharpe-ratio reward, S&P 500; the closest existing precedent for doing RL directly on financial time-series
- Schulman et al. (2017) — PPO

Plus, read during lit-review expansion:
- Demir (2021) — Self Memory Management: push/skip memory control driven by an *intrinsic* curiosity reward for rare observations. Useful contrast case for RQ2's related work: Demir motivates memory retention via novelty, Adaptive Stacking lets the task reward alone teach the agent what's worth keeping. Both reject fixed windows; the mechanism differs.

**Literature gap, as identified:** Adaptive Stacking has not been evaluated on real-world, continuous, non-stationary financial time-series. Existing RL memory approaches use either fixed frame stacking (rigid, discards potentially critical past events) or implicit gating via LSTM (compresses but doesn't explicitly select). No prior work gives an agent explicit control over what to remember in a financial forecasting/positioning context.

## 6. Where the Actual Edge Is — Say This Plainly

The AS-vs-FS comparison itself is *not* new — Nangue Tasse et al. already answered it, on stationary, episodic, controlled benchmarks. What's actually being tested here is narrower and sharper: does the *sparsity assumption* underlying Adaptive Stacking (that a small κ ≪ k\* suffices) survive contact with a domain where the environment is exogenous (the agent's actions don't affect what happens next), noisy, and non-stationary, rather than clean and episodic? That's the open empirical question. This was flagged as early as the Week 2 proposal deck (18 May 2026) as the single biggest project risk — and it cuts both ways: if the sparsity assumption *doesn't* hold on financial data, that is itself a legitimate, honest, reportable finding, not a failed project. This was raised with Geraud directly as an accepted possible outcome. Don't let this framing get lost or flattened into "we're just redoing AS vs FS."

## 7. Known Open Concerns / Standing Watch Items

- **Dataset span.** Roughly 20 years of daily VIX/S&P data. Flagged (Aug 2026) as potentially short relative to standard practice elsewhere in comparable published RL-for-finance and financial-ML work. This needs checking against the actual literature and against how comparable published benchmarks size their data — without expanding scope past what's locked in Section 2. Treat this as one example of the kind of thing Section 8 below should be catching, not the only item on the list.
- **Memory interface unification** (Section 4) — FrameStack and AdaptiveStack need a shared, config-selectable interface with a single place where the matched-budget constraint is enforced. Not yet resolved in code as of the current smoke-testing phase.
- **LSTM baseline fairness.** For RQ1 to mean anything, the LSTM needs to be a genuinely *tuned* baseline — its own small hyperparameter search, multiple seeds reported with spread not a single number, gradient clipping, and scaling statistics computed per walk-forward split only. A single untuned run answers "does PPO beat a lazy LSTM," not RQ1 as actually written.

## 8. Standing Instruction — Surface Concerns Like Section 7 Automatically

When running the search prompts in Section 9, don't just answer them narrowly and move on. Actively watch for anything that puts a methodology choice already locked into this project (data span, architecture, evaluation protocol, hyperparameter ranges, reward design, whatever) at odds with common published or industry practice. When something surfaces, add it — dated — under a new `## Concerns Bubbled Up From Research` section at the bottom of this file, rather than silently absorbing it or leaving it unmentioned. Section 7's dataset-size item is an example of the *category* of thing to watch for, not a checklist to clear and stop.

## 9. Search Prompts to Run

Weighted roughly 80% CS/RL, 20% finance — the domain is financial, but the contribution is a memory-mechanism result, and grounding should reflect that balance.

**CS / RL (80%):**
1. PPO sample efficiency on non-stationary time-series environments
2. learned memory selection reinforcement learning partial observability benchmarks
3. frame stacking vs recurrent policy comparison POMDP
4. catastrophic forgetting / non-stationarity in deep RL — recent approaches
5. reward shaping Sharpe-ratio-style objectives in RL trading agents
6. sample complexity PPO small or limited-size training data
7. adaptive / learned memory mechanisms in sequential decision-making, 2025–2026
8. exogenous state / action-independent transitions in reinforcement learning formulations

**Finance / industry practice (20%):**
9. typical historical data window length in quantitative trading backtests, industry practice
10. how much historical data is considered sufficient for backtesting statistical significance
11. VIX and S&P 500 dataset lengths used in published financial machine-learning papers
12. walk-forward analysis as standard practice in quant finance, common window sizes

## 10. Deliverables Produced So Far

- Annotated bibliography, 6 sources, LaTeX (IEEEtran, migrated to biblatex/biber)
- Literature review draft (due 19 April 2026)
- Research proposal submitted 18 May 2026 (abstract, proposal draft, research questions)
- Scope-reduction document, 24 June 2026 (Section 2 above)
- Weekly accountability presentation decks (at least through Week 2)
- Data pipeline: VIX + S&P loader, date alignment, log-return transforms
- Gymnasium `TimeSeries` env — in progress
- PPO/SB3 smoke test — environment handshake confirmed (not a real training run)
- Cluster onboarding — Wits mscluster, SLURM-based, GPU partitions `stampede`/`bigbatch`/`biggpu`

## 11. Working Conventions

- Append-only, dated entries, format `- [update, YYYY-MM-DD] ...`
- Plain, direct English in all guidance and comments, no verbosity
- Do not write academic prose on Benny's behalf — skeletons, structure, comment blocks, and explicit correction of definitional/theoretical errors only. This convention was established during the literature-review phase and still holds for the write-up.
- IEEE citation style, IEEEtran/biblatex+biber as already configured in the LaTeX setup

## 12. Persistent Project Memory — Where It Lives

Claude Code: keep doing your normal persistent-memory thing. In addition to that, this project's durable context lives in two files in this repo — this `CLAUDE.md` and the companion dated log `research_log.md` — so it stays in reach for me directly, not only in your own memory. When you learn something across a session worth persisting for this project, make sure it lands in the right one of these two files (context and decisions here, dated journey entries in the log), on top of whatever you'd normally do.

- [update, 2026-10-19] Project memory for this work lives **in the repo**, not in Claude's global per-project memory folder. Write durable notes here or in `research_log.md`.

## 13. Planned Retrospective Agent

- [plan, 2026-10-19] Benny intends to build a **separate external agent** that points at this project directory and runs a retrospective. Not built yet; recorded so it isn't lost.

  **Scope that agent should cover:**
  - What went wrong across the project lifecycle.
  - What went well.
  - Whether this was a good start for his Honours overall.
  - Whether quick fixes existed before submission that would have meaningfully improved results.

  **Context behind the ask:** raised while the results felt weaker than hoped — neither memory mechanism beat the buy-and-hold floor (0.841), and the RQ2 margins between Adaptive Stacking and Frame Stacking sit inside one standard deviation. He wants an honest external read rather than a self-assessment, and deliberately deferred it rather than going back and forth during crunch.

  **How to apply:** if he later asks to build a retrospective agent, or asks "was this project any good", surface this section and the scope above. Do not volunteer retrospective analysis while he is still mid-submission.

## 14. Visualisation Tooling

- [plan, 2026-10-20] Benny wants "TLog Tensor bot" (as dictated; read as TensorBoard) used for visualisation going forward. Context: SB3 training curves currently land as `progress.csv` under `results/*/training_curves/`, `tensorboard==2.20.0` is already in `requirements.txt`, and `tb_logs/` is gitignored. Not wired up yet. If the term meant something else, correct this line.
