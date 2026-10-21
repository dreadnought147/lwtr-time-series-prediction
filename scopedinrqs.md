# Scoped Research Questions

**Date scoped:** 2026-06-24
**Status:** Scope reduction from the original proposal (submitted 2026-05-18). Four configurations → three. Three research questions → two. To be discussed with supervisor before implementation locks in.

---

## RQ1 — Paradigm: Does RL Stand a Chance Against a Strong Sequential Supervised Baseline?

**Question:** On a non-stationary financial time-series positioning task, can a reinforcement learning agent with Frame Stacking match or outperform a strong LSTM-based sequential supervised predictor (with a threshold trading rule for converting forecasts to positions)?

**Why LSTM as the baseline:** LSTM is the established, well-understood sequential supervised predictor in financial RL literature (used as the reference class in Moody & Saffell and follow-ups). It captures the "top-of-the-line supervised sequential model" framing without expanding scope into Transformer or foundation-model territory that would blow the crunch budget.

**Objectives that answer this question** *(mapped from the proposal, with numbers preserved for traceability)*:

1. **Build a financial time-series environment.** Wrap the aligned VIX + S&P 500 dataset into a data-agnostic Gymnasium environment with observation space, action space, and reward function.
2. **Implement a supervised learning baseline.** Train an LSTM forecaster on the observation stream to predict next-day return; convert forecasts to positions via a fixed threshold rule.
4. **Implement a Frame Stacking RL baseline.** Train PPO agents with Frame Stacking at stack sizes k ∈ {3, 5, 10, 20}, sharing the env and PPO configuration.
6. **Compare using financial + RL metrics.** Cumulative return, Sharpe ratio, maximum drawdown as primary; sample efficiency and training-time memory footprint as secondary.

---

## RQ2 — Mechanism (Primary Contribution): Does Adaptive Stacking Beat Frame Stacking on Non-Stationary Financial Time-Series?

**Question:** At a matched memory budget (κ = k), does Adaptive Stacking outperform Frame Stacking when the environment is a non-stationary, noisy financial time-series stream, rather than the stationary, episodic benchmarks (T-Maze, MiniGrid-Memory, Rubik's Cube, FetchReach) on which it has been validated so far?

**Why this is the primary contribution:** the FS-vs-AS comparison has already been answered in stationary, controlled environments (Nangue Tasse et al., 2025). It has not been answered in noisy, non-stationary time-series. Whether the sparsity assumption (κ ≪ k*) transfers to financial data — where informative signals may be diffuse rather than concentrated — is the open empirical question this research investigates.

**Objectives that answer this question:**

1. **Build a financial time-series environment.** *(Shared with RQ1.)*
4. **Implement a Frame Stacking RL baseline.** *(Shared with RQ1; used as the head-to-head control.)*
5. **Implement Adaptive Stacking.** Train PPO agents with Adaptive Stacking at matched memory budgets κ ∈ {3, 5, 10, 20}, mirroring the Frame Stacking sweep so the comparison is honest and controlled.
6. **Compare using financial + RL metrics.** *(Shared with RQ1.)*
7. **Analyse the learned memory policy.** Inspect which observations the Adaptive Stacking agent chooses to retain over time — do retentions cluster around identifiable market events (VIX spikes, regime transitions), or are they uniform across the trailing history?
8. **Report a comparison of memory mechanisms in the financial setting.** Situate results against the controlled-benchmark numbers in Nangue Tasse et al., and identify whether the sparsity assumption holds in this domain.

---

## Dropped from the Proposal

**Original RQ2 (dropped):** *"Does giving the reinforcement learning agent a fixed memory window through Frame Stacking improve performance over a memoryless agent that observes only the current timestep?"*

**Original Objective 3 (dropped):** *"Implement a no-memory reinforcement learning baseline."*

**Reasoning:**
- The memoryless-vs-FS comparison has already been established in prior work (DQN and subsequent literature on partially observable tasks). Replicating it on financial data was framed in the proposal itself as *"a sanity check more than a contribution."*
- The primary contribution lives at the FS-vs-AS comparison, not at the memoryless baseline.
- Time budget: dropping this configuration removes one full PPO sweep from the July workload, freeing time for the AS integration in August (which is where the contribution actually lives).
- **Trade-off accepted:** cannot make the standalone claim *"memory helps in financial RL"* — only *"learned memory beats fixed memory in financial RL"*. That is the claim that matters.

---

## Configuration Summary (Post-Scope)

| # | Configuration            | Role                          | Answers  |
|---|--------------------------|-------------------------------|----------|
| 1 | Supervised LSTM + threshold rule | Sequential supervised baseline | RQ1      |
| 2 | PPO + Frame Stacking, k ∈ {3, 5, 10, 20} | RL memory-mechanism control    | RQ1, RQ2 |
| 3 | PPO + Adaptive Stacking, κ ∈ {3, 5, 10, 20} | Primary contribution           | RQ2      |

Buy-and-hold on S&P 500 is reported alongside all three as a financial sanity floor. No configuration that fails to beat buy-and-hold earns a "useful trader" label, independent of how the memory comparison lands.

---

## Action Items Before Implementation Locks

- [ ] Discuss scope reduction with supervisor (Geraud Nangue Tasse).
- [ ] Confirm LSTM is an acceptable "top-of-the-line supervised sequential baseline" or whether he wants a Transformer variant.
- [ ] Confirm the workplan revision: memoryless PPO removed from June; June is now supervised LSTM build and Frame Stacking scaffold.
