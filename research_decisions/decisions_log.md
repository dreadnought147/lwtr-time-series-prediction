# Research Decisions — Running Log

How this file works:

- Append-only. New entries go at the bottom with the next ID and a date. Never
  rewrite or delete an entry — if a decision changes, add a new entry and set
  the old one's Status to "superseded by Dxxx".
- One entry per decision, in this shape:
  Decision / Backing / Citation / Why valid / Status.
- "Backing" is the reason from the literature or from our own data. "Why valid"
  is why that backing actually licenses the decision in *this* setting.
- Status is one of: in force, superseded by Dxxx, open (not yet decided),
  deviation (we knowingly depart from the source).
- Obvious decisions get logged too (the method under test, the base algorithm,
  the data) so the log is a complete record, not a list of edge cases.
- Dates before 2026-10-21 are reconstructed from the repo (proposal, slides,
  configs, research_log.md, takeoff/status.md); from 2026-10-21 onward they are
  logged as the session happens.

Full citation strings are at the bottom under References.

---

## D001 — Adaptive Stacking is the method under test (2026-04)
- Decision: the project evaluates Adaptive Stacking (AS), an explicit memory
  action over a bounded stack of size κ, against the standard alternative.
- Backing: AS is validated only on stationary, episodic benchmarks (T-Maze,
  MiniGrid-Memory, Rubik's Cube, FetchReach), with a convergence result proved
  for stationary k*-Markov environments.
- Citation: Nangue Tasse et al. (2025).
- Why valid: the untested transfer to noisy, non-stationary streams is a real
  and narrow gap, and the method's own κ ≪ k* premise is falsifiable on data —
  so both a positive and a negative outcome are informative.
- Status: in force.

## D002 — Frame Stacking is the control (2026-04)
- Decision: compare AS against Frame Stacking (FIFO window of the last k
  observations) at matched budget κ = k.
- Backing: frame stacking has been the default memory mechanism in deep RL
  since DQN, where the last 4 frames are concatenated into one input.
- Citation: Mnih et al. (2015); Nangue Tasse et al. (2025) use the same control.
- Why valid: it is the mechanism AS claims to improve on, so it is the
  comparison that carries the contribution; using the same control as the
  source paper keeps our result commensurable with theirs.
- Status: in force.

## D003 — PPO as the base RL algorithm (2026-04)
- Decision: all RL configurations use PPO.
- Backing: PPO's clipped objective keeps updates near the previous policy,
  which stabilises on-policy training; it is also the algorithm used in the AS
  experiments.
- Citation: Schulman et al. (2017); Nangue Tasse et al. (2025).
- Why valid: matching the source paper's algorithm means any FS-vs-AS
  difference is attributable to the memory mechanism rather than to a new
  algorithmic variable.
- Status: in force.

## D004 — LSTM as the supervised baseline for RQ1 (2026-05-18, narrowed 2026-06-24)
- Decision: the supervised arm is an LSTM forecaster of next-day return plus a
  threshold rule converting forecasts to positions.
- Backing: LSTM is the established sequential supervised predictor in financial
  RL work and the reference class that the memory literature argues against;
  Transformer/foundation-model baselines were ruled out on compute.
- Citation: Hochreiter & Schmidhuber (1997); Moody & Saffell (2001); Hambly,
  Xu & Yang (2023).
- Why valid: RQ1 only needs a credible sequential supervised predictor, not the
  state of the art; LSTM is the one the literature already treats as the
  comparison point for learned memory.
- Status: in force. Caveat: see D027, D028 — the baseline as implemented is
  handicapped and must be fixed or caveated before the RQ1 claim is made.

## D005 — VIX as the observation stream, S&P 500 as the traded underlying (2026-05)
- Decision: the agent observes VIX and takes positions on the S&P 500.
- Backing: VIX is the market's forward-looking volatility expectation (the
  "investor fear gauge") with a clear regime structure — long mean-reverting
  low-volatility stretches punctuated by spikes. Pure VIX-to-VIX forecasting was
  rejected because a prediction has no sequential consequence, so the RL framing
  would earn nothing over the supervised baseline.
- Citation: Whaley (2000, 2009); Moody & Saffell (2001) for the position-taking
  formulation.
- Why valid: it creates a genuine sequential decision problem (positions
  accumulate, costs compound, drawdowns matter) while keeping the data to two
  daily series, and it is the setting where the sparse-signal-in-noisy-history
  pattern AS depends on is most plausible in finance.
- Status: in force. Caveat: D008 undercuts the regime-structure half of this
  rationale.

## D006 — Discrete action set {short, flat, long} = {-1, 0, +1} (2026-05)
- Decision: three discrete positions; continuous position sizing rejected.
- Backing: the canonical direct-RL trading formulation uses exactly this
  discrete position set.
- Citation: Moody & Saffell (2001).
- Why valid: it keeps the number of moving variables down — adding continuous
  sizing would introduce a second difference between configurations and muddy
  the memory-mechanism comparison, which is the load-bearing part.
- Status: in force.

## D007 — Reward = position x next-day return − cost x |position change| (2026-08)
- Decision: per-step reward is the realised PnL of the held position minus a
  transaction cost charged on position changes.
- Backing: direct-reinforcement trading scores the policy on realised,
  cost-adjusted PnL rather than on forecast error, which is what makes the
  sequential framing meaningful.
- Citation: Moody & Saffell (2001); Hambly, Xu & Yang (2023).
- Why valid: it is the same objective the financial metrics are later computed
  on, so training target and evaluation metric are consistent.
- Status: in force.

## D008 — Log returns throughout, including for the observation stream (2026-08)
- Decision: both the observation series and the asset series are converted to
  log returns.
- Backing: log returns are additive across time, symmetric around zero, cannot
  imply a negative price, and are the convention in the lineage this work builds
  on; additivity is what makes cumulative return a simple sum.
- Citation: Campbell, Lo & MacKinlay (1997); Moody & Saffell (2001).
- Why valid for the asset series: the reward and all three financial metrics
  depend on additivity.
- Why questionable for the observation series: differencing removes the VIX
  *level*, which is the part of D005's rationale that carries regime
  information. This is a known tension, not a settled decision.
- Status: open — see D031.

## D009 — Transaction cost fixed at 1 basis point per unit of position change (2026-08)
- Decision: `transaction_cost: 0.0001` in every config.
- Backing: a non-zero cost is required so that flipping positions is not free;
  the exact level was set low to avoid suppressing trading entirely.
- Citation: Moody & Saffell (2001) (costs as a first-class term); Arnott, Harvey
  & Markowitz (2019) (costs and robustness in backtest protocol).
- Why valid: it is identical across all configurations, so it cannot bias the
  FS-vs-AS comparison. It is optimistic in absolute terms, so absolute
  profitability claims need a cost sensitivity sweep.
- Status: in force, with a sensitivity sweep outstanding.

## D010 — Walk-forward evaluation: 1260-day train, 252-day test, non-overlapping (2026-08)
- Decision: 5 trading years train, 1 trading year test, step = test size,
  yielding 31 windows over 1990-2026; a fresh model is trained per window.
- Backing: rolling-origin evaluation is the standard out-of-sample protocol for
  time series, because a single split hides regime change; our own data shows
  the variance of returns drifting across the span (volatility clustering in
  2008 and 2020) while the mean stays near zero.
- Citation: Tashman (2000); Bergmeir & Benítez (2012); López de Prado (2018);
  Hambly, Xu & Yang (2023).
- Why valid: 31 windows is well above the 6-8 minimum we set ourselves, the 5:1
  train:test ratio sits inside the 2:1-to-5:1 band, and non-overlapping test
  slices mean no test day is scored twice.
- Status: in force. Caveats: adjacent windows share 4/5 of their training data,
  so windows are not independent; and there is no embargo at the train/test
  boundary (see D029).

## D011 — Three primary financial metrics: cumulative return, Sharpe, max drawdown (2026-05)
- Decision: report all three, with Sharpe as the comparison metric.
- Backing: cumulative return is the headline, Sharpe gives the risk-adjusted
  comparison, max drawdown captures the worst realised loss — the standard
  triple in RL-for-finance evaluations.
- Citation: Sharpe (1994); Hambly, Xu & Yang (2023).
- Why valid: no single metric is sufficient — a high return with a 40% drawdown
  is not the same result as the same return with 10% — and all three are
  computed identically for every arm including buy-and-hold.
- Status: in force.

## D012 — Sharpe annualised by sqrt(252) with zero risk-free rate (2026-08)
- Decision: `sqrt(252) * mean / std(ddof=1)`, no risk-free rate.
- Backing: 252 is the US trading-day convention; a zero risk-free rate is
  standard in research comparisons where the same adjustment would apply to
  every arm.
- Citation: Sharpe (1994); Lo (2002) on the IID assumptions behind sqrt-time
  annualisation.
- Why valid: it is applied identically to agents and to buy-and-hold, so it
  cannot change the ranking. Lo (2002) is the citation to acknowledge that
  sqrt-time scaling assumes IID returns, which daily returns only approximately
  satisfy — a stated limitation, not an error.
- Status: in force.

## D013 — Buy-and-hold reported alongside every configuration as a sanity floor (2026-05)
- Decision: B&H Sharpe / cumulative return / max drawdown logged per test window
  in `windows.csv`.
- Backing: a trained trading agent that cannot beat doing nothing has not earned
  the label; the passive benchmark is the first thing a reader checks.
- Citation: Moody & Saffell (2001) (B&H as the reference baseline); Arnott,
  Harvey & Markowitz (2019).
- Why valid: it is computed from the same test slice and the same metric code,
  so it is a like-for-like floor rather than a quoted external number.
- Status: in force. Current finding: neither mechanism clears it (B&H mean
  Sharpe 0.841 vs FS 0.387, AS 0.465; agents win ~25% of windows).

## D014 — Memory budget grid k ∈ {3, 5, 10, 20}, identical for both mechanisms (2026-05)
- Decision: sweep both FS and AS over the same four budgets and report RQ2 at
  matched budget κ = k.
- Backing: the source paper's central claim is about performance at a *matched*
  budget, and a grid spanning small to large budgets is what shows whether any
  advantage is budget-dependent.
- Citation: Nangue Tasse et al. (2025).
- Why valid: matched budget is the only comparison that isolates the mechanism;
  an unmatched comparison would confound memory size with memory policy.
- Status: in force.

## D015 — 7 random seeds per configuration (2026-08)
- Decision: seeds 0-6 for every (mechanism, k) and every LSTM shape.
- Backing: single-seed deep RL results are not reproducible; variance across
  seeds is often comparable to the differences being claimed, and 3-5 seeds is
  frequently too few to detect a real effect.
- Citation: Henderson et al. (2018); Colas, Sigaud & Oudeyer (2018); Patterson
  et al. (2024).
- Why valid: 7 seeds x 31 windows per configuration is above the usual practice
  in the field and lets seed spread be separated from window spread. It does not
  by itself license treating 217 rows as 217 independent samples (see D030).
- Status: in force.

## D016 — PPO hyperparameters held fixed across the whole sweep (2026-08)
- Decision: total_timesteps 100000, n_steps 2048, batch_size 64, lr 3e-4,
  default MLP policy — identical in the FS and AS configs.
- Backing: the proposal commits to holding everything but the memory mechanism
  constant so the RQ2 difference is attributable to the mechanism.
- Citation: Schulman et al. (2017); Nangue Tasse et al. (2025).
- Why valid: it protects internal validity of the comparison. Open point: the
  proposal says these follow the AS paper's defaults, which has not been
  verified against the reference repository — and AS carries an extra action
  head at equal compute, so a null result is partly a budget question.
- Status: in force; verification open.

## D017 — Cold start fills every stack slot with the initial observation (2026-08)
- Decision: on `reset()`, both FS and AS fill all k slots with the first
  observation, rather than zero-filling the first k-1.
- Backing: the reference AS implementation zero-fills; zero in log-return space
  is a value the network sees nowhere else and would inject an artificial
  extreme during burn-in.
- Citation: deviation from Nangue Tasse et al. (2025) reference implementation.
- Why valid: both mechanisms are made cold-start-identical, so any RQ2
  difference comes from the memory mechanism rather than from a different
  burn-in. The deviation is deliberate, documented in the wrapper's docstring,
  and applied symmetrically.
- Status: deviation, in force.

## D018 — AS observation flattened to shape (k,), identical to FS (2026-08)
- Decision: the AS stack is concatenated into the same flat k-vector the FS
  wrapper produces.
- Backing: the networks must see the same input shape for the comparison to be
  about memory policy rather than about input dimensionality.
- Citation: Nangue Tasse et al. (2025) (architecture-agnostic claim).
- Why valid: it makes the comparison honest at the input layer — the only
  architectural difference left is the extra memory head, which is intrinsic to
  the method being tested.
- Status: in force.

## D019 — AS action space is MultiDiscrete([3, k]) (2026-08)
- Decision: one head for the trading action, one for the slot to overwrite;
  both learned from the same reward.
- Backing: this is the method's definition — memory management as a decision
  problem driven by task reward, not by an auxiliary or intrinsic objective.
- Citation: Nangue Tasse et al. (2025); contrast with Demir (2022), where
  retention is driven by an intrinsic novelty reward.
- Why valid: it reproduces the mechanism faithfully, and SB3's MlpPolicy handles
  MultiDiscrete natively so no algorithm change was needed.
- Status: in force.

## D020 — Stable-Baselines3 + Gymnasium rather than a from-scratch PPO (2026-08)
- Decision: use SB3's PPO and the Gymnasium env API.
- Backing: a tested, widely used implementation removes a large class of silent
  algorithmic bugs, and the env API gives the memory wrappers a standard
  interface.
- Citation: Raffin et al. (2021); Towers et al. (2024).
- Why valid: at honours scope the contribution is the empirical comparison, not
  a PPO implementation; using a reference implementation makes the result easier
  for a reader to trust and reproduce.
- Status: in force.

## D021 — Data-agnostic environment: generic arrays, not "the VIX file" (2026-08)
- Decision: `TimeSeries(observations, returns, transaction_cost)` takes plain
  arrays; all data-source specifics live in the loader/transform layer.
- Backing: own design constraint — the env, memory wrappers and metrics should
  not know which asset they are running on.
- Citation: none (engineering decision).
- Why valid: it makes the single-source dependency a swap in one file, which is
  the cheapest available defence against "your result is one dataset".
- Status: in force.

## D022 — CPU training, not GPU (2026-08)
- Decision: `device: cpu` in every config.
- Backing: measured in our own smoke tests — a small MLP-policy PPO agent does
  not benefit from GPU acceleration at this scale; CPU was faster.
- Citation: own measurement; consistent with SB3's own guidance that MlpPolicy
  is usually faster on CPU.
- Why valid: it is an empirical finding on this workload, not an assumption, and
  it removes a dependency on cluster GPU availability.
- Status: in force.

## D023 — Resume-safe, checkpointed sweeps (2026-08)
- Decision: row-by-row CSV appends with flushes, atomic summary.json write,
  strict schema check, and skipping of already-completed (k, seed, window)
  triplets.
- Backing: the Wits mscluster incident history (repeated power and network
  failures, a failed core server, a month-long shutdown) makes interruption the
  normal case, not the exception.
- Citation: own operational record (research_log.md, 2026-08).
- Why valid: it was a first-day requirement rather than an optimisation, and it
  is why the 868-row sweeps survived to completion.
- Status: in force. Known side effect: the resume path is what left the
  `summary.json` champion blocks computed on a partial subset — see D026.

## D024 — Scope cut: memoryless PPO baseline and its RQ dropped (2026-06-24)
- Decision: three configurations instead of four, two research questions instead
  of three; the k=1 memoryless PPO arm is removed.
- Backing: the memoryless-vs-FS comparison is already established in the
  DQN/POMDP literature, and the proposal itself called replicating it on
  financial data "a sanity check more than a contribution"; the time freed went
  to the AS integration where the contribution lives.
- Citation: Mnih et al. (2015); Hausknecht & Stone (2015).
- Why valid: it preserves the comparison that carries the contribution while
  cutting the one that does not. Trade-off accepted on record: the standalone
  claim "memory helps in financial RL" can no longer be made — only "learned
  memory vs fixed memory".
- Status: in force. Note: the Week 2 slides and the proposal still describe four
  configurations and three RQs; the report must state this cut explicitly.

## D025 — Retention diagnostic as the mechanism evidence for RQ2 (2026-08)
- Decision: log which slot the AS agent overwrites, per day and in aggregate, as
  direct evidence about whether memory is being managed or not.
- Backing: AS's efficiency claim rests on the premise that only κ ≪ k*
  observations matter; a flat overwrite distribution would mean the agent has
  degenerated to "overwrite anything" and the mechanism is not being used.
- Citation: Nangue Tasse et al. (2025).
- Why valid: it tests the method's own premise rather than only its outcome, so
  it can explain a null performance result instead of merely reporting one.
- Status: in force, but the diagnostic as implemented is insufficient — see
  D026, D032, D033.

---

## Entries from the 2026-10-21 evaluation review

## D026 — Headline numbers are taken from windows.csv, never from summary.json (2026-10-21)
- Decision: all reported results are recomputed from `results/*/windows.csv`;
  the `champion` blocks in `summary.json` are not quoted.
- Backing: own audit. `windows.csv` holds the complete balanced record (868 rows
  = 4 x 7 x 31 for each sweep), while the champion blocks were built only from
  rows executed in the final resumed session — FS ranks k=5 on 14 rows against
  217 for k=10/20, AS contains only k=10 and k=20, and the LSTM summary has an
  empty per-window list. Complete-data means differ from the champion blocks
  (FS best is k=3 at 0.484, not k=5 at 0.601).
- Citation: own audit of the artifacts; cause is the resume logic in D023.
- Why valid: it is a verified property of the files, reproducible by grouping
  windows.csv by (k, seed) and counting.
- Status: in force.

## D027 — RQ1 claim is held back until the LSTM information set is matched (2026-10-21)
- Decision: do not claim "RL beats the supervised baseline" while the LSTM's
  forecast for day t is made from observations only up to day t-1, whereas the
  PPO agent sees day t.
- Backing: own code audit of the LSTM rollout against the env's step contract.
  The measured gap is large (LSTM mean Sharpe 0.082 vs FS 0.387), which makes
  ruling out a one-day handicap necessary before attributing the gap to the
  paradigm.
- Citation: own audit; RQ1's own requirement of a *strong* baseline, per the
  scoped-RQ document.
- Why valid: a comparison in which one arm has strictly less information is a
  confound, not a result.
- Status: open — fix or caveat before reporting RQ1.

## D028 — LSTM threshold must be swept, not fixed at zero (2026-10-21)
- Decision: treat `threshold: 0.0` as a configuration to vary rather than a
  default.
- Backing: at threshold zero the baseline can never take the flat position; this
  is confirmed in the data — `pos_flat` is exactly 0 in all 868 LSTM rows, while
  the RL agents go flat on 18-34 days per test year.
- Citation: own audit; Moody & Saffell (2001) for the three-position action set
  the project adopted in D006.
- Why valid: the two arms are otherwise claimed to produce "the same kind of
  output"; a dead-band threshold restores the third action and removes an
  uncontrolled difference.
- Status: open.

## D029 — Train/test boundary needs an embargo, or an explicit footnote (2026-10-21)
- Decision: either drop one row at each walk-forward boundary or state the
  one-day overlap as a known limitation.
- Backing: the label is a next-day return, so the final training day's target
  covers the first test day's price move; purging and embargoing is the standard
  remedy.
- Citation: López de Prado (2018).
- Why valid: the leak is small but removable at no cost, and naming it is
  cheaper than having it found.
- Status: open.

## D030 — Unit of analysis for RQ2 is the window, with seeds averaged inside it (2026-10-21)
- Decision: report the paired AS−FS difference at window level (n=31 per k) with
  a bootstrap confidence interval, not as a pooled standard deviation over
  seeds x windows.
- Backing: own recomputation. Pooling all 868 pairs gives mean +0.077 with
  t-test p=0.024 (Wilcoxon p=0.030); averaging the 7 seeds within each window
  first gives per-k p-values of 0.71, 0.41, 0.23 and 0.37 — none significant.
  Pooling treats 217 non-independent rows as independent samples.
- Citation: Agarwal et al. (2021); Patterson et al. (2024); Colas, Sigaud &
  Oudeyer (2018). Multiple-testing inflation across the 10+ configurations
  selected against one test set: Bailey & López de Prado (2014); Harvey & Liu
  (2015).
- Why valid: the correct inference changes the conclusion, which is exactly when
  the choice of unit has to be made explicit. `defence/make_defence_figures.py`
  already implements the window-level bootstrap version.
- Status: in force.

## D031 — Observation stream representation is reopened (2026-10-21)
- Decision: revisit D008 for the observation series — consider the VIX level
  (train-slice z-scored) or a level-plus-change pair instead of the log return.
- Backing: the project's own motivation (D005, Week 2 slides, proposal §6.1)
  rests on the VIX *level* carrying regime information that persists for weeks.
  Differencing removes that persistence, which weakens the κ ≪ k* premise before
  the agent sees anything.
- Citation: Whaley (2000, 2009) on VIX level as a regime/fear measure; Nangue
  Tasse et al. (2025) for the sparsity premise being tested.
- Why valid: additivity (the reason for log returns) is required for the *reward*
  and the metrics, not for the observation; the two series can be represented
  differently without breaking anything downstream.
- Status: open — highest-value single change if time allows.

## D032 — A random-memory-action control arm is required for RQ2 (2026-10-21)
- Decision: add an AS arm with uniformly random memory actions, everything else
  identical.
- Backing: without it, any AS-vs-FS difference cannot be attributed to *learning*
  what to keep rather than to AS's different stack composition. Our own data
  makes this urgent: within a single (k, seed, window) run the modal slot absorbs
  90% / 82% / 74% / 63% of all overwrites at k = 3 / 5 / 10 / 20, i.e. individual
  agents collapse to a near-constant eviction rule.
- Citation: own analysis of `results/rq_rl_as_sweep/steps.csv`; Nangue Tasse et
  al. (2025) for the claim being isolated.
- Why valid: it is the ablation that makes the mechanism claim falsifiable, and
  it costs one sweep arm at the same budget as an existing one.
- Status: open.

## D033 — Retention must be reported per run, with a null, and not only from deterministic rollouts (2026-10-21)
- Decision: replace the pooled slot histogram and the min/max "concentration"
  scalar with (a) per-run distributions, (b) a comparison against a random-policy
  null, and (c) a stochastic-policy or observation-dependence measure.
- Backing: own analysis. Pooled across runs the histogram looks mildly
  non-uniform (max slot share 0.41 / 0.28 / 0.18 / 0.10 against uniform 0.33 /
  0.20 / 0.10 / 0.05), which hides the per-run collapse in D032. `concentration`
  = 1 − min/max has no sampling distribution, so "strongly skewed" is an
  assertion. Retention is also measured under `deterministic=True`, where argmax
  guarantees low-entropy slot choice.
- Citation: own analysis; Agarwal et al. (2021) on reporting distributions rather
  than point statistics.
- Why valid: the corrected reading is a genuine mechanism finding (AS behaves as
  a fixed non-FIFO rule rather than a state-dependent memory policy), which is
  more informative than the current figure and supports the pre-registered
  negative result.
- Status: open.

## D034 — Conclusion is framed as a pre-registered null result (2026-10-21)
- Decision: report "no detectable difference between learned and fixed memory at
  matched budget on this task", with the sparsity premise weakened by D031,
  rather than "AS wins".
- Backing: the observed paired effect is +0.08 Sharpe against a per-window
  standard deviation of roughly 1.0; with 31 windows the minimum detectable
  paired difference at 80% power is on the order of 0.5 Sharpe.
- Citation: Colas, Sigaud & Oudeyer (2018); Patterson et al. (2024); the
  proposal's own statement that a negative result is an acceptable contribution.
- Why valid: the project pre-committed to this outcome being publishable, and
  the data supports a null far better than it supports an effect.
- Status: in force.

---

## Entries from the 2026-10-21 project pass (back-filled from claude-read/ and verified against the data)

## D035 — Fairness rule: equal-quality champions, not same-knob-count champions (2026-09-08)
- Decision: each paradigm gets the tuning treatment appropriate to its family, then champions face off — LSTM gets an HP grid, FS gets a k grid, AS gets a κ grid with the PPO base held identical to FS.
- Backing: own reasoning, recorded in research_log.md 2026-09-08. Sweeping hyperparameters on one side and not the other is the same category of mistake as any unfair benchmark; the fix is to sweep both sides, not to sweep neither.
- Citation: own reasoning; Henderson et al. (2018) on comparison hygiene in deep RL.
- Why valid: it keeps the RQ1 comparison about paradigms rather than about who got tuned, while still holding the FS/AS pair identical so RQ2 stays a mechanism comparison.
- Status: in force, with a known asymmetry to disclose — PPO was never tuned (stock SB3 defaults) while the LSTM got a 4-combo grid, so the residual tuning asymmetry favours the LSTM, not the contribution.

## D036 — The k grid was fixed before any results were seen (2026-09-08)
- Decision: predefine k ∈ {3, 5, 10, 20}, run all four, and pick the best mean at the end — never "try k=5, and if it loses try k=10".
- Backing: iterative config-adding against test numbers is early stopping on the test set, i.e. p-hacking; the number of trials has to be fixed and countable for any later correction to be meaningful.
- Citation: Bailey & López de Prado (2014); Arnott, Harvey & Markowitz (2019).
- Why valid: it is the one protection against snooping that was actually in place from the start, and it makes the trial count honest (4 k values per mechanism, 4 LSTM shapes) so a multiple-testing caveat can be stated precisely.
- Status: in force.

## D037 — The memory action is forced: no "skip" slot (2026-08-20)
- Decision: every step the AS agent must overwrite one of the k slots; there is no option to leave the stack untouched.
- Backing: a skip action would be trivially safe — nothing risked, nothing lost — so the memory head would collapse to always-skip and no learning signal would ever reach the memory decision. The forced overwrite *is* the learning signal.
- Citation: matches Nangue Tasse et al. (2025); reasoning recorded in research_log.md 2026-08-20.
- Why valid: it reproduces the method as published rather than inventing a variant, and the reasoning explains why the variant would be degenerate. "Does a skip slot help?" is a separate research question, not a fix.
- Status: in force.

## D038 — Matched budget enforced by config convention, not by a code assertion (2026-08-20)
- Decision: κ = k is guaranteed by paired config files and by the comment blocks forbidding a change to one PPO block without the other, not by an invariant in code.
- Backing: own decision; encoding it as a code invariant would block mismatched-budget experiments if those become interesting.
- Citation: own design note (claude-read/design_notes.md, 2026-08-20).
- Why valid: it is honest about where the guarantee lives, which matters because the guarantee is the basis of the RQ2 claim. The weakness is that a convention can be broken silently — so the paired configs are the thing to show a reader, not the code.
- Status: in force.

## D039 — `build_memory_wrapper` is the single swap point; no shared base class (2026-08-20)
- Decision: one dispatcher function selects FS or AS from a YAML key; everything upstream and downstream is memory-agnostic. No abstract `MemoryWrapper` interface.
- Backing: this was the answer to the standing open question about unifying a stateless observation transform (FS) with a learned in-policy decision (AS). Both satisfy `gym.Wrapper` and compatible signatures, which was judged good enough for two mechanisms.
- Citation: own design note; Towers et al. (2024) for the wrapper contract both satisfy.
- Why valid: it is proportionate — a formal interface buys nothing with two implementations and would be worth adding at three.
- Status: in force; formalise only if a third mechanism is added.

## D040 — Two output formats per run: windows.csv and summary.json (2026-08-20)
- Decision: a tidy one-row-per-window CSV for analysis, plus a nested JSON with the config embedded for human debugging.
- Backing: different consumers. One `pd.concat` across runs gives the cross-run comparison DataFrame; the JSON answers "what exactly did this run do".
- Citation: own design note.
- Why valid: no attempt is made to force one file to serve both purposes. In practice this decision is also what saved the results — the CSV stayed complete and correct when the JSON aggregate did not (see D026).
- Status: in force.

## D041 — Seeds raised 3 → 7 by resuming, not by re-running from scratch (2026-09-08)
- Decision: edit the YAML to `seeds: [0..6]` and rerun; the resume logic trains only the new seeds and reuses the existing ones.
- Backing: the reported mean and spread aggregate all seven seeds, not a chosen subset, so the statistical result is identical to running seven from scratch at roughly half the wall clock.
- Citation: own reasoning; Colas, Sigaud & Oudeyer (2018) and Patterson et al. (2024) for the 5-10 seed band being the target.
- Why valid: it is not cherry-picking, because no seed is dropped and the seed list is declared in the config that ships with the results. The answer to "why 7?" is on record: 5-10 is standard, 3 was prototyping.
- Status: in force.

## D042 — LSTM hyperparameter grid is "informed by" the literature, not "justified by" it (2026-09-03, wording corrected 2026-10-21)
- Decision: hidden ∈ {64, 128}, layers ∈ {1, 2}, dropout 0.2, seq_len 20 (~1 trading month), grad_clip 5.0; a deliberately compact 4-combo grid.
- Backing: a 2026-09-03 web search surfaced convergent practitioner recommendations (hidden 128-256 for financial forecasting, 1-2 layers, dropout ~0.2, L2-norm clipping ~5.0). Several of those sources are blog posts rather than peer-reviewed work — one attributed as "Chowdhury 2023" is a Medium article with an unverified year.
- Citation: Hochreiter & Schmidhuber (1997) for the architecture; practitioner sources for the ranges, cited as such.
- Why valid: the grid is proportionate because the LSTM is a control for RQ1, not the research object. The claim has to be worded as "informed by common practice". Note the champion sits at hidden=128, layers=1 — at the top edge of the hidden range, which is the pre-agreed signal that the grid would need widening if the LSTM were the contribution.
- Status: in force, wording corrected.

## D043 — Plot design: sparse by intent for the baseline, dense for the contribution (2026-09-08)
- Decision: for the LSTM sweep, only a per-shape metric timeline (median across seeds, min-max band, B&H reference) and an aggregate champion bar chart. No ROC curves, confusion matrices, or threshold analyses.
- Backing: own decision — the LSTM is a control, so plot effort belongs on RQ2.
- Citation: own design note.
- Why valid: it keeps the figure budget on the contribution. The one thing to add is forecast-quality metrics, because without them a weak baseline cannot be distinguished from a strong RL result (see D027, D028).
- Status: in force.

## D044 — Dataset span is 36.5 years, not ~20-24; the "span too short" concern is closed (2026-10-21)
- Decision: report the span as 1990-01-03 to 2026-06-29, 9188 aligned trading days, 31 test windows spanning 1994-12-27 to 2026-01-09.
- Backing: verified directly from the loader. The long-standing watch item about the data span being short relative to published practice was based on an incorrect "~20 years" figure.
- Citation: own verification; Arnott, Harvey & Markowitz (2019) for the data-constraint concern that prompted the item.
- Why valid: the number is checkable in one command, and 36.5 years is comfortably within published practice for daily VIX/S&P work.
- Status: in force. Follow-ups: the deck (slide 4), takeoff/status.md, plots/data_integrity/disclaimer.tex and earlier log entries all still say "24 years" and must be corrected; `scripts/data_integrity_check.py` also fetches from 2000-01-01 while the local CSVs start in 1990, so its "only local" date count is misleading.

## D045 — The reported VIX/S&P correlation must be the predictive one (2026-10-21)
- Decision: quote the correlation between today's VIX log-return and the *next* day's S&P log-return, which is +0.03. The same-day correlation is -0.71.
- Backing: verified from the aligned arrays. Only the predictive direction is available to the agent, because the reward at day t uses the return from t to t+1.
- Citation: own verification; the env's own `shift(-1)` alignment rule (D007).
- Why valid: the two numbers describe different things, and the large one is the one the agent cannot use. Quoting -0.7 as evidence of signal would be a real overclaim.
- Status: in force. Follow-up: slide 4 and plots/DATA_TERMINAL.md §1.7 currently cite the -0.7 to -0.8 figure in the predictive context.

## D046 — Variance is dominated by the test period, not by the seed — at small k (2026-10-21)
- Decision: attribute the wide error bars primarily to which test year a window covers, and say so, rather than to model randomness.
- Backing: verified per k. Standard deviation across the 31 seed-averaged windows vs mean standard deviation across the 7 seeds within a window: FS 1.13 vs 0.36 (k=3), 1.03 vs 0.47 (k=5), 0.68 vs 0.57 (k=10), 0.73 vs 0.67 (k=20); AS is similar. The market-dominates reading is clear at k=3 and k=5, but the two sources are nearly equal at k=20.
- Citation: own analysis; Agarwal et al. (2021) and Patterson et al. (2024) on separating sources of variation.
- Why valid: it follows directly from the decomposition, and it has a practical consequence — more seeds will not buy significance, more test periods would. State it per k rather than as a blanket claim, because at k=20 it no longer holds cleanly.
- Status: in force.

## D047 — The behavioural difference is promoted as a primary RQ2 result (2026-10-21)
- Decision: report trading behaviour, not only performance. Mean position flips per test year, FS vs AS: 33 vs 29 (k=3), 54 vs 31 (k=5), 88 vs 40 (k=10), 118 vs 52 (k=20) — verified from windows.csv.
- Backing: the performance difference is inside the noise (D030, D034), but the behavioural difference is large, monotone in k, and in the same direction at every budget: FS churns more as its window grows, AS does not.
- Citation: own analysis; Moody & Saffell (2001) for turnover as a first-class property of a trading policy.
- Why valid: it is a real, measurable, mechanism-level consequence of giving the agent control of its memory, and it is the most defensible positive finding in the project. It is not yet significance-tested, and it must be reported as a behavioural difference rather than as a performance claim — lower turnover also means lower transaction costs, which ties back to the cost sensitivity sweep in D009.
- Status: in force; significance test outstanding.

## D048 — RQ1 is reported as directional only (2026-10-21)
- Decision: say "directionally, RL matched or exceeded the tuned LSTM", not "RL earns its keep".
- Backing: seed-averaged window-level paired tests give FS(k=3) − LSTM +0.28 at p≈0.10, and AS(k=5) − LSTM +0.31 at p=0.04 — but the latter is one of eight comparisons run against the same test windows, so it does not survive a multiple-testing correction.
- Citation: Bailey & López de Prado (2014); Harvey & Liu (2015); Agarwal et al. (2021).
- Why valid: the honest reading of p=0.04 out of eight tests is "not established". Combined with D027 and D028 (the LSTM's one-day handicap and its inability to go flat), the RQ1 result is a direction, not a finding.
- Status: in force.

## D049 — No number or citation is stated without being checked against the data or the source (2026-10-21)
- Decision: standing rule for the write-up, the deck and this log. Every figure quoted is recomputed; every attribution is verified.
- Backing: three live examples found in the project's own materials — "24 years" for a 36.5-year span, a -0.7 correlation quoted in the predictive direction, and "Henderson et al. 2018, How Many Random Seeds?" which is actually Colas, Sigaud & Oudeyer (2018), arXiv:1806.08295.
- Citation: own audit.
- Why valid: in a project whose contribution is an honest null result, the credibility of the write-up *is* the contribution; a misattributed citation on a slide costs more here than a weak effect size.
- Status: in force.

## D050 — Retention has two readings, and both are reported (2026-10-21)
- Decision: report the across-run positional preference *and* the within-run degeneracy, because they are different facts and only the pair is honest.
- Backing: verified. Pooled over all runs at k=5 the overwrite fractions by slot position (0 = oldest survivor) are [.104, .176, .201, .241, .278] — the agent overwrites the newest slots most and protects the oldest, which is the opposite of FIFO (FIFO would be 100% on slot 0). But within a single (k, seed, window) run the modal slot absorbs 90% / 82% / 74% / 63% of overwrites at k = 3 / 5 / 10 / 20, so each individual agent is close to a fixed rule.
- Citation: own analysis of windows.csv and steps.csv; Nangue Tasse et al. (2025) for the premise under test.
- Why valid: the pooled fractions show a preference exists and is not FIFO; the per-run concentration shows that preference is close to constant rather than state-dependent. Neither alone is the full picture, and neither shows the preference *causes* any performance gain — which is what D032's random-memory control would test.
- Status: in force. Refines D025, D033.

## D051 — S&P series uses `close`, not `adj_close` (2026-10-21)
- Decision: currently the price return excludes dividends; either switch to the adjusted series or state the omission.
- Backing: own code audit of the loader.
- Citation: own audit.
- Why valid: it understates the buy-and-hold benchmark that every configuration is judged against, and buy-and-hold is already beating every agent — so the direction of the error works against the project's own result, which is worth saying out loud.
- Status: open.

## D052 — Durable project context lives in the repo, not in a tool's memory (2026-10-19)
- Decision: context, decisions and the dated journey log live in tracked files in this repository.
- Backing: own working preference — the notes must stay in reach directly and travel with the repo.
- Citation: own convention (claude-read/CLAUDE.md §12).
- Why valid: it keeps the record portable and reviewable by a supervisor, and it is why this log is a file rather than a tool-side note.
- Status: in force.

## D053 — This log's place among the existing project documents (2026-10-21)
- Decision: `research_decisions/decisions_log.md` is the single index of *decisions and their justification*. It does not replace the four existing documents: claude-read/CLAUDE.md (standing project context and scope), research_log.md (dated first-person journey), claude-read/design_notes.md (why the code looks the way it does), claude-read/handoff.md (what was found after the others were last updated).
- Backing: own audit of overlap. The reasoning behind decisions was spread across all four, which is why some decisions were justified in two places and others nowhere.
- Citation: own audit.
- Why valid: one append-only index with a fixed entry shape is checkable; four prose documents with different conventions are not. The rule going forward: a decision gets an entry here, and the other files keep their own jobs.
- Status: in force.

## D054 — Deferred work, recorded so it is not lost (2026-10-19, 2026-10-20)
- Decision: three items are deliberately deferred rather than dropped — a sealed final holdout (ask the supervisor, currently documented as a limitation), TensorBoard for training visualisation (SB3 curves land as progress.csv today; `tensorboard` is already pinned), and an external retrospective agent to be run after submission.
- Backing: all three are out of scope during crunch; the retrospective in particular was deferred deliberately rather than run mid-submission.
- Citation: own record (claude-read/CLAUDE.md §13, §14; handoff.md open decisions).
- Why valid: deferring with a written reason is a decision; dropping silently is not. Each has a named trigger — supervisor conversation, post-submission, post-submission.
- Status: open by design.

---

## Entries from the 2026-10-21 action-items sprint (to-do-asap.md)

## D055 — The summary.json aggregates are rebuilt from windows.csv, not deleted (2026-10-21)
- Decision: regenerate the `per_window` and `champion` blocks in all three
  `results/*/summary.json` from the complete `windows.csv` record, rather than
  deleting the champion blocks. New script: `scripts/rebuild_summaries.py`. Each
  rebuilt file carries a `rebuilt` provenance block (source, row count, git
  commit, reason) and the stale original is kept beside it as
  `summary.partial-<UTC timestamp>.json`.
- Backing: own audit, confirmed by re-running the aggregation. The stale blocks
  ranked configurations on unequal samples — FS ranked k=5 champion on 14 rows
  from 1 seed (Sharpe +0.601) against 217 rows for k=10/20, with k=3 absent
  entirely; AS held only k=10 (87 rows, 3 seeds) and k=20; the LSTM block was
  null with an empty `per_window`. Rebuilt over the full 868 rows per sweep the
  champions are FS k=3 (+0.484), AS k=5 (+0.515), LSTM hp_id=2 / h128L1
  (+0.208) — all 217 rows, 7 seeds, 31 windows. The FS champion changes
  identity, not just its value.
- Citation: own audit; cause is the resume logic in D023, as recorded in D026.
- Why valid: D026 already committed to windows.csv as the source of truth, but
  left a contradicting artefact on disk; rebuilding removes the contradiction
  without discarding evidence of what the bug produced. Rebuilding is preferred
  to deletion because the shape of summary.json is referenced by
  `src/evaluation/retention.py` and by the write-up's reproducibility claim.
  The rebuilt `per_window` omits the SB3 curve path and the forecast min/max,
  which existed only in memory during training — they are left out rather than
  invented.
- Status: in force. Root cause not yet fixed: `train.py` and `train_lstm.py`
  still build `per_window_records` only from triplets executed in the current
  session, so any future resumed sweep reintroduces the same artefact. Rerun
  this script after any resumed run, or fix the writers.

## D056 — Every figure shows the seed spread as its error bar and names the window spread beside it (2026-10-21)
- Decision: error bars, bands and whiskers in `scripts/generate_plots.py` are now
  ±1 std across the 7 seeds, computed by collapsing each seed's 31 windows to one
  number first. Every affected figure's title states the unit of analysis, and
  the larger window spread (n=31) is drawn as a faint second whisker on the two
  bar charts and quoted in the other captions. New helpers: `spread_table`,
  `spread_caption`, `by_window`, `equity_band`.
- Backing: the supervisor's instruction, and his reason for it — each walk-forward
  window is a *separately trained model* on different data, so the 7 seeds are
  true replicates of one setup while the 31 windows are not; only the seed spread
  is an error bar in the ordinary sense. The previous figures pooled all 217 rows
  into one std, which mixes the two and treats non-independent rows as independent
  samples. Measured on our data the two differ by roughly an order of magnitude:
  at FS k=3, seed std 0.068 against window std 1.126 (pooled was 1.194); at AS
  k=5, 0.113 against 1.044. Full table for all twelve configurations is printed by
  `spread_table`.
- Citation: supervisor instruction (transcription recording 5); Agarwal et al.
  (2021) and Patterson et al. (2024) on reporting the right unit; D030 for the
  paired significance test using the window as the unit.
- Why valid: the two spreads answer different questions — "would a different
  random start give this?" and "would a different year give this?" — and the
  instruction is well defined at the aggregate only once each seed is collapsed.
  Showing both in one figure means the tighter seed bars can never be mistaken
  for the whole uncertainty. See `disagreements.md` entry 3 for why the literal
  single-number reading of the rule has no implementation.
- Also changed, in the same pass, because the old versions showed no variance at
  all or an incomparable one: the two equity-curve figures now compound each seed
  separately and band the spread across those curves (log y-axis, since compounded
  equity over 31 years is multiplicative); the three Sharpe timelines replaced a
  min-max seed range with mean ±1 seed std, because a min-max range is an extreme
  value that widens with the number of seeds and cannot be compared between
  configurations.
- Status: in force.

## D057 — Memory-slot columns are selected by number, never by sorting their names (2026-10-21)
- Decision: `mem_slot_*` columns are addressed as `[f"mem_slot_{i}" for i in range(k)]`.
  Fixed in `scripts/generate_plots.py`; `src/evaluation/retention.py` still has
  the old form and is next.
- Backing: the columns are padded to the sweep's largest k (20), so sorting the
  names as text orders them mem_slot_0, mem_slot_1, mem_slot_10, mem_slot_11 …
  A k=3 run therefore read columns 0, 1 and 10, and column 10 is empty for that
  run — the plotted third bar was always zero, and the same applied at k=5 and
  k=10. Corrected, the distributions are k=3 [.29 .31 .41], k=5 [.10 .18 .20 .24 .28],
  k=10 rising from .04 to .18, k=20 rising from .03 to .10.
- Citation: own audit; the corrected indexing was already in use in
  `defence/make_defence_figures.py`.
- Why valid: it is a column-selection bug with a verifiable signature (an exactly
  zero bar), and the corrected figure changes the reading — the preference for
  overwriting the newest slots is monotone across all four k, which the broken
  figure obscured.
- Note: the retention figure now pools runs and bands the spread across seeds.
  That pooled view still hides the per-run collapse recorded in D032/D033, so
  this fix makes the figure correct, not yet sufficient.
- Status: in force; `retention.py` pending.

## D058 — The RQ2 headline is a paired per-window difference with a block-bootstrap interval (2026-10-21)
- Decision: the AS-vs-FS claim is made from the paired difference at each
  walk-forward window (seeds averaged inside the window, n=31 per k), reported as
  mean, interquartile mean, median, a 95% bootstrap CI and a win rate — not from
  a t-test over pooled runs. The interval is a moving-block bootstrap with block
  length 5. New figure `plots/rq2/07_paired_as_minus_fs.png` with its table and
  CSV beside it. Implemented in `plot_rq2_paired()`.
- Backing: own computation over the complete sweeps. Every interval contains
  zero and every win-rate interval contains 50%:
      k=3   -0.024  block CI [-0.111, +0.043]  win 42% [26%, 58%]
      k=5   +0.080  block CI [-0.079, +0.251]  win 52% [35%, 68%]
      k=10  +0.138  block CI [-0.146, +0.263]  win 61% [42%, 77%]
      k=20  +0.116  block CI [-0.178, +0.284]  win 52% [32%, 68%]
  The IQM is smaller than the mean at every k (+0.036 vs +0.080 at k=5, +0.040 vs
  +0.116 at k=20), so the mean is being carried by a few extreme years.
- Citation: Agarwal et al. (2021) for stratified bootstrap CIs, IQM and
  probability of improvement; Patterson et al. (2024); Künsch (1989) for the
  moving-block bootstrap; López de Prado (2018) on overlapping-sample dependence.
- Why valid: block length 5 is not a free parameter — `train_size` 1260 with
  `step` 252 means exactly five consecutive windows share training data, so five
  is the dependence length the design creates. Resampling whole blocks keeps that
  dependence in the resampled series, where an ordinary bootstrap would destroy
  it and return an interval that is too narrow. Both versions are printed; the
  block interval is wider at k=10 and k=20 and narrower at k=3, which is the
  expected behaviour when the dependence is in the data rather than assumed.
- Why this and not the t-test: the t-test in D030 gives the same verdict
  (p = 0.71 / 0.41 / 0.23 / 0.37), but it assumes normal, independent windows and
  per-window Sharpe is neither. The bootstrap needs neither assumption and
  reports an effect size with an interval rather than a yes/no.
- Status: in force. This is the number the RQ2 conclusion is written from, and it
  supports D034's pre-registered null.

## D059 — There is one LSTM in the project, and it is the supervised baseline (2026-10-21)
- Decision: T1.5 ("there are two LSTM implementations in the code base — try
  both") is closed as answered, not as outstanding work. The project contains a
  single LSTM, `src/baselines/lstm.py` (`LSTMForecaster`), used only by the RQ1
  supervised arm.
- Backing: Benny answered this on the recording — it was Geraud asking, not
  instructing. Verified against the repository: a search for LSTM definitions
  across `src/` and `scripts/` returns only `src/baselines/lstm.py`; the other
  matches are references to it or to the results directory name.
- Citation: transcription recording v4; own audit of the repository.
- Why valid: the item was transcribed into the action list as an instruction
  because the recordings do not mark who is asking. Closing it on the answer
  already given is correct; carrying it as work would have invented a task.
- Related, but a different thing: the new RL arm uses `RecurrentPPO`'s
  `MlpLstmPolicy`, which does expose a genuine implementation choice —
  `shared_lstm` (one LSTM feeding actor and critic) against separate actor and
  critic LSTMs. That is a second LSTM *in the new arm*, not a second one that
  existed when he asked. Default here is separate; it is not swept, consistent
  with the no-second-sweep instruction (D060).
- Status: closed.

## D060 — The RL LSTM arm reuses the supervised champion's shape and is not swept (2026-10-21)
- Decision: add `policy.type` to `scripts/train.py` so one code path serves both
  architectures (`mlp` → PPO + MlpPolicy, `lstm` → RecurrentPPO + MlpLstmPolicy).
  New arms `configs/rq_rl_fs_lstm_sweep.yaml` and `configs/rq_rl_as_lstm_sweep.yaml`
  at matched k=3, 7 seeds, LSTM shape copied from the supervised champion
  (hidden 128, 1 layer), PPO block byte-identical to the MLP arms.
- Backing: supervisor instruction in recording 4, which supersedes his earlier
  v4 position that the RL LSTM should get its own sweep. His stated reason:
  beating the baseline on the baseline's own best settings is the fair and
  sufficient result, and sweeping would only move the comparison in our favour.
- Citation: transcription recordings 3, 4, 6, and the "guidance changed" note;
  own reasoning on the k grid.
- Why valid: extending `train.py` rather than forking it is what makes the claim
  defensible — the env, wrapper, walk-forward split, seeding and PPO
  hyperparameters are provably the same objects, so the architecture is the only
  free variable. A separate script could drift.
- Why k=3 only, not the full grid: the memory-budget sweep is already answered
  by the MLP arms; this run answers the architecture question, which needs one
  budget. k=3 is the FS champion on the complete-data record. Running both
  mechanisms at the *same* k additionally lets RQ2 be re-asked at the LSTM
  architecture — if the null in D058 survives a change of network, it is a
  result about the mechanism rather than about the MLP.
- Known unmatched item, carried over from the MLP arms: AS still has an extra
  action head at equal timesteps. Not fixed here.
- Status: in force; FS arm to be run first so RQ1 has a deliverable even if the
  AS arm cannot finish.

---

## References

- Agarwal, R., Schwarzer, M., Castro, P. S., Courville, A., Bellemare, M. G.
  (2021). "Deep Reinforcement Learning at the Edge of the Statistical
  Precipice." NeurIPS 2021.
- Arnott, R., Harvey, C. R., Markowitz, H. (2019). "A Backtesting Protocol in
  the Era of Machine Learning." Journal of Financial Data Science 1(1), 64-74.
- Bailey, D. H., López de Prado, M. (2014). "The Deflated Sharpe Ratio:
  Correcting for Selection Bias, Backtest Overfitting and Non-Normality."
  Journal of Portfolio Management 40(5), 94-107.
- Bakker, B. (2002). "Reinforcement Learning with Long Short-Term Memory."
  NIPS 14.
- Bergmeir, C., Benítez, J. M. (2012). "On the Use of Cross-Validation for Time
  Series Predictor Evaluation." Information Sciences 191, 192-213.
- Campbell, J. Y., Lo, A. W., MacKinlay, A. C. (1997). The Econometrics of
  Financial Markets. Princeton University Press.
- Colas, C., Sigaud, O., Oudeyer, P.-Y. (2018). "How Many Random Seeds?
  Statistical Power Analysis in Deep Reinforcement Learning Experiments."
  arXiv:1806.08295.
- Demir, A. (2022). "Learning What to Memorize: Using Intrinsic Motivation to
  Form Useful Memory in Partially Observable Reinforcement Learning."
  arXiv:2110.12810v2.
- Hambly, B., Xu, R., Yang, H. (2023). "Recent Advances in Reinforcement
  Learning in Finance." Mathematical Finance 33(3).
- Harvey, C. R., Liu, Y. (2015). "Backtesting." Journal of Portfolio Management
  42(1).
- Hausknecht, M., Stone, P. (2015). "Deep Recurrent Q-Learning for Partially
  Observable MDPs." arXiv:1507.06527.
- Henderson, P., Islam, R., Bachman, P., Pineau, J., Precup, D., Meger, D.
  (2018). "Deep Reinforcement Learning that Matters." AAAI 2018.
- Hochreiter, S., Schmidhuber, J. (1997). "Long Short-Term Memory." Neural
  Computation 9(8), 1735-1780.
- Lo, A. W. (2002). "The Statistics of Sharpe Ratios." Financial Analysts
  Journal 58(4), 36-52.
- López de Prado, M. (2018). Advances in Financial Machine Learning. Wiley.
- Mnih, V. et al. (2015). "Human-Level Control through Deep Reinforcement
  Learning." Nature 518(7540).
- Moody, J., Saffell, M. (2001). "Learning to Trade via Direct Reinforcement."
  IEEE Transactions on Neural Networks 12(4).
- Nangue Tasse, G., Riemer, M., Rosman, B., Klinger, T. (2025). "Beyond Sliding
  Windows: Learning to Manage Memory in Non-Markovian Environments."
  arXiv:2512.19154v1.
- Patterson, A., Neumann, S., White, M., White, A. (2024). "Empirical Design in
  Reinforcement Learning." JMLR 25(318), 1-63.
- Raffin, A., Hill, A., Gleave, A., Kanervisto, A., Ernestus, M., Dormann, N.
  (2021). "Stable-Baselines3: Reliable Reinforcement Learning Implementations."
  JMLR 22(268).
- Schulman, J., Wolski, F., Dhariwal, P., Radford, A., Klimov, O. (2017).
  "Proximal Policy Optimization Algorithms." arXiv:1707.06347.
- Sharpe, W. F. (1994). "The Sharpe Ratio." Journal of Portfolio Management
  21(1), 49-58.
- Sutton, R. S., Barto, A. G. (2018). Reinforcement Learning: An Introduction,
  2nd ed. MIT Press.
- Tashman, L. J. (2000). "Out-of-Sample Tests of Forecasting Accuracy: An
  Analysis and Review." International Journal of Forecasting 16(4), 437-450.
- Towers, M. et al. (2024). "Gymnasium: A Standard Interface for Reinforcement
  Learning Environments." arXiv:2407.17032.
- Whaley, R. E. (2000). "The Investor Fear Gauge." Journal of Portfolio
  Management 26(3), 12-17. See also Whaley (2009), "Understanding the VIX,"
  Journal of Portfolio Management 35(3).
