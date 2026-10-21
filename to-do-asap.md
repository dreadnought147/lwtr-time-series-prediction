# To Do, ASAP

Two sources, joined. Nothing is paraphrased — every line is cut verbatim from
its source file.

- `SUP` = `transcription/refined/action-items.md` (Geraud). 56 action items. His
  own recording cites `(3, 6)` are kept on the line.
- `EVAL` = `evaluation_report.md` (my own pass, 2026-10-21). 62 items. Their
  `[DO]/[BUG]/[GAP]/[STAT]` tags and section numbers are kept.

Ordering method:

1. The spine is Geraud's items, in his own tiers — the things he flagged as
   biggest or as conclusion-invalidating first, then Must-do, then sanity
   checks and pins, then report requirements, then the items he explicitly
   called "not hard requirements".
2. Each EVAL item is nested under the SUP item it serves, so it inherits that
   item's priority. An EVAL item is nested there because it is either a
   prerequisite for his item, or the same concern stated from the code side.
3. EVAL items with no SUP counterpart go in the last tier, T5. Low in the order
   because he did not raise them — not because they are small. Two of them
   (selection on test, final holdout) are severe; read T5 before deciding to
   skip it.
4. `NOTE (mine)` lines are my own connective text, never his and never from the
   evaluation report.

Counts: 56 SUP + 62 EVAL = 118 items. Geraud's 11 standing judgements and 3
"guidance changed" notes are not tasks and are not included here — read them in
the source file.

---

# T0 — Conclusion-invalidating, or his stated biggest emphasis

## T0.1
- ~~Fix the standard deviation on every plot and table. Mean and std are to be computed **across the 7 seeds only** - not across the 31 windows. Several plots currently show the mean with no variance at all (the equity curves among them). Tables get plus-minus std. Bar charts get error bars. (5)~~ **DONE 2026-10-21** — all plots; the results table is T1.9, still open. D056.
  - ~~`[EVAL §1]` [BUG] The `champion` block inside every `results/*/summary.json` is computed from a partial subset of runs, because `train.py` only appends to `per_window_records` for triplets executed in the final resumed session (train.py:263, 434, 452). Reason: FS summary.json ranks k=5 as champion on 14 window-rows while k=10/20 have 217 each, and k=3 is missing entirely; AS summary.json contains only k=10 (87 rows) and k=20 (217); the LSTM summary.json has `per_window: []`. Any number quoted from these blocks is a selection artefact.~~ **CONFIRMED 2026-10-21** — reproduced exactly. Root cause still live in `train.py`; see D055.
  - ~~`[EVAL §1]` [DO] Recompute every headline number from windows.csv and either regenerate summary.json by a post-hoc aggregation script or delete the champion blocks.~~ **DONE 2026-10-21** — `scripts/rebuild_summaries.py`; originals kept as `summary.partial-*.json`. D055. (Note: "enough to put all results in doubt" is overstated — nothing downstream ever read those blocks; see `disagreements.md` entry 2.)
  - ~~`[EVAL §1]` [DO] Complete-data means (from windows.csv, mean agent Sharpe over 7 seeds x 31 windows): FS k=3 0.484, k=5 0.435, k=10 0.310, k=20 0.321; AS k=3 0.459, k=5 0.515, k=10 0.448, k=20 0.437; LSTM best HP 0.208, LSTM overall 0.082; buy-and-hold 0.841.~~ **VERIFIED 2026-10-21** — every figure reproduced independently from windows.csv.
  - ~~`[EVAL §5]` [STAT] Error bars in the current plots are std over seeds x windows pooled (generate_plots.py:238-293: "mean Sharpe ± std across seeds × windows"). Reason: this pools two different variance sources (which environment year it was, and which random seed) and treats 217 rows as 217 independent samples — pseudo-replication. The honest unit of analysis is the window (n=31), with seeds averaged inside it.~~ **DONE 2026-10-21** — D056. Both spreads now shown; seed std draws the bar, window std the faint whisker.
  - `[EVAL §5]` [STAT] The significance of the headline AS-vs-FS difference depends entirely on that choice. Paired AS−FS mean Sharpe at matched k is +0.077; pooled over all 868 pairs a one-sample t-test gives p=0.024 (Wilcoxon p=0.030), but after averaging the 7 seeds within each window (n=31 per k, the correct unit) the per-k p-values are 0.71 (k=3), 0.41 (k=5), 0.23 (k=10), 0.37 (k=20) — none significant. Reason: the "AS wins" claim is not supported once pseudo-replication is removed. Per-k win rates (38%, 49%, 56%, 54% of pairs) say the same thing.
  - ~~`[EVAL §11]` [DO] Label every figure with its unit of analysis and what the error bar is (n windows, n seeds, std vs SEM vs bootstrap CI). Reason: the current rq1/rq2 captions say "mean ± std across seeds × windows", which is the pseudo-replication in §5 stated in the caption itself.~~ **DONE 2026-10-21** — every regenerated figure states n and which spread it draws. D056.
  - ~~`[EVAL §5]` [DO] Report a probability-of-improvement / win-rate plus a bootstrap CI on the mean paired difference, not just mean ± std.~~ **DONE 2026-10-21** — `plots/rq2/07_paired_as_minus_fs.{png,txt,csv}`. Every CI contains 0; every win-rate CI contains 50%. D058.
  - ~~`[EVAL §5]` [DO] Prefer interquartile mean or median alongside the mean for Sharpe. Reason: per-window Sharpe std is ~0.9-1.3, i.e. the spread is an order of magnitude larger than the effect, so the mean is outlier-driven.~~ **DONE 2026-10-21** — IQM and median in the same table; IQM is below the mean at every k, confirming the mean is outlier-driven. D058.
  - NOTE (mine): his rule (std across the 7 seeds only) and my EVAL §5 items (window as the unit) are in direct conflict. Seed-std is the smaller number at every k, so it produces tighter error bars than window-std. Do it his way as the headline since he asked for it, show the window-level version alongside, and put the question to him — this is the single decision that most changes how the RQ2 result reads.

## T0.2
- Plot all 31 walk-forward windows individually, for the RL runs as well as the LSTM sweep. Then decide how to present them. Geraud's guess is a timeline is the right presentation, because it *is* a timeline - and a trend across the years may show up. A 6-by-5 grid is the fallback layout. He expects this to shrink the variance problem and make the story clean. (4, 5, 6)
  - ~~`[EVAL §4]` [STAT] The 31 windows are not independent: train_size 1260 with step 252 means adjacent windows share 4/5 of their training data. Reason: any error bar or test that treats windows as i.i.d. overstates confidence; say so, and prefer a block bootstrap over windows.~~ **DONE 2026-10-21** — moving-block bootstrap, block length 5, in `plot_rq2_paired()`. Both plain and block intervals reported. D058.
  - `[EVAL §8]` [DO] Report the regime composition of the 31 test windows (bull/bear/high-vol years, and B&H Sharpe per window). Reason: the agents are long-biased, so knowing how many test years were up-years tells the reader how much of the result is sample composition.

## T0.3
- Check the reward curves for every RL training run, and the loss curves for the supervised baseline. Loss should decrease then plateau; reward should increase then plateau, or at least be stable. **If the reward curves are noisy and flat, every conclusion drawn from the results has to be re-rated** - it would mean the models never learned and the numbers are randomness and luck. Geraud called this the basic sanity check. (5, 6)
  - `[EVAL §9]` [DO] Report explained_variance and entropy at end of training per config, not just plot them. Reason: if explained variance is near zero the value function learned nothing and the policy is effectively noise — that single number changes how every result should be read.
  - `[EVAL §9]` [DO] Plot learning curves as test-window Sharpe vs training timesteps for at least one (k, seed). Reason: it is the sample-efficiency claim in the proposal, and it also shows whether 100k timesteps was enough or too many.
  - `[EVAL §4]` [DO] Report a train-slice vs test-slice Sharpe gap per config. Reason: each PPO agent sees one 1260-day trajectory for 100k timesteps (~79 passes over the same data) with no early stopping, so overfitting is the default expectation and the gap is the only evidence either way.

## T0.4
- Fix the explained-variance plot - it is currently not legible, possibly misplotted. (5)

## T0.5
- **31 X/Y pairs is extremely small.** Geraud pushed on how many data points the training set actually has. Benny's answer was 31 - one per walk-forward window. Geraud's response: abstracting away the fact that it's financial data, 31 samples sounds very, very small for *any* machine learning model training, and he asked for a note on it. His reasoning: a model needs to see many examples to learn which features to extract and how to use them, and the more complex the problem the more examples it needs. The underlying daily series being 9,180 days long does not change this - in that framing one window is one X. This is the deepest methodological challenge in any of the seven recordings and it is not yet addressed anywhere in the project. (v1)
  - `[EVAL §5]` [DO] Power statement: with 31 windows and per-window Sharpe sd ~1.0, the minimum detectable paired difference at 80% power is roughly 0.5 Sharpe. Reason: the observed +0.08 is far below that, so the correct conclusion is "no detectable difference", not "AS is better". This framing turns a weak result into a defensible negative result — which the proposal already pre-registered as a valid contribution.
  - `[EVAL §5]` [STAT] State the number of configurations tried and deflate or at least caveat the best Sharpe accordingly. Reason: 4 LSTM shapes + 8 RL configs selected against one test set is textbook multiple testing (Bailey & Lopez de Prado's deflated Sharpe; Harvey & Liu on t-stat thresholds in finance).

## T0.6
- **Benchmark the baseline Sharpe against prior work, and enumerate every difference.** Geraud said he puts "a huge emphasis" on this. The steps he gave: find the papers using S&P that are popular in prior work; look at what Sharpe ratio *their* baselines get; compare against ours. If theirs is better, extract (i) which dataset they used, (ii) how big their dataset was in X/Y pairs, (iii) their history length versus our 24-25 days, (iv) their prediction target versus ours (next day). Then list every point of difference between their experimentation and ours. His stated motivation: at the poster presentation there will be people from banks, and showing a baseline Sharpe that low after hyperparameter tuning will draw exactly this question. (v3)
  - `[EVAL §2]` [DO] Report prominently that neither mechanism beats the buy-and-hold floor: mean B&H Sharpe 0.841 vs FS 0.387 and AS 0.465 across all 868 rows; the agent beats B&H in only 27.2% (FS) and 24.5% (AS) of the 217 (seed, window) pairs per mechanism. Reason: the proposal itself set this rule ("no configuration that fails to beat buy-and-hold earns a 'useful trader' label"), so the project is graded against its own stated criterion.
  - `[EVAL §2]` [DO] State explicitly that this does not invalidate RQ2. Reason: RQ2 is a *relative* mechanism comparison (AS vs FS at matched k), which remains meaningful even when both are below a passive benchmark — but the report must say this rather than letting the reader assume the agents are profitable.
  - `[EVAL §2]` [DO] Add a transaction-cost sensitivity curve (recompute rewards from steps.csv at 1, 5, 10, 25 bps). Reason: the configs use 0.0001 (1 bp) per unit of position change, which is optimistic; agents flip positions often, so the ranking may be cost-dependent. This is a standard robustness check and costs no retraining.
  - `[EVAL §8]` [DO] Answer the question already standing open in research_log.md: whether ~36 years of daily data is a defensible span versus comparable published work. Reason: it is the obvious "is the sample big enough" challenge, and Arnott/Harvey/Markowitz's backtesting protocol opens with exactly this data-constraint point for ML in finance.
  - `[EVAL §8]` [BUG] The loader uses `close`, not `adj_close`, for the S&P (loader.py:16-19). Reason: price return excludes dividends, which slightly understates the buy-and-hold benchmark the project is judged against. Either switch or state it.

---

# T1 — Must do, new work

## T1.1
- Run the RL agent with an LSTM policy, not just MLP, so there is an apples-to-apples architecture comparison against the supervised LSTM baseline. This is the single biggest time-consuming item left. (3, 6)
  - `[EVAL §3]` [STAT] AS is not parameter-matched or action-space-matched to FS: it carries an extra memory head and a MultiDiscrete([3, k]) action space at the same 100k timesteps. Reason: AS has a harder learning problem at equal compute, so a null result is partly confounded with sample budget. Either note it as a limitation or add one AS run at a larger timestep budget to show the result is not budget-limited.

## T1.2
- For that RL+LSTM run, do not do a fresh hyperparameter sweep. Reuse the exact best hyperparameters already found for the supervised LSTM baseline. If RL still wins using the baseline's own best settings, the comparison is fair and covered. (3, 4)

## T1.3
- Keep the existing MLP RL results too. Two experiments side by side: (a) same-architecture comparison, LSTM vs LSTM, isolates the RL-vs-SL paradigm effect; (b) different-architecture comparison, shows whether RL wins regardless of architecture. Nothing gets thrown away. (3)

## T1.4
- Run the RL agent with an LSTM instead of MLP. This is where the instruction originates; Geraud's reasoning is the research-question discipline below. He judged it worth doing given FS took ~1 day, AS ~2 days, LSTM ~1 day or less. (v4)

## T1.5
- ~~**There are two LSTM implementations in the code base.** When switching the RL agent to LSTM, try both and see how the implementation choice affects performance. (v4)~~ **CLOSED 2026-10-21, not work** — this was Geraud asking a question, not issuing an instruction. Benny answered on the recording that there is only one, the supervised baseline (`src/baselines/lstm.py`). Confirmed against the repo: no second implementation exists. Nothing to run. D059.

## T1.6
- Add an RL run with the full history as the memory budget - set k to the length of an episode, in addition to k=20. The AS paper's framing is that FS with full history is usually the strongest but slowest to learn, so the full-history point is needed to make the memory story complete. (1, 3)
  - `[EVAL §3]` [DO] A k=1 PPO run is cheap (31 windows x 7 seeds at the same 100k timesteps) and restores the dropped step of the staircase. Reason: it also gives the only clean answer to "is any of this better than reacting to today's VIX alone", which is the first question any examiner asks.

## T1.7
- Add a comparison at k=25 (or k = full history) against the supervised model, on the grounds that the supervised model is effectively already using the full history: in the supervised case the last 25 days *is* one X at a given time step, so in time-series terms it sees the full history. This is the origin of the full-history item in Part A. (v4)

## T1.8
- Make the LSTM baseline able to output flat. Currently it emits a continuous number and the rounding rule means it is effectively two-position (long/short) - it structurally can never go flat, because the chance of an exact zero is nil. Two parts: (i) check what prior work does about this; (ii) consider making the LSTM predict the three discrete classes directly instead of a continuous value, labelled from the known correct forecast. (5)
  - `[EVAL §7]` [BUG] With `threshold: 0.0`, `predictions_to_postsions` can never output flat (lstm.py:97-104): >0 → long, <0 → short. Confirmed in the data — `pos_flat` is exactly 0 in all 868 LSTM rows, while RL agents go flat on 18-34 days per window. Reason: the baseline is forced into a 2-action space while the agents have 3, which is an uncontrolled difference. Sweep a dead-band threshold (e.g. 0, 0.25σ, 0.5σ of the forecast) and report the best.
  - `[EVAL §7]` [GAP] No validation-based early stopping for the LSTM: 15 epochs fixed, and hp_id selected on test Sharpe. Reason: same in-sample-selection problem as §4, and a baseline whose capacity was never tuned on held-out data is not a "strong" baseline in the sense RQ1 needs.
  - `[EVAL §7]` [GAP] No forecast-quality metrics at all: no MSE/MAE, no directional accuracy, no information coefficient, no comparison against a naive (always-long, or lagged-return) forecaster. Reason: if the LSTM is a bad forecaster the RQ1 result says nothing about paradigms; forecast metrics separate "RL framing wins" from "my baseline was weak".
  - `[EVAL §7]` [DO] Documentation mismatch: train_lstm.py's docstring claims obs *and targets* are z-scored, but only obs are (train_lstm.py:296-303). Reason: small, but it is exactly the kind of claim a marker will check against the code.

## T1.9
- Build one combined comparison plot rather than separate per-method plots: MLP, LSTM, FS, AS and buy-and-hold all on the same axes. The x-axis is still to be decided. (4, 6)
  - `[EVAL §11]` [DO] Put the buy-and-hold line on every performance figure, including the RQ2 by-k plots. Reason: without it the reader sees Sharpe 0.5 and assumes success.
  - `[EVAL §11]` [DO] Add one results table with, per config: n runs, mean and IQM test Sharpe, bootstrap 95% CI, mean cumulative return, mean max drawdown, win rate vs B&H, mean position flips. Reason: a single table makes the whole result auditable and is what a marker looks for first.

## T1.10
- Investigate *when* AS makes its position flips - does flip timing overlap with regime starts and ends? Geraud flagged this as one of the cool experiments and wants it checked. (5)
  - `[EVAL §6]` [GAP] Objective 7 of the proposal — "do retentions cluster around identifiable market events (VIX spikes, regime transitions)?" — is not implemented anywhere. Reason: this is the only analysis that directly tests the κ ≪ k* sparsity premise, which the proposal names as the project's central assumption. steps.csv has date, memory_action and position per day, so an event-alignment plot (overwrite behaviour vs VIX percentile / high-vol regime indicator) is a few lines of pandas.
  - `[EVAL §6]` [GAP] No measure of the effective memory horizon — the age in days of the oldest surviving slot. Reason: this is the cleanest single quantity for "does the agent keep old information", and it is the natural quantitative form of the κ ≪ k* claim. It needs a short replay of the saved policy, not a retrain.

## T1.11
- Separate out what is actually causing AS's lower flip rate: the learned policy, or the transaction-cost term already in the reward function. Possibly both. (5)
  - `[EVAL §2]` [GAP] No constant-long control separate from B&H, and no random-policy or untrained-policy control. Reason: the agents are long on ~70-78% of test days (FS k=3: 191 long / 43 short / 18 flat out of 252), so most of the measured Sharpe is market beta in a mostly-rising sample, not skill. A random-position and a frozen-init-policy control cost one rollout each and separate "learned something" from "happened to be long".

## T1.12
- Create the Overleaf project for the report as soon as possible, so there is enough time to polish. (6)
  - `[EVAL §11]` [DO] Fix the docs so one story is told: Week 2 slides say 4 configurations / 3 RQs, scopedinrqs.md says 3 / 2, design_doc.md's Picture 7 checklist still lists a final-holdout hard rule and a memory-footprint metric that were not delivered. Reason: a reader comparing the deck, the proposal and the results will find three different experiment designs.
  - `[EVAL §1]` [BUG] `takeoff/status.md` (2026-10-05) says sweeps are complete and the diagnostic data is logged, but also still lists the final-holdout rule as unenforced. The claim "champion identified by mean Sharpe across seeds" is true of the code but the code picked it on test data (see §4). Reason: status docs that overstate completeness are the first thing a supervisor cross-checks.

## T1.13
- Write the research report while the LSTM runs are training - don't serialise them. (v4)

## T1.14
- Put all the hyperparameter-sweep detail in the appendix, after the references. One or two sentences in the main body. (v3)

## T1.15
- Put a note on the counterintuitive LSTM result: one layer beating two layers is surprising, because one layer is almost linear and cannot learn non-linear features. (v3)

---

# T2 — Sanity checks, verification, and pins

## T2.1
- Index-bias sanity check on AS retention. The concern someone could raise: AS may look biased toward old slots purely because of how slots are indexed in the code. Geraud concluded he does not need compute spent on this, since PPO picks any slot with equal probability - but it is worth being able to answer, because a reader will not know the internals. The test: flip the indexing so slot 0 maps to position k in the frame stack, retrain, and see if the preference flips with it. If it does, AS is genuinely learning. Implement as a fixed permutation (`np.random.permutation` over the index list, kept fixed) applied as an environment wrapper. Must stay fixed, not re-randomised per step, so the model has a consistent notion of oldest and newest. (5)
  - `[EVAL §6]` [DO] The most important missing control in the whole project: AS with uniformly random memory actions, everything else identical. Reason: it is the only way to show that any AS-vs-FS difference comes from *learning* what to keep rather than from AS's different stack composition. One extra sweep arm, same cost as an existing one.
  - `[EVAL §6]` [BUG] Within a single (k, seed, window) run the memory policy is near-degenerate: the modal slot takes 90% (k=3), 82% (k=5), 74% (k=10) and 63% (k=20) of all overwrites. Pooled across runs the histogram looks mildly non-uniform (max slot share 0.41 / 0.28 / 0.18 / 0.10 vs uniform 0.33 / 0.20 / 0.10 / 0.05). Reason: the pooled figure therefore hides the real finding — individual agents collapse to a nearly *constant* eviction slot, i.e. AS behaves as a fixed (non-FIFO) rule rather than a state-dependent memory policy. That is a genuinely interesting, reportable negative mechanism result, and it is the opposite of what the current pooled histogram suggests.
  - `[EVAL §6]` [BUG] Retention is measured from `deterministic=True` rollouts (train.py:81). Reason: argmax over the memory head guarantees low-entropy slot choice, so the observed concentration is partly an artefact of evaluation mode. Report stochastic-policy retention too, or measure dependence between observation and memory action (e.g. mutual information / a logistic fit) rather than marginal counts.
  - `[EVAL §6]` [BUG] `concentration()` = 1 − min/max (retention.py:99-111) is a range statistic on counts with no null distribution. Reason: it has no sampling distribution, so "0.6 = strongly skewed" is an assertion, not a result. Use normalised entropy or a chi-square / multinomial test against uniform, and report the value a *random* memory policy produces as the null.

## T2.2
- Confirm the slot positions the model picks actually map to the memory indices as assumed in the retention plot. Benny had not checked this. (5)
  - ~~`[EVAL §6]` [BUG] The plotted retention histogram in plots/rq2/03 sorts `mem_slot_*` columns as text, so mem_slot_10 lands before mem_slot_2 and small-k runs read empty columns. Reason: already identified and fixed in defence/make_defence_figures.py — use the corrected figure and delete or regenerate the broken one so two contradictory figures are not both in the repo.~~ **DONE 2026-10-21** — fixed in `generate_plots.py` and regenerated; `src/evaluation/retention.py` still has the old form. D057.
  - `[EVAL §6]` [BUG] `retention_from_summary` reads `doc["aggregate"]["agent_memory_action_counts_total"]` (retention.py:73-75) but `train.py` never writes an `aggregate` key. Reason: the documented fast path silently returns None for every real run — the diagnostic as shipped does not run on the sweep output.

## T2.3
- Check whether the VIX/S&P dataset has been used in prior work. (1)
  - `[EVAL §8]` [DO] Name the data vintage and the download date; `scripts/downlaod_data.py` (filename typo) is the only provenance record. Reason: Yahoo Finance series are revised and re-downloads are not reproducible; a one-line provenance note (ticker, field, download date, row count, checksum) makes the result reproducible.
  - `[EVAL §8]` [GAP] No check that the VIX-S&P inner join did not silently drop days. Reason: the join is `how="inner"` with no assertion on the row count; 9190 raw rows vs 9188 after dropna is consistent, but an explicit assertion is cheap insurance for a pipeline whose correctness everything downstream depends on.

## T2.4
- Check what prior work did with the reward formula, and whether their result is equivalent to the RL Frame Stacking baseline. (2)

## T2.5
- Find the paper(s) behind the reward formula. Geraud's method: check at most three, at least one, picking papers that are either very recent (so few citations yet) or a couple of years old with many citations. Find one showing that reward formula. Then establish (i) a citation that backs the choice of reward function, and (ii) whether they were doing reinforcement learning - a reward signal implies they were - and if so, the details of their RL process. Benny could not say for certain whether the papers he took it from were RL papers. (v2)

## T2.6
- Check whether prior work has used the VIX index, and justify the dataset choice explicitly in the report. Geraud's verdict on the sentiment/regime justification Benny gave: reasonable, not the strongest, but fine. His caveat: for publishable results at least one of the datasets would have to be the standard one people use. He said this is not a huge problem. (v1)
  - `[EVAL §8]` [DO] Add a formal stationarity test (ADF and/or KPSS) alongside the rolling-window visual. Reason: stats_report.txt itself calls the rolling plot a "visual proxy for ADF/KPSS"; running the actual test is two lines of statsmodels and removes a soft spot, given that non-stationarity is the project's stated reason for walk-forward.

## T2.7
- Find and cite the research backing the ~5-year regime length that motivates the 5-year training window. Benny said he researched it and could back it up but could not recall the source. (v1)

## T2.8
- Is the LSTM output a single continuous number or a vector? If it is a vector, what makes it scalar for the threshold rule? (2)

## T2.9
- The input/output spec of the LSTM in full: what is one input, how is it obtained at each epoch, and what is its dimensionality (assumed 20 x something - 20 days, each day a vector of features)? (2)
  - `[EVAL §7]` [BUG] Off-by-one information set: in `rollout_lstm` the forecast for test day 0 is made from the window ending on the last *train* day, so each forecast predicts `ret[t]` from observations up to `t-1`, while the PPO agent at day t sees `obs[t]` and earns `ret[t]` (train_lstm.py:77-99 vs timeseries_env.py:52-63). Reason: the LSTM is systematically one day behind the RL agents. RQ1 is "does RL beat a *strong* supervised baseline", so a one-day information handicap undermines the comparison — and the measured gap is large (LSTM 0.082 vs FS 0.387 mean Sharpe), which makes the handicap worth ruling out before the claim is made.

## T2.10
- Is the look-back 20 trading days or 5 years? The slides said both in different places. (2)

## T2.11
- Which loss function is the LSTM trained with? Geraud's point: regression on a continuous output implies MSE; discrete classes imply cross-entropy. The loss choice follows from the output type. (2)
  - `[EVAL §7]` [STAT] The LSTM minimises MSE on raw log-returns (std 0.011, mean 0.0003) with no target scaling. Reason: near-zero-variance targets give tiny gradients and push the forecaster toward a constant, which is consistent with the observed near-coin-flip Sharpe. Either standardise the target or report directional accuracy / IC alongside MSE so the reader can see whether the baseline learned anything at all.

## T2.12
- PPO's episode length / max steps per episode - Benny glossed over this. Geraud: it's an important hyperparameter, go find what it was set to. (3)
  - `[EVAL §9]` [STAT] Episode structure is one 1260-day episode per window with n_steps=2048 > episode length. Reason: rollouts straddle episode resets and the agent sees the same trajectory ~79 times; worth one sentence because it determines whether "sample efficiency" means anything here.

---

# T3 — Report requirements

## T3.1
- Overleaf project, created as soon as possible, shared with Geraud.

## T3.2
- First review pass: he reads **only the title and the abstract paragraph**. Everything else he checks for structure only.
  - `[EVAL §11]` [DO] Frame the conclusion as a pre-registered negative/null result: at matched memory budget, no detectable difference between learned and fixed memory on this task, with the sparsity premise weakened by the differenced observation. Reason: the proposal pre-committed to a negative result being a contribution, and the data supports "no detectable difference" far better than it supports "AS wins".

## T3.3
- Introduction: bullet points fine, same shape as the proposal, **plus** the three main contributions of the work at the end of the introduction, correlating with the research questions.
  - `[EVAL §3]` [GAP] Slides and proposal describe four configurations and three RQs; `scopedinrqs.md` drops the memoryless PPO (k=1) baseline and RQ2-original. Reason: the deck and the proposal are now inconsistent with what was run, and the scope cut must appear in the report with its reason, or a marker will read the gap as an omission. Also: with no k=1 run, the claim "memory helps on financial data" cannot be made at all — only "learned memory vs fixed memory".

## T3.4
- Background: bullet points of what is planned, don't waste time on prose yet. Same as the proposal.

## T3.5
- Methodology: section headings plus bullet points of what goes in each.

## T3.6
- Results: include the figures with full captions - he will look at those.

## T3.7
- Write in third person. "We", not "I".

## T3.8
- Aim at publishable standard: it should read like the best papers you have read. Room for aesthetic creativity, but not to the point of distracting from conveying information clearly. Run any specific creative idea past him early - that's the point of starting now.
  - `[EVAL §11]` [DO] Write the limitations section around the three items that genuinely bound the conclusions: single observation stream (one differenced feature), model selection on test windows, and no cost/robustness sweep. Reason: naming your own three biggest limitations is the cheapest credibility in an honours defence, and all three are already true and documented above.

## T3.9
- Poster for Innovation Day: deadline unknown, double-check with Helen. Geraud will also check and send PowerPoint templates.

---

# T4 — Suggestions and framing, his "not hard requirements"

## T4.1
- Use the negative of the supervised loss as the RL reward function. Geraud listed this as one of the things he was interested in seeing tried: the loss states what metric you care about and want minimised, so the RL agent maximises its negative. (2)

## T4.2
- Use TensorBoard for quick visualisation during presentations: `tensorboard --logdir <dir>`, open the localhost URL. It will not do the seed averaging, so paper and report figures still get hand-plotted with mean plus shaded std. (3)

## T4.3
- Frame the k sweep as an experiment measuring the effect of memory, not as hyperparameter tuning. Better story. (4)
  - `[EVAL §9]` [GAP] Objective 6 promised sample efficiency and training-time memory footprint; only `train_wall_seconds` is logged. Reason: memory footprint vs k is the headline efficiency argument of Adaptive Stacking (O(κ) vs O(k*)), so it is both promised and on-topic. Peak RSS per run, or even just parameter count and observation-vector size vs k, is enough at this scope.

## T4.4
- Look at *which* observations AS chose to remember and check whether they coincide with significant market changes while ignoring the most stochastic ones. Geraud's stretch suggestion: apply the probabilistic graphical model approach from Benny's PGM seminar project to the AS retention choices, to measure how important each remembered observation is to the decision made. If it's plug-and-play, he said it would "put your stuff to the next level". (5)

## T4.5
- The real contribution he wants surfaced: how correlated is the learned policy with what you'd expect a good strategy to be for the given test period. Using an actual real-world problem is what gives this work its contribution. (5)

## T4.6
- On the entropy-loss plot, for now just say it makes sense that the model should not become extremely confident, because the data has a lot of underlying uncertainty - confidence should rise with learning but not saturate. (5)

## T4.7
- Fairness rules, for the write-up: either use library defaults (e.g. PPO Stable Baselines defaults), or hyperparameters from prior work, or tune them. If architectures differ and you don't sweep, keep the total parameter counts similar in size - otherwise a reviewer can argue either way (more capacity, or more complexity). (3, 4)
  - `[EVAL §9]` [GAP] No PPO hyperparameter justification. The proposal says hyperparameters follow the Adaptive Stacking paper's defaults; the configs use SB3 defaults (lr 3e-4, n_steps 2048, batch 64, no entropy coefficient set, default [64,64] net). Reason: either verify the claim against the reference repo or change the sentence in the report. It is a one-line claim that is easy to check and embarrassing if wrong.
  - `[EVAL §4]` [BUG] The RL side applies no observation normalisation at all (no VecNormalize, raw log-returns with std 0.068 into an MLP). Reason: this is an unequal-footing difference between the LSTM baseline and the PPO agents, not just a tuning choice; mention it, or add VecNormalize to both arms.
  - NOTE (mine): the normalisation item is a code fix, not a write-up line, and it is cheap. It sits this low only because he filed parameter fairness under suggestions.

## T4.8
- Not sweeping is defensible: it handicaps you, so beating the baseline anyway means you are covered. Tuning would only have improved your side. (3)

## T4.9
- Move the less useful plots (the AS-minus-FS per-year dot plot was named) to the appendix rather than dropping them. Geraud will say which graphs to move once he reads the report. (5)

## T4.10
- **Never say "I blindly followed" prior work** - not in the report, and not when defending the research at the poster presentation. Reframe as: this is the standard in financial research, [one sentence on why it is the standard], and here is the prior work you can go to for deeper reasons. One sentence of justification is enough. Geraud raised this specifically about the log-returns choice and Moody & Saffell, and praised the correct version of the explanation Benny had given moments earlier. (v1)

## T4.11
- Include the negative VIX/S&P correlation in the argument: because there is no positive linear relationship between the observation stream and the traded asset, the result can be attributed more surgically to the memory contribution rather than to the two series being linearly correlated. (v1)

## T4.12
- Research-question discipline, as Geraud phrased it: a research question compares one thing with everything else held constant. Comparing architectures means holding the learning algorithm and dataset fixed. Comparing supervised learning against RL Frame Stacking means holding the architecture and dataset fixed. Changing both the architecture (MLP vs LSTM) and the learning algorithm means the performance difference can no longer be attributed to the memory mechanism. (v4)

## T4.13
- Benny's own admission on record: he froze PPO and MLP across FS and AS - inside the RL side only - and did not think to freeze across the two learning paradigms. Geraud: "you should have also done the same for the supervised side." (v4)

## T4.14
- The supervised baseline could have used an MLP, and LSTM being the more standard choice is the reason it didn't. Geraud: "the fact that you *could* is the exact point." (v4)

---

# T5 — My items with no counterpart in his list

NOTE (mine): low in the order only because he did not raise them. T5.1 and T5.2
are the two most serious findings in the whole evaluation — read them before
deciding this tier can wait.

## T5.1
- `[EVAL §4]` [BUG] Model selection happens on the test data. `compute_champion` ranks k (and LSTM hp_id) by mean Sharpe over the walk-forward *test* windows (train.py:174-197; train_lstm.py:174-199), and the same windows are then used to report the result. Reason: this is in-sample selection — the reported champion Sharpe is biased upward, and the bias grows with the number of configurations tried (4 k values x 2 mechanisms x 4 LSTM shapes = 10 selections against one test set). Bailey & Lopez de Prado's deflated-Sharpe argument is exactly this: a hold-out does not remove data-snooping bias when many strategies are tested against the same hold-out.
- `[EVAL §4]` [DO] Minimum credible fix at honours scope: carve a validation slice inside each training window (e.g. last 252 train days), select k / hp there, and report the selected config once on the test windows. Reason: no retraining of the sweep is needed if you instead do the cheaper version — report *all* k values rather than a champion, and label any "best k" as selected in-sample.

## T5.2
- `[EVAL §4]` [GAP] The final-holdout rule the project set for itself (last ~2 years untouched) was never enforced; all sweeps use the full 1990-2026 range. `takeoff/status.md` already admits this. Reason: it has to be either implemented or written up as a deviation with the reason, because the design_doc explicitly draws it as a "hard rule" line.
- `[EVAL §4]` [BUG] No gap/embargo between train and test slices. Walk-forward windows are contiguous (walk_forward.py:69-92) and the target is a shifted next-day return (transforms.py:39), so the last training day's label covers the first test day's price move. Reason: a one-day leak is small but free to remove — drop one row at each boundary (purged/embargoed walk-forward, Lopez de Prado). Worth a footnote either way.

## T5.3
- `[EVAL §3]` [BUG] The agent never observes the VIX *level* — it observes the VIX *log-return* (transforms.py:31, fed to TimeSeries as `observations`). Reason: the entire motivation in the proposal and Week 2 slides is that the VIX level encodes regime and that "a spike from six weeks back still shapes today's level". Log-differencing removes exactly that persistence, so the sparse-informative-observation premise that Adaptive Stacking depends on is destroyed before the agent sees anything. Either (a) feed the level (z-scored on the train slice) or a level-plus-change pair, or (b) state plainly in the report that the observation is a differenced series and that this weakens the sparsity premise. (a) is one line of code and is the single most defensible experimental improvement available.
- `[EVAL §3]` [BUG] The agent's own current position is not in the observation (timeseries_env.py:29, `_obs` returns only the observation value), yet the reward depends on it through the transaction cost (timeseries_env.py:60-63). Reason: this makes the env non-Markov in a way that was not intended and that no memory mechanism can fix — the agent cannot know whether an action incurs a cost. Standard practice in financial RL (Moody & Saffell onward) includes the current position in the state. Fix it, or document it as a known limitation that penalises all three configurations equally.

## T5.4
- `[EVAL §10]` [GAP] `tests/` contains only `__init__.py` — no unit tests. Reason: the correctness of this project rests almost entirely on index alignment (the `shift(-1)` target, the FrameStack/AdaptiveStack cold start, the LSTM window-to-target offset), which is precisely the class of bug that produces plausible-looking wrong results. Three small tests — (a) reward at day t equals position x next-day return, (b) FS and AS reset to identical observations, (c) LSTM forecast index aligns with the day it trades — would catch the two off-by-one issues above and are directly defensible as rigour.
- `[EVAL §10]` [BUG] `adaptive_stack.py`'s `__main__` smoke test imports `TimeSeriesEnv`, but the class is named `TimeSeries` (adaptive_stack.py:189 vs timeseries_env.py:5). Reason: the documented self-check cannot run; also a sign the two modules drifted apart.
- `[EVAL §10]` [DO] Pin dependency versions in requirements.txt and record the Python version. Reason: SB3/Gymnasium APIs move; without pins the sweeps are not re-runnable, and reproducibility is cheap marks.
- `[EVAL §10]` [DO] Record a git commit hash (and the config) in each summary.json. Reason: `config` is saved but not the code version, so a result cannot be tied to the code that produced it.
- `[EVAL §10]` [DO] Note the known env quirks in the limitations section: `terminated` is computed after incrementing `t`, so the final day's observation is a zero placeholder, and the last row's reward uses the final available next-day return. Reason: harmless but visible in steps.csv, and better named by you than found by a marker.
