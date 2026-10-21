# What Geraud said to do - all seven recordings

Two sections. The `v`-series (videos v1-v4) came **earlier** chronologically; the numbered recordings 1-6 came after. Each item cites its recording. "Pin" = Benny said on the recording that he would go back and check.

**Read the "Where the guidance changed between the two sittings" section at the bottom before acting on anything about LSTM hyperparameters.**

---

# Part A - recordings 1-6 (later sitting)

## Must do - new work

- Run the RL agent with an LSTM policy, not just MLP, so there is an apples-to-apples architecture comparison against the supervised LSTM baseline. This is the single biggest time-consuming item left. (3, 6)
- For that RL+LSTM run, do not do a fresh hyperparameter sweep. Reuse the exact best hyperparameters already found for the supervised LSTM baseline. If RL still wins using the baseline's own best settings, the comparison is fair and covered. (3, 4)
- Keep the existing MLP RL results too. Two experiments side by side: (a) same-architecture comparison, LSTM vs LSTM, isolates the RL-vs-SL paradigm effect; (b) different-architecture comparison, shows whether RL wins regardless of architecture. Nothing gets thrown away. (3)
- Add an RL run with the full history as the memory budget - set k to the length of an episode, in addition to k=20. The AS paper's framing is that FS with full history is usually the strongest but slowest to learn, so the full-history point is needed to make the memory story complete. (1, 3)
- Make the LSTM baseline able to output flat. Currently it emits a continuous number and the rounding rule means it is effectively two-position (long/short) - it structurally can never go flat, because the chance of an exact zero is nil. Two parts: (i) check what prior work does about this; (ii) consider making the LSTM predict the three discrete classes directly instead of a continuous value, labelled from the known correct forecast. (5)
- Check the reward curves for every RL training run, and the loss curves for the supervised baseline. Loss should decrease then plateau; reward should increase then plateau, or at least be stable. **If the reward curves are noisy and flat, every conclusion drawn from the results has to be re-rated** - it would mean the models never learned and the numbers are randomness and luck. Geraud called this the basic sanity check. (5, 6)
- Fix the standard deviation on every plot and table. Mean and std are to be computed **across the 7 seeds only** - not across the 31 windows. Several plots currently show the mean with no variance at all (the equity curves among them). Tables get plus-minus std. Bar charts get error bars. (5)
- Plot all 31 walk-forward windows individually, for the RL runs as well as the LSTM sweep. Then decide how to present them. Geraud's guess is a timeline is the right presentation, because it *is* a timeline - and a trend across the years may show up. A 6-by-5 grid is the fallback layout. He expects this to shrink the variance problem and make the story clean. (4, 5, 6)
- Build one combined comparison plot rather than separate per-method plots: MLP, LSTM, FS, AS and buy-and-hold all on the same axes. The x-axis is still to be decided. (4, 6)
- Investigate *when* AS makes its position flips - does flip timing overlap with regime starts and ends? Geraud flagged this as one of the cool experiments and wants it checked. (5)
- Separate out what is actually causing AS's lower flip rate: the learned policy, or the transaction-cost term already in the reward function. Possibly both. (5)
- Create the Overleaf project for the report as soon as possible, so there is enough time to polish. (6)

## Sanity checks and verification

- Index-bias sanity check on AS retention. The concern someone could raise: AS may look biased toward old slots purely because of how slots are indexed in the code. Geraud concluded he does not need compute spent on this, since PPO picks any slot with equal probability - but it is worth being able to answer, because a reader will not know the internals. The test: flip the indexing so slot 0 maps to position k in the frame stack, retrain, and see if the preference flips with it. If it does, AS is genuinely learning. Implement as a fixed permutation (`np.random.permutation` over the index list, kept fixed) applied as an environment wrapper. Must stay fixed, not re-randomised per step, so the model has a consistent notion of oldest and newest. (5)
- Confirm the slot positions the model picks actually map to the memory indices as assumed in the retention plot. Benny had not checked this. (5)
- Check whether the VIX/S&P dataset has been used in prior work. (1)
- Check what prior work did with the reward formula, and whether their result is equivalent to the RL Frame Stacking baseline. (2)
- Fix the explained-variance plot - it is currently not legible, possibly misplotted. (5)

## Pins - things Benny said he would go back and confirm

- Is the LSTM output a single continuous number or a vector? If it is a vector, what makes it scalar for the threshold rule? (2)
- The input/output spec of the LSTM in full: what is one input, how is it obtained at each epoch, and what is its dimensionality (assumed 20 x something - 20 days, each day a vector of features)? (2)
- Is the look-back 20 trading days or 5 years? The slides said both in different places. (2)
- Which loss function is the LSTM trained with? Geraud's point: regression on a continuous output implies MSE; discrete classes imply cross-entropy. The loss choice follows from the output type. (2)
- PPO's episode length / max steps per episode - Benny glossed over this. Geraud: it's an important hyperparameter, go find what it was set to. (3)

## Suggestions and framing - not hard requirements

- Use the negative of the supervised loss as the RL reward function. Geraud listed this as one of the things he was interested in seeing tried: the loss states what metric you care about and want minimised, so the RL agent maximises its negative. (2)
- Use TensorBoard for quick visualisation during presentations: `tensorboard --logdir <dir>`, open the localhost URL. It will not do the seed averaging, so paper and report figures still get hand-plotted with mean plus shaded std. (3)
- Frame the k sweep as an experiment measuring the effect of memory, not as hyperparameter tuning. Better story. (4)
- Look at *which* observations AS chose to remember and check whether they coincide with significant market changes while ignoring the most stochastic ones. Geraud's stretch suggestion: apply the probabilistic graphical model approach from Benny's PGM seminar project to the AS retention choices, to measure how important each remembered observation is to the decision made. If it's plug-and-play, he said it would "put your stuff to the next level". (5)
- The real contribution he wants surfaced: how correlated is the learned policy with what you'd expect a good strategy to be for the given test period. Using an actual real-world problem is what gives this work its contribution. (5)
- On the entropy-loss plot, for now just say it makes sense that the model should not become extremely confident, because the data has a lot of underlying uncertainty - confidence should rise with learning but not saturate. (5)
- Fairness rules, for the write-up: either use library defaults (e.g. PPO Stable Baselines defaults), or hyperparameters from prior work, or tune them. If architectures differ and you don't sweep, keep the total parameter counts similar in size - otherwise a reviewer can argue either way (more capacity, or more complexity). (3, 4)
- Not sweeping is defensible: it handicaps you, so beating the baseline anyway means you are covered. Tuning would only have improved your side. (3)
- Move the less useful plots (the AS-minus-FS per-year dot plot was named) to the appendix rather than dropping them. Geraud will say which graphs to move once he reads the report. (5)

## Current standing judgements from Geraud on the results

- The AS vs FS difference is **not statistically significant** as currently plotted - the variance is too large to claim it. It may genuinely be that for this dataset FS and AS have no big difference. Fixing the averaging and plotting all 31 windows may change this picture; it may not. (5)
- It is impressive that AS does what it does on much less memory. (5)
- Worth checking whether AS has simply collapsed to the FS strategy. FS always forgets the oldest; AS was found to overwrite the newest, so it has not collapsed. (5)
- On AS keeping old information: plausible that what AS retained was new information once, and it kept it because it judged that information relevant - so "new information is noise" and "it keeps old data" are not in contradiction. (5)
- AS's behaviour capturing systemic risk, lower flip rate, and alignment with buy-and-hold / non-myopic investing is a real finding worth developing. (5)

## Report requirements

- Overleaf project, created as soon as possible, shared with Geraud.
- First review pass: he reads **only the title and the abstract paragraph**. Everything else he checks for structure only.
- Introduction: bullet points fine, same shape as the proposal, **plus** the three main contributions of the work at the end of the introduction, correlating with the research questions.
- Background: bullet points of what is planned, don't waste time on prose yet. Same as the proposal.
- Methodology: section headings plus bullet points of what goes in each.
- Results: include the figures with full captions - he will look at those.
- Write in third person. "We", not "I".
- Aim at publishable standard: it should read like the best papers you have read. Room for aesthetic creativity, but not to the point of distracting from conveying information clearly. Run any specific creative idea past him early - that's the point of starting now.
- Poster for Innovation Day: deadline unknown, double-check with Helen. Geraud will also check and send PowerPoint templates.

---

# Part B - recordings v1-v4 (earlier sitting, the videos)

## The two biggest things raised here

- **31 X/Y pairs is extremely small.** Geraud pushed on how many data points the training set actually has. Benny's answer was 31 - one per walk-forward window. Geraud's response: abstracting away the fact that it's financial data, 31 samples sounds very, very small for *any* machine learning model training, and he asked for a note on it. His reasoning: a model needs to see many examples to learn which features to extract and how to use them, and the more complex the problem the more examples it needs. The underlying daily series being 9,180 days long does not change this - in that framing one window is one X. This is the deepest methodological challenge in any of the seven recordings and it is not yet addressed anywhere in the project. (v1)
- **Benchmark the baseline Sharpe against prior work, and enumerate every difference.** Geraud said he puts "a huge emphasis" on this. The steps he gave: find the papers using S&P that are popular in prior work; look at what Sharpe ratio *their* baselines get; compare against ours. If theirs is better, extract (i) which dataset they used, (ii) how big their dataset was in X/Y pairs, (iii) their history length versus our 24-25 days, (iv) their prediction target versus ours (next day). Then list every point of difference between their experimentation and ours. His stated motivation: at the poster presentation there will be people from banks, and showing a baseline Sharpe that low after hyperparameter tuning will draw exactly this question. (v3)

## Must do - new work

- Run the RL agent with an LSTM instead of MLP. This is where the instruction originates; Geraud's reasoning is the research-question discipline below. He judged it worth doing given FS took ~1 day, AS ~2 days, LSTM ~1 day or less. (v4)
- **There are two LSTM implementations in the code base.** When switching the RL agent to LSTM, try both and see how the implementation choice affects performance. (v4)
- Add a comparison at k=25 (or k = full history) against the supervised model, on the grounds that the supervised model is effectively already using the full history: in the supervised case the last 25 days *is* one X at a given time step, so in time-series terms it sees the full history. This is the origin of the full-history item in Part A. (v4)
- Write the research report while the LSTM runs are training - don't serialise them. (v4)
- Put all the hyperparameter-sweep detail in the appendix, after the references. One or two sentences in the main body. (v3)
- Put a note on the counterintuitive LSTM result: one layer beating two layers is surprising, because one layer is almost linear and cannot learn non-linear features. (v3)
- Find the paper(s) behind the reward formula. Geraud's method: check at most three, at least one, picking papers that are either very recent (so few citations yet) or a couple of years old with many citations. Find one showing that reward formula. Then establish (i) a citation that backs the choice of reward function, and (ii) whether they were doing reinforcement learning - a reward signal implies they were - and if so, the details of their RL process. Benny could not say for certain whether the papers he took it from were RL papers. (v2)
- Check whether prior work has used the VIX index, and justify the dataset choice explicitly in the report. Geraud's verdict on the sentiment/regime justification Benny gave: reasonable, not the strongest, but fine. His caveat: for publishable results at least one of the datasets would have to be the standard one people use. He said this is not a huge problem. (v1)
- Find and cite the research backing the ~5-year regime length that motivates the 5-year training window. Benny said he researched it and could back it up but could not recall the source. (v1)

## Framing and presentation instructions

- **Never say "I blindly followed" prior work** - not in the report, and not when defending the research at the poster presentation. Reframe as: this is the standard in financial research, [one sentence on why it is the standard], and here is the prior work you can go to for deeper reasons. One sentence of justification is enough. Geraud raised this specifically about the log-returns choice and Moody & Saffell, and praised the correct version of the explanation Benny had given moments earlier. (v1)
- Include the negative VIX/S&P correlation in the argument: because there is no positive linear relationship between the observation stream and the traded asset, the result can be attributed more surgically to the memory contribution rather than to the two series being linearly correlated. (v1)
- Research-question discipline, as Geraud phrased it: a research question compares one thing with everything else held constant. Comparing architectures means holding the learning algorithm and dataset fixed. Comparing supervised learning against RL Frame Stacking means holding the architecture and dataset fixed. Changing both the architecture (MLP vs LSTM) and the learning algorithm means the performance difference can no longer be attributed to the memory mechanism. (v4)
- Benny's own admission on record: he froze PPO and MLP across FS and AS - inside the RL side only - and did not think to freeze across the two learning paradigms. Geraud: "you should have also done the same for the supervised side." (v4)
- The supervised baseline could have used an MLP, and LSTM being the more standard choice is the reason it didn't. Geraud: "the fact that you *could* is the exact point." (v4)

## Geraud's standing judgements from this sitting

- The best result seems very low to him. (v3)
- The results are hard to interpret precisely because the dataset is not one people have used - so it can't be told apart from an error on our side. (v3)
- It may simply be a hard dataset: predicting the next day from the last 24 days is a very hard prediction problem, and with financial data there is always the possibility that what we are trying to predict is not really predictable. (v3)
- 868 models trained in a day and some is "not the worst case scenario" - he was fine with the compute. (v3)
- On the sweep itself: he would not have done the meta-sweep extension without a results-driven reason for it, and would rather have seen time spent on the evaluation we care about before multi-seed and parameter sweeps. He still called the work good and said the detail belongs in the appendix. (v3)
- Using standard dropout and clip values found in similar finance papers, rather than tuning them, is fine. He called Benny's running log of the research behind each decision "awesome". (v2)

---

# Where the guidance changed between the two sittings

- **LSTM hyperparameters for the RL agent.** In v4 (earlier) Geraud said that because the LSTM was hyperparameter tuned for the supervised baseline, you should *also* hyperparameter tune the LSTM for the RL agent. In recording 4 (later) he revised this: do **not** sweep again, reuse the exact best hyperparameters already found for the supervised baseline, because beating the baseline on the baseline's own best settings is the fair and sufficient result. **Follow the later version.**
- **The k sweep.** In v3 he questioned whether the sweep should have been done at all without a results-driven motivation. In recording 4 he settled on how to frame it: present it as an experiment measuring the effect of memory, not as hyperparameter tuning.
- **Dataset size.** The v1 challenge (31 X/Y pairs) and the v3 challenge (best result seems very low, possibly too little data) are the same thread. By recordings 1-6 the conversation had moved on to variance and plotting, and the 31-samples objection was not revisited. It is still open and nothing in the project addresses it.
