# Research Log — Adaptive Stacking for Financial Time-Series RL

> **Purpose of this file**
> A dated, running account of the actual research journey — decisions, turning points, and the one real insight or revelation of each week or month, whether in RL, finance, or data science. Written dense enough that it can be lifted almost directly into presentation slides ("here's my journey, here's my line of thinking") without needing to be reconstructed from memory later.
>
> **Rule for this file: append-only, always dated.** New entries go at the bottom under a new dated heading. Never rewrite or remove an existing entry — if something changed, say so in the new entry and reference the old one. Claude Code should add a new entry roughly weekly as work happens, in this same first-person, journey-style voice, focused on the one thing most worth remembering that week — not a full changelog of every action taken.
>
> Early dates below are reconstructed from actual session history rather than a diary kept in real time, so they're grouped by the week they happened in rather than claiming a precise day. From this point forward, entries should be written as the week actually happens.

---

## Week of 2 April 2026 — Setting the rhythm

Set up the weekly accountability structure: a reusable presentation template with a running papers-read tally (target 16 total) and a countdown to the literature review deadline (19 April). Six sources already anchored the annotated bibliography: Sutton & Barto for RL foundations, Mnih et al. for DQN and fixed frame stacking, Hochreiter & Schmidhuber for LSTM, Moody & Saffell for direct RL trading, Schulman et al. for PPO, and Nangue Tasse et al. for Adaptive Stacking itself — my supervisor's own paper, and the one this whole project extends.

## Week of 6 April 2026 — Naming the actual intuition

Finished the annotated bibliography across all six sources. The moment that mattered this week wasn't procedural, it was finally being able to say *why* fixed frame stacking bothers me in plain terms: in a financial series, a 10-day window of small, boring fluctuation is exactly the information worth keeping, but a real spike from 11 days ago gets silently dropped the moment a fixed window slides past it — even though that spike might be the most decision-relevant thing that happened all month. That single example became the throughline for the whole project's motivation, and it's the version of the pitch that actually lands when I explain this to someone cold.

Also set up the first `CLAUDE.md` for Claude Code, append-only from day one, so nothing gets silently lost between sessions — the same convention this file and its companion now follow.

## Mid-April 2026 — The literature gap, stated explicitly

Literature review drafted ahead of the 19 April deadline. The gap I ended up defending: Adaptive Stacking has never been evaluated on real-world, continuous, non-stationary financial time-series — every existing memory approach in RL is either a rigid fixed window or an implicit, non-selective LSTM gate. Nobody has given an agent explicit control over what to remember in a financial forecasting context specifically.

## Early May 2026 — Reading Demir, and open questions for Geraud

Read Demir (2021) on Self Memory Management — an agent that gets an explicit push/skip action over a fixed-capacity memory queue, guided by an *intrinsic* reward for storing rare observations. The useful realization: this is a close cousin of Adaptive Stacking, but the mechanism is genuinely different — Demir motivates memory retention through novelty-seeking, while Adaptive Stacking lets the ordinary task reward alone teach the agent what's worth keeping, no curiosity bonus needed. Reading it side by side with Nangue Tasse et al. clarified the actual design space this project sits inside, rather than treating "learned memory" as one monolithic idea.

Also logged the open questions I wanted answered before committing further: is VIX an acceptable dataset or is there a preferred alternative, should the reward be binary direction or continuous magnitude, does a supervised baseline belong in the final comparison or just as motivation, and — the one I think mattered most — if Adaptive Stacking doesn't beat Frame Stacking on financial data, is that negative result an acceptable contribution on its own. Wanted this on record early, since my supervisor tends to be hard to reach.

## Mid-May 2026 — Proposal submitted, and naming the real risk

Proposal submission due 17 May: abstract, full proposal draft (methodology as the most fleshed-out section, per Geraud's guidance), and three research questions each mapped to a measurable experiment. Submitted 18 May.

The Week 2 deck, built the same week, is where I first ranked project risks explicitly — and put "the sparsity assumption might not hold on financial data" at the top of that list. Sitting with that risk directly is what made me realize it isn't just a risk to manage, it's arguably the most interesting possible outcome of the whole project either way: Adaptive Stacking already works on stationary, episodic benchmarks, so a clean negative result here would say something real about where that method's assumptions break down, not just that my project didn't "win."

## 24 June 2026 — Cutting scope on purpose

Dropped the original third configuration entirely — a memoryless PPO baseline, and the research question it was meant to answer ("does Frame Stacking beat a memoryless agent"). That comparison is already established in the DQN/POMDP literature, and my own proposal had already called it "a sanity check more than a contribution." Cutting it freed a full sweep's worth of July time for the part that actually matters: getting Adaptive Stacking properly built and integrated in August.

What's left is two research questions instead of three: RQ1 asks whether PPO with Frame Stacking can even match a strong LSTM baseline — establishing that RL is a reasonable paradigm here at all. RQ2, the actual contribution, asks whether Adaptive Stacking beats Frame Stacking at a matched memory budget, specifically in this noisy, non-stationary setting rather than the clean benchmarks it's been proven on so far. Trade-off accepted on record: I can no longer claim "memory helps" as a standalone finding, only "learned memory beats fixed memory" — which, honestly, is the claim that was always the point.

## Early-to-mid August 2026 — From paper to code

Coding phase started in earnest. Built the data pipeline first: aligning VIX and S&P history, and — after actually sitting down and working through it properly — settling on log returns rather than simple percent returns, for reasons that turned out to matter more than I expected: additivity across days, symmetry around zero, no risk of implying a negative price, and it's the convention Moody & Saffell already uses, so the choice keeps this work comparable to the lineage it's building on.

The single trickiest thing to get right in the environment design: keeping the observation stream (what the agent sees, derived from VIX) strictly separate from the asset-return stream used to score the agent's action, and making sure the reward at day t is computed against the *next* day's return, not the return that already existed the moment the agent acted. Getting the `shift(-1)` alignment wrong here would have quietly leaked future information into training without throwing any error — the kind of bug that invalidates a whole set of results without ever announcing itself.

Ran a PPO/Stable-Baselines3 smoke test: not real training, just proof that the environment's observation/action contract actually matches what SB3 expects. It passed, and along the way, confirmed something worth remembering — a small MLP-policy PPO agent doesn't meaningfully benefit from GPU acceleration, CPU is genuinely faster at this scale, contrary to the instinct to reach for a GPU by default.

Onboarded onto Wits' mscluster for the actual training runs ahead. The cluster's incident history going back to March made one thing very clear before writing a single line of training code: interruption here is the normal case, not the exception (multiple power and network failures, a failed core server twice over, a month-long full shutdown in May from a security incident affecting national HPC infrastructure). Checkpointing and periodic progress logging aren't optional extras for this project, they're a first-day requirement.

Still open, unresolved as of this entry: how Frame Stacking (a stateless, unlearned transform sitting outside the agent) and Adaptive Stacking (a learned policy decision sitting inside the agent) share one clean, config-selectable interface — and, more importantly, where the matched-memory-budget constraint between them actually gets enforced so the RQ2 comparison stays honest. Also open: whether roughly 20 years of daily data is actually a defensible span by the standards of comparable published work, which is now a standing item for literature grounding rather than something to quietly assume.
