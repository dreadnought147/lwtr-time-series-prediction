# Metrics Prep Guide — read before the slides

> Private reading. Not in the deck. Use this to explain in your own words why each metric matters when questioned.

---

## Sharpe Ratio

**Formula:** `(mean of returns − risk-free rate) / std of returns × √(trading days per year)`
We use risk-free = 0 (research convention) and 252 trading days per year, so it's annualised.

**What it answers:** "how much return did you get *per unit of volatility you had to swallow to get it?*"

**Intuition — tell this to anyone who asks:**
- Two agents both earn +5%. Agent A got there smoothly (std 2%). Agent B got there with wild +15% / −10% swings (std 12%). Both have the same cumulative return. **Sharpe says A is better** because the +5% came with less risk.
- A positive Sharpe means you got paid for the volatility. A negative Sharpe means volatility ate you. **>+1 is good, >+2 is excellent, <0 is losing money adjusted for risk**.

**Why it matters for memory research specifically:**
- If AS's memory choices let the agent "see" regime transitions earlier, it should take **more decisive** positions and avoid flipping back and forth. Fewer flips → less noise in the per-day reward series → lower std → higher Sharpe for the same cumulative return.
- Sharpe is therefore the metric where a *selective memory* mechanism would show up most clearly — not necessarily by making more money, but by making the same money *more calmly*.

---

## Cumulative Return

**Formula:** `Σ (per-day log-returns)` across the test period. On log-returns, this equals `log(end equity / start equity)`.

**What it answers:** "how much money did this agent make in total?" The headline, no risk adjustment.

**Why it matters for memory research:**
- If memory doesn't translate to extra cumulative return at all, there's not much to show. Sharpe alone could win just by being "less bad" — cumret keeps us honest about whether memory actually added profit.
- Also where buy-and-hold comparison lives: during a long bull market, B&H cumret is huge. We don't expect to beat it; we expect to earn fair-ish cumret with less risk (which is the Sharpe story).

---

## Maximum Drawdown

**Formula:** largest peak-to-trough loss in the equity curve, as a fraction. 0.15 = at some point the strategy was 15% below its own prior high.

**What it answers:** "what's the biggest hit a real trader would have taken if they ran this?" Captures the real-world *pain* of actually trading the strategy.

**Why it matters for memory research:**
- A memory mechanism that identifies regime shifts (e.g. VIX spiking before a crash) should let the agent reduce exposure (go flat or short) before the drawdown deepens. **Lower max DD on AS vs FS at matched k = direct evidence that AS used its memory to see risk coming.**
- Even if Sharpe and cumret are similar, a 10% smaller max drawdown is a meaningful real-world win.

---

## Position Stats (long / flat / short / flips)

**What they are:** per test window, count of days spent in each position + count of position changes.

**What they answer:** "*how* did the agent trade, not just how well?"

**Why they matter for memory research:**
- **Flips:** a flip is a trade — a decision reversal. AS with better memory should flip less because it's more *committed* once it sees signal. Fewer flips at matched k = direct evidence that memory produced decisiveness.
- **Long/flat/short distribution:** reveals bias. An agent that only ever goes long in a bull market is secretly a buy-and-hold strategy that got lucky. An agent that uses all three positions is actually responding to signal.

---

## Memory Action Distribution (AS only)

**What it is:** fraction of overwrites that landed in each memory slot across the whole run.

**What it answers:** "*did the agent actually learn a selection preference, or did it just randomly overwrite?*"

**Why it matters — this is the proposal's primary qualitative diagnostic:**
- If the overwrite distribution is uniform (1/k per slot), the agent is effectively doing FIFO with extra steps. Memory wasn't exploited.
- If the distribution is **skewed** — some slots overwritten rarely, some often — the agent learned that *some observations are worth keeping longer than others*. That's exactly what Adaptive Stacking is supposed to do.
- **This is unique to our contribution** — Frame Stacking has no equivalent. Being able to show a clearly non-uniform retention histogram is one of the strongest pieces of evidence for RQ2.

---

## The Four-Metric Picture (how they fit together)

If AS beats FS at matched k, the ideal story the four metrics tell together is:

| Metric | Ideal AS vs FS result | Interpretation |
|---|---|---|
| Sharpe | AS higher | memory smoothed decisions |
| Cumret | AS similar or higher | memory didn't just reduce risk, it kept the gains |
| Max DD | AS lower | memory saw regime risk coming |
| Flips  | AS lower | memory produced decisive, committed positions |
| Retention | AS skewed, not uniform | memory was *selective*, not just FIFO |

Our actual results show AS wins on Sharpe at 3 of 4 k values, with retention histograms worth inspecting per k.

---

## Quick numerical anchors (memorise these for Q&A)

- **Buy-and-hold Sharpe** (floor): **+0.841**
- **LSTM champion Sharpe:** +0.208 (h=128, L=1)
- **FS champion Sharpe:** +0.484 (k=3)
- **AS champion Sharpe:** +0.515 (k=5)
- **AS vs FS at matched k:** FS wins k=3 (+0.484 vs +0.459); AS wins k=5 (+0.515 vs +0.435), k=10 (+0.447 vs +0.310), k=20 (+0.437 vs +0.321)
- **AS advantage grows with k** — the stronger pattern than just the champion comparison.
