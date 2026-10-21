# Journey Since Geraud

A running record of what changed in the project after the seven supervisor
recordings, written plainly. Not a decision log — the backed decisions live in
`research_decisions/decisions_log.md` and are referenced by ID here. This file
is for the reasoning in the form it was actually worked out, so it can be
explained again later without re-deriving it.

---

## 2026-10-21 — Why "AS wins" became "we can't tell"

### The example, with no project in it

You want to know if boys are taller than girls. You line up 31 boys and 31
girls.

You walk down the line with a tape measure. But instead of measuring each child
once, you measure each child **7 times** — same child, same day, you just do it
seven times over. So for the boys your sheet now has 31 × 7 = **217 numbers** on
it, and the same for the girls.

Here's what you did wrong. You looked at your sheet, saw 217 numbers, and told
yourself "I have 217 boys' worth of evidence." Then you worked out how sure you
could be — and a bigger pile of numbers always makes you look surer. So you
announced boys are taller.

But you never met 217 boys. You met 31. Measuring Tom seven times gives you
seven numbers about Tom, not seven boys. All it tells you is how steady your
tape measure is.

So you go back and do it properly: average Tom's 7 measurements into one height
for Tom, do that for all 31 boys, and now you have 31 heights — one per child,
which is what you actually collected. Same for the girls. Line them up, and the
two groups overlap so heavily you can't honestly say either is taller.

### The same thing, in this project

You want to know if AS does better than FS. You line up 31 test years for AS and
the same 31 test years for FS.

You walk down the line scoring each one. But instead of training each year's
model once, you train it **7 times** — same year, same five years of history
behind it, you just do it seven times with a different random start. So for AS
your sheet now has 31 × 7 = **217 scores** on it, and you did exactly the same
for FS, so FS has its own 217.

Here's what you did wrong. You looked at your sheet, saw 217 scores, and told
yourself "I have 217 years' worth of evidence." Then you worked out how sure you
could be — and a bigger pile of numbers always makes you look surer. So you
announced AS is better.

But you never tested 217 years. You tested 31. Training 1998 seven times gives
you seven numbers about 1998, not seven years. All it tells you is how steady
your training is to the random start.

So you go back and do it properly: average 1998's 7 scores into one score for
1998, do that for all 31 years, and now you have 31 scores for AS — one per
year, which is what you actually collected. Do exactly the same for FS, so it
also has 31. Line them up year against year, 1998 against 1998, and the two
overlap so heavily you can't honestly say either is better.

---

### What actually changed, and what did not

Nothing was retrained. No data changed. The averages themselves did not move —
AS still scores higher than FS at k = 5, 10 and 20. The only thing that changed
is how the *uncertainty* around those averages was worked out.

| | before | after |
|---|---|---|
| what counts as one result | one trained model (217 per mechanism) | one test year (31 per mechanism) |
| AS − FS at k=5 | +0.077 pooled over all 868 pairs | +0.080 per year |
| how sure | p = 0.024, looked real | CI [−0.079, +0.251], contains zero |
| win rate | — | 52% of years, CI [35%, 68%] |

Full table for all four memory budgets is in
`plots/rq2/07_paired_as_minus_fs.txt`.

### Why the new version is the defensible one

Averaging across years was never the problem — that is a fine way to get the
number, and the number is unchanged. The problem was counting the 7 reruns of a
year as 7 separate pieces of evidence about how the mechanism does *on years in
general*. That inflates the apparent sample by about five times (√31 ≈ 5.6),
which shrinks the apparent wobble by the same factor, which is exactly enough to
push a gap that sits inside the noise to looking like it sits outside it.

A referee would find this in one pass. Better to have found it ourselves.

### Does this prove the two mechanisms are the same?

No, and the write-up must not say so. With 31 test years and a year-to-year
spread of roughly 1.0 Sharpe, the smallest gap this study could reliably detect
is around 0.5. The observed gap is about 0.1. So the honest statement is **"too
small a sample to tell"**, not "no difference exists". Those are different
claims and only the first one is supported.

This is the same thread as Geraud's challenge in recording v1 — that 31 X/Y
pairs is very small for any machine learning result. It arrives here from the
statistics side rather than the training side, but it is the same objection and
it now has a number attached to it.

### Can it be fixed before submission?

No. Retraining would not help: the limit is the number of genuinely different
test periods, and 36.5 years of one market cannot produce many more than 31
non-overlapping ones. More seeds buy nothing — they measure training stability,
which is already the small number. More markets or more assets would be a new
study.

So this is a limitation to state clearly and own, not a bug to fix. Doing so is
also what the proposal pre-committed to: a negative result was named in advance
as an acceptable contribution.

### Where this lands in the decision log

- **D030** — the window is the unit of analysis, seeds averaged inside it.
- **D056** — every figure's error bar is the seed spread, with the year spread
  shown beside it so the tighter bars are never mistaken for the whole picture.
- **D058** — the RQ2 headline is the paired per-window difference with a
  block-bootstrap interval. This is the number the conclusion is written from.
- **D034** — the conclusion is framed as a pre-registered null result.
