# Disagreements

Where I think `to-do-asap.md` (or the `evaluation_report.md` items folded into
it) is wrong, overstated, or already satisfied by work that was done correctly.

One entry per disagreement. Shape: what the item says / what is actually true /
what to do about it. Append-only, dated. If I later find I was the one who was
wrong, I add a correction below the entry rather than deleting it.

Numbers quoted here were checked against the code or the data on the date of
the entry, not taken from any note.

---

## 2026-10-21

### 1. "Pin dependency versions in requirements.txt and record the Python version" is mostly already done

**The item** — `to-do-asap.md` T5.4, from `[EVAL §10] [DO]`: "Pin dependency
versions in requirements.txt and record the Python version. Reason: SB3/Gymnasium
APIs move; without pins the sweeps are not re-runnable."

**What is actually true** — `requirements.txt` already pins every dependency that
could move the results, exactly:

    gymnasium==1.2.3      stable_baselines3==2.7.1    sb3_contrib==2.7.1
    numpy==1.24.3         matplotlib==3.7.5           tensorboard==2.20.0
    minigrid==3.0.0       gymnasium-robotics==1.4.1   opencv-python==4.10.0.84
    imageio==2.35.0       imageio-ffmpeg==0.5.1

and line 2 of the file records "Python 3.10". The header says these are pinned to
match the Adaptive Stacking reference repo and must not be changed without a
reason. That is a stronger reproducibility position than the item assumes, and it
was a deliberate choice, not an accident.

**What is genuinely missing** is narrower than the item claims:

- `torch` is not in the file at all. The header defers it to a separate CUDA-index
  install command, but that command is not recorded anywhere in the repo. The
  LSTM baseline cannot be reproduced without it.
- Five support packages are ranges, not pins: `pandas>=2.0,<2.3`,
  `yfinance>=0.2.40`, `scipy>=1.10`, `pyyaml>=6.0`, `tqdm>=4.66`. Of these only
  `scipy` touches a reported number (the p-values), and only via standard tests.

**What to do** — add the torch version and its install command; optionally pin the
five. Do not write this up as "dependencies were not pinned" — that would
misrepresent the repo to a marker who then opens the file.

---

### 2. "Any number quoted from these blocks is a selection artefact" is true of the file, but nothing downstream used it

**The item** — `to-do-asap.md` T0.1, from `[EVAL §1]`: the `champion` blocks in
`summary.json` are built from a partial subset of runs, and "that single
inconsistency is enough to put all results in doubt".

**What is actually true** — the first half is correct and I confirmed it
(FS ranked k=5 champion on 14 rows from 1 seed; AS held only k=10 and k=20; the
LSTM block was null). That has now been rebuilt — see D055.

The second half overstates the blast radius. Every consumer of the results in
this repo already reads `windows.csv`, not `summary.json`:

- `scripts/generate_plots.py` loads `results/*/windows.csv` directly
  (lines 234-236, 433). Every PNG in `plots/` was built from the complete record.
- `defence/make_defence_figures.py` does the same.
- `claude-read/handoff.md` quotes FS k=3 +0.484, AS k=5 +0.515, LSTM h128L1
  +0.208 — the complete-data numbers, which I reproduced exactly.

So no plot, figure or note in the repo was ever built on the broken blocks. The
exposure was a reader opening `summary.json` by hand, which is real but is not
"all results in doubt".

**What to do** — keep the fix (it was worth doing), but do not carry the "all
results in doubt" framing into the report or into a conversation with the
supervisor. The honest statement is: a stale aggregate existed on disk, it was
never consumed, and it has been rebuilt with provenance.

---

### 3. Geraud's std rule cannot be applied literally to a single headline number

**The item** — `to-do-asap.md` T0.1: "Mean and std are to be computed **across
the 7 seeds only** - not across the 31 windows." The NOTE under it calls this
"in direct conflict" with reporting the window as the unit of analysis.

**What is actually true** — taken literally for one headline number per
configuration, the instruction has no implementation. There is no "Sharpe per
seed": each seed produces 31 window Sharpes. To get one number per seed you must
first collapse the 31 windows, and the moment you do that the spread you are
left with measures only initialisation luck. Our own data shows how much that
discards: at FS k=3 the std across test years is **1.126**, the mean within-window
std across seeds is **0.361**. Reporting only the second makes the error bars
about three times tighter than the uncertainty that actually matters, which is
which years the agent happened to be tested on.

I also do not think the two views are in direct conflict. They answer different
questions, and his own next instruction resolves it:

- Seed-std answers "would I get this again with a different random start?"
- Window-std answers "would I get this again in a different year?"

T0.2 asks for all 31 windows plotted individually. On that plot the natural error
bar at each window **is** the seed spread — which is exactly his rule, applied
per window, where it is well defined. The aggregate significance test then uses
the window as the unit (n=31), which is what D030 already committed to.

**What to do** — plot per-window with seed error bars (his rule, in the only
place it is well defined), report the aggregate with the window as the unit, and
show both numbers in the same figure caption so nothing is hidden. Put the
question to him with the 1.126-vs-0.361 pair in hand; that single comparison is
the whole argument and he will settle it in one sentence.

---
