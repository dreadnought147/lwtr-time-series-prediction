# Handoff, 2026-10-21

Read CLAUDE.md, research_log.md and design_notes.md in this folder too. This file holds what was found AFTER those were last updated.

## State
- Training is finished: LSTM, FS and AS sweeps, 868 runs each (4 configs x 7 seeds x 31 walk-forward windows). Results in `results/rq1_lstm_sweep`, `rq_rl_fs_sweep`, `rq_rl_as_sweep`.
- Plots: `plots/` with landing page `plots/DATA_TERMINAL.md`. Regenerate with `python scripts/generate_plots.py`.
- Deck: `takeoff/make_pptx.html` (PptxGenJS, open in browser, click button). Notes: `takeoff/speaker_notes.md`, `takeoff/metrics_guide.md`. Defence guide: `defence/`.
- Champions (mean Sharpe over 217 test slices): LSTM h128 L1 +0.208, FS k=3 +0.484, AS k=5 +0.515, buy-and-hold +0.841.

## Findings from the paired test (same test year for both, seed-averaged over 31 windows)
| Comparison | Sharpe diff | Windows won | p (t-test) |
|---|---|---|---|
| AS - FS k=3 | -0.02 | 13/31 | 0.71 |
| AS - FS k=5 | +0.08 | 16/31 | 0.41 |
| AS - FS k=10 | +0.14 | 19/31 | 0.23 |
| AS - FS k=20 | +0.12 | 16/31 | 0.37 |
| FS(k3) - LSTM | +0.28 | 20/31 | 0.10 |
| AS(k5) - LSTM | +0.31 | 19/31 | 0.04 (1 of 8 tests) |

- RQ2 is not significant at any k. RQ1 is directional (p about 0.10).
- Gap does NOT grow monotonically with k: -0.02, +0.08, +0.14, then +0.12 at k=20.
- Whiskers: std across test years 1.13 vs across seeds within a year 0.36. The spread is the market, not model randomness. More seeds will not buy significance, more test periods would.
- Behaviour is the clearest real difference. Mean position flips per test window, FS vs AS: k3 33 vs 29, k5 54 vs 31, k10 88 vs 40, k20 118 vs 52. Not yet significance-tested.
- Correct retention fractions (slot 0 = oldest survivor): k=5 [.104 .176 .201 .241 .278], k=10 [.041 .046 .070 .092 .078 .097 .120 .136 .138 .181]. AS protects old memories and refreshes the newest. FIFO would be 100% on slot 0. Shows a preference, not that it causes any gain.
- Predictive correlation (VIX move today vs S&P move next day) is +0.03. Same-day is -0.71. The agent can only use the first.
- Data is 36.5 years (1990-01-03 to 2026-06-29, 9188 days). Test slices span 1994-12-27 to 2026-01-09.

## Bugs and errors to fix (not yet fixed)
1. Retention plot wrong for k=3,5,10: `generate_plots.py` (and `src/evaluation/retention.py`) use `sorted()` on `mem_slot_*` columns, so slot 10 sorts before slot 2 and `[:k]` reads the wrong columns. Fix: use `[f"mem_slot_{i}" for i in range(k)]`. Do not show `plots/rq2/03_retention_histogram_by_k.png` until regenerated.
2. Deck and docs say "24 years" (slide 4, `takeoff/status.md`, `disclaimer.tex`, logs). Correct is 36.5 years. CLAUDE.md section 7 "dataset span too short" concern is therefore moot.
3. Slide 4 and `plots/DATA_TERMINAL.md` section 1.7 claim a -0.7 to -0.8 rolling correlation. That is the same-day number, wrong for the predictive direction.
4. Slide 5 says "2 layers". The sweep used 1 or 2 and the champion has 1.
5. Slide 6 / speaker notes cite "Henderson et al. 2018, How Many Random Seeds?". That paper is Colas, Sigaud & Oudeyer 2018 (arXiv 1806.08295).
6. "Chowdhury 2023" (dropout 0.2) is a Medium blog, year unverified. Hyperparameter sources came from search snippets, so wording should be "informed by", not "justified by".
7. "Gap grows with k" (slides 13, 15, notes) and "RQ1 answered, RL earns its keep" overclaim. Use "directionally".
8. `summary.json` "champion" block only reflects the last resume session. `windows.csv` is the ground truth.
9. Likely, unverified: `scripts/data_integrity_check.py` fetches from 2000-01-01 but local CSVs start 1990, so it will report about 2500 "only local" dates.

## Corrections to things the user said while rehearsing
36 days -> 36 years. LSTM champion is 0.208 (0.28 is the gap). 217 slices, 868 agents. Moody & Saffell. Reward is dense, not sparse. AS contribution is NOT "continuous environments" (the paper already has FetchReach), it is noisy, non-stationary, exogenous finance. Earlier test years DO become later training years (allowed, but windows share about 80% of training days, so the 31 Sharpes are not independent). PPO was not tuned at all (stock SB3 defaults); LSTM got a 4-combo grid, so tuning asymmetry favours the LSTM.

## Open decisions
- Sealed final holdout: not done. Industry practice is walk-forward plus a sealed holdout. Currently documented as a limitation. Ask Geraud.
- Submission deadline not known to me; ask the user. Also what evidence level Geraud wants for RQ2.
- `disclaimer.tex` title and preamble to be cleaned against `proposal.md`.
- Deck still has 72 em-dashes and an image-load error when opened as file:// (images need inlining as base64). `takeoff/finalise_deck.py` fixes both, never run; user said leave as is.
- User wants TensorBoard (dictated as "TLog Tensor bot") for visualisation. Not wired up.
- Planned retrospective agent: see CLAUDE.md section 13.

## Working preferences
- Memory and notes live in this repo, not Claude's global memory folder.
- Short answers, simple English, bullets; say so if an answer needs more than 5 lines. No em-dashes in presentation material.
- Mentor mode (see `feedback_mentor_persona` in the global memory folder): guide, explain why, user types the code unless it is pure boilerplate.
- Do not state numbers or citations without checking them against the data or the source.
