"""
Retention diagnostic for Adaptive Stacking runs.

WHAT THIS ANSWERS
-----------------
"Which memory slots does the AS agent tend to overwrite over training?"

If AS is doing what its paper claims (learning to preserve informative
observations and evict uninformative ones), the histogram of slot-overwrite
counts should be NON-UNIFORM. If it's flat, the agent has degenerated to
"overwrite anything" — effectively FIFO with extra steps.

The k=5 seed=0 run on 2026-08-24 already surfaced a non-uniform pattern:
    [779, 1264, 2141, 1625, 2003]
Slot 0 preserved (least overwritten). Slot 2 churned hardest. This is the
kind of signal that turns into a reportable figure for RQ2's write-up.

WHAT'S HERE
-----------
- load_windows_csv:            read one run's windows.csv into a DataFrame.
- retention_from_summary:      pull the aggregate slot counts from a run's
                               summary.json (the fast path — one dict, no
                               row iteration).
- retention_from_windows_csv:  sum per-slot counts across all windows from
                               windows.csv (slower path, per-window resolution).
- concentration:               a single scalar in [0, 1] measuring how far a
                               retention histogram is from uniform. 0 = perfectly
                               uniform (FIFO-like); 1 = one slot takes everything.
- print_report:                CLI-friendly summary of one run.

RUN
---
    python -m src.evaluation.retention <run_dir>
    e.g. python -m src.evaluation.retention results/rq2_as_k5_seed0

Produces stdout only — no plotting library dependency. Plotting can bolt on
later if needed.
"""

from __future__ import annotations
import json
import sys
from pathlib import Path
import numpy as np
import pandas as pd


# ---------- readers ----------

def load_windows_csv(run_dir: Path) -> pd.DataFrame:
    """Read windows.csv from a run's results directory.

    # think of this as: load the tidy per-window table for further analysis.
    """
    path = Path(run_dir) / "windows.csv"
    if not path.exists():
        raise FileNotFoundError(f"windows.csv not found in {run_dir}")
    return pd.read_csv(path)


def retention_from_summary(run_dir: Path) -> list[int] | None:
    """Fast path — pull the aggregate slot-overwrite counts from summary.json.

    Returns:
        List of counts per slot (length k), OR None if this run wasn't AS
        (no memory-action counts recorded).
    """
    path = Path(run_dir) / "summary.json"
    if not path.exists():
        raise FileNotFoundError(f"summary.json not found in {run_dir}")
    with open(path) as f:
        doc = json.load(f)
    agg = doc.get("aggregate") or {}
    counts = agg.get("agent_memory_action_counts_total")
    return list(counts) if counts is not None else None


def retention_from_windows_csv(run_dir: Path) -> list[int] | None:
    """Slower path — sum the per-window mem_slot_* columns from windows.csv.

    Useful when you want per-window resolution too (return the whole DataFrame
    and let the caller slice). Here we return just the total for convenience.

    Returns:
        List of totals per slot, OR None if no mem_slot_* columns present
        (FS/LSTM runs).
    """
    df = load_windows_csv(run_dir)
    slot_cols = sorted([c for c in df.columns if c.startswith("mem_slot_")])
    if not slot_cols:
        return None
    # empty strings (FS-in-AS-schema, shouldn't happen for AS run, but be defensive)
    # coerce to numeric, non-numeric becomes NaN, sum ignores NaN.
    return [int(pd.to_numeric(df[c], errors="coerce").sum()) for c in slot_cols]


# ---------- analysis ----------

def concentration(counts: list[int]) -> float:
    """How far from uniform is this retention histogram? Scalar in [0, 1].

    # think of this as: 0.0 = every slot was overwritten equally often (agent
    # behaves like FIFO), 1.0 = one slot took every single overwrite (extreme
    # skew, agent has strong preference).

    Formula: 1 - (min(counts) / max(counts)) if max > 0 else 0.
    Simple, interpretable, doesn't require entropy machinery.
    """
    if not counts or max(counts) == 0:
        return 0.0
    return 1.0 - (min(counts) / max(counts))


def normalised_counts(counts: list[int]) -> list[float]:
    """Return per-slot fractions summing to 1 — easier to eyeball than raw counts."""
    total = sum(counts)
    if total == 0:
        return [0.0] * len(counts)
    return [c / total for c in counts]


# ---------- CLI ----------

def print_report(run_dir: Path) -> None:
    """Pretty stdout report for one run.

    # think of this as: the human-facing readout you'd paste into a status update.
    """
    run_dir = Path(run_dir)
    print(f"\nRetention report for: {run_dir}")
    print("-" * 60)

    # Fast path first — summary.json aggregate.
    try:
        counts = retention_from_summary(run_dir)
    except FileNotFoundError as e:
        print(f"  {e}")
        return

    if counts is None:
        print("  This run has no memory-action counts (not an AS run).")
        return

    k = len(counts)
    total = sum(counts)
    frac = normalised_counts(counts)
    conc = concentration(counts)

    print(f"  Memory budget k = {k}")
    print(f"  Total overwrites across all test windows: {total}")
    print(f"  Slot counts:       {counts}")
    print(f"  Slot fractions:    {[f'{f:.3f}' for f in frac]}")
    print(f"  Concentration:     {conc:.3f}   (0 = uniform / FIFO-like, 1 = one slot takes all)")

    # Rough classification for a quick verdict.
    if conc < 0.2:
        verdict = "≈ uniform — agent behaves like FIFO; not exploiting the mechanism."
    elif conc < 0.5:
        verdict = "mildly skewed — some preference emerging."
    else:
        verdict = "strongly skewed — agent has learned a clear retention policy."
    print(f"  Verdict:           {verdict}")


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print("Usage: python -m src.evaluation.retention <run_dir> [<run_dir> ...]")
        sys.exit(1)
    for run_dir in sys.argv[1:]:
        print_report(Path(run_dir))
