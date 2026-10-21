"""
Rebuild the aggregate blocks in results/*/summary.json from windows.csv.

WHY THIS EXISTS
---------------
`train.py` and `train_lstm.py` collect `per_window_records` in memory during a
run, then write `summary.json` at the end. Both support resuming: a triplet
already present in windows.csv is skipped. Skipped triplets never enter the
in-memory list, so after a resumed session `summary.json` describes only the
work done in that final session, not the sweep.

The damage, measured:
    FS   champion block ranks k=5 on 14 window-rows, k=10/20 on 217, k=3 absent
    AS   champion block contains only k=10 (87 rows) and k=20 (217)
    LSTM per_window is empty, champion is null

`windows.csv` is the ground truth — it is appended and flushed per window, so
it holds the complete balanced record (868 rows = 4 configs x 7 seeds x 31
windows for each of the three sweeps).

WHAT THIS SCRIPT DOES
---------------------
For each run directory:
  1. Copy summary.json to summary.partial-<timestamp>.json (nothing is lost).
  2. Rebuild `per_window` from every row of windows.csv.
  3. Recompute `champion` / `ranking_by_sharpe_mean` over the complete record,
     using the same aggregation train.py uses (mean, sample std ddof=1).
  4. Add a `rebuilt` provenance block saying where the numbers came from.
  5. Write atomically (.tmp -> rename), matching the original writer.

The grouping key is `k` for the RL sweeps and `hp_id` for the LSTM sweep; the
script picks it from the CSV's columns rather than being told.

Run:
    python scripts/rebuild_summaries.py
    python scripts/rebuild_summaries.py --dry-run
    python scripts/rebuild_summaries.py results/rq_rl_fs_sweep
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]

DEFAULT_RUNS = [
    "results/rq1_lstm_sweep",
    "results/rq_rl_fs_sweep",
    "results/rq_rl_as_sweep",
]


# ---------- provenance ----------

def git_commit() -> str | None:
    """Short hash of the current commit, or None outside a git checkout."""
    try:
        out = subprocess.run(
            ["git", "rev-parse", "--short", "HEAD"],
            cwd=ROOT, capture_output=True, text=True, timeout=10,
        )
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


# ---------- rebuild ----------

def group_key(df: pd.DataFrame) -> str:
    """Which column separates the configurations in this sweep."""
    if "k" in df.columns:
        return "k"
    if "hp_id" in df.columns:
        return "hp_id"
    raise ValueError("windows.csv has neither a 'k' nor an 'hp_id' column")


def _metrics(row: pd.Series) -> dict:
    return {
        "cumulative_return": float(row["agent_cumret"]),
        "sharpe_ratio": float(row["agent_sharpe"]),
        "max_drawdown": float(row["agent_maxdd"]),
    }


def _bh_metrics(row: pd.Series) -> dict:
    return {
        "cumulative_return": float(row["bh_cumret"]),
        "sharpe_ratio": float(row["bh_sharpe"]),
        "max_drawdown": float(row["bh_maxdd"]),
    }


def build_per_window(df: pd.DataFrame, key: str) -> list[dict]:
    """One record per windows.csv row, in the shape the training scripts wrote.

    Fields that only existed in memory during training (the SB3 curve path, the
    forecast min/max) are omitted rather than invented.
    """
    records = []
    for _, row in df.iterrows():
        rec = {
            "run_id": row["run_id"],
            key: int(row[key]),
            "seed": int(row["seed"]),
            "window_index": int(row["window_index"]),
            "dates": {
                "train_start": str(row["train_start_date"]),
                "train_end": str(row["train_end_date"]),
                "test_start": str(row["test_start_date"]),
                "test_end": str(row["test_end_date"]),
            },
            "sizes": {"train": int(row["train_days"]), "test": int(row["test_days"])},
            "agent_test": {
                "metrics": _metrics(row),
                "positions": {
                    "n_long": int(row["pos_long"]),
                    "n_flat": int(row["pos_flat"]),
                    "n_short": int(row["pos_short"]),
                    "n_flips": int(row["pos_flips"]),
                },
            },
            "buy_and_hold_test": {"metrics": _bh_metrics(row)},
        }
        if "memory_type" in df.columns:
            rec["memory_type"] = row["memory_type"]
        if "hp_hidden_size" in df.columns:
            rec["hp"] = {
                "hidden_size": int(row["hp_hidden_size"]),
                "num_layers": int(row["hp_num_layers"]),
                "dropout": float(row["hp_dropout"]),
                "seq_len": int(row["hp_seq_len"]),
            }
        # AS runs record how often each memory slot was overwritten. Columns are
        # padded to the sweep's largest k, so read them by number, not by the
        # alphabetical order of the column names (mem_slot_10 sorts before
        # mem_slot_2 as text).
        if key == "k":
            k_val = int(row[key])
            slot_cols = [f"mem_slot_{i}" for i in range(k_val)]
            if all(c in df.columns for c in slot_cols):
                counts = pd.to_numeric(row[slot_cols], errors="coerce")
                if counts.notna().all():
                    rec["agent_test"]["memory_action_counts"] = [int(c) for c in counts]
        records.append(rec)
    return records


def build_champion(df: pd.DataFrame, key: str) -> dict:
    """Mean/std per configuration over the complete record, best Sharpe first.

    Same aggregation as compute_champion() in the training scripts: plain mean
    and sample standard deviation (ddof=1) over every row of the group.
    """
    summaries = []
    for key_val, rows in df.groupby(key):
        sharpes = rows["agent_sharpe"].to_numpy(dtype=float)
        cumrets = rows["agent_cumret"].to_numpy(dtype=float)
        maxdds = rows["agent_maxdd"].to_numpy(dtype=float)

        def stat(arr):
            return {
                "mean": float(arr.mean()),
                "std": float(arr.std(ddof=1)) if arr.size > 1 else 0.0,
            }

        entry = {
            key: int(key_val),
            "seeds": sorted(int(s) for s in rows["seed"].unique()),
            "n_windows_total": int(len(rows)),
            "sharpe": stat(sharpes),
            "cumret": stat(cumrets),
            "maxdd": stat(maxdds),
        }
        if "hp_hidden_size" in rows.columns:
            first = rows.iloc[0]
            entry["hp"] = {
                "hidden_size": int(first["hp_hidden_size"]),
                "num_layers": int(first["hp_num_layers"]),
                "dropout": float(first["hp_dropout"]),
                "seq_len": int(first["hp_seq_len"]),
            }
        summaries.append(entry)

    summaries.sort(key=lambda s: s["sharpe"]["mean"], reverse=True)
    return {"ranking_by_sharpe_mean": summaries, "champion": summaries[0]}


def describe_old(doc: dict, key: str) -> str:
    """One line per configuration the stale champion block covered."""
    champ = doc.get("champion")
    if not champ:
        return "champion: null, per_window: %d rows" % len(doc.get("per_window") or [])
    parts = [
        f"{key}={s.get(key)}:{s['n_windows_total']}r/{len(s['seeds'])}s"
        f" sharpe {s['sharpe']['mean']:+.3f}"
        for s in champ["ranking_by_sharpe_mean"]
    ]
    return "; ".join(parts)


def rebuild(run_dir: Path, dry_run: bool = False) -> None:
    windows_csv = run_dir / "windows.csv"
    summary_json = run_dir / "summary.json"
    if not windows_csv.exists():
        print(f"[skip] {run_dir.name}: no windows.csv")
        return
    if not summary_json.exists():
        print(f"[skip] {run_dir.name}: no summary.json")
        return

    df = pd.read_csv(windows_csv)
    key = group_key(df)
    doc = json.loads(summary_json.read_text())

    print(f"\n[{run_dir.name}] windows.csv: {len(df)} rows, "
          f"{df[key].nunique()} {key} values, {df['seed'].nunique()} seeds, "
          f"{df['window_index'].nunique()} windows")
    print(f"  was:  {describe_old(doc, key)}")

    new_per_window = build_per_window(df, key)
    new_champion = build_champion(df, key)

    print(f"  now:  " + "; ".join(
        f"{key}={s[key]}:{s['n_windows_total']}r/{len(s['seeds'])}s"
        f" sharpe {s['sharpe']['mean']:+.3f}"
        for s in new_champion["ranking_by_sharpe_mean"]
    ))
    c = new_champion["champion"]
    print(f"  champion: {key}={c[key]}  Sharpe {c['sharpe']['mean']:+.4f} "
          f"± {c['sharpe']['std']:.4f} over {c['n_windows_total']} rows")

    if dry_run:
        print("  (dry run — nothing written)")
        return

    # Keep the partial original. summary.json is not under version control
    # (results/ is gitignored), so this copy is the only way back.
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
    backup = run_dir / f"summary.partial-{stamp}.json"
    shutil.copy2(summary_json, backup)

    doc["per_window"] = new_per_window
    doc["champion"] = new_champion
    doc["rebuilt"] = {
        "rebuilt_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "rebuilt_by": "scripts/rebuild_summaries.py",
        "source": "windows.csv",
        "n_rows": int(len(df)),
        "group_key": key,
        "git_commit": git_commit(),
        "reason": (
            "The original per_window and champion blocks were built only from "
            "triplets executed in the final resumed session, so they ranked "
            "configurations on unequal and incomplete samples. Rebuilt from the "
            "complete windows.csv record."
        ),
        "original_kept_as": backup.name,
    }

    tmp = summary_json.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(doc, indent=2, default=str))
    tmp.replace(summary_json)
    print(f"  wrote {summary_json.name}  (original kept as {backup.name})")


def main():
    ap = argparse.ArgumentParser(description=__doc__.split("\n")[1])
    ap.add_argument("runs", nargs="*", default=None,
                    help="run directories; default is the three sweeps")
    ap.add_argument("--dry-run", action="store_true",
                    help="print what would change, write nothing")
    args = ap.parse_args()

    runs = args.runs or DEFAULT_RUNS
    for r in runs:
        p = Path(r)
        if not p.is_absolute():
            p = ROOT / p
        rebuild(p, dry_run=args.dry_run)


if __name__ == "__main__":
    main()
