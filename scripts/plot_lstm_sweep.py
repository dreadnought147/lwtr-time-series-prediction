"""
Plot generator for LSTM sweep runs.

Reads a sweep run's windows.csv, optionally filters to the first N seeds,
and produces per-HP-shape + aggregate plots into a seed_{N} subfolder.

Design principle (matches user's plotting convention):
  results/{run_dir}/seed_{N}/
    aggregate_sharpe_by_hp.png       — bar chart with error bars, champion picker
    hp_{id}_sharpe_timeline.png      — one per HP shape: Sharpe across windows
    hp_{id}_cumret_timeline.png      — one per HP shape: cumret across windows

The sweep YAML is the source of truth for what was trained — no README next
to the plots. Reading windows.csv + the sweep YAML gives full reproducibility.

Run:
    python scripts/plot_lstm_sweep.py --run-dir results/rq1_lstm_sweep --seeds 3
    python scripts/plot_lstm_sweep.py --run-dir results/rq1_lstm_sweep --seeds 7
"""

from __future__ import annotations
import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")   # non-interactive backend — safe when run headless / on cluster
import matplotlib.pyplot as plt


# ---------- data loading ----------

def load_sweep_windows(run_dir: Path, max_seed_exclusive: int | None) -> pd.DataFrame:
    """Load windows.csv, optionally filter to seeds < max_seed_exclusive.

    # think of this as: 'give me the rows for the first N seeds only,
    # so plots for the 3-seed snapshot don't mix in future 7-seed data.'
    """
    path = Path(run_dir) / "windows.csv"
    if not path.exists():
        raise FileNotFoundError(f"windows.csv not found in {run_dir}")
    df = pd.read_csv(path)
    if max_seed_exclusive is not None:
        df = df[df["seed"] < max_seed_exclusive].copy()
    if df.empty:
        raise RuntimeError(f"No rows in windows.csv after filtering to seeds < {max_seed_exclusive}")
    return df


def hp_label(row: dict | pd.Series) -> str:
    """Compact HP label for legends and filenames."""
    return f"h{int(row['hp_hidden_size'])}L{int(row['hp_num_layers'])}"


# ---------- per-HP-shape plots ----------

def plot_metric_timeline(
    df: pd.DataFrame,
    hp_id: int,
    metric_col: str,
    bh_col: str,
    title_metric: str,
    out_path: Path,
) -> None:
    """One HP shape, one metric — median line across seeds + min-max shaded band.

    x-axis: walk-forward window index (chronological).
    y-axis: the metric (Sharpe / cumret).
    Overlay: buy-and-hold as a dashed reference line (median across seeds).
    """
    sub = df[df["hp_id"] == hp_id].sort_values(["window_index", "seed"])
    if sub.empty:
        return

    # Aggregate across seeds within each window
    grouped = sub.groupby("window_index")
    med = grouped[metric_col].median()
    lo = grouped[metric_col].min()
    hi = grouped[metric_col].max()
    bh_med = grouped[bh_col].median()

    x = med.index.to_numpy()
    fig, ax = plt.subplots(figsize=(9, 4.5))
    ax.fill_between(x, lo.to_numpy(), hi.to_numpy(), alpha=0.20, label="agent seed spread (min-max)")
    ax.plot(x, med.to_numpy(), lw=1.8, label="agent (median across seeds)")
    ax.plot(x, bh_med.to_numpy(), lw=1.2, ls="--", label="buy & hold")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)

    label = hp_label(sub.iloc[0])
    ax.set_title(f"{title_metric} across walk-forward windows — {label}")
    ax.set_xlabel("walk-forward window index (chronological)")
    ax.set_ylabel(title_metric)
    ax.legend(loc="best", fontsize=8)
    ax.grid(alpha=0.25)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------- aggregate plot ----------

def plot_aggregate_bar(df: pd.DataFrame, out_path: Path) -> None:
    """Bar chart with error bars — mean Sharpe per HP shape (aggregate over seeds & windows).

    Champion is the tallest bar. Buy-and-hold shown as a horizontal dashed reference.
    """
    # Aggregate across (seeds × windows) per HP
    grouped = df.groupby("hp_id")
    means = grouped["agent_sharpe"].mean()
    stds = grouped["agent_sharpe"].std(ddof=1).fillna(0.0)
    bh_overall = df["bh_sharpe"].mean()

    # Build labels from HP columns
    labels = []
    for hp_id in means.index:
        row = df[df["hp_id"] == hp_id].iloc[0]
        labels.append(f"hp{hp_id}\n{hp_label(row)}\nd{row['hp_dropout']}")

    fig, ax = plt.subplots(figsize=(1.6 + 1.6 * len(means), 4.5))
    xpos = np.arange(len(means))
    ax.bar(xpos, means.to_numpy(), yerr=stds.to_numpy(), capsize=6,
           alpha=0.85, edgecolor="black")
    ax.axhline(bh_overall, color="red", ls="--", lw=1.2,
               label=f"buy & hold (mean Sharpe = {bh_overall:.3f})")
    ax.axhline(0.0, color="k", lw=0.6, alpha=0.4)

    ax.set_xticks(xpos)
    ax.set_xticklabels(labels, fontsize=9)
    ax.set_ylabel("mean Sharpe (± std across seeds × windows)")
    ax.set_title("LSTM sweep — champion picker (aggregate over seeds × windows)")
    ax.legend(loc="best", fontsize=9)
    ax.grid(axis="y", alpha=0.25)

    # Highlight champion (tallest mean)
    champion_idx = int(np.argmax(means.to_numpy()))
    ax.get_children()[champion_idx].set_edgecolor("green")
    ax.get_children()[champion_idx].set_linewidth(2.5)

    fig.tight_layout()
    fig.savefig(out_path, dpi=140)
    plt.close(fig)


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Plot LSTM sweep results.")
    parser.add_argument("--run-dir", type=str, required=True,
                        help="Sweep results directory containing windows.csv (e.g. results/rq1_lstm_sweep).")
    parser.add_argument("--seeds", type=int, required=True,
                        help="Number of seeds to include (plots go in seed_{N}/). "
                             "Filters windows.csv to seed < N.")
    args = parser.parse_args()

    run_dir = Path(args.run_dir).resolve()
    n_seeds = int(args.seeds)

    df = load_sweep_windows(run_dir, max_seed_exclusive=n_seeds)
    out_dir = run_dir / f"seed_{n_seeds}"
    out_dir.mkdir(parents=True, exist_ok=True)

    hp_ids = sorted(df["hp_id"].unique())
    print(f"[plot_lstm_sweep] run={run_dir.name}  seeds<{n_seeds}  "
          f"HP combos={hp_ids}  windows/HP≈{len(df)//(len(hp_ids)*n_seeds)}")

    # per HP shape — Sharpe and cumret timelines
    for hp_id in hp_ids:
        plot_metric_timeline(
            df, hp_id,
            metric_col="agent_sharpe", bh_col="bh_sharpe",
            title_metric="Sharpe",
            out_path=out_dir / f"hp_{hp_id}_sharpe_timeline.png",
        )
        plot_metric_timeline(
            df, hp_id,
            metric_col="agent_cumret", bh_col="bh_cumret",
            title_metric="Cumulative return",
            out_path=out_dir / f"hp_{hp_id}_cumret_timeline.png",
        )

    # aggregate — champion picker
    plot_aggregate_bar(df, out_dir / "aggregate_sharpe_by_hp.png")

    # champion identification for the console
    champion_id = int(df.groupby("hp_id")["agent_sharpe"].mean().idxmax())
    champion_row = df[df["hp_id"] == champion_id].iloc[0]
    mean_ch = df[df["hp_id"] == champion_id]["agent_sharpe"].mean()
    std_ch = df[df["hp_id"] == champion_id]["agent_sharpe"].std(ddof=1)

    print(f"[plot_lstm_sweep] wrote {2 * len(hp_ids) + 1} plots to {out_dir}")
    print(f"[plot_lstm_sweep] champion  hp{champion_id}  {hp_label(champion_row)}  "
          f"Sharpe {mean_ch:+.3f} ± {std_ch:.3f}")


if __name__ == "__main__":
    main()
