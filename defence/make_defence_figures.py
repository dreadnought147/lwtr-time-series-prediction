"""
Figures for the defence guide (defence/defence_guide.tex).

Writes ONLY into defence/figures/. Never touches plots/ or results/.

Figures produced:
    retention_corrected.png    AS slot-overwrite distribution per k, with the
                               slot columns read in NUMERIC order. The version in
                               plots/rq2/03_retention_histogram_by_k.png sorts the
                               column names as text (mem_slot_0, mem_slot_1,
                               mem_slot_10, ...), so for k=3,5,10 it reads empty
                               columns and shows zeros. This one does not.
    paired_difference.png      AS minus FS Sharpe at matched k, averaged over the
                               7 seeds inside each walk-forward window, with a
                               bootstrap 95% interval on the mean.
    where_spread_comes_from.png  Splits each config's error bar into
                               "window-to-window" spread vs "seed-to-seed" spread.

Run from the project root:
    python defence/make_defence_figures.py
"""

from __future__ import annotations
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

PROJECT_ROOT = Path(__file__).resolve().parents[1]
RESULTS_DIR = PROJECT_ROOT / "results"
FIGURES_DIR = Path(__file__).resolve().parent / "figures"
FIGURES_DIR.mkdir(parents=True, exist_ok=True)

K_GRID = [3, 5, 10, 20]
FS_COLOUR = "tab:blue"
AS_COLOUR = "tab:purple"

plt.rcParams.update({
    "font.size": 11,
    "axes.grid": True,
    "grid.alpha": 0.25,
    "axes.spines.top": False,
    "axes.spines.right": False,
})


def load_sweep(sweep_name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS_DIR / sweep_name / "windows.csv")


frame_stacking_rows = load_sweep("rq_rl_fs_sweep")
adaptive_stacking_rows = load_sweep("rq_rl_as_sweep")
lstm_rows = load_sweep("rq1_lstm_sweep")


# ---------------------------------------------------------------------------
# 1. Corrected retention histogram
# ---------------------------------------------------------------------------

def plot_corrected_retention() -> None:
    figure, axes = plt.subplots(1, 4, figsize=(17, 4.2))
    for axis, memory_budget in zip(axes, K_GRID):
        rows_for_this_k = adaptive_stacking_rows[adaptive_stacking_rows["k"] == memory_budget]
        # Numeric order: slot 0, slot 1, ..., slot k-1. This is the fix.
        slot_columns_in_numeric_order = [f"mem_slot_{slot}" for slot in range(memory_budget)]
        overwrite_totals = (
            rows_for_this_k[slot_columns_in_numeric_order]
            .apply(pd.to_numeric, errors="coerce")
            .sum()
            .to_numpy()
        )
        overwrite_fractions = overwrite_totals / overwrite_totals.sum()

        axis.bar(range(memory_budget), overwrite_fractions,
                 color=AS_COLOUR, edgecolor="black", alpha=0.85)
        axis.axhline(1 / memory_budget, color="red", ls="--", lw=1,
                     label=f"uniform = 1/k = {1 / memory_budget:.2f}")
        axis.set_title(f"k = {memory_budget}")
        tick_positions = list(range(memory_budget)) if memory_budget <= 10 else [0, 5, 10, 15, 19]
        axis.set_xticks(tick_positions)
        axis.set_xlabel("slot position (0 = oldest kept, k-1 = newest)")
        axis.legend(fontsize=8, loc="upper left")
        print(f"[retention] k={memory_budget}: " + " ".join(f"{f:.3f}" for f in overwrite_fractions))
    axes[0].set_ylabel("fraction of all overwrites")
    figure.suptitle("Adaptive Stacking: which slot does the agent throw away? "
                    "(test years, all 7 seeds x 31 windows, slots read in numeric order)", y=1.03)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "retention_corrected.png", dpi=150, bbox_inches="tight")
    plt.close(figure)


# ---------------------------------------------------------------------------
# 2. Paired AS - FS difference per k
# ---------------------------------------------------------------------------

def window_level_mean_sharpe(rows: pd.DataFrame, memory_budget: int) -> pd.Series:
    """Average the 7 seeds inside each window, giving 31 numbers (one per test year)."""
    return rows[rows["k"] == memory_budget].groupby("window_index")["agent_sharpe"].mean()


def bootstrap_interval_of_mean(values: np.ndarray, repetitions: int = 10000) -> tuple[float, float]:
    random_generator = np.random.default_rng(0)
    resampled_means = [
        random_generator.choice(values, size=values.size, replace=True).mean()
        for _ in range(repetitions)
    ]
    return tuple(np.percentile(resampled_means, [2.5, 97.5]))


def plot_paired_difference() -> None:
    figure, axis = plt.subplots(figsize=(9, 5))
    for position, memory_budget in enumerate(K_GRID):
        difference_per_window = (
            window_level_mean_sharpe(adaptive_stacking_rows, memory_budget)
            - window_level_mean_sharpe(frame_stacking_rows, memory_budget)
        ).to_numpy()
        jitter = np.random.default_rng(memory_budget).uniform(-0.12, 0.12, difference_per_window.size)
        axis.scatter(position + jitter, difference_per_window, s=14, color="grey", alpha=0.6,
                     label="one test year (7-seed average)" if position == 0 else None)
        mean_difference = difference_per_window.mean()
        lower, upper = bootstrap_interval_of_mean(difference_per_window)
        axis.errorbar(position, mean_difference,
                      yerr=[[mean_difference - lower], [upper - mean_difference]],
                      fmt="o", color="black", capsize=8, lw=2, ms=8,
                      label="mean difference with 95% bootstrap interval" if position == 0 else None)
        wins = int((difference_per_window > 0).sum())
        axis.annotate(f"AS better in\n{wins}/31 years", (position, upper + 0.15),
                      ha="center", fontsize=9)
        print(f"[paired] k={memory_budget}: mean {mean_difference:+.3f}  "
              f"95% CI [{lower:+.3f}, {upper:+.3f}]  AS better {wins}/31")
    axis.axhline(0, color="red", lw=1.2, ls="--")
    axis.set_xticks(range(len(K_GRID)))
    axis.set_xticklabels([f"k = {k}" for k in K_GRID])
    axis.set_ylabel("Sharpe(AS) minus Sharpe(FS), same test year")
    axis.set_title("RQ2 paired view: above the red line = Adaptive Stacking did better that year")
    axis.legend(loc="lower right", fontsize=9)
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "paired_difference.png", dpi=150)
    plt.close(figure)


# ---------------------------------------------------------------------------
# 3. Where does the error bar come from?
# ---------------------------------------------------------------------------

def spread_components(rows: pd.DataFrame, group_column: str, group_value) -> tuple[float, float]:
    rows_for_config = rows[rows[group_column] == group_value]
    across_windows = rows_for_config.groupby("window_index")["agent_sharpe"].mean().std()
    within_window_across_seeds = rows_for_config.groupby("window_index")["agent_sharpe"].std().mean()
    return across_windows, within_window_across_seeds


def plot_where_spread_comes_from() -> None:
    labels, market_spread, seed_spread = [], [], []
    lstm_champion_id = int(lstm_rows.groupby("hp_id")["agent_sharpe"].mean().idxmax())
    configurations = [("LSTM h128L1", lstm_rows, "hp_id", lstm_champion_id)]
    configurations += [(f"FS k={k}", frame_stacking_rows, "k", k) for k in K_GRID]
    configurations += [(f"AS k={k}", adaptive_stacking_rows, "k", k) for k in K_GRID]
    for label, rows, column, value in configurations:
        across, within = spread_components(rows, column, value)
        labels.append(label)
        market_spread.append(across)
        seed_spread.append(within)
        print(f"[spread] {label}: across windows {across:.3f}  across seeds {within:.3f}")

    positions = np.arange(len(labels))
    bar_width = 0.4
    figure, axis = plt.subplots(figsize=(11, 4.8))
    axis.bar(positions - bar_width / 2, market_spread, bar_width, color="tab:orange",
             edgecolor="black", label="spread between test years (the market)")
    axis.bar(positions + bar_width / 2, seed_spread, bar_width, color="tab:grey",
             edgecolor="black", label="spread between seeds in the same year (luck of the start)")
    axis.set_xticks(positions)
    axis.set_xticklabels(labels, rotation=20)
    axis.set_ylabel("standard deviation of Sharpe")
    axis.set_title("What the error bars are made of")
    axis.legend()
    figure.tight_layout()
    figure.savefig(FIGURES_DIR / "where_spread_comes_from.png", dpi=150)
    plt.close(figure)


def print_flip_and_position_numbers() -> None:
    for name, rows in [("FS", frame_stacking_rows), ("AS", adaptive_stacking_rows)]:
        flips = rows.groupby("k")["pos_flips"].mean().round(1).to_dict()
        print(f"[flips] {name} mean flips per test year by k: {flips}")
    lstm_champion_id = int(lstm_rows.groupby("hp_id")["agent_sharpe"].mean().idxmax())
    print(f"[flips] LSTM champion: {lstm_rows[lstm_rows.hp_id == lstm_champion_id].pos_flips.mean():.1f}")
    print(f"[wall] mean training seconds per model: FS {frame_stacking_rows.train_wall_seconds.mean():.0f}, "
          f"AS {adaptive_stacking_rows.train_wall_seconds.mean():.0f}, "
          f"LSTM {lstm_rows.train_wall_seconds.mean():.0f}")
    print(f"[maxdd] FS k3 {frame_stacking_rows[frame_stacking_rows.k == 3].agent_maxdd.mean():.3f}  "
          f"AS k5 {adaptive_stacking_rows[adaptive_stacking_rows.k == 5].agent_maxdd.mean():.3f}  "
          f"B&H {frame_stacking_rows.bh_maxdd.mean():.3f}")


if __name__ == "__main__":
    plot_corrected_retention()
    plot_paired_difference()
    plot_where_spread_comes_from()
    print_flip_and_position_numbers()
    print(f"Figures written to {FIGURES_DIR}")
