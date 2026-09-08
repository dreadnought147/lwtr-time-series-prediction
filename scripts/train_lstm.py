"""
Sweep-native LSTM baseline training — the STRONG RQ1 baseline.

Replaces the earlier single-run script. This runner reads a "sweep" YAML that
specifies a hyperparameter grid + a list of seeds, then for every combination
runs the walk-forward evaluation. That gives us:
  - proper hyperparameter comparison across shapes
  - proper seed-spread reporting (mean ± std across seeds per shape)
  - one champion configuration reported as the RQ1 baseline number

WHY THIS EXISTS
---------------
Earlier train_lstm.py was a single-run script — one shape, one seed. That's
a "lazy LSTM": beating it doesn't prove RL beat a strong SL predictor, only
that it beat a weak one. Proposal (per CLAUDE.md §7) explicitly requires the
LSTM baseline to be tuned + multi-seed for RQ1 to be defensible. This sweep
script closes that gap.

FAIRNESS ITEMS ADDED
--------------------
- Multiple random seeds per HP combo (default 3, configurable).
- Per-window standardisation: obs (and target) z-scored using TRAIN-slice
  statistics only. No test-slice info leaks into training.
- Gradient clipping (default 1.0) to stop occasional exploding gradients on
  noisy financial data.
- Weight decay (Adam L2) for mild regularisation.

Outputs (all under results/{run_name}/):
  windows.csv    one row per (hp_id, seed, walk-forward window) — flat, tidy.
  steps.csv      one row per (hp_id, seed, window, day).
  summary.json   nested rich record, plus a "champion" section identifying
                 the best HP combo by mean Sharpe across seeds.

CLUSTER-FRIENDLY
----------------
- CSVs appended row-by-row and flushed after each window.
- summary.json written atomically (.tmp → rename) at the end.
- Resume: if windows.csv already exists, skips (hp_id, seed, window) triplets
  already completed. Schema check on resume is strict.

Run:
    python scripts/train_lstm.py --config configs/rq1_lstm_sweep.yaml
"""

from __future__ import annotations
import argparse
import csv
import json
import sys
import time
from pathlib import Path
import yaml
import numpy as np
import torch
from torch.utils.data import DataLoader

# so this script can import from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.data.loader import load_aligned
from src.data.transforms import to_env_arrays
from src.baselines.lstm import (
    SequenceDataset,
    LSTMForecaster,
    train_forecaster,
    predictions_to_postsions,
    fit_standardiser,
    apply_standardiser,
)
from src.evaluation.metrics import summary as metric_summary
from src.evaluation.walk_forward import walk_forward_windows


# ---------- LSTM rollout on the test slice ----------

def rollout_lstm(model, warmup_obs, test_obs, test_ret, seq_len, threshold, transaction_cost):
    """Run trained LSTM through the test slice, return per-day rewards/positions/forecasts.

    warmup_obs: last seq_len days of the TRAIN slice, prepended so every
                test-day forecast has enough history.
    """
    all_obs = np.concatenate([warmup_obs[-seq_len:], test_obs])
    all_targets = np.concatenate([np.zeros(seq_len, dtype=np.float32), test_ret])
    dataset = SequenceDataset(all_obs.astype(np.float32), all_targets.astype(np.float32), k=seq_len)

    model.eval()
    with torch.no_grad():
        loader = DataLoader(dataset, batch_size=256, shuffle=False)
        forecasts = torch.cat([model(w) for w, _ in loader]).cpu().numpy()

    assert len(forecasts) == len(test_ret), (
        f"forecast/test_ret length mismatch: {len(forecasts)} vs {len(test_ret)}"
    )

    positions = predictions_to_postsions(forecasts, thresh=threshold).astype(np.int64)
    prev_positions = np.concatenate([[0], positions[:-1]])
    per_step_rewards = positions * test_ret - transaction_cost * np.abs(positions - prev_positions)

    return {
        "rewards": [float(r) for r in per_step_rewards],
        "positions": [int(p) for p in positions],
        "forecasts": [float(f) for f in forecasts],
    }


# ---------- helpers mirroring train.py's semantics ----------

def position_stats(positions):
    arr = np.asarray(positions)
    return {
        "n_long":  int((arr == 1).sum()),
        "n_flat":  int((arr == 0).sum()),
        "n_short": int((arr == -1).sum()),
        "n_flips": int(np.count_nonzero(np.diff(arr))) if arr.size > 1 else 0,
    }


def buy_and_hold_returns(returns_slice):
    return returns_slice.astype(np.float64)


# ---------- CSV schemas ----------

def windows_csv_fieldnames() -> list[str]:
    return [
        "run_id", "hp_id", "seed",
        "hp_hidden_size", "hp_num_layers", "hp_dropout", "hp_seq_len",
        "window_index",
        "train_start_date", "train_end_date", "train_days",
        "test_start_date", "test_end_date", "test_days",
        "total_epochs", "train_wall_seconds",
        "agent_sharpe", "agent_cumret", "agent_maxdd",
        "bh_sharpe", "bh_cumret", "bh_maxdd",
        "pos_long", "pos_flat", "pos_short", "pos_flips",
        "forecast_mean", "forecast_std",
    ]


def steps_csv_fieldnames() -> list[str]:
    return ["run_id", "hp_id", "seed", "window_index", "step", "date", "reward", "position", "forecast"]


# ---------- resume logic ----------

def load_completed_triplets(windows_csv_path: Path, expected_fields: list[str]) -> set[tuple[int, int, int]]:
    """Read (hp_id, seed, window_index) triplets already completed."""
    done: set[tuple[int, int, int]] = set()
    if not windows_csv_path.exists():
        return done
    with open(windows_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return done
        if header != expected_fields:
            raise RuntimeError(
                f"Existing {windows_csv_path.name} has different schema than this run expects.\n"
                f"  existing: {header}\n  expected: {expected_fields}\n"
                f"Delete the results dir if you meant to start fresh."
            )
        # column indices we care about
        idx_hp = header.index("hp_id")
        idx_seed = header.index("seed")
        idx_win = header.index("window_index")
        for row in reader:
            if row:
                done.add((int(row[idx_hp]), int(row[idx_seed]), int(row[idx_win])))
    return done


# ---------- champion selection ----------

def compute_champion(per_window_records: list[dict], hp_grid: list[dict]) -> dict | None:
    """Group by hp_id, aggregate across seeds and windows, pick best mean Sharpe."""
    if not per_window_records:
        return None
    per_hp = {}
    for rec in per_window_records:
        hp_id = rec["hp_id"]
        per_hp.setdefault(hp_id, []).append(rec)

    summaries = []
    for hp_id, rows in per_hp.items():
        sharpes = np.array([r["agent_test"]["metrics"]["sharpe_ratio"] for r in rows])
        cumrets = np.array([r["agent_test"]["metrics"]["cumulative_return"] for r in rows])
        maxdds  = np.array([r["agent_test"]["metrics"]["max_drawdown"]     for r in rows])
        seeds_seen = sorted({r["seed"] for r in rows})
        summaries.append({
            "hp_id": hp_id,
            "hp": hp_grid[hp_id],
            "seeds": seeds_seen,
            "n_windows_total": len(rows),
            "sharpe":  {"mean": float(sharpes.mean()), "std": float(sharpes.std(ddof=1)) if sharpes.size > 1 else 0.0},
            "cumret":  {"mean": float(cumrets.mean()), "std": float(cumrets.std(ddof=1)) if cumrets.size > 1 else 0.0},
            "maxdd":   {"mean": float(maxdds.mean()),  "std": float(maxdds.std(ddof=1))  if maxdds.size > 1 else 0.0},
        })
    summaries.sort(key=lambda s: s["sharpe"]["mean"], reverse=True)
    return {"ranking_by_sharpe_mean": summaries, "champion": summaries[0]}


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Sweep-native LSTM baseline training.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML sweep config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = Path(args.config).stem
    results_dir = ROOT / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)

    print(f"[{run_name}] loading data...")
    df = load_aligned(vix_file=cfg["data"]["vix_file"], sp_file=cfg["data"]["sp_file"])
    obs_full, ret_full, dates = to_env_arrays(
        df,
        observation_col=cfg["data"]["observation_col"],
        asset_col=cfg["data"]["asset_col"],
    )
    n = len(obs_full)
    print(f"[{run_name}] {n} aligned trading days ({dates[0].date()} .. {dates[-1].date()})")

    # ---- sweep config ----
    sweep = cfg["lstm_sweep"]
    hp_grid: list[dict] = sweep["hp_grid"]
    seeds: list[int] = list(cfg["seeds"])
    epochs = int(sweep.get("epochs", 10))
    batch_size = int(sweep.get("batch_size", 64))
    learning_rate = float(sweep.get("learning_rate", 1e-3))
    weight_decay = float(sweep.get("weight_decay", 0.0))
    grad_clip = float(sweep.get("grad_clip", 0.0))
    threshold = float(sweep.get("threshold", 0.0))
    standardise = bool(sweep.get("standardise", True))

    print(f"[{run_name}] sweep: {len(hp_grid)} HP combos × {len(seeds)} seeds = {len(hp_grid)*len(seeds)} runs, "
          f"each over walk-forward windows")

    # ---- CSV paths + resume ----
    windows_csv_path = results_dir / "windows.csv"
    steps_csv_path = results_dir / "steps.csv"
    windows_fields = windows_csv_fieldnames()
    steps_fields = steps_csv_fieldnames()

    completed = load_completed_triplets(windows_csv_path, windows_fields)
    if completed:
        print(f"[{run_name}] resuming — {len(completed)} (hp_id, seed, window) triplets already done, will skip")

    fresh_run = not completed
    windows_f = open(windows_csv_path, "w" if fresh_run else "a", newline="")
    steps_f = open(steps_csv_path, "w" if fresh_run else "a", newline="")
    windows_writer = csv.DictWriter(windows_f, fieldnames=windows_fields)
    steps_writer = csv.DictWriter(steps_f, fieldnames=steps_fields)
    if fresh_run:
        windows_writer.writeheader()
        steps_writer.writeheader()
        windows_f.flush()
        steps_f.flush()

    per_window_records = []   # rich in-memory copies for summary.json

    try:
        for hp_id, hp in enumerate(hp_grid):
            hidden_size = int(hp["hidden_size"])
            num_layers = int(hp["num_layers"])
            dropout = float(hp.get("dropout", 0.0))
            seq_len = int(hp["seq_len"])

            for seed in seeds:
                run_id = f"hp{hp_id}_seed{seed}"
                # Set both torch and numpy seeds each run for reproducibility.
                torch.manual_seed(seed)
                np.random.seed(seed)

                for w in walk_forward_windows(
                    n_samples=n,
                    train_size=cfg["walk_forward"]["train_size"],
                    test_size=cfg["walk_forward"]["test_size"],
                    max_windows=cfg["walk_forward"].get("max_windows"),
                ):
                    if (hp_id, seed, w.index) in completed:
                        continue

                    tr, te = w.sizes()
                    train_dates = (dates[w.train_start].date(), dates[w.train_end - 1].date())
                    test_dates = (dates[w.test_start].date(), dates[w.test_end - 1].date())

                    obs_train_raw = obs_full[w.train_start : w.train_end]
                    ret_train = ret_full[w.train_start : w.train_end]
                    obs_test_raw = obs_full[w.test_start : w.test_end]
                    ret_test = ret_full[w.test_start : w.test_end]
                    test_dates_arr = dates[w.test_start : w.test_end]

                    # --- per-window standardisation, fit on TRAIN slice only ---
                    if standardise:
                        mu, sigma = fit_standardiser(obs_train_raw)
                        obs_train = apply_standardiser(obs_train_raw, mu, sigma)
                        obs_test = apply_standardiser(obs_test_raw, mu, sigma)
                    else:
                        obs_train = obs_train_raw
                        obs_test = obs_test_raw

                    # --- build + train LSTM ---
                    train_dataset = SequenceDataset(obs_train, ret_train, k=seq_len)
                    model = LSTMForecaster(
                        inputsize=1,
                        hiddensze=hidden_size,
                        numLayers=num_layers,
                        dropout=dropout,
                    )
                    t0 = time.time()
                    train_forecaster(
                        model,
                        train_dataset,
                        epochs=epochs,
                        batchsze=batch_size,
                        learn_rate=learning_rate,
                        weight_decay=weight_decay,
                        grad_clip=grad_clip,
                        verbose=False,   # sweep runs are noisy; per-epoch prints get spammy
                    )
                    train_wall = time.time() - t0

                    # --- rollout ---
                    rollout_data = rollout_lstm(
                        model=model,
                        warmup_obs=obs_train,
                        test_obs=obs_test,
                        test_ret=ret_test,
                        seq_len=seq_len,
                        threshold=threshold,
                        transaction_cost=cfg["env"]["transaction_cost"],
                    )

                    agent_metrics = metric_summary(np.asarray(rollout_data["rewards"]))
                    agent_positions = position_stats(rollout_data["positions"])
                    forecasts_arr = np.asarray(rollout_data["forecasts"])
                    bh_metrics = metric_summary(buy_and_hold_returns(ret_test))

                    # --- windows.csv row ---
                    windows_writer.writerow({
                        "run_id": run_id,
                        "hp_id": hp_id,
                        "seed": seed,
                        "hp_hidden_size": hidden_size,
                        "hp_num_layers": num_layers,
                        "hp_dropout": dropout,
                        "hp_seq_len": seq_len,
                        "window_index": w.index,
                        "train_start_date": str(train_dates[0]),
                        "train_end_date": str(train_dates[1]),
                        "train_days": tr,
                        "test_start_date": str(test_dates[0]),
                        "test_end_date": str(test_dates[1]),
                        "test_days": te,
                        "total_epochs": epochs,
                        "train_wall_seconds": round(train_wall, 2),
                        "agent_sharpe": round(agent_metrics["sharpe_ratio"], 6),
                        "agent_cumret": round(agent_metrics["cumulative_return"], 6),
                        "agent_maxdd":  round(agent_metrics["max_drawdown"], 6),
                        "bh_sharpe":    round(bh_metrics["sharpe_ratio"], 6),
                        "bh_cumret":    round(bh_metrics["cumulative_return"], 6),
                        "bh_maxdd":     round(bh_metrics["max_drawdown"], 6),
                        "pos_long":  agent_positions["n_long"],
                        "pos_flat":  agent_positions["n_flat"],
                        "pos_short": agent_positions["n_short"],
                        "pos_flips": agent_positions["n_flips"],
                        "forecast_mean": round(float(forecasts_arr.mean()), 6),
                        "forecast_std":  round(float(forecasts_arr.std()),  6),
                    })
                    windows_f.flush()

                    # --- steps.csv rows ---
                    for j, (r_step, p_step, f_step) in enumerate(
                        zip(rollout_data["rewards"], rollout_data["positions"], rollout_data["forecasts"])
                    ):
                        steps_writer.writerow({
                            "run_id": run_id,
                            "hp_id": hp_id,
                            "seed": seed,
                            "window_index": w.index,
                            "step": j,
                            "date": str(test_dates_arr[j].date()),
                            "reward": round(r_step, 6),
                            "position": p_step,
                            "forecast": round(f_step, 6),
                        })
                    steps_f.flush()

                    per_window_records.append({
                        "run_id": run_id,
                        "hp_id": hp_id,
                        "seed": seed,
                        "hp": {
                            "hidden_size": hidden_size, "num_layers": num_layers,
                            "dropout": dropout, "seq_len": seq_len,
                        },
                        "window_index": w.index,
                        "dates": {
                            "train_start": str(train_dates[0]), "train_end": str(train_dates[1]),
                            "test_start":  str(test_dates[0]),  "test_end":  str(test_dates[1]),
                        },
                        "sizes": {"train": tr, "test": te},
                        "training": {"total_epochs": epochs, "wall_seconds": round(train_wall, 2)},
                        "agent_test": {
                            "n_steps": int(agent_metrics["n_steps"]),
                            "metrics": {
                                "cumulative_return": agent_metrics["cumulative_return"],
                                "sharpe_ratio": agent_metrics["sharpe_ratio"],
                                "max_drawdown": agent_metrics["max_drawdown"],
                            },
                            "positions": agent_positions,
                            "forecast_stats": {
                                "mean": float(forecasts_arr.mean()),
                                "std":  float(forecasts_arr.std()),
                                "min":  float(forecasts_arr.min()),
                                "max":  float(forecasts_arr.max()),
                            },
                        },
                        "buy_and_hold_test": {
                            "n_steps": int(bh_metrics["n_steps"]),
                            "metrics": {
                                "cumulative_return": bh_metrics["cumulative_return"],
                                "sharpe_ratio": bh_metrics["sharpe_ratio"],
                                "max_drawdown": bh_metrics["max_drawdown"],
                            },
                        },
                    })

                    print(
                        f"[{run_name}] hp{hp_id} seed{seed} W{w.index:2d}  "
                        f"Sharpe {agent_metrics['sharpe_ratio']:+.3f}  "
                        f"cumret {agent_metrics['cumulative_return']:+.4f}  "
                        f"maxDD {agent_metrics['max_drawdown']:.4f}  "
                        f"B&H {bh_metrics['sharpe_ratio']:+.3f}  "
                        f"train {train_wall:.1f}s"
                    )
    finally:
        windows_f.close()
        steps_f.close()

    # ---- summary.json + champion identification ----
    champion_info = compute_champion(per_window_records, hp_grid)

    output = {
        "run_name": run_name,
        "config": cfg,
        "data_dates_full": {"start": str(dates[0].date()), "end": str(dates[-1].date()), "n": int(n)},
        "sweep_shape": {"n_hp_combos": len(hp_grid), "n_seeds": len(seeds)},
        "per_window": per_window_records,
        "champion": champion_info,
    }
    summary_path = results_dir / "summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    tmp_path.replace(summary_path)

    print(f"\n[{run_name}] wrote {windows_csv_path.name}, {steps_csv_path.name}, {summary_path.name}")
    if champion_info:
        c = champion_info["champion"]
        print(
            f"[{run_name}] CHAMPION hp{c['hp_id']}  "
            f"{c['hp']}  "
            f"Sharpe {c['sharpe']['mean']:+.3f} ± {c['sharpe']['std']:.3f}  "
            f"cumret {c['cumret']['mean']:+.4f} ± {c['cumret']['std']:.4f}  "
            f"across {len(c['seeds'])} seeds"
        )


if __name__ == "__main__":
    main()
