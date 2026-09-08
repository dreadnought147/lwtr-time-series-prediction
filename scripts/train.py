"""
Sweep-native RL training entry point (PPO + FrameStack OR AdaptiveStack).

Rewritten from single-config → sweep-native so one command trains the full
matched-budget grid for RQ2. Config specifies:
  - memory.type ∈ {fs, as}                (one paradigm per config)
  - memory.k_grid = list of k values      (memory budget sweep)
  - seeds = list of seed ints             (statistical spread)
  - ppo.*   fixed across the whole sweep  (isolates the memory mechanism)

For each (k, seed, walk-forward window) triplet:
  1. Build fresh env + wrapper at this k.
  2. Instantiate PPO with the FIXED ppo config.
  3. Train for total_timesteps, evaluate on the test slice.
  4. Log to CSVs + emit SB3 training curves per window.

Outputs (all under results/{run_name}/):
  windows.csv                          one row per (k, seed, window). Flat, tidy.
  steps.csv                            one row per (k, seed, window, day).
  training_curves/w{w}_k{k}_s{seed}/progress.csv
                                       SB3's per-rollout training stats.
  summary.json                         nested rich record + champion-by-k.

CLUSTER-FRIENDLY
----------------
- CSVs appended row-by-row and flushed after each window.
- summary.json written atomically (.tmp → rename) at the end.
- Resume: skips (k, seed, window) triplets already in windows.csv.

Run:
    python scripts/train.py --config configs/rq_rl_fs_sweep.yaml
    python scripts/train.py --config configs/rq_rl_as_sweep.yaml
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

# so this script can import from src/
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.logger import configure as sb3_configure_logger

from src.data.loader import load_aligned
from src.data.transforms import to_env_arrays
from src.envs.timeseries_env import TimeSeries
from src.memory.framestack import FrameStack
from src.memory.adaptive_stack import AdaptiveStack
from src.evaluation.metrics import summary as metric_summary
from src.evaluation.walk_forward import walk_forward_windows


# ---------- wrappers + rollouts ----------

def build_memory_wrapper(env, memory_type: str, k: int):
    """Wrap `env` with the memory mechanism the config asks for."""
    if memory_type == "none":
        return env
    if memory_type == "fs":
        return FrameStack(env, k=k)
    if memory_type == "as":
        return AdaptiveStack(env, num_stack=k)
    raise ValueError(f"Unknown memory_type: {memory_type!r}")


def rollout(model, env) -> dict:
    """Replay the test period once with the trained policy, collect per-day details."""
    obs, _ = env.reset()
    rewards, positions, memory_actions = [], [], []
    done = False
    while not done:
        action, _ = model.predict(obs, deterministic=True)
        obs, reward, terminated, truncated, info = env.step(action)
        rewards.append(float(reward))
        positions.append(int(info.get("position", 0)))
        if "memory_action" in info:
            memory_actions.append(int(info["memory_action"]))
        done = terminated or truncated
    return {
        "rewards": rewards,
        "positions": positions,
        "memory_actions": memory_actions if memory_actions else None,
    }


def position_stats(positions):
    arr = np.asarray(positions)
    return {
        "n_long":  int((arr == 1).sum()),
        "n_flat":  int((arr == 0).sum()),
        "n_short": int((arr == -1).sum()),
        "n_flips": int(np.count_nonzero(np.diff(arr))) if arr.size > 1 else 0,
    }


def memory_action_counts(actions: list[int], num_stack: int) -> list[int]:
    counts = [0] * num_stack
    for a in actions:
        counts[a] += 1
    return counts


def buy_and_hold_returns(returns_slice):
    return returns_slice.astype(np.float64)


# ---------- CSV schemas ----------

def windows_csv_fieldnames(max_k: int) -> list[str]:
    """Column order for windows.csv — stable across the whole sweep.

    mem_slot_* columns padded to max_k so rows for smaller k values keep the
    same schema (extra slots stay empty). This lets a single pd.read_csv
    load the whole sweep as one DataFrame without column-set gymnastics.
    """
    base = [
        "run_id", "memory_type", "k", "seed",
        "window_index",
        "train_start_date", "train_end_date", "train_days",
        "test_start_date", "test_end_date", "test_days",
        "total_timesteps", "train_wall_seconds",
        "agent_sharpe", "agent_cumret", "agent_maxdd",
        "bh_sharpe", "bh_cumret", "bh_maxdd",
        "pos_long", "pos_flat", "pos_short", "pos_flips",
    ]
    for i in range(max_k):
        base.append(f"mem_slot_{i}")
    return base


def steps_csv_fieldnames() -> list[str]:
    return ["run_id", "memory_type", "k", "seed", "window_index", "step",
            "date", "reward", "position", "memory_action"]


# ---------- resume logic ----------

def load_completed_triplets(windows_csv_path: Path, expected_fields: list[str]) -> set[tuple[int, int, int]]:
    """Read (k, seed, window_index) triplets already completed."""
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
        idx_k = header.index("k")
        idx_seed = header.index("seed")
        idx_win = header.index("window_index")
        for row in reader:
            if row:
                done.add((int(row[idx_k]), int(row[idx_seed]), int(row[idx_win])))
    return done


# ---------- champion selection ----------

def compute_champion(per_window_records: list[dict]) -> dict | None:
    """Group by k, aggregate across seeds and windows, pick best mean Sharpe."""
    if not per_window_records:
        return None
    per_k: dict[int, list[dict]] = {}
    for rec in per_window_records:
        per_k.setdefault(rec["k"], []).append(rec)

    summaries = []
    for k_val, rows in per_k.items():
        sharpes = np.array([r["agent_test"]["metrics"]["sharpe_ratio"] for r in rows])
        cumrets = np.array([r["agent_test"]["metrics"]["cumulative_return"] for r in rows])
        maxdds  = np.array([r["agent_test"]["metrics"]["max_drawdown"]     for r in rows])
        seeds_seen = sorted({r["seed"] for r in rows})
        summaries.append({
            "k": k_val,
            "seeds": seeds_seen,
            "n_windows_total": len(rows),
            "sharpe": {"mean": float(sharpes.mean()), "std": float(sharpes.std(ddof=1)) if sharpes.size > 1 else 0.0},
            "cumret": {"mean": float(cumrets.mean()), "std": float(cumrets.std(ddof=1)) if cumrets.size > 1 else 0.0},
            "maxdd":  {"mean": float(maxdds.mean()),  "std": float(maxdds.std(ddof=1))  if maxdds.size > 1 else 0.0},
        })
    summaries.sort(key=lambda s: s["sharpe"]["mean"], reverse=True)
    return {"ranking_by_sharpe_mean": summaries, "champion": summaries[0]}


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Sweep-native RL training run.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML sweep config.")
    args = parser.parse_args()

    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    run_name = Path(args.config).stem
    results_dir = ROOT / "results" / run_name
    results_dir.mkdir(parents=True, exist_ok=True)
    curves_root = results_dir / "training_curves"
    curves_root.mkdir(parents=True, exist_ok=True)

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
    memory_type = cfg["memory"]["type"]
    k_grid: list[int] = [int(k) for k in cfg["memory"]["k_grid"]]
    seeds: list[int] = [int(s) for s in cfg["seeds"]]
    max_k = max(k_grid)   # for padded mem_slot_* columns
    is_as = memory_type == "as"

    ppo_cfg = cfg["ppo"]
    total_timesteps = int(ppo_cfg["total_timesteps"])
    n_steps = int(ppo_cfg.get("n_steps", 2048))
    batch_size = int(ppo_cfg.get("batch_size", 64))
    learning_rate = float(ppo_cfg.get("learning_rate", 3e-4))

    print(f"[{run_name}] sweep: memory={memory_type}, k_grid={k_grid}, seeds={seeds} "
          f"→ {len(k_grid) * len(seeds)} (k, seed) combos across walk-forward windows")

    # ---- CSV paths + resume ----
    windows_csv_path = results_dir / "windows.csv"
    steps_csv_path = results_dir / "steps.csv"
    windows_fields = windows_csv_fieldnames(max_k)
    steps_fields = steps_csv_fieldnames()

    completed = load_completed_triplets(windows_csv_path, windows_fields)
    if completed:
        print(f"[{run_name}] resuming — {len(completed)} (k, seed, window) triplets already done, will skip")

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

    per_window_records: list[dict] = []

    try:
        for k in k_grid:
            for seed in seeds:
                run_id = f"{memory_type}_k{k}_seed{seed}"
                # Seed everything each (k, seed) start for reproducibility.
                set_random_seed(seed)
                np.random.seed(seed)

                for w in walk_forward_windows(
                    n_samples=n,
                    train_size=cfg["walk_forward"]["train_size"],
                    test_size=cfg["walk_forward"]["test_size"],
                    max_windows=cfg["walk_forward"].get("max_windows"),
                ):
                    if (k, seed, w.index) in completed:
                        continue

                    tr, te = w.sizes()
                    train_dates = (dates[w.train_start].date(), dates[w.train_end - 1].date())
                    test_dates = (dates[w.test_start].date(), dates[w.test_end - 1].date())

                    # slice
                    obs_train = obs_full[w.train_start : w.train_end]
                    ret_train = ret_full[w.train_start : w.train_end]
                    obs_test = obs_full[w.test_start : w.test_end]
                    ret_test = ret_full[w.test_start : w.test_end]
                    test_dates_arr = dates[w.test_start : w.test_end]

                    # build + train
                    train_env = TimeSeries(
                        obs_train, ret_train,
                        transaction_cost=cfg["env"]["transaction_cost"],
                    )
                    train_env = build_memory_wrapper(train_env, memory_type, k)

                    model = PPO(
                        "MlpPolicy",
                        train_env,
                        verbose=0,
                        seed=seed,
                        device=cfg.get("device", "cpu"),
                        n_steps=n_steps,
                        batch_size=batch_size,
                        learning_rate=learning_rate,
                    )

                    # SB3 training-curve capture — writes progress.csv into
                    # results/{run}/training_curves/w{w}_k{k}_s{seed}/
                    curve_dir = curves_root / f"w{w.index}_k{k}_s{seed}"
                    curve_dir.mkdir(parents=True, exist_ok=True)
                    curve_logger = sb3_configure_logger(str(curve_dir), ["csv"])
                    model.set_logger(curve_logger)

                    t0 = time.time()
                    model.learn(total_timesteps=total_timesteps)
                    train_wall = time.time() - t0

                    # eval
                    eval_env = TimeSeries(
                        obs_test, ret_test,
                        transaction_cost=cfg["env"]["transaction_cost"],
                    )
                    eval_env = build_memory_wrapper(eval_env, memory_type, k)

                    rollout_data = rollout(model, eval_env)
                    agent_metrics = metric_summary(np.asarray(rollout_data["rewards"]))
                    agent_positions = position_stats(rollout_data["positions"])
                    agent_mem_counts = (
                        memory_action_counts(rollout_data["memory_actions"], k)
                        if is_as and rollout_data["memory_actions"] is not None
                        else None
                    )

                    bh_returns = buy_and_hold_returns(ret_test)
                    bh_metrics = metric_summary(bh_returns)

                    # ---- windows.csv row ----
                    windows_row = {
                        "run_id": run_id,
                        "memory_type": memory_type,
                        "k": k,
                        "seed": seed,
                        "window_index": w.index,
                        "train_start_date": str(train_dates[0]),
                        "train_end_date": str(train_dates[1]),
                        "train_days": tr,
                        "test_start_date": str(test_dates[0]),
                        "test_end_date": str(test_dates[1]),
                        "test_days": te,
                        "total_timesteps": total_timesteps,
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
                    }
                    for i in range(max_k):
                        # fill only the slots this run's k uses; leave the rest empty
                        if agent_mem_counts is not None and i < k:
                            windows_row[f"mem_slot_{i}"] = agent_mem_counts[i]
                        else:
                            windows_row[f"mem_slot_{i}"] = ""
                    windows_writer.writerow(windows_row)
                    windows_f.flush()

                    # ---- steps.csv rows ----
                    for j, (r_step, p_step) in enumerate(
                        zip(rollout_data["rewards"], rollout_data["positions"])
                    ):
                        steps_writer.writerow({
                            "run_id": run_id,
                            "memory_type": memory_type,
                            "k": k,
                            "seed": seed,
                            "window_index": w.index,
                            "step": j,
                            "date": str(test_dates_arr[j].date()),
                            "reward": round(r_step, 6),
                            "position": p_step,
                            "memory_action": (
                                rollout_data["memory_actions"][j]
                                if rollout_data["memory_actions"] is not None else ""
                            ),
                        })
                    steps_f.flush()

                    # ---- rich record for summary.json ----
                    window_record = {
                        "run_id": run_id,
                        "memory_type": memory_type,
                        "k": k,
                        "seed": seed,
                        "window_index": w.index,
                        "dates": {
                            "train_start": str(train_dates[0]), "train_end": str(train_dates[1]),
                            "test_start":  str(test_dates[0]),  "test_end":  str(test_dates[1]),
                        },
                        "sizes": {"train": tr, "test": te},
                        "training": {
                            "total_timesteps": total_timesteps,
                            "wall_seconds": round(train_wall, 2),
                            "curves_path": str(curve_dir.relative_to(results_dir)),
                        },
                        "agent_test": {
                            "n_steps": int(agent_metrics["n_steps"]),
                            "metrics": {
                                "cumulative_return": agent_metrics["cumulative_return"],
                                "sharpe_ratio": agent_metrics["sharpe_ratio"],
                                "max_drawdown": agent_metrics["max_drawdown"],
                            },
                            "positions": agent_positions,
                        },
                        "buy_and_hold_test": {
                            "n_steps": int(bh_metrics["n_steps"]),
                            "metrics": {
                                "cumulative_return": bh_metrics["cumulative_return"],
                                "sharpe_ratio": bh_metrics["sharpe_ratio"],
                                "max_drawdown": bh_metrics["max_drawdown"],
                            },
                        },
                    }
                    if agent_mem_counts is not None:
                        window_record["agent_test"]["memory_action_counts"] = agent_mem_counts
                    per_window_records.append(window_record)

                    print(
                        f"[{run_name}] {run_id} W{w.index:2d}  "
                        f"Sharpe {agent_metrics['sharpe_ratio']:+.3f}  "
                        f"cumret {agent_metrics['cumulative_return']:+.4f}  "
                        f"maxDD {agent_metrics['max_drawdown']:.4f}  "
                        f"B&H {bh_metrics['sharpe_ratio']:+.3f}  "
                        f"train {train_wall:.1f}s"
                    )
                    if agent_mem_counts is not None:
                        print(f"[{run_name}]   mem overwrite counts: {agent_mem_counts}")

    finally:
        windows_f.close()
        steps_f.close()

    # ---- summary.json + champion identification ----
    champion_info = compute_champion(per_window_records)

    output = {
        "run_name": run_name,
        "config": cfg,
        "data_dates_full": {"start": str(dates[0].date()), "end": str(dates[-1].date()), "n": int(n)},
        "sweep_shape": {"memory_type": memory_type, "k_grid": k_grid, "seeds": seeds},
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
            f"[{run_name}] CHAMPION k={c['k']}  "
            f"Sharpe {c['sharpe']['mean']:+.3f} ± {c['sharpe']['std']:.3f}  "
            f"cumret {c['cumret']['mean']:+.4f} ± {c['cumret']['std']:.4f}  "
            f"across {len(c['seeds'])} seeds"
        )


if __name__ == "__main__":
    main()
