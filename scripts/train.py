"""
Config-driven training entry point.

Reads a YAML config, loads data, splits into walk-forward windows, trains PPO
on each window, evaluates on the test slice, writes rich outputs to disk.

Outputs (all under results/{run_name}/):
  windows.csv    one row per completed window — tidy, pandas-friendly.
  steps.csv      one row per (window, day) — for per-step analysis / plots.
  summary.json   nested rich record — narrative, self-documenting.

CLUSTER-FRIENDLY DESIGN
-----------------------
1. CSVs are appended row-by-row and flushed after each window. If the process
   dies mid-run, the CSV is valid up to the last completed window — never
   corrupt, always readable.
2. summary.json is written atomically at the end (write to .tmp, then rename)
   so a partial JSON never exists.
3. Resume: if windows.csv already exists in the run's results dir, the script
   reads which windows were completed and picks up from the next one. The
   schema of the existing CSV must match this run's config; if it doesn't,
   the script fails loudly rather than silently drifting.

Run:
    python scripts/train.py --config configs/rq2_fs_k5_seed0.yaml
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

from src.data.loader import load_aligned
from src.data.transforms import to_env_arrays
from src.envs.timeseries_env import TimeSeries
from src.memory.framestack import FrameStack
from src.memory.adaptive_stack import AdaptiveStack
from src.evaluation.metrics import summary as metric_summary
from src.evaluation.walk_forward import walk_forward_windows


# ---------- wrappers + rollouts ----------

def build_memory_wrapper(env, memory_type: str, k: int):
    """Wrap `env` with the memory mechanism the config asks for.

    memory_type: "none" (no wrapper), "fs" (Frame Stacking), "as" (Adaptive Stacking).

    # this is the ONE PLACE where the memory choice happens.
    # everything upstream and downstream stays memory-agnostic.
    """
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


def position_stats(positions: list[int]) -> dict:
    """Count how the agent used its position choices over one test window."""
    arr = np.asarray(positions)
    return {
        "n_long": int((arr == 1).sum()),
        "n_flat": int((arr == 0).sum()),
        "n_short": int((arr == -1).sum()),
        "n_flips": int(np.count_nonzero(np.diff(arr))) if arr.size > 1 else 0,
    }


def memory_action_counts(actions: list[int], num_stack: int) -> list[int]:
    """Histogram of overwrite-choices, one entry per slot 0..k-1."""
    counts = [0] * num_stack
    for a in actions:
        counts[a] += 1
    return counts


def buy_and_hold_returns(returns_slice: np.ndarray) -> np.ndarray:
    """Buy-and-hold benchmark: always long, no trading cost."""
    return returns_slice.astype(np.float64)


# ---------- CSV schemas ----------

def windows_csv_fieldnames(memory_k: int) -> list[str]:
    """Column order for windows.csv. Stable across runs with same memory_k."""
    base = [
        "window_index",
        "train_start_date", "train_end_date", "train_days",
        "test_start_date", "test_end_date", "test_days",
        "total_timesteps", "train_wall_seconds",
        "agent_sharpe", "agent_cumret", "agent_maxdd",
        "bh_sharpe", "bh_cumret", "bh_maxdd",
        "pos_long", "pos_flat", "pos_short", "pos_flips",
    ]
    # AS-only columns kept in the schema for both FS and AS runs
    # (empty for FS) so many-run CSVs can concat cleanly later.
    for i in range(memory_k):
        base.append(f"mem_slot_{i}")
    return base


def steps_csv_fieldnames() -> list[str]:
    return ["window_index", "step", "date", "reward", "position", "memory_action"]


# ---------- resume logic ----------

def determine_start_index(
    windows_csv_path: Path, expected_fields: list[str]
) -> int:
    """If windows.csv exists, figure out which window to start from.

    Returns 0 if no prior run. Returns max completed window_index + 1 otherwise.
    Fails loudly if the existing header doesn't match — we don't silently
    stitch mismatched schemas.
    """
    if not windows_csv_path.exists():
        return 0
    with open(windows_csv_path, "r", newline="") as f:
        reader = csv.reader(f)
        header = next(reader, None)
        if header is None:
            return 0
        if header != expected_fields:
            raise RuntimeError(
                f"Existing {windows_csv_path.name} has different schema than this run "
                f"expects.\n  existing: {header}\n  expected: {expected_fields}\n"
                f"Delete the results dir if you meant to start fresh."
            )
        completed = [int(row[0]) for row in reader if row]
    if not completed:
        return 0
    return max(completed) + 1


# ---------- main ----------

def main():
    parser = argparse.ArgumentParser(description="Config-driven training run.")
    parser.add_argument("--config", type=str, required=True, help="Path to YAML config.")
    args = parser.parse_args()

    # load the recipe for this run
    with open(args.config) as f:
        cfg = yaml.safe_load(f)

    # results go in a folder named after the config file
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

    set_random_seed(cfg["seed"])
    np.random.seed(cfg["seed"])

    memory_type = cfg["memory"]["type"]
    memory_k = cfg["memory"]["k"]
    is_as = memory_type == "as"

    # --- CSV paths + resume logic ---
    windows_csv_path = results_dir / "windows.csv"
    steps_csv_path = results_dir / "steps.csv"
    windows_fields = windows_csv_fieldnames(memory_k)
    steps_fields = steps_csv_fieldnames()

    start_index = determine_start_index(windows_csv_path, windows_fields)
    if start_index > 0:
        print(f"[{run_name}] resuming — {start_index} window(s) already completed, "
              f"continuing from window {start_index}")

    # Open CSVs in append mode. Write headers only if starting fresh.
    fresh_run = start_index == 0
    windows_f = open(windows_csv_path, "w" if fresh_run else "a", newline="")
    steps_f = open(steps_csv_path, "w" if fresh_run else "a", newline="")
    windows_writer = csv.DictWriter(windows_f, fieldnames=windows_fields)
    steps_writer = csv.DictWriter(steps_f, fieldnames=steps_fields)
    if fresh_run:
        windows_writer.writeheader()
        steps_writer.writeheader()
        windows_f.flush()
        steps_f.flush()

    per_window = []  # in-memory copy for the final summary.json

    try:
        for w in walk_forward_windows(
            n_samples=n,
            train_size=cfg["walk_forward"]["train_size"],
            test_size=cfg["walk_forward"]["test_size"],
            max_windows=cfg["walk_forward"].get("max_windows"),
        ):
            if w.index < start_index:
                # already done in a previous partial run — skip.
                continue

            tr, te = w.sizes()
            train_dates = (dates[w.train_start].date(), dates[w.train_end - 1].date())
            test_dates = (dates[w.test_start].date(), dates[w.test_end - 1].date())
            print(
                f"\n[{run_name}] window {w.index}: "
                f"train {tr}d ({train_dates[0]}..{train_dates[1]}), "
                f"test {te}d ({test_dates[0]}..{test_dates[1]})"
            )

            # slice
            obs_train = obs_full[w.train_start : w.train_end]
            ret_train = ret_full[w.train_start : w.train_end]
            obs_test = obs_full[w.test_start : w.test_end]
            ret_test = ret_full[w.test_start : w.test_end]
            test_dates_arr = dates[w.test_start : w.test_end]

            # build + train
            train_env = TimeSeries(
                obs_train, ret_train, transaction_cost=cfg["env"]["transaction_cost"]
            )
            train_env = build_memory_wrapper(train_env, memory_type, memory_k)

            model = PPO(
                "MlpPolicy",
                train_env,
                verbose=0,
                seed=cfg["seed"],
                device=cfg.get("device", "cpu"),
                n_steps=cfg["ppo"].get("n_steps", 2048),
                batch_size=cfg["ppo"].get("batch_size", 64),
                learning_rate=cfg["ppo"].get("learning_rate", 3e-4),
            )
            t0 = time.time()
            model.learn(total_timesteps=cfg["ppo"]["total_timesteps"])
            train_wall = time.time() - t0

            # eval
            eval_env = TimeSeries(
                obs_test, ret_test, transaction_cost=cfg["env"]["transaction_cost"]
            )
            eval_env = build_memory_wrapper(eval_env, memory_type, memory_k)

            rollout_data = rollout(model, eval_env)
            agent_metrics = metric_summary(np.asarray(rollout_data["rewards"]))
            agent_positions = position_stats(rollout_data["positions"])
            agent_mem_counts = (
                memory_action_counts(rollout_data["memory_actions"], memory_k)
                if is_as and rollout_data["memory_actions"] is not None
                else None
            )

            bh_returns = buy_and_hold_returns(ret_test)
            bh_metrics = metric_summary(bh_returns)

            # ---- write to windows.csv (one row) ----
            windows_row = {
                "window_index": w.index,
                "train_start_date": str(train_dates[0]),
                "train_end_date": str(train_dates[1]),
                "train_days": tr,
                "test_start_date": str(test_dates[0]),
                "test_end_date": str(test_dates[1]),
                "test_days": te,
                "total_timesteps": cfg["ppo"]["total_timesteps"],
                "train_wall_seconds": round(train_wall, 2),
                "agent_sharpe": round(agent_metrics["sharpe_ratio"], 6),
                "agent_cumret": round(agent_metrics["cumulative_return"], 6),
                "agent_maxdd": round(agent_metrics["max_drawdown"], 6),
                "bh_sharpe": round(bh_metrics["sharpe_ratio"], 6),
                "bh_cumret": round(bh_metrics["cumulative_return"], 6),
                "bh_maxdd": round(bh_metrics["max_drawdown"], 6),
                "pos_long": agent_positions["n_long"],
                "pos_flat": agent_positions["n_flat"],
                "pos_short": agent_positions["n_short"],
                "pos_flips": agent_positions["n_flips"],
            }
            for i in range(memory_k):
                # empty for FS runs — schema stable, semantics honest
                windows_row[f"mem_slot_{i}"] = (
                    agent_mem_counts[i] if agent_mem_counts is not None else ""
                )
            windows_writer.writerow(windows_row)
            windows_f.flush()

            # ---- write to steps.csv (one row per day) ----
            for j, (r_step, p_step) in enumerate(zip(rollout_data["rewards"], rollout_data["positions"])):
                step_row = {
                    "window_index": w.index,
                    "step": j,
                    "date": str(test_dates_arr[j].date()),
                    "reward": round(r_step, 6),
                    "position": p_step,
                    "memory_action": (
                        rollout_data["memory_actions"][j]
                        if rollout_data["memory_actions"] is not None else ""
                    ),
                }
                steps_writer.writerow(step_row)
            steps_f.flush()

            # ---- rich per-window record for summary.json ----
            window_record = {
                "window_index": w.index,
                "dates": {
                    "train_start": str(train_dates[0]), "train_end": str(train_dates[1]),
                    "test_start": str(test_dates[0]),   "test_end": str(test_dates[1]),
                },
                "sizes": {"train": tr, "test": te},
                "training": {
                    "total_timesteps": cfg["ppo"]["total_timesteps"],
                    "wall_seconds": round(train_wall, 2),
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
            per_window.append(window_record)

            # ---- console readout ----
            print(
                f"  agent: Sharpe {agent_metrics['sharpe_ratio']:+.3f}  "
                f"cumret {agent_metrics['cumulative_return']:+.4f}  "
                f"maxDD {agent_metrics['max_drawdown']:.4f}  "
                f"pos L/F/S/flips {agent_positions['n_long']}/{agent_positions['n_flat']}/"
                f"{agent_positions['n_short']}/{agent_positions['n_flips']}  "
                f"train {train_wall:.1f}s"
            )
            if agent_mem_counts is not None:
                print(f"  mem overwrite counts by slot: {agent_mem_counts}")
            print(
                f"  B&H:   Sharpe {bh_metrics['sharpe_ratio']:+.3f}  "
                f"cumret {bh_metrics['cumulative_return']:+.4f}  "
                f"maxDD {bh_metrics['max_drawdown']:.4f}"
            )

    finally:
        windows_f.close()
        steps_f.close()

    # ---- write summary.json atomically at the end ----
    # If resuming, load prior windows from CSV so summary.json is complete.
    all_windows = _load_prior_windows_for_summary(windows_csv_path) if start_index > 0 else per_window

    aggregate = _aggregate(all_windows, is_as, memory_k) if all_windows else None

    output = {
        "run_name": run_name,
        "config": cfg,
        "data_dates_full": {
            "start": str(dates[0].date()), "end": str(dates[-1].date()), "n": int(n),
        },
        "per_window": all_windows,
        "aggregate": aggregate,
    }
    summary_path = results_dir / "summary.json"
    tmp_path = summary_path.with_suffix(".json.tmp")
    with open(tmp_path, "w") as f:
        json.dump(output, f, indent=2, default=str)
    tmp_path.replace(summary_path)  # atomic on Windows + POSIX

    print(f"\n[{run_name}] wrote {windows_csv_path.name}, {steps_csv_path.name}, {summary_path.name}")
    if aggregate:
        print(
            f"[{run_name}] agent mean Sharpe: "
            f"{aggregate['agent']['sharpe_ratio']['mean']:+.3f}  "
            f"vs B&H {aggregate['buy_and_hold']['sharpe_ratio']['mean']:+.3f}"
        )


def _load_prior_windows_for_summary(windows_csv_path: Path) -> list[dict]:
    """After a resume, reconstruct minimal per-window records from windows.csv
    so summary.json reflects the whole run, not only the newly-completed windows."""
    records = []
    with open(windows_csv_path, "r", newline="") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append({
                "window_index": int(row["window_index"]),
                "dates": {
                    "train_start": row["train_start_date"], "train_end": row["train_end_date"],
                    "test_start": row["test_start_date"],   "test_end": row["test_end_date"],
                },
                "sizes": {"train": int(row["train_days"]), "test": int(row["test_days"])},
                "training": {
                    "total_timesteps": int(row["total_timesteps"]),
                    "wall_seconds": float(row["train_wall_seconds"]),
                },
                "agent_test": {
                    "n_steps": int(row["test_days"]),
                    "metrics": {
                        "cumulative_return": float(row["agent_cumret"]),
                        "sharpe_ratio": float(row["agent_sharpe"]),
                        "max_drawdown": float(row["agent_maxdd"]),
                    },
                    "positions": {
                        "n_long": int(row["pos_long"]), "n_flat": int(row["pos_flat"]),
                        "n_short": int(row["pos_short"]), "n_flips": int(row["pos_flips"]),
                    },
                },
                "buy_and_hold_test": {
                    "n_steps": int(row["test_days"]),
                    "metrics": {
                        "cumulative_return": float(row["bh_cumret"]),
                        "sharpe_ratio": float(row["bh_sharpe"]),
                        "max_drawdown": float(row["bh_maxdd"]),
                    },
                },
            })
    return records


def _aggregate(records: list[dict], is_as: bool, memory_k: int) -> dict:
    def agg(key: str, side: str) -> dict:
        vals = np.array([w[side]["metrics"][key] for w in records])
        return {
            "mean": float(vals.mean()),
            "std": float(vals.std(ddof=1) if vals.size > 1 else 0.0),
        }

    result = {
        "n_windows": len(records),
        "agent": {k: agg(k, "agent_test") for k in ("sharpe_ratio", "cumulative_return", "max_drawdown")},
        "buy_and_hold": {k: agg(k, "buy_and_hold_test") for k in ("sharpe_ratio", "cumulative_return", "max_drawdown")},
        "agent_positions_total": {
            "n_long": int(sum(w["agent_test"]["positions"]["n_long"] for w in records)),
            "n_flat": int(sum(w["agent_test"]["positions"]["n_flat"] for w in records)),
            "n_short": int(sum(w["agent_test"]["positions"]["n_short"] for w in records)),
            "n_flips": int(sum(w["agent_test"]["positions"]["n_flips"] for w in records)),
        },
    }
    if is_as:
        total_counts = [0] * memory_k
        for w in records:
            slot_counts = w["agent_test"].get("memory_action_counts")
            if slot_counts is not None:
                for i, c in enumerate(slot_counts):
                    total_counts[i] += c
        result["agent_memory_action_counts_total"] = total_counts
    return result


if __name__ == "__main__":
    main()
