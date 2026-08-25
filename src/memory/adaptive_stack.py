"""
Adaptive Stacking wrapper for time-series RL environments.

Ported and adapted from the reference implementation by Nangue Tasse et al. (2025):
    https://github.com/geraudnt/adaptive-stacking

WHAT ADAPTIVE STACKING DOES
---------------------------
Frame Stacking (fixed): shows the agent the last k observations, dropping the
oldest as new ones arrive. The agent has no say in what stays.

Adaptive Stacking (learned): also shows the agent k observations at a time,
but the agent CHOOSES WHICH SLOT TO OVERWRITE each step. Memory management
becomes part of the policy, learned via the same reward signal as trading.

ACTION SPACE
------------
Base env:  Discrete(3)  →  trading action in {short, flat, long}

With AS wrapper (multi-head):
    MultiDiscrete([3, k])
        - index 0: trading action (same 3 options as base env)
        - index 1: memory slot 0..k-1 to overwrite

SB3's MlpPolicy handles MultiDiscrete natively — no algorithm changes needed.

OBSERVATION SPACE
-----------------
Base env: Box(shape=(1,))    →  current log-return
With AS:  Box(shape=(k,))    →  flat vector of the k stored observations

We flatten to (k,) — same shape as our FrameStack produces — so PPO's network
input is identical across the two wrappers. That is what makes the RQ2
comparison honest at the input layer.

COLD-START POLICY
-----------------
On reset() we fill all k slots with the initial observation, matching our
FrameStack. The reference implementation uses zero-fill for the first k-1
slots; we deviate here so FrameStack and AdaptiveStack are cold-start-identical
(any RQ2 difference should come from the memory mechanism, not from a
different burn-in). Zero-fill would inject artificial extreme values into
log-return space that the network sees nowhere else.

INFO DICT
---------
Each step returns:
    info["pre_frames"]     : list of k arrays, the stack BEFORE this step's
                             memory action was applied. Feeds the retention
                             diagnostic ("which slots does the agent overwrite
                             over time?").
    info["memory_action"]  : the slot the agent chose to overwrite.
    info["trading_action"] : the trading action taken this step.
"""

from __future__ import annotations
import numpy as np
import gymnasium as gym
from gymnasium import spaces


class AdaptiveStack(gym.Wrapper):
    """Adaptive Stacking wrapper.

    Args:
        env:        Any Gymnasium environment with a 1-D observation space and
                    a Discrete action space. TimeSeriesEnv satisfies both.
        num_stack:  k — memory budget (number of slots in the stack).

    Two contract changes are visible to whoever uses this wrapper:
      1. observation_space becomes Box(shape=(k * obs_dim,))
      2. action_space becomes MultiDiscrete([env.action_space.n, k])
    """

    def __init__(self, env: gym.Env, num_stack: int):
        super().__init__(env)

        # --- Sanity guards on the underlying env ---
        # We only support Discrete trading actions (matches TimeSeriesEnv).
        # A MultiDiscrete base would need extra plumbing; not needed for RQ2.
        assert isinstance(env.action_space, spaces.Discrete), (
            f"AdaptiveStack requires a Discrete action space; "
            f"got {type(env.action_space).__name__}."
        )
        # 1-D observation only. Multi-feature would need reshape logic.
        base_obs_shape = env.observation_space.shape
        assert len(base_obs_shape) == 1, (
            f"AdaptiveStack requires a 1-D observation; got shape {base_obs_shape}."
        )

        self.num_stack = num_stack
        self.obs_dim = base_obs_shape[0]  # 1 for TimeSeriesEnv

        # --- Observation space: k slots, flattened for PPO's MLP input ---
        # Bounds tiled from the base env so the Box declaration is honest.
        # Convention: rightmost element of the flat vector is the most recent
        # observation (matches FrameStack).
        low = np.tile(env.observation_space.low, num_stack)
        high = np.tile(env.observation_space.high, num_stack)
        self.observation_space = spaces.Box(
            low=low,
            high=high,
            shape=(num_stack * self.obs_dim,),
            dtype=env.observation_space.dtype,
        )

        # --- Action space: [trading_action, memory_slot_to_overwrite] ---
        # PPO learns both heads simultaneously via the same reward signal.
        self.action_space = spaces.MultiDiscrete(
            [env.action_space.n, num_stack]
        )

        # Internal state — plain list, NOT deque.
        # Reason: we need index-based pop() (self.buffer.pop(memory_action)).
        # deque only supports popleft/pop from the ends. Length is enforced
        # by our own logic (we pop one and append one on each step, so length
        # stays fixed at num_stack).
        self.buffer: list[np.ndarray] = []

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Cold-start: fill every slot with the initial observation.
        # See module docstring for the rationale (matches FrameStack cold-start).
        self.buffer = [obs.copy() for _ in range(self.num_stack)]
        return self._stacked(), info

    def step(self, action):
        # `action` arrives as a length-2 array from PPO's MultiDiscrete sampling.
        # Safety wrap: some callers pass tuples/lists/numpy scalars.
        action = np.asarray(action).flatten()
        trading_action = int(action[0])
        memory_action = int(action[1])

        # Snapshot the stack BEFORE we mutate it — used by the retention
        # diagnostic to answer "what did the agent see when it decided
        # which slot to overwrite?"
        pre_frames = [frame.copy() for frame in self.buffer]

        # Advance the underlying env using ONLY the trading action.
        # The wrapped env has no knowledge that a memory action exists.
        new_obs, reward, terminated, truncated, env_info = self.env.step(trading_action)

        # Apply the memory action:
        #   1. Remove the slot the agent chose (memory_action ∈ [0, num_stack))
        #   2. Append the newly observed observation on the right
        # The buffer's length stays at num_stack because we removed and added one.
        self.buffer.pop(memory_action)
        self.buffer.append(new_obs.copy())

        # Enrich info dict with our own diagnostics on top of the env's info.
        info = dict(env_info)
        info["pre_frames"] = pre_frames
        info["memory_action"] = memory_action
        info["trading_action"] = trading_action

        # Build the observation returned to PPO.
        # On terminal steps we return a zero vector as a filler — the episode
        # is over, nothing will consume this observation, but Gymnasium's
        # contract still expects a valid-shaped array.
        if terminated:
            obs = np.zeros(
                self.num_stack * self.obs_dim,
                dtype=self.observation_space.dtype,
            )
        else:
            obs = self._stacked()

        return obs, reward, terminated, truncated, info

    def _stacked(self) -> np.ndarray:
        """Concatenate the k slots into one flat observation vector.

        Layout (for obs_dim=1):
            [slot_0, slot_1, ..., slot_{k-1}]

        For obs_dim>1, each slot contributes obs_dim consecutive elements.
        Rightmost slot is always the most recently written observation.
        """
        return np.concatenate(self.buffer, axis=0).astype(
            self.observation_space.dtype
        )


if __name__ == "__main__":
    # Smoke test on synthetic data — proves the Gymnasium contract holds.
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.envs.timeseries_env import TimeSeriesEnv

    N = 30
    rng = np.random.default_rng(0)
    obs_arr = rng.standard_normal(N).astype(np.float32)
    ret_arr = rng.standard_normal(N).astype(np.float32) * 0.01

    base = TimeSeriesEnv(obs_arr, ret_arr, transaction_cost=0.0001)
    env = AdaptiveStack(base, num_stack=5)

    print(f"observation_space: {env.observation_space}")
    print(f"action_space:      {env.action_space}\n")

    o, _ = env.reset()
    print(f"reset obs: {np.round(o, 4)}\n")

    for step_i in range(10):
        a = env.action_space.sample()
        o, r, term, trunc, info = env.step(a)
        print(
            f"step {step_i:2d}  "
            f"trading={info['trading_action']} "
            f"memory={info['memory_action']}  "
            f"obs={np.round(o, 4)}  "
            f"reward={r:+.4f}"
        )
        if term or trunc:
            print("episode ended")
            break