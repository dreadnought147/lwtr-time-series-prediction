import numpy as np
import gymnasium as gym
from gymnasium import spaces
from collections import deque


class FrameStack(gym.Wrapper):
    """
    Wraps an env so the agent sees the last k observations concatenated.

    - observation_space changes from (obs_dim,) to (k * obs_dim,).
    - FIFO buffer: oldest observation drops off when a new one arrives.
    - Agent has no control over what's kept; recency-only eviction.
    """
    def __init__(self, env: gym.Env, k: int):
        super().__init__(env)
        self.k = k #size of our frame_stck

        # Original obs shape (e.g. (1,) for our TimeSeriesEnv)
        base_shape = env.observation_space.shape
        assert len(base_shape) == 1, "FrameStack expects a 1-D observation space"
        self.obs_dim = base_shape[0]

        # New observation_space: k stacked observations, flattened
        low  = np.tile(env.observation_space.low,  k)
        high = np.tile(env.observation_space.high, k)
        self.observation_space = spaces.Box(
            low=low, high=high, shape=(k * self.obs_dim,), dtype=env.observation_space.dtype
        )

        # FIFO buffer for the last k observations
        self.buffer = deque(maxlen=k)

        # Rest the episode, we tart our with a astck of initial observatios [init_obs, init_obs, init_obs ...]
    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        # Fill the buffer with k copies of the initial observation.
        # Avoids the cold-start problem where early stack slots would be zeros.
        for _ in range(self.k):
            self.buffer.append(obs)
        return self._stacked(), info

    
    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        self.buffer.append(obs)
        return self._stacked(), reward, terminated, truncated, info

    def _stacked(self) -> np.ndarray:
        # Concatenate the k observations into a single flat array.
        # Rightmost = most recent (deque appends to the right).
        return np.concatenate(list(self.buffer), axis=0).astype(np.float32)
    
if __name__ == "__main__":
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.envs.timeseries_env import TimeSeries

    N = 100
    obs = np.random.randn(N).astype(np.float32)
    rets = np.random.randn(N).astype(np.float32) * 0.01
    base = TimeSeries(obs, rets)
    env = FrameStack(base, k=5)

    print(f"observation_space: {env.observation_space}")
    o, _ = env.reset()
    print(f"reset obs shape: {o.shape}, first 5: {o[:5]}")

    for _ in range(10):
        o, r, term, trunc, info = env.step(env.action_space.sample())
        print(f"step obs: {o}, reward: {r:.4f}")
#could use framestack observation by Gym warpa but we want to see internal mechanmisn



