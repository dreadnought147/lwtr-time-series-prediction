import numpy as np
import gymnasium as gym
from gymnasium import spaces

class TimeSeries(gym.Env):
    """Data aganostic
       Args:
            observation:
            returns: 
            transaction_cost: 
    
        Action space: 
        Reward:
        Episode ends: 
    """

    metadata = {"render_modes": []}

    def __init__(self, observations: np.ndarray, returns: np.ndarray, transaction_cost: float = 0.0):
        super().__init__()
        assert len(observations) == len(returns), "observations and returns must align"
        self.observations =observations.astype(np.float32)
        self.returns = returns.astype(np.float32)
        self.transaction_cost = float(transaction_cost)

        # --- Declare the two Gymnasium "spaces" ---
        # observation_space: what the agent sees at each step. Shape (1,) = one float.
        # Bounds are -inf..inf because log-returns can be any real number.
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(1,), dtype=np.float32)
        # action_space: three discrete options mapped later to {-1, 0, +1}.
        self.action_space = spaces.Discrete(3)      

        # the time and position o f the agent
        self.t = 0 #agent indexes into data arrays using t
        self.position = 0        #transaction cost calcutation will need what [ostion the ganet took]


 
    def reset(self, seed=None, options=None):
        # Reset the parent env's RNG (Gymnasium ≥0.29 requires this line).
        super().reset(seed=seed)
        
        # Wind everything back to the start.
        self.t = 0
        self.position = 0
        
        # Return the first observation + empty info dict.
        # Gymnasium's reset() contract: (obs, info).
        return self._obs(), {}


    def step(self, action: int):
        # --- Map discrete action {0,1,2} sell hold buy to signed position {-1,0,+1} ---
        new_position = int(action) - 1

        # --- Reward = trading PnL - transaction cost ---
        # Trading PnL: hold `new_position` for one day, earn `returns[t]` per unit position.
        # Transaction cost: charged proportional to how much the position changed.
        # abs(new_position - self.position) is 0 if same, 1 or 2 if changed.
        reward = (
            new_position * self.returns[self.t]
            - self.transaction_cost * abs(new_position - self.position)
        )

        self.position = new_position

        self.t = self.t  +1 

                # --- Terminal condition ---
        # We ran out of data — no more days to trade. Not a failure, just end of stream.
        terminated = self.t >= len(self.observations) 
        # We never let a wrapper cut the episode short, so truncated is always False.
        truncated = False
        # --- Build the next observation ---
        # If terminated, self.t is out of bounds. Return zeros as a placeholder
        # (agent won't act on it — the episode is over).
        obs = self._obs() if not terminated else np.zeros(1, dtype=np.float32)
        
        # --- Info dict for post-hoc logging / evaluation ---
        # step - 1 because we already incremented t; this is the step just executed.
        info = {"position": new_position, "step": self.t - 1}

        # Gymnasium's step() contract: (obs, reward, terminated, truncated, info).
        return obs, float(reward), terminated, truncated, info

    def _obs(self) -> np.ndarray:
        # Wrap the scalar observation as a shape-(1,) float32 array to match observation_space.
        # Leading underscore = private helper; not part of the public env interface.
        return np.array([self.observations[self.t]], dtype=np.float32)


if __name__ == "__main__":
    # smoke test with fake data
    N = 1000
    obs = np.random.randn(N).astype(np.float32)
    rets = np.random.randn(N).astype(np.float32) * 0.01 

    env = TimeSeries(obs, rets, transaction_cost=0.001)

    o, _ = env.reset()
    print(f"initial obs: {o}, shape: {o.shape}")

    total_reward = 0.0
    done = False
    while not done:
        a = env.action_space.sample()   # random action
        o, r, terminated, truncated, info = env.step(a)
        total_reward += r
        done = terminated or truncated
    print(f"episode done at t={env.t}, total reward: {total_reward:.4f}")
