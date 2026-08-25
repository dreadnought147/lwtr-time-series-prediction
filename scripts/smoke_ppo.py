# Smoke test to prove that the Timeseries env handshakes with SB3(lib that ships standard PPO and other RL algorithms )"
#  not a real training run jusu enough to see if our Env handshakes and contracts matches the SB3 lib

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import numpy as np
from stable_baselines3 import PPO
from stable_baselines3.common.env_checker import check_env

from src.data.loader import load_aligned
from src.data.transforms import to_env_arrays
from src.envs.timeseries_env import TimeSeries

# loading our data

df = load_aligned()
obs, rets = to_env_arrays(df, observation_col="vix_close", asset_col="sp_close")
print(f"data: {len(obs), len(rets)} steps ready")



#Build tyhe env and make sure the Env we are giving is handshaking with 
env = TimeSeries(obs, rets, transaction_cost=0.0001)
check_env(env, warn=True)
print("env passes SB3's Gymnasium contract check")

# laoding the PPO algorithm and Train our env via ppo bereifly 
model = PPO("MlpPolicy", env, verbose=1, n_steps=256, batch_size=64) 
model.learn(total_timesteps=2000)

#Sanity check , run one episode so all steps with teh trianed policy 

obs_reset, _ = env.reset()
total_reward = 0.0
done = False
while not done:
    action, _ = model.predict(obs_reset, deterministic=True)
    obs_reset, reward, terminated, truncated, _ = env.step(int(action))
    total_reward += reward
    done = terminated or truncated
    #get clratiy on term and trucn
print(f"smoke test done: total reward over one episode = {total_reward:.4f}")
