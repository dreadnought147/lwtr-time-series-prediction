import numpy as np
import pandas as pd
from pathlib import Path

def to_env_arrays(
    df: pd.DataFrame,
    observation_col: str,
    asset_col: str,
) -> tuple[np.ndarray, np.ndarray, pd.DatetimeIndex]:
    """
    Turn an aligned price DataFrame into (observations, returns, dates) for TimeSeriesEnv.

    Args:
        df: aligned DataFrame from loader.load_aligned(). Date-indexed;
            must contain the two named columns as price series.
        observation_col: column whose log-returns the agent sees each step.
            Example: "vix_close".
        asset_col: column whose next-day log-return produces the reward.
            Example: "sp_close".

    Returns:
        observations: 1-D float32 array of log-returns of observation_col.
        returns:      1-D float32 array of next-day log-returns of asset_col,
                      index-aligned 1:1 with observations.
        dates:        DatetimeIndex aligned 1:1 with observations. Same length.
                      # dates[i] is the trading day at which obs[i] is observed
                      # and rets[i] is the return realised over the following day.
    """

      # Log-returns of the observation stream (what the agent SEES at each step)
    obs_returns = np.log(df[observation_col] / df[observation_col].shift(1))

        # Log-returns of the asset (what the agent TRADES on)
    asset_returns = np.log(df[asset_col] / df[asset_col].shift(1))


        # shift(-1) so that at index t we hold the NEXT DAY's return.
    # Reward at day t uses the return from day t to day t+1.
    next_day_asset_returns = asset_returns.shift(-1)

    # Drop rows where either series is NaN (first row of obs_returns, last of next_day_returns)
    combined = pd.concat([obs_returns, next_day_asset_returns], axis=1).dropna()

    obs  = combined.iloc[:, 0].to_numpy(dtype=np.float32)
    rets = combined.iloc[:, 1].to_numpy(dtype=np.float32)
    dates = combined.index  # DatetimeIndex, aligned 1:1 with obs and rets
    return obs, rets, dates


if __name__ == "__main__":
    # sys.path hack so this file runs directly (python src/data/transforms.py).
    # For the proper way, prefer: python -m src.data.transforms  (from project root).
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from src.data.loader import load_aligned

    df = load_aligned()
    obs, rets, dates = to_env_arrays(df, observation_col="vix_close", asset_col="sp_close")

    print(f"obs   -> shape: {obs.shape}, dtype: {obs.dtype}")
    print(f"rets  -> shape: {rets.shape}, dtype: {rets.dtype}")
    print(f"dates -> {dates[0].date()} .. {dates[-1].date()} (len {len(dates)})")
    print(f"first 5 obs:   {obs[:5]}")
    print(f"first 5 rets:  {rets[:5]}")
    print(f"any NaN in obs?  {np.isnan(obs).any()}")
    print(f"any NaN in rets? {np.isnan(rets).any()}")
