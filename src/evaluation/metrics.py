"""
Financial evaluation metrics.

WHAT'S HERE
-----------
All functions operate on a 1-D array of per-step rewards (or per-step returns —
in our setup they coincide once transaction cost is separated, so callers can
pass whichever they have).

METRICS
-------
- cumulative_return : sum of per-step returns. Simplest scalar.
- sharpe_ratio      : annualised risk-adjusted return. Industry-standard.
- max_drawdown      : largest peak-to-trough loss over the equity curve.
- summary           : returns all three plus n_steps in one dict, for reporting.

CONVENTIONS AND ASSUMPTIONS
---------------------------
- Trading days per year: 252 (US convention). Configurable per call.
- Sharpe uses zero risk-free rate. Standard for research; add it later if the
  supervisor asks.
- Returns are log-returns throughout the project. On log-returns:
    * cumulative_return = log(P_end / P_start), the total log-growth.
    * Sharpe has a very small bias vs simple returns at daily frequency, but
      the ranking of configs is unaffected — which is all we need for the
      RQ1 / RQ2 comparisons.
- Max drawdown is returned as a POSITIVE number in [0, 1]:
    0.15  means "worst drawdown was a 15% loss from the running peak."
"""

from __future__ import annotations
import numpy as np


TRADING_DAYS_PER_YEAR = 252


def cumulative_return(returns: np.ndarray) -> float:
    """Sum of per-step returns.

    On log-returns this equals log(P_end / P_start) — total log-growth.
    """
    return float(np.sum(returns))


def sharpe_ratio(
    returns: np.ndarray,
    periods_per_year: int = TRADING_DAYS_PER_YEAR,
) -> float:
    """Annualised Sharpe ratio (zero risk-free rate).

    Formula:  sqrt(periods_per_year) * mean(returns) / std(returns)

    Returns 0.0 if:
      - the array is empty, or
      - std(returns) is 0 (no volatility — degenerate constant sequence).

    Uses sample std (ddof=1) for an unbiased estimator when possible.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    if r.size == 1:
        return 0.0  # std undefined; a single-day Sharpe is not meaningful.
    std = float(r.std(ddof=1))
    if std == 0.0:
        return 0.0
    return float(np.sqrt(periods_per_year) * r.mean() / std)


def max_drawdown(returns: np.ndarray) -> float:
    """Largest peak-to-trough decline in cumulative equity.

    On log-returns:
      1. Cumulative sum → running total log-growth.
      2. exp(...) → equity curve (starts at ~1.0).
      3. Running max → the ledger of "all-time highs" up to each timestep.
      4. Drawdown series = (peak - current) / peak, per timestep.
      5. Max drawdown = the deepest drawdown ever reached.

    Returns a number in [0, 1]. Higher = worse. 0.0 = never made a loss from
    any prior peak.
    """
    r = np.asarray(returns, dtype=np.float64)
    if r.size == 0:
        return 0.0
    equity = np.exp(np.cumsum(r))          # equity curve
    peaks = np.maximum.accumulate(equity)  # running max
    drawdowns = (peaks - equity) / peaks   # fractional drop from most recent peak
    return float(drawdowns.max())


def summary(returns: np.ndarray) -> dict:
    """One call, all three primary metrics plus n_steps.

    Shape of returned dict matches what scripts/train.py expects downstream.
    """
    return {
        "n_steps": int(np.asarray(returns).size),
        "cumulative_return": cumulative_return(returns),
        "sharpe_ratio": sharpe_ratio(returns),
        "max_drawdown": max_drawdown(returns),
    }


if __name__ == "__main__":
    # Sanity checks with synthetic data.
    rng = np.random.default_rng(0)

    # Case 1: noisy daily returns with a small positive drift.
    fake = rng.normal(loc=0.0005, scale=0.01, size=1000).astype(np.float32)
    print("Case 1 — 1000 noisy days, small positive drift:")
    for k, v in summary(fake).items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Case 2: constant 4 bps per day — infinite Sharpe (std=0), zero drawdown.
    steady = np.full(1000, 0.0004, dtype=np.float32)
    print("\nCase 2 — constant 4 bps/day (edge case: std=0):")
    for k, v in summary(steady).items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Case 3: all zeros (no trading) — everything should be 0.
    idle = np.zeros(1000, dtype=np.float32)
    print("\nCase 3 — all zeros (agent never traded):")
    for k, v in summary(idle).items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")

    # Case 4: single big crash mid-series — max drawdown should be substantial.
    crash = np.concatenate([
        np.full(500, 0.001, dtype=np.float32),
        np.full(50, -0.02, dtype=np.float32),   # 50-day crash, ~65% loss cumulatively
        np.full(450, 0.001, dtype=np.float32),
    ])
    print("\nCase 4 — steady drift with a mid-series crash:")
    for k, v in summary(crash).items():
        print(f"  {k}: {v:.6f}" if isinstance(v, float) else f"  {k}: {v}")
