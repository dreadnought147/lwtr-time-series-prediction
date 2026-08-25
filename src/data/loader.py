import pandas as pd
from pathlib import Path

RAW = Path(__file__).resolve().parents[2] / "data" / "raw"

COLS = ["date", "adj_close", "close", "high", "low", "open", "volume"]

def _read_ticker(filename: str) -> pd.DataFrame:
    """Read one yfinance CSV, handle its 3-row header, return a clean DataFrame."""
    df = pd.read_csv(RAW / filename, skiprows=3, header=None, names=COLS,
                     parse_dates=["date"], index_col="date")
    return df

def load_aligned(vix_file: str = "vix.csv", sp_file: str = "sp500.csv") -> pd.DataFrame:
    """Load VIX + S&P, align on trading dates, return one DataFrame."""
    vix = _read_ticker(vix_file)[["close"]].rename(columns={"close": "vix_close"})
    sp  = _read_ticker(sp_file)[["close"]].rename(columns={"close": "sp_close"})
    # inner join on sp and vix
    return vix.join(sp, how="inner")   



if __name__ == "__main__":
    df = load_aligned()
    print(df.head())
    print(f"rows: {len(df)}")

