import yfinance as yf
from pathlib import Path

RAW_DATA = Path(__file__).resolve().parents[1] / "data" / "raw"
RAW_DATA.mkdir(parents=True, exist_ok=True)

vix = yf.download("^VIX", start="1990-01-01", end="2026-07-01", auto_adjust=False)
sp = yf.download("^GSPC", start="1990-01-01", end="2026-07-01", auto_adjust=False)\

vix.to_csv(RAW_DATA / "vix.csv")
sp.to_csv(RAW_DATA / "sp500.csv")

print(f"saved {len(vix)} VIX rows, {len(sp)} S&P rows to {RAW_DATA}")
