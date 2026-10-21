"""
Data integrity audit — re-fetch VIX + S&P from Yahoo Finance and compare
against the local CSVs. Writes a text report into plots/data_integrity/.

Why this exists:
  Raw data is the foundation of every result in this project. If Yahoo
  changes historical closes (dividend adjustments, data corrections),
  our training-time CSVs could silently drift from reality. This script
  re-downloads the same tickers and tells you where (and by how much)
  the live Yahoo data differs from what's on disk.

Run:
    python scripts/data_integrity_check.py

Output:
    plots/data_integrity/integrity_report.txt  — summary + any mismatches
"""

from __future__ import annotations
from pathlib import Path
import sys
import numpy as np
import pandas as pd
import yfinance as yf

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
RAW = ROOT / "data" / "raw"
OUT = ROOT / "plots" / "data_integrity"
OUT.mkdir(parents=True, exist_ok=True)


def _load_local(name: str) -> pd.DataFrame:
    """Read a locally saved yfinance CSV using the project's loader convention."""
    cols = ["date", "adj_close", "close", "high", "low", "open", "volume"]
    df = pd.read_csv(
        RAW / name, skiprows=3, header=None, names=cols,
        parse_dates=["date"], index_col="date",
    )
    return df


def _fetch_fresh(ticker: str) -> pd.DataFrame:
    """Download fresh data from Yahoo and shape it to match local-CSV columns."""
    df = yf.download(ticker, start="2000-01-01", progress=False, auto_adjust=False)
    df.columns = [c.lower() if isinstance(c, str) else c[0].lower() for c in df.columns]
    df = df.rename(columns={"adj close": "adj_close"})
    df.index.name = "date"
    return df


def _compare(local: pd.DataFrame, fresh: pd.DataFrame, label: str) -> list[str]:
    """Return a human-readable diff report between local and fresh close series."""
    lines = [f"\n=== {label} ===",
             f"  local rows   : {len(local)}  ({local.index.min().date()} .. {local.index.max().date()})",
             f"  fresh rows   : {len(fresh)}  ({fresh.index.min().date()} .. {fresh.index.max().date()})"]

    common = local.index.intersection(fresh.index)
    only_local = local.index.difference(fresh.index)
    only_fresh = fresh.index.difference(local.index)

    lines.append(f"  shared dates : {len(common)}")
    lines.append(f"  only local   : {len(only_local)}  (first few: {list(only_local[:3].date) if len(only_local) else []})")
    lines.append(f"  only fresh   : {len(only_fresh)}  (first few: {list(only_fresh[:3].date) if len(only_fresh) else []})")

    # Close-price diffs on shared dates
    if len(common):
        l = local.loc[common, "close"].astype(float)
        f = fresh.loc[common, "close"].astype(float)
        diff = (l - f).abs()
        pct = diff / f.abs().replace(0, np.nan)
        lines.append(f"  close abs-diff summary (shared dates):")
        lines.append(f"    max         : {diff.max():.6f}")
        lines.append(f"    mean        : {diff.mean():.6f}")
        lines.append(f"    median      : {diff.median():.6f}")
        lines.append(f"    % diff max  : {(pct.max() * 100):.4f}%")
        lines.append(f"    % diff mean : {(pct.mean() * 100):.4f}%")

        # Days that differ by more than 0.1%
        big_diffs = pct[pct > 0.001].sort_values(ascending=False)
        if len(big_diffs):
            lines.append(f"  dates with >0.1% close diff: {len(big_diffs)}")
            for d, p in big_diffs.head(5).items():
                lines.append(f"    {d.date()}  local={l.loc[d]:.4f}  fresh={f.loc[d]:.4f}  diff={p*100:.3f}%")
        else:
            lines.append("  no shared date differs by more than 0.1% — data is stable.")
    return lines


def main():
    print("Re-fetching VIX and S&P 500 from Yahoo (this needs internet)...")
    try:
        fresh_vix = _fetch_fresh("^VIX")
        fresh_sp = _fetch_fresh("^GSPC")
    except Exception as e:
        msg = f"ERROR: Yahoo fetch failed: {e}"
        print(msg)
        (OUT / "integrity_report.txt").write_text(msg, encoding="utf-8")
        return

    local_vix = _load_local("vix.csv")
    local_sp = _load_local("sp500.csv")

    report = ["DATA INTEGRITY REPORT — Yahoo refetch vs local CSV",
              "=" * 70,
              f"Run date: {pd.Timestamp.utcnow().strftime('%Y-%m-%d %H:%M UTC')}"]
    report += _compare(local_vix, fresh_vix, "VIX (^VIX)")
    report += _compare(local_sp, fresh_sp, "S&P 500 (^GSPC)")

    report += [
        "",
        "=" * 70,
        "INTERPRETATION GUIDE",
        "- Yahoo occasionally corrects historical closes (dividend adjustments,",
        "  recorded errors). Small percentage differences (<0.1%) are normal.",
        "- Large percentage differences on specific dates indicate the local",
        "  CSV is stale and should be re-downloaded (run scripts/downlaod_data.py).",
        "- Extra dates in 'fresh' (not in local) = Yahoo has data for days after",
        "  the last local download. Re-download to catch up.",
        "- Extra dates in 'local' (not in fresh) = Yahoo may have removed days",
        "  (very rare). Worth investigating.",
    ]

    out_path = OUT / "integrity_report.txt"
    out_path.write_text("\n".join(report), encoding="utf-8")
    print(f"wrote {out_path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
