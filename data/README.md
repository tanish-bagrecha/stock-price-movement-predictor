# Data Folder

This project uses **daily OHLCV (Open, High, Low, Close, Volume)** data.

## How data is obtained

`src/data_loader.py` tries, in order:

1. **Live download via `yfinance`** for the configured `TICKER`, `START_DATE`, and `END_DATE`
   (set in the notebook's Configuration cell).
2. **Bundled local CSV fallback** — `AAPL_raw.csv` in this folder — used automatically if
   `yfinance` is unavailable (no internet access, package not installed, or the request is
   rate-limited/empty).

This guarantees the notebook is reproducible for every reviewer, regardless of network
conditions, while still following the project's preferred live-data-source approach whenever
possible.

## Regenerating / changing the data

- To use a **different ticker with live data**, just change `TICKER` in the notebook's
  configuration cell — no code changes needed, as long as you have internet access and
  `yfinance` installed.
- To add a **local fallback CSV for a different ticker**, place a file named
  `{TICKER}_raw.csv` in this folder with columns: `date, open, high, low, close, volume`
  (lowercase headers), one row per trading day, sorted or unsorted (the loader sorts it).

## About `AAPL_raw.csv`

- **Ticker**: AAPL (Apple Inc.)
- **Frequency**: Daily bars
- **Range bundled**: 2012-01-03 onward
- **Columns**: `date, open, high, low, close, volume`
- **Source**: A publicly available, version-controlled historical OHLCV dataset (originally
  compiled similarly to standard market-data providers). Provided here purely as a
  reproducibility fallback — for live/current data, prefer the `yfinance` path.

This file is intentionally kept small enough to commit to version control.
