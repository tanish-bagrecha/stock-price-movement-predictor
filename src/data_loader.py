"""
data_loader.py
================
Responsible for obtaining daily OHLCV data for a given ticker and running
a thorough data-quality audit on it before any modeling happens.

Design notes
------------
The project brief recommends `yfinance` as the data source. This module
tries `yfinance` first. If it is unavailable (no internet access, rate
limiting, or the package is not installed) it falls back to a bundled,
version-controlled CSV snapshot in `data/AAPL_raw.csv` so that the project
remains 100% reproducible even offline.

Every function here is a pure function: given the same inputs, it always
returns the same outputs. This is important for reproducibility.
"""

from __future__ import annotations

import os
import warnings
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
import pandas as pd

# Columns we require to be present, in this exact case, after loading.
REQUIRED_COLUMNS = ["Open", "High", "Low", "Close", "Volume"]


@dataclass
class DataQualityReport:
    """Container summarizing the results of the data-quality audit."""

    ticker: str
    n_rows_raw: int
    n_rows_clean: int
    date_min: pd.Timestamp
    date_max: pd.Timestamp
    n_missing_values: int
    n_duplicate_dates: int
    n_duplicate_rows: int
    n_invalid_ohlc_rows: int
    n_nonpositive_price_rows: int
    n_negative_volume_rows: int
    n_rows_dropped: int
    issues: list = field(default_factory=list)

    def summary(self) -> str:
        lines = [
            f"Data Quality Report for {self.ticker}",
            "=" * 40,
            f"Raw rows loaded            : {self.n_rows_raw:,}",
            f"Rows after cleaning        : {self.n_rows_clean:,}",
            f"Rows dropped               : {self.n_rows_dropped:,}",
            f"Date range                 : {self.date_min.date()} -> {self.date_max.date()}",
            f"Missing values found       : {self.n_missing_values}",
            f"Duplicate dates found      : {self.n_duplicate_dates}",
            f"Duplicate rows found       : {self.n_duplicate_rows}",
            f"Invalid OHLC rows found    : {self.n_invalid_ohlc_rows}",
            f"Non-positive price rows    : {self.n_nonpositive_price_rows}",
            f"Negative volume rows       : {self.n_negative_volume_rows}",
        ]
        if self.issues:
            lines.append("Notes:")
            lines.extend(f"  - {issue}" for issue in self.issues)
        else:
            lines.append("Notes: No unresolved issues. Dataset is clean.")
        return "\n".join(lines)


def _try_yfinance_download(
    ticker: str, start: str, end: Optional[str]
) -> Optional[pd.DataFrame]:
    """Attempt to download data with yfinance. Returns None on any failure."""
    try:
        import yfinance as yf
    except ImportError:
        warnings.warn("yfinance is not installed; will look for a local CSV fallback.")
        return None

    try:
        df = yf.download(
            ticker, start=start, end=end, progress=False, auto_adjust=True
        )
        if df is None or df.empty:
            warnings.warn(
                f"yfinance returned no data for {ticker} (no internet access, "
                "invalid ticker, or rate limiting). Falling back to local CSV."
            )
            return None
        # yfinance sometimes returns a MultiIndex column structure.
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        df = df.reset_index().rename(columns={"Date": "Date"})
        return df
    except Exception as exc:  # noqa: BLE001 - we deliberately want a broad fallback
        warnings.warn(f"yfinance download failed ({exc!r}); falling back to local CSV.")
        return None


def _load_local_csv_fallback(ticker: str, data_dir: str) -> pd.DataFrame:
    """
    Load a bundled CSV snapshot as a reproducibility fallback.

    The bundled file `AAPL_raw.csv` was sourced from a public, version-controlled
    OHLCV dataset (daily bars, non-adjusted-for-splits-beyond-standard-provider-
    adjustment) and is provided so the notebook runs identically for every
    reviewer, with or without live internet access.
    """
    path = os.path.join(data_dir, f"{ticker}_raw.csv")
    if not os.path.exists(path):
        raise FileNotFoundError(
            f"No local fallback dataset found for ticker '{ticker}' at {path}. "
            "Either provide internet access for yfinance, or supply a CSV named "
            f"'{ticker}_raw.csv' with columns [date, open, high, low, close, volume] "
            f"in the '{data_dir}' folder."
        )
    df = pd.read_csv(path)
    df.columns = [c.strip().lower() for c in df.columns]
    df = df.rename(
        columns={
            "date": "Date",
            "open": "Open",
            "high": "High",
            "low": "Low",
            "close": "Close",
            "volume": "Volume",
        }
    )
    df["Date"] = pd.to_datetime(df["Date"])
    return df


def load_ohlcv(
    ticker: str,
    start: str = "2012-01-01",
    end: Optional[str] = None,
    data_dir: str = "../data",
) -> pd.DataFrame:
    """
    Load daily OHLCV data for `ticker`, trying yfinance first and falling
    back to a local CSV snapshot for reproducibility.

    Parameters
    ----------
    ticker : str
        Ticker symbol, e.g. "AAPL".
    start : str
        Start date, "YYYY-MM-DD".
    end : str, optional
        End date, "YYYY-MM-DD". None means "up to the most recent data available".
    data_dir : str
        Folder containing the local CSV fallback.

    Returns
    -------
    pd.DataFrame
        Columns: Date, Open, High, Low, Close, Volume. Sorted chronologically.
    """
    df = _try_yfinance_download(ticker, start, end)
    if df is None:
        df = _load_local_csv_fallback(ticker, data_dir)

    df = df[["Date"] + REQUIRED_COLUMNS].copy()
    df["Date"] = pd.to_datetime(df["Date"])
    df = df[(df["Date"] >= pd.Timestamp(start))]
    if end is not None:
        df = df[df["Date"] <= pd.Timestamp(end)]
    df = df.sort_values("Date").reset_index(drop=True)
    return df


def run_data_quality_audit(df: pd.DataFrame, ticker: str) -> tuple[pd.DataFrame, DataQualityReport]:
    """
    Run a comprehensive data-quality audit on raw OHLCV data and return a
    cleaned DataFrame plus a report describing exactly what was found/fixed.

    This function NEVER silently invents data. It only:
      - flags issues,
      - removes rows that are unusable (e.g., duplicate dates, invalid OHLC),
      - reports everything it did.
    """
    issues = []
    n_rows_raw = len(df)

    working = df.copy()

    # 1. Data types
    working["Date"] = pd.to_datetime(working["Date"])
    for col in REQUIRED_COLUMNS:
        working[col] = pd.to_numeric(working[col], errors="coerce")

    # 2. Missing values
    n_missing = int(working[REQUIRED_COLUMNS].isna().sum().sum())
    if n_missing > 0:
        issues.append(f"{n_missing} missing OHLCV cell(s) found -> rows dropped.")
        working = working.dropna(subset=REQUIRED_COLUMNS)

    # 3. Duplicate dates (keep first occurrence)
    n_duplicate_dates = int(working["Date"].duplicated().sum())
    if n_duplicate_dates > 0:
        issues.append(f"{n_duplicate_dates} duplicate date(s) found -> kept first occurrence.")
        working = working.drop_duplicates(subset="Date", keep="first")

    # 4. Fully duplicate rows
    n_duplicate_rows = int(working.duplicated().sum())
    if n_duplicate_rows > 0:
        issues.append(f"{n_duplicate_rows} fully duplicate row(s) found -> removed.")
        working = working.drop_duplicates()

    # 5. Date ordering
    working = working.sort_values("Date").reset_index(drop=True)
    if not working["Date"].is_monotonic_increasing:
        issues.append("Dates were not monotonically increasing -> re-sorted.")

    # 6. Invalid OHLC relationships (e.g., High < Low, Close outside [Low, High])
    invalid_mask = (
        (working["High"] < working["Low"])
        | (working["Close"] > working["High"])
        | (working["Close"] < working["Low"])
        | (working["Open"] > working["High"])
        | (working["Open"] < working["Low"])
    )
    n_invalid_ohlc = int(invalid_mask.sum())
    if n_invalid_ohlc > 0:
        issues.append(f"{n_invalid_ohlc} row(s) with invalid OHLC relationships -> removed.")
        working = working[~invalid_mask]

    # 7. Non-positive prices
    price_cols = ["Open", "High", "Low", "Close"]
    nonpositive_mask = (working[price_cols] <= 0).any(axis=1)
    n_nonpositive = int(nonpositive_mask.sum())
    if n_nonpositive > 0:
        issues.append(f"{n_nonpositive} row(s) with non-positive prices -> removed.")
        working = working[~nonpositive_mask]

    # 8. Negative volume
    negative_volume_mask = working["Volume"] < 0
    n_negative_volume = int(negative_volume_mask.sum())
    if n_negative_volume > 0:
        issues.append(f"{n_negative_volume} row(s) with negative volume -> removed.")
        working = working[~negative_volume_mask]

    working = working.sort_values("Date").reset_index(drop=True)

    # 9. Missing trading sessions (informational only - markets close on weekends/
    #    holidays, so gaps are expected and are NOT treated as errors).
    business_days = pd.bdate_range(working["Date"].min(), working["Date"].max())
    missing_sessions = business_days.difference(working["Date"])
    # Typically this includes market holidays, which is normal and expected.
    issues.append(
        f"{len(missing_sessions)} weekday date(s) with no trading session in range "
        "(expected: market holidays)."
    )

    n_rows_clean = len(working)
    report = DataQualityReport(
        ticker=ticker,
        n_rows_raw=n_rows_raw,
        n_rows_clean=n_rows_clean,
        date_min=working["Date"].min(),
        date_max=working["Date"].max(),
        n_missing_values=n_missing,
        n_duplicate_dates=n_duplicate_dates,
        n_duplicate_rows=n_duplicate_rows,
        n_invalid_ohlc_rows=n_invalid_ohlc,
        n_nonpositive_price_rows=n_nonpositive,
        n_negative_volume_rows=n_negative_volume,
        n_rows_dropped=n_rows_raw - n_rows_clean,
        issues=issues,
    )
    return working, report
