"""
features.py
============
Everything related to:
  1. Leak-free target construction
  2. The RAW feature set (plain OHLCV-derived transformations)
  3. The ENGINEERED feature set (technical indicators)

GOLDEN RULE ENFORCED THROUGHOUT THIS FILE
------------------------------------------
Every feature for row `t` must be computable using ONLY information available
at the close of day `t` (or earlier). We never use `.shift(-k)` for a FEATURE.
The only place `.shift(-1)` is allowed in the entire project is in the target
construction, because the target is explicitly a look-ahead LABEL, not a
model input.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

RAW_FEATURE_COLUMNS = [
    "Open",
    "High",
    "Low",
    "Close",
    "Volume",
    "daily_return",
    "hl_range",
    "oc_return",
    "volume_change",
]

TECHNICAL_INDICATOR_COLUMNS = [
    "sma_5",
    "sma_20",
    "ema_12",
    "ema_26",
    "rsi_14",
    "macd_line",
    "macd_signal",
    "macd_hist",
    "bb_middle",
    "bb_upper",
    "bb_lower",
    "bb_width",
    "atr_14",
    "momentum_10",
    "roc_10",
    "volatility_10",
]


# ---------------------------------------------------------------------------
# 1. Target construction
# ---------------------------------------------------------------------------
def add_leak_free_target(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add the binary target column: direction of the NEXT trading day's close.

        Target_t = 1  if Close_(t+1) > Close_t
        Target_t = 0  otherwise

    The final row's target is undefined (there is no t+1) and is set to NaN
    here; it must be dropped by the caller before training (see
    `drop_undefined_target_rows`). This function does not drop it itself so
    that the leakage-audit table can still display that final row explicitly.
    """
    out = df.copy()
    next_close = out["Close"].shift(-1)  # the ONLY forward-looking operation allowed
    out["Next_Close"] = next_close
    out["Target"] = (next_close > out["Close"]).astype("float")
    out.loc[next_close.isna(), "Target"] = np.nan
    return out


def drop_undefined_target_rows(df: pd.DataFrame) -> pd.DataFrame:
    """Remove the (single) final row whose target cannot be computed."""
    before = len(df)
    out = df.dropna(subset=["Target"]).copy()
    out["Target"] = out["Target"].astype(int)
    dropped = before - len(out)
    assert dropped <= 1, (
        f"Expected to drop at most 1 row (the last one) for undefined target, "
        f"but dropped {dropped}. Check for other NaNs in the target column."
    )
    return out


# ---------------------------------------------------------------------------
# 2. Raw feature set
# ---------------------------------------------------------------------------
def add_raw_features(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add simple, leak-free transformations of the current day's OHLCV data.

    All of these use ONLY information available by the close of day t:
      - daily_return   : today's close-over-yesterday's-close return
      - hl_range       : today's (High - Low) / Close  -> intraday volatility proxy
      - oc_return      : today's (Close - Open) / Open  -> intraday direction
      - volume_change  : percentage change in volume vs. the prior day
    """
    out = df.copy()
    out["daily_return"] = out["Close"].pct_change()
    out["hl_range"] = (out["High"] - out["Low"]) / out["Close"]
    out["oc_return"] = (out["Close"] - out["Open"]) / out["Open"]
    out["volume_change"] = out["Volume"].pct_change()
    return out


# ---------------------------------------------------------------------------
# 3. Technical indicators (engineered features)
# ---------------------------------------------------------------------------
def _rsi(close: pd.Series, window: int = 14) -> pd.Series:
    """
    Relative Strength Index.

    RSI = 100 - (100 / (1 + RS))
    RS  = average gain over `window` days / average loss over `window` days

    Uses only past and current closes via a rolling/EWM window - no future data.
    """
    delta = close.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / window, min_periods=window, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    rsi = 100 - (100 / (1 + rs))
    return rsi


def _atr(high: pd.Series, low: pd.Series, close: pd.Series, window: int = 14) -> pd.Series:
    """
    Average True Range - a volatility indicator.

    True Range = max(High-Low, |High - PrevClose|, |Low - PrevClose|)
    ATR = rolling mean of True Range over `window` days.

    PrevClose uses `.shift(1)` (yesterday), which is safe: it looks BACKWARD.
    """
    prev_close = close.shift(1)
    tr = pd.concat(
        [
            (high - low),
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window=window).mean()


def add_technical_indicators(df: pd.DataFrame) -> pd.DataFrame:
    """
    Add a compact, interpretable set of technical indicators. Every indicator
    is computed using `.rolling()`, `.ewm()`, or `.shift(k)` with k >= 1
    (backward-looking only) - never `.shift(-k)`.
    """
    out = df.copy()
    close = out["Close"]

    # --- Trend: Simple / Exponential Moving Averages ---
    out["sma_5"] = close.rolling(5).mean()
    out["sma_20"] = close.rolling(20).mean()
    out["ema_12"] = close.ewm(span=12, adjust=False).mean()
    out["ema_26"] = close.ewm(span=26, adjust=False).mean()

    # --- Momentum: RSI ---
    out["rsi_14"] = _rsi(close, window=14)

    # --- Trend/Momentum: MACD ---
    macd_line = out["ema_12"] - out["ema_26"]
    macd_signal = macd_line.ewm(span=9, adjust=False).mean()
    out["macd_line"] = macd_line
    out["macd_signal"] = macd_signal
    out["macd_hist"] = macd_line - macd_signal

    # --- Volatility: Bollinger Bands ---
    bb_middle = close.rolling(20).mean()
    bb_std = close.rolling(20).std()
    out["bb_middle"] = bb_middle
    out["bb_upper"] = bb_middle + 2 * bb_std
    out["bb_lower"] = bb_middle - 2 * bb_std
    out["bb_width"] = (out["bb_upper"] - out["bb_lower"]) / bb_middle

    # --- Volatility: ATR ---
    out["atr_14"] = _atr(out["High"], out["Low"], close, window=14)

    # --- Momentum / Rate of Change ---
    out["momentum_10"] = close - close.shift(10)
    out["roc_10"] = close.pct_change(periods=10)

    # --- Volatility: rolling std of daily returns ---
    daily_return = close.pct_change()
    out["volatility_10"] = daily_return.rolling(10).std()

    return out


# ---------------------------------------------------------------------------
# 4. Convenience builders
# ---------------------------------------------------------------------------
def build_feature_frame(raw_ohlcv: pd.DataFrame) -> pd.DataFrame:
    """
    Full pipeline: raw OHLCV -> raw features -> technical indicators -> target.
    Returns a single DataFrame containing everything; callers slice out the
    RAW_FEATURE_COLUMNS or TECHNICAL_INDICATOR_COLUMNS subsets as needed.
    """
    df = raw_ohlcv.copy()
    df = add_raw_features(df)
    df = add_technical_indicators(df)
    df = add_leak_free_target(df)
    return df


def leakage_audit_table(df: pd.DataFrame, n: int = 8) -> pd.DataFrame:
    """
    Build the small "Date | Close_t | Close_t+1 | Target" table required by
    the leakage-audit section of the notebook.
    """
    table = df[["Date", "Close", "Next_Close", "Target"]].head(n).copy()
    table = table.rename(columns={"Close": "Close_t", "Next_Close": "Close_t+1"})
    return table
