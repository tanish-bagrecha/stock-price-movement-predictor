# Stock Price Movement Predictor

**Leak-Free Stock Price Movement Prediction Using Time-Series Machine Learning**

A methodologically rigorous, portfolio-level project demonstrating correct time-series ML
practice for financial data: leak-free target construction, chronological validation, honest
multi-metric evaluation, and walk-forward robustness testing.

> **Disclaimer**: This is an educational / research project, **not financial advice** and
> **not a trading strategy**. Directional classification accuracy does not imply, and should
> never be interpreted as, trading profitability.

---

## Overview

This project builds a binary classifier that predicts whether a stock's closing price will be
**higher (UP = 1)** or **lower/equal (DOWN = 0)** on the **next trading day**, using only
information available up to the current day. The emphasis throughout is on **zero temporal
data leakage** and **honest evaluation** — not on maximizing a headline accuracy number.

## Problem Statement

Financial markets are widely believed to be close to informationally efficient, which makes
short-horizon direction extremely hard to predict from price/volume history alone. This
project treats that difficulty as the point: it builds a fully correct pipeline and reports
whatever the data honestly shows, rather than engineering a result.

## Objective

Binary classification of next-day directional movement, comparing:

- Two "no machine learning" baselines (Persistence, Majority Class)
- Machine learning models trained on **raw** OHLCV-derived features
- The same models trained on an **engineered** feature set (raw + technical indicators)

## Dataset

- **Ticker (default)**: `AAPL` — configurable via one variable in the notebook
- **Frequency**: Daily OHLCV bars
- **Range**: 2012‑01‑03 to the most recent available trading day
- **Source**: Live via `yfinance` when internet access is available; otherwise a bundled,
  version-controlled CSV snapshot (`data/AAPL_raw.csv`) is used automatically as a
  reproducibility fallback. See `data/README.md`.
- **Final cleaned dataset size**: 3,694 trading days (after data-quality checks)

## Methodology

### Target Construction

```python
Target_t = 1 if Close_(t+1) > Close_t else 0
```

Implemented as `(Close.shift(-1) > Close).astype(int)`. This is the **only** place in the
entire codebase where a forward-looking (`.shift(-1)`) operation is used, and it is used
exclusively to build the *label*, never a *feature*. The final row (whose "tomorrow" doesn't
exist yet) is dropped.

### Feature Engineering

**Raw feature set** — current-day OHLCV plus simple leak-free transformations:
`Open, High, Low, Close, Volume, daily_return, hl_range, oc_return, volume_change`

**Engineered feature set** — raw features **+** technical indicators, all computed with
backward-looking rolling/EWM windows only (no unmaintained `pandas-ta` dependency):

| Indicator | Purpose |
|---|---|
| SMA 5 / SMA 20 | Short/medium-term trend |
| EMA 12 / EMA 26 | Reactive trend, feeds MACD |
| RSI 14 | Overbought / oversold momentum |
| MACD (line, signal, histogram) | Trend/momentum shifts |
| Bollinger Bands (mid/upper/lower/width) | Volatility-relative price position |
| ATR 14 | Volatility magnitude |
| Momentum 10 / ROC 10 | Recent trend strength |
| Volatility 10 | Rolling std. dev. of returns |

### Technical Indicators

See the notebook's "Technical Indicators" section for the formula, rationale, and explicit
leak-free verification of every indicator.

### Baselines

- **Persistence**: predicts tomorrow repeats today's already-realized direction (no ML).
- **Majority Class**: always predicts the majority class *from the training set only*.

### Machine Learning Models

- Logistic Regression (scaled, inside a `Pipeline`)
- Random Forest (`class_weight="balanced"`)
- HistGradientBoostingClassifier

Each is trained once on the raw feature set and once on the engineered feature set.

### Time-Based Validation

- **Chronological 80/20 split** — the earliest ~80% of days train the model; the most recent
  ~20% are held out and touched exactly once, at final evaluation.
- **No shuffling, ever.** `train_test_split(shuffle=True)` is never used.
- **`TimeSeriesSplit` cross-validation** for hyperparameter tuning, applied only within the
  training set.
- **Expanding-window walk-forward evaluation** on top of the single split, to check that
  performance is stable across time rather than an artifact of one lucky test window.

### Leakage Prevention

- Scalers live inside `sklearn.Pipeline` objects and are fit only on `X_train`.
- Hyperparameter tuning never sees the test set.
- The majority-class baseline is computed from training labels only.
- A dedicated, programmatically-verified **Leakage Audit** section in the notebook checks all
  of the above explicitly (see Section 26 of the notebook).

## Results

Four-way comparison table (chronological 80/20 split, AAPL, 2012–2026):

| Model | Accuracy | Balanced Accuracy | Precision | Recall | F1 | ROC-AUC |
|---|---|---|---|---|---|---|
| Persistence | 0.5102 | 0.5070 | 0.5466 | 0.5466 | 0.5466 | — |
| Majority Class | 0.5401 | 0.5000 | 0.5401 | 1.0000 | 0.7014 | — |
| Logistic Regression (Raw) | 0.4871 | 0.4752 | 0.5221 | 0.6200 | 0.5669 | 0.4660 |
| Random Forest (Raw) | 0.4655 | 0.5054 | 0.6923 | 0.0225 | 0.0436 | 0.4766 |
| Gradient Boosting (Raw) | 0.5088 | 0.5188 | 0.5658 | 0.3975 | 0.4670 | 0.5002 |
| Logistic Regression (Engineered) | 0.4912 | 0.4793 | 0.5242 | 0.6272 | 0.5711 | 0.4612 |
| Random Forest (Engineered) | 0.4707 | 0.4973 | 0.5323 | 0.1662 | 0.2534 | 0.5003 |
| Gradient Boosting (Engineered) | 0.4721 | 0.4942 | 0.5273 | 0.2191 | 0.3096 | 0.4965 |

**Honest reading of this table**: no model — raw or engineered — reliably beats the naive
baselines on **Balanced Accuracy**, and ROC-AUC values cluster tightly around 0.50 (random
chance). The Majority Class baseline's high headline Accuracy is a direct consequence of class
imbalance (~53% UP days), not predictive skill — exactly the trap this project's methodology
is designed to expose rather than hide.

Walk-forward evaluation (expanding window, ~monthly blocks) reinforces this: mean accuracy
0.460 and mean balanced accuracy 0.495 across blocks, hovering around chance level with no
clear improving trend.

## Visualizations

The notebook produces (saved to `results/figures/`): closing price over time, return
distribution, target class balance, indicator overlay, train/test split timeline, model
comparison bars, confusion matrices, Random Forest feature importance, Logistic Regression
coefficients, actual-vs-predicted direction over the test window, and walk-forward rolling
accuracy.

## Generalization Analysis

The walk-forward analysis (Section 24–25 of the notebook) shows performance fluctuating close
to the 0.5 "random guess" line across different time blocks, with no model consistently and
substantially outperforming Persistence in a way that holds up out-of-sample across regimes.
This is consistent with the broader literature on weak-form market efficiency for daily
OHLCV-only signals.

## Limitations

- Daily OHLCV captures a small fraction of what actually drives next-day price moves (no
  news, macro, sentiment, or order-flow data).
- Historical statistical relationships are not guaranteed to persist (non-stationarity).
- No transaction costs, slippage, or market-impact modeling.
- Directional accuracy is not the same as trading profitability.
- Single-asset study; results may not generalize to other tickers or periods.

## Reproducibility

- Random seeds fixed (`RANDOM_SEED = 42`) everywhere randomness is used.
- All preprocessing wrapped in `sklearn.Pipeline` objects, fit only on training data.
- Data loading falls back to a bundled CSV if live download is unavailable, so results are
  reproducible offline too.
- Python version: 3.10+ recommended (developed/tested on 3.11).

## Installation

```bash
git clone <this-repo-url>
cd stock-price-movement-predictor
python -m venv .venv
source .venv/bin/activate   # on Windows: .venv\Scripts\activate
pip install -r requirements.txt
```

## How to Run

```bash
jupyter notebook notebooks/stock_direction_prediction.ipynb
```

Then run all cells top to bottom (Kernel → Restart & Run All). To analyze a different ticker,
edit the `TICKER` variable in the "Project Configuration" cell near the top of the notebook and
re-run.

Generated figures are saved to `results/figures/`, and the four-way comparison / walk-forward
tables are saved to `results/tables/`.

## Repository Structure

```
stock-price-movement-predictor/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── stock_direction_prediction.ipynb
│
├── src/
│   ├── data_loader.py       # download + data-quality audit
│   ├── features.py          # target construction + raw/engineered features
│   ├── models.py            # baselines + ML pipelines
│   ├── evaluation.py        # metrics, tuning, walk-forward evaluation
│   └── visualization.py     # all plotting functions
│
├── data/
│   ├── README.md
│   └── AAPL_raw.csv         # small, reproducibility fallback snapshot
│
└── results/
    ├── figures/              # PNG plots generated by the notebook
    └── tables/               # CSV comparison / walk-forward tables
```

## Future Improvements

- Feature ablation study (raw vs. raw+SMA vs. raw+RSI vs. full engineered set) to isolate
  which indicators, if any, add incremental value.
- Probability calibration and ROC/Precision-Recall curve analysis.
- Multi-asset validation to check whether any signal found is asset-specific or general.
- Confidence-threshold analysis (only act on high-confidence predictions).

## Disclaimer

This project is for **educational and research purposes only**. It is **not financial
advice**, is **not a trading system**, and past performance/backtested results are **not**
indicative of future results.
