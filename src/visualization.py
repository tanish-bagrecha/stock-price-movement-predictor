"""
visualization.py
=================
All plotting functions used in the notebook. Kept separate from analysis
logic so the notebook can focus on narrating results rather than matplotlib
boilerplate.
"""

from __future__ import annotations

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import numpy as np
import pandas as pd
from sklearn.metrics import ConfusionMatrixDisplay


def plot_closing_price(df: pd.DataFrame, ticker: str, ax=None):
    ax = ax or plt.gca()
    ax.plot(df["Date"], df["Close"], color="#1f77b4", linewidth=1)
    ax.set_title(f"{ticker} — Daily Closing Price")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price ($)")
    ax.xaxis.set_major_locator(mdates.YearLocator())
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.grid(alpha=0.3)
    return ax


def plot_train_test_split(df: pd.DataFrame, split_date: pd.Timestamp, ticker: str, ax=None):
    ax = ax or plt.gca()
    train = df[df["Date"] < split_date]
    test = df[df["Date"] >= split_date]
    ax.plot(train["Date"], train["Close"], color="#1f77b4", label="Training Period", linewidth=1)
    ax.plot(test["Date"], test["Close"], color="#d62728", label="Test Period", linewidth=1)
    ax.axvline(split_date, color="black", linestyle="--", linewidth=1)
    ax.set_title(f"{ticker} — Chronological Train / Test Split")
    ax.set_xlabel("Date")
    ax.set_ylabel("Close Price ($)")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax


def plot_target_distribution(y: pd.Series, ax=None):
    ax = ax or plt.gca()
    counts = y.value_counts().sort_index()
    labels = ["DOWN (0)", "UP (1)"]
    colors = ["#d62728", "#2ca02c"]
    bars = ax.bar(labels, [counts.get(0, 0), counts.get(1, 0)], color=colors)
    for bar in bars:
        height = bar.get_height()
        pct = height / counts.sum() * 100
        ax.annotate(
            f"{int(height)}\n({pct:.1f}%)",
            (bar.get_x() + bar.get_width() / 2, height),
            ha="center",
            va="bottom",
        )
    ax.set_title("Target Class Distribution")
    ax.set_ylabel("Number of Observations")
    ax.grid(alpha=0.3, axis="y")
    return ax


def plot_indicator_overlay(df: pd.DataFrame, ticker: str, ax=None):
    ax = ax or plt.gca()
    ax.plot(df["Date"], df["Close"], label="Close", color="black", linewidth=1)
    ax.plot(df["Date"], df["sma_20"], label="SMA 20", color="#1f77b4", linewidth=1)
    ax.plot(df["Date"], df["ema_12"], label="EMA 12", color="#ff7f0e", linewidth=1)
    ax.plot(df["Date"], df["bb_upper"], label="Bollinger Upper", color="#7f7f7f", linestyle="--", linewidth=0.8)
    ax.plot(df["Date"], df["bb_lower"], label="Bollinger Lower", color="#7f7f7f", linestyle="--", linewidth=0.8)
    ax.set_title(f"{ticker} — Price with Key Technical Indicators")
    ax.set_xlabel("Date")
    ax.set_ylabel("Price ($)")
    ax.legend(fontsize=8)
    ax.grid(alpha=0.3)
    return ax


def plot_confusion_matrices(results: list, ax_list=None):
    """results: list of EvalResult objects."""
    n = len(results)
    if ax_list is None:
        fig, ax_list = plt.subplots(1, n, figsize=(4.2 * n, 4))
        if n == 1:
            ax_list = [ax_list]
    for ax, result in zip(ax_list, results):
        cm = result.confusion()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["DOWN", "UP"])
        disp.plot(ax=ax, colorbar=False, cmap="Blues")
        ax.set_title(result.name, fontsize=10)
    return ax_list


def plot_model_comparison_bars(comparison_table, metrics_to_plot=None, ax=None):
    metrics_to_plot = metrics_to_plot or ["Accuracy", "Balanced Accuracy", "F1"]
    ax = ax or plt.gca()
    comparison_table[metrics_to_plot].plot(kind="bar", ax=ax)
    ax.set_title("Model Comparison Across Key Metrics")
    ax.set_ylabel("Score")
    ax.set_xlabel("")
    ax.set_ylim(0, 1)
    ax.legend(loc="lower right")
    ax.grid(alpha=0.3, axis="y")
    plt.setp(ax.get_xticklabels(), rotation=20, ha="right")
    return ax


def plot_actual_vs_predicted(dates, y_true, y_pred, ticker: str, model_name: str, ax=None):
    ax = ax or plt.gca()
    ax.step(dates, y_true, where="post", label="Actual Direction", color="black", linewidth=1.4)
    ax.step(dates, y_pred, where="post", label="Predicted Direction", color="#d62728", linewidth=1.0, alpha=0.7, linestyle="--")
    ax.set_yticks([0, 1])
    ax.set_yticklabels(["DOWN", "UP"])
    ax.set_title(f"{ticker} — Actual vs. Predicted Next-Day Direction ({model_name})")
    ax.set_xlabel("Date")
    ax.set_ylabel("Direction")
    ax.legend(loc="upper right")
    ax.grid(alpha=0.3)
    return ax


def plot_feature_importance(importances, title, ax=None):
    if ax is None:
        fig, ax = plt.subplots(figsize=(8, 6))
    importances.sort_values(ascending=True).tail(10).plot(kind='barh', ax=ax)
    ax.set_title(title)
    ax.set_xlabel("Importance")
    return ax


def plot_walk_forward_accuracy(wf_df: pd.DataFrame, ticker: str, ax=None):
    ax = ax or plt.gca()
    ax.plot(wf_df["window_end"], wf_df["accuracy"], marker="o", label="Accuracy", color="#1f77b4")
    ax.plot(wf_df["window_end"], wf_df["balanced_accuracy"], marker="s", label="Balanced Accuracy", color="#ff7f0e")
    ax.axhline(0.5, color="gray", linestyle="--", linewidth=1, label="Random Guess (0.5)")
    ax.set_title(f"{ticker} — Walk-Forward Rolling Performance")
    ax.set_xlabel("Prediction Window End Date")
    ax.set_ylabel("Score")
    ax.legend()
    ax.grid(alpha=0.3)
    return ax
