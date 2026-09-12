"""
evaluation.py
=============
Metric computation, model tuning (leakage-safe), and walk-forward /
robustness evaluation utilities.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV

from models import RANDOM_SEED, ModelSpec


@dataclass
class EvalResult:
    name: str
    y_true: np.ndarray
    y_pred: np.ndarray
    y_proba: Optional[np.ndarray] = None  # probability of class 1 (UP), if available

    def metrics(self) -> dict:
        m = {
            "Model": self.name,
            "Accuracy": accuracy_score(self.y_true, self.y_pred),
            "Balanced Accuracy": balanced_accuracy_score(self.y_true, self.y_pred),
            "Precision": precision_score(self.y_true, self.y_pred, zero_division=0),
            "Recall": recall_score(self.y_true, self.y_pred, zero_division=0),
            "F1": f1_score(self.y_true, self.y_pred, zero_division=0),
        }
        if self.y_proba is not None and len(np.unique(self.y_true)) > 1:
            m["ROC-AUC"] = roc_auc_score(self.y_true, self.y_proba)
        else:
            m["ROC-AUC"] = np.nan
        return m

    def confusion(self) -> np.ndarray:
        return confusion_matrix(self.y_true, self.y_pred, labels=[0, 1])


def build_comparison_table(results: list[EvalResult]) -> pd.DataFrame:
    """Build the mandatory four-(or more)-way comparison table."""
    rows = [r.metrics() for r in results]
    table = pd.DataFrame(rows).set_index("Model")
    return table.round(4)


def tune_model_with_time_series_cv(
    spec: ModelSpec,
    X_train: pd.DataFrame,
    y_train: pd.Series,
    n_splits: int = 5,
    scoring: str = "balanced_accuracy",
):
    """
    Tune a model's hyperparameters using TimeSeriesSplit cross-validation,
    which respects chronological order: every validation fold is strictly
    LATER in time than the training folds that precede it within that fold.

    This function only ever touches `X_train`/`y_train` - the held-out test
    set is never passed in and never influences model selection.
    """
    tscv = TimeSeriesSplit(n_splits=n_splits)
    search = GridSearchCV(
        estimator=spec.pipeline,
        param_grid=spec.param_grid,
        scoring=scoring,
        cv=tscv,
        n_jobs=-1,
    )
    search.fit(X_train, y_train)
    return search.best_estimator_, search.best_params_, search.best_score_


def walk_forward_evaluation(
    df: pd.DataFrame,
    feature_columns: list[str],
    build_pipeline_fn,
    initial_train_size: int,
    step_size: int,
) -> pd.DataFrame:
    """
    Expanding-window walk-forward evaluation.

    At each step:
      1. Train on all data from the start up to the current cutoff.
      2. Predict the NEXT `step_size` rows (strictly future, unseen data).
      3. Move the cutoff forward by `step_size` and repeat.

    The model is retrained from scratch at each step using only data that
    was available *before* the block being predicted - no future leakage.

    Returns a DataFrame with one row per walk-forward block, containing the
    prediction-window dates and accuracy/balanced-accuracy for that block.
    """
    records = []
    n = len(df)
    cutoff = initial_train_size

    while cutoff < n:
        train_slice = df.iloc[:cutoff]
        test_slice = df.iloc[cutoff : cutoff + step_size]
        if len(test_slice) == 0:
            break

        X_train = train_slice[feature_columns]
        y_train = train_slice["Target"]
        X_test = test_slice[feature_columns]
        y_test = test_slice["Target"]

        pipeline = build_pipeline_fn()
        pipeline.fit(X_train, y_train)
        y_pred = pipeline.predict(X_test)

        records.append(
            {
                "window_start": test_slice["Date"].iloc[0],
                "window_end": test_slice["Date"].iloc[-1],
                "n_test_obs": len(test_slice),
                "accuracy": accuracy_score(y_test, y_pred),
                "balanced_accuracy": balanced_accuracy_score(y_test, y_pred),
            }
        )
        cutoff += step_size

    return pd.DataFrame(records)
