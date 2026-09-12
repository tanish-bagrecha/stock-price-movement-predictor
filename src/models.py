"""
models.py
=========
Baseline strategies (no ML) and machine-learning model pipelines.

All ML models are wrapped in `sklearn.Pipeline` objects that bundle scaling
(where relevant) together with the classifier. This guarantees that, as long
as the caller only ever calls `.fit(X_train, y_train)`, the scaler can never
see the test set. See `evaluation.py` / the notebook's Leakage Audit section
for the explicit check that this held.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

RANDOM_SEED = 42


# ---------------------------------------------------------------------------
# Baselines (no machine learning)
# ---------------------------------------------------------------------------
def persistence_baseline(df: pd.DataFrame) -> pd.Series:
    """
    BASELINE 1 - PERSISTENCE.

    Predicts that tomorrow's direction repeats today's direction, i.e.
    "today's direction" = 1 if Close_t > Close_(t-1) else 0.

    This uses ONLY already-known past information (today's close vs.
    yesterday's close) - no machine learning, no future data.
    """
    today_direction = (df["Close"] > df["Close"].shift(1)).astype(int)
    return today_direction


def majority_class_baseline(y_train: pd.Series, n_predictions: int) -> np.ndarray:
    """
    BASELINE 2 - MAJORITY CLASS.

    Always predicts whichever class (UP=1 or DOWN=0) is more frequent in the
    TRAINING data only. The test set is never inspected to make this choice.
    """
    majority_class = int(y_train.mode().iloc[0])
    return np.full(shape=n_predictions, fill_value=majority_class)


# ---------------------------------------------------------------------------
# Machine learning model pipelines
# ---------------------------------------------------------------------------
@dataclass
class ModelSpec:
    name: str
    pipeline: Pipeline
    param_grid: dict


def build_model_specs(random_seed: int = RANDOM_SEED) -> list[ModelSpec]:
    """
    Build the three classical ML pipelines used in this project, each paired
    with a small hyperparameter grid for TimeSeriesSplit-based tuning.

    Logistic Regression and effectively-distance-based scikit-learn models
    are scaled; tree ensembles do not require scaling but we keep a consistent
    Pipeline interface for all models regardless.
    """
    specs = []

    logreg_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            ("model", LogisticRegression(max_iter=2000, random_state=random_seed)),
        ]
    )
    specs.append(
        ModelSpec(
            name="Logistic Regression",
            pipeline=logreg_pipeline,
            param_grid={"model__C": [0.01, 0.1, 1.0, 10.0]},
        )
    )

    rf_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),  # harmless for RF, kept for pipeline consistency
            (
                "model",
                RandomForestClassifier(
                    random_state=random_seed,
                    n_jobs=1,  # keep single-threaded here: GridSearchCV already
                               # parallelizes across folds/params (n_jobs=-1 there);
                               # nested parallel workers on Windows actually slow
                               # things down and cause joblib memmapping warnings.
                    class_weight="balanced",
                ),
            ),
        ]
    )
    specs.append(
        ModelSpec(
            name="Random Forest",
            pipeline=rf_pipeline,
            param_grid={
                "model__n_estimators": [200, 400],
                "model__max_depth": [3, 5, 8],
            },
        )
    )

    hgb_pipeline = Pipeline(
        [
            ("scaler", StandardScaler()),
            (
                "model",
                HistGradientBoostingClassifier(random_state=random_seed),
            ),
        ]
    )
    specs.append(
        ModelSpec(
            name="Gradient Boosting",
            pipeline=hgb_pipeline,
            param_grid={
                "model__max_depth": [3, 5, None],
                "model__learning_rate": [0.03, 0.1],
            },
        )
    )

    return specs
