from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.base import BaseEstimator, TransformerMixin
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, RobustScaler, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATABASE_DIR = ROOT_DIR / "app" / "database"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "outlier_study"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLIGHTS_TREATED_PATH = DATABASE_DIR / "flights_treated.csv"

RANDOM_STATE = 42
USE_SAMPLE = False
SAMPLE_SIZE = 120_000
CHUNK_SIZE = 250_000
TARGET = "IS_DELAYED_15"

NUMERIC_FEATURES = [
    "MONTH",
    "DAY",
    "DAY_OF_WEEK",
    "SCHEDULED_DEPARTURE",
    "SCHEDULED_ARRIVAL",
    "SCHEDULED_DEPARTURE_HOUR",
    "SCHEDULED_ARRIVAL_HOUR",
    "SCHEDULED_TIME",
    "DISTANCE",
    "ORIGIN_LATITUDE",
    "ORIGIN_LONGITUDE",
    "DESTINATION_LATITUDE",
    "DESTINATION_LONGITUDE",
    "DEPARTURE_DELAY",
    "TAXI_OUT",
    "WHEELS_OFF",
]

CATEGORICAL_FEATURES = [
    "AIRLINE",
    "ORIGIN_AIRPORT",
    "DESTINATION_AIRPORT",
    "ORIGIN_STATE",
    "DESTINATION_STATE",
    "ROUTE",
    "IS_WEEKEND",
]

# Columns where a rare value can be real operational information, not bad data.
OUTLIER_FEATURES = [
    "SCHEDULED_TIME",
    "DISTANCE",
    "DEPARTURE_DELAY",
    "TAXI_OUT",
]

USECOLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


class QuantileClipper(BaseEstimator, TransformerMixin):
    """Clip numeric columns using quantiles learned only from the training data."""

    def __init__(self, lower_quantile: float = 0.01, upper_quantile: float = 0.99):
        self.lower_quantile = lower_quantile
        self.upper_quantile = upper_quantile

    def fit(self, X, y=None):
        frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        self.lower_bounds_ = frame.quantile(self.lower_quantile)
        self.upper_bounds_ = frame.quantile(self.upper_quantile)
        return self

    def transform(self, X):
        frame = pd.DataFrame(X).apply(pd.to_numeric, errors="coerce")
        clipped = frame.clip(self.lower_bounds_, self.upper_bounds_, axis=1)
        return clipped.to_numpy()


@dataclass(frozen=True)
class Experiment:
    name: str
    scaler: str
    clip_quantiles: tuple[float, float] | None = None
    remove_train_quantiles: tuple[float, float] | None = None


EXPERIMENTS = [
    Experiment(name="baseline_keep_extremes", scaler="standard"),
    Experiment(name="robust_scaler_keep_extremes", scaler="robust"),
    Experiment(name="winsorize_p01_p99", scaler="standard", clip_quantiles=(0.01, 0.99)),
    Experiment(name="winsorize_p05_p95", scaler="standard", clip_quantiles=(0.05, 0.95)),
    Experiment(
        name="remove_train_outside_p01_p99",
        scaler="standard",
        remove_train_quantiles=(0.01, 0.99),
    ),
    Experiment(
        name="remove_train_outside_p05_p95",
        scaler="standard",
        remove_train_quantiles=(0.05, 0.95),
    ),
]


def normalize_target(series: pd.Series) -> pd.Series:
    if series.dtype == bool:
        return series.astype(int)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "sim", "yes"])
        .astype(int)
    )


def load_supervised_data() -> pd.DataFrame:
    if not FLIGHTS_TREATED_PATH.exists():
        raise FileNotFoundError(f"Missing treated flights file: {FLIGHTS_TREATED_PATH}")

    chunks: list[pd.DataFrame] = []
    rng = np.random.default_rng(RANDOM_STATE)

    for chunk in pd.read_csv(
        FLIGHTS_TREATED_PATH,
        usecols=lambda col: col in USECOLS,
        chunksize=CHUNK_SIZE,
    ):
        chunk[TARGET] = normalize_target(chunk[TARGET])
        chunk = chunk.dropna(subset=[TARGET])

        if USE_SAMPLE:
            expected_chunks = max(1, FLIGHTS_TREATED_PATH.stat().st_size // 85_000_000)
            frac = min(0.20, max(0.01, SAMPLE_SIZE / (expected_chunks * CHUNK_SIZE)))
            chunk = chunk.sample(
                frac=frac,
                random_state=int(rng.integers(0, 1_000_000)),
            )

        chunks.append(chunk)

    data = pd.concat(chunks, ignore_index=True)
    if USE_SAMPLE and len(data) > SAMPLE_SIZE:
        data = data.sample(SAMPLE_SIZE, random_state=RANDOM_STATE)

    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].fillna("UNKNOWN").astype(str)
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")

    return data.reset_index(drop=True)


def make_preprocessor(experiment: Experiment) -> ColumnTransformer:
    numeric_steps: list[tuple[str, object]] = [("imputer", SimpleImputer(strategy="median"))]

    if experiment.clip_quantiles:
        lower, upper = experiment.clip_quantiles
        numeric_steps.append(("clipper", QuantileClipper(lower, upper)))

    if experiment.scaler == "robust":
        numeric_steps.append(("scaler", RobustScaler()))
    else:
        numeric_steps.append(("scaler", StandardScaler()))

    numeric_pipeline = Pipeline(steps=numeric_steps)
    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                    sparse_output=True,
                ),
            )
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def filter_train_rows(
    X_train: pd.DataFrame,
    y_train: pd.Series,
    quantiles: tuple[float, float],
) -> tuple[pd.DataFrame, pd.Series, dict[str, float]]:
    lower_q, upper_q = quantiles
    bounds = X_train[OUTLIER_FEATURES].quantile([lower_q, upper_q])
    keep_mask = pd.Series(True, index=X_train.index)

    for col in OUTLIER_FEATURES:
        keep_mask &= X_train[col].between(bounds.loc[lower_q, col], bounds.loc[upper_q, col])

    removed_rows = int((~keep_mask).sum())
    diagnostics = {
        "train_rows_before": float(len(X_train)),
        "train_rows_after": float(keep_mask.sum()),
        "train_rows_removed": float(removed_rows),
        "train_rows_removed_pct": float(removed_rows / len(X_train)),
    }
    return X_train.loc[keep_mask], y_train.loc[keep_mask], diagnostics


def evaluate_experiment(
    experiment: Experiment,
    X_train: pd.DataFrame,
    X_test: pd.DataFrame,
    y_train: pd.Series,
    y_test: pd.Series,
) -> dict[str, float | str]:
    train_diagnostics = {
        "train_rows_before": float(len(X_train)),
        "train_rows_after": float(len(X_train)),
        "train_rows_removed": 0.0,
        "train_rows_removed_pct": 0.0,
    }

    X_train_fit = X_train
    y_train_fit = y_train
    if experiment.remove_train_quantiles:
        X_train_fit, y_train_fit, train_diagnostics = filter_train_rows(
            X_train,
            y_train,
            experiment.remove_train_quantiles,
        )

    model = Pipeline(
        steps=[
            ("preprocess", make_preprocessor(experiment)),
            (
                "model",
                LogisticRegression(
                    max_iter=400,
                    class_weight="balanced",
                    random_state=RANDOM_STATE,
                ),
            ),
        ]
    )
    model.fit(X_train_fit, y_train_fit)

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    return {
        "experiment": experiment.name,
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        **train_diagnostics,
    }


def create_tail_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for col in OUTLIER_FEATURES:
        series = data[col]
        p95 = series.quantile(0.95)
        p99 = series.quantile(0.99)
        for label, threshold in [("p95", p95), ("p99", p99)]:
            tail_mask = series >= threshold
            rows.append(
                {
                    "feature": col,
                    "tail": label,
                    "threshold": threshold,
                    "tail_rows": int(tail_mask.sum()),
                    "tail_pct": float(tail_mask.mean()),
                    "target_rate_tail": float(data.loc[tail_mask, TARGET].mean()),
                    "target_rate_rest": float(data.loc[~tail_mask, TARGET].mean()),
                    "target_rate_lift": float(
                        data.loc[tail_mask, TARGET].mean()
                        / data.loc[~tail_mask, TARGET].mean()
                    ),
                }
            )
    return pd.DataFrame(rows)


def main() -> None:
    data = load_supervised_data()
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    metrics = [
        evaluate_experiment(experiment, X_train, X_test, y_train, y_test)
        for experiment in EXPERIMENTS
    ]
    metrics_df = pd.DataFrame(metrics).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "outlier_strategy_metrics.csv", index=False)

    tail_diagnostics = create_tail_diagnostics(data)
    tail_diagnostics.to_csv(OUTPUT_DIR / "outlier_tail_diagnostics.csv", index=False)

    print("Outlier strategy metrics")
    print(metrics_df.to_string(index=False))
    print("\nTail diagnostics")
    print(tail_diagnostics.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
