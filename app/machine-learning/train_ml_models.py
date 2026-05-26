from __future__ import annotations

from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.compose import ColumnTransformer
from sklearn.decomposition import PCA
from sklearn.ensemble import RandomForestClassifier
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
    roc_auc_score,
    roc_curve,
    silhouette_score,
)
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATABASE_DIR = ROOT_DIR / "app" / "database"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"

FLIGHTS_TREATED_PATH = DATABASE_DIR / "flights_treated.csv"
ROUTE_PROFILE_PATH = DATABASE_DIR / "outputs" / "completed_route_delay_profile.csv"

RANDOM_STATE = 42
SUPERVISED_SAMPLE_SIZE = 180_000
CHUNK_SIZE = 250_000
ROUTE_MIN_FLIGHTS = 500

TARGET = "IS_DELAYED_15"

# Features used in the supervised study. The prediction scenario here is:
# "after takeoff". Because of that, we keep variables already known by then,
# such as DEPARTURE_DELAY, TAXI_OUT and WHEELS_OFF, and exclude arrival/final
# delay columns that would reveal the target directly.
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

SUPERVISED_USECOLS = NUMERIC_FEATURES + CATEGORICAL_FEATURES + [TARGET]


def ensure_output_dir() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)


def normalize_bool_target(series: pd.Series) -> pd.Series:
    """Normalize boolean-like CSV values to integer 0/1."""
    if series.dtype == bool:
        return series.astype(int)
    return (
        series.astype(str)
        .str.strip()
        .str.lower()
        .isin(["true", "1", "sim", "yes"])
        .astype(int)
    )


def load_supervised_sample(sample_size: int = SUPERVISED_SAMPLE_SIZE) -> pd.DataFrame:
    """Load a reproducible sample from the large treated flights dataset."""
    if not FLIGHTS_TREATED_PATH.exists():
        raise FileNotFoundError(f"Missing treated flights file: {FLIGHTS_TREATED_PATH}")

    samples: list[pd.DataFrame] = []
    rng = np.random.default_rng(RANDOM_STATE)

    for chunk in pd.read_csv(
        FLIGHTS_TREATED_PATH,
        usecols=lambda col: col in SUPERVISED_USECOLS,
        chunksize=CHUNK_SIZE,
    ):
        if TARGET not in chunk.columns:
            raise ValueError(f"Target column {TARGET} was not found.")

        chunk[TARGET] = normalize_bool_target(chunk[TARGET])
        chunk = chunk.dropna(subset=[TARGET])

        # Dynamic fraction keeps memory bounded while still scanning the full file.
        expected_chunks = max(1, FLIGHTS_TREATED_PATH.stat().st_size // 85_000_000)
        frac = min(0.25, max(0.01, sample_size / (expected_chunks * CHUNK_SIZE)))
        sampled = chunk.sample(frac=frac, random_state=int(rng.integers(0, 1_000_000)))
        samples.append(sampled)

    data = pd.concat(samples, ignore_index=True)
    if len(data) > sample_size:
        data = data.sample(sample_size, random_state=RANDOM_STATE)

    return data.reset_index(drop=True)


def build_preprocessor() -> ColumnTransformer:
    numeric_pipeline = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_pipeline = Pipeline(
        steps=[
            (
                "onehot",
                OneHotEncoder(
                    handle_unknown="infrequent_if_exist",
                    min_frequency=50,
                    sparse_output=True,
                ),
            ),
        ]
    )

    return ColumnTransformer(
        transformers=[
            ("num", numeric_pipeline, NUMERIC_FEATURES),
            ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
        ],
        remainder="drop",
    )


def model_definitions() -> dict[str, Pipeline]:
    preprocessor = build_preprocessor()
    return {
        "logistic_regression": Pipeline(
            steps=[
                ("preprocess", preprocessor),
                (
                    "model",
                    LogisticRegression(
                        max_iter=400,
                        class_weight="balanced",
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
        "random_forest": Pipeline(
            steps=[
                ("preprocess", build_preprocessor()),
                (
                    "model",
                    RandomForestClassifier(
                        n_estimators=140,
                        max_depth=18,
                        min_samples_leaf=40,
                        class_weight="balanced_subsample",
                        n_jobs=-1,
                        random_state=RANDOM_STATE,
                    ),
                ),
            ]
        ),
    }


def evaluate_classifier(model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series) -> dict[str, float]:
    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]
    return {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
    }


def save_supervised_plots(best_model: Pipeline, X_test: pd.DataFrame, y_test: pd.Series, model_name: str) -> None:
    probability = best_model.predict_proba(X_test)[:, 1]
    prediction = best_model.predict(X_test)

    fpr, tpr, _ = roc_curve(y_test, probability)
    roc_fig = go.Figure()
    roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=model_name))
    roc_fig.add_trace(
        go.Scatter(
            x=[0, 1],
            y=[0, 1],
            mode="lines",
            name="baseline",
            line=dict(dash="dash"),
        )
    )
    roc_fig.update_layout(
        title=f"ROC Curve - {model_name}",
        xaxis_title="False positive rate",
        yaxis_title="True positive rate",
        template="plotly_dark",
    )
    roc_fig.write_html(OUTPUT_DIR / "supervised_roc_curve.html")

    matrix = confusion_matrix(y_test, prediction)
    matrix_fig = px.imshow(
        matrix,
        text_auto=True,
        labels=dict(x="Predicted", y="Actual", color="Flights"),
        x=["on_time", "delayed_15"],
        y=["on_time", "delayed_15"],
        title=f"Confusion Matrix - {model_name}",
        color_continuous_scale="Blues",
    )
    matrix_fig.update_layout(template="plotly_dark")
    matrix_fig.write_html(OUTPUT_DIR / "supervised_confusion_matrix.html")


def run_supervised_modeling() -> pd.DataFrame:
    data = load_supervised_sample()
    data[CATEGORICAL_FEATURES] = data[CATEGORICAL_FEATURES].fillna("UNKNOWN").astype(str)
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
    X = data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = data[TARGET].astype(int)

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.25,
        stratify=y,
        random_state=RANDOM_STATE,
    )

    rows = []
    trained_models: dict[str, Pipeline] = {}
    for name, model in model_definitions().items():
        model.fit(X_train, y_train)
        metrics = evaluate_classifier(model, X_test, y_test)
        metrics["model"] = name
        rows.append(metrics)
        trained_models[name] = model
        joblib.dump(model, OUTPUT_DIR / f"{name}_classifier.joblib")

    metrics_df = pd.DataFrame(rows).sort_values("roc_auc", ascending=False)
    metrics_df.to_csv(OUTPUT_DIR / "supervised_classification_metrics.csv", index=False)

    best_model_name = metrics_df.iloc[0]["model"]
    save_supervised_plots(trained_models[best_model_name], X_test, y_test, best_model_name)

    sample_profile = pd.DataFrame(
        [
            {"metric": "sample_rows", "value": len(data)},
            {"metric": "train_rows", "value": len(X_train)},
            {"metric": "test_rows", "value": len(X_test)},
            {"metric": "delayed_15_rate_sample", "value": y.mean()},
        ]
    )
    sample_profile.to_csv(OUTPUT_DIR / "supervised_sample_profile.csv", index=False)

    return metrics_df


def load_route_profile() -> pd.DataFrame:
    if not ROUTE_PROFILE_PATH.exists():
        raise FileNotFoundError(f"Missing route profile file: {ROUTE_PROFILE_PATH}")
    route_df = pd.read_csv(ROUTE_PROFILE_PATH)
    required = [
        "ROUTE",
        "flights",
        "delayed_flights",
        "avg_arrival_delay",
        "delayed_15_pct",
        "adjusted_delay_pct",
        "delay_impact_score",
    ]
    missing = [col for col in required if col not in route_df.columns]
    if missing:
        raise ValueError(f"Missing route columns: {missing}")
    return route_df


def run_unsupervised_modeling() -> tuple[pd.DataFrame, pd.DataFrame]:
    route_df = load_route_profile()
    route_df = route_df[route_df["flights"] >= ROUTE_MIN_FLIGHTS].copy()
    route_df["avg_arrival_delay_positive"] = route_df["avg_arrival_delay"].clip(lower=0)

    cluster_features = [
        "flights",
        "delayed_flights",
        "avg_arrival_delay_positive",
        "delayed_15_pct",
        "adjusted_delay_pct",
        "delay_impact_score",
    ]

    X = route_df[cluster_features].apply(pd.to_numeric, errors="coerce")
    X = X.fillna(X.median(numeric_only=True))
    scaled = StandardScaler().fit_transform(X)

    pca = PCA(n_components=2, random_state=RANDOM_STATE)
    components = pca.fit_transform(scaled)

    kmeans = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
    route_df["cluster"] = kmeans.fit_predict(scaled)
    route_df["pca_1"] = components[:, 0]
    route_df["pca_2"] = components[:, 1]

    clustered_path = OUTPUT_DIR / "unsupervised_route_clusters.csv"
    route_df.sort_values(["cluster", "delay_impact_score"], ascending=[True, False]).to_csv(
        clustered_path,
        index=False,
    )

    cluster_summary = (
        route_df.groupby("cluster", as_index=False)
        .agg(
            routes=("ROUTE", "count"),
            avg_flights=("flights", "mean"),
            avg_arrival_delay_positive=("avg_arrival_delay_positive", "mean"),
            avg_delayed_15_pct=("delayed_15_pct", "mean"),
            avg_delay_impact_score=("delay_impact_score", "mean"),
            top_route=("ROUTE", "first"),
        )
        .sort_values("avg_delay_impact_score", ascending=False)
    )
    cluster_summary["silhouette_score"] = silhouette_score(scaled, route_df["cluster"])
    cluster_summary["pca_explained_variance"] = pca.explained_variance_ratio_.sum()
    cluster_summary.to_csv(OUTPUT_DIR / "unsupervised_cluster_summary.csv", index=False)

    fig = px.scatter(
        route_df,
        x="pca_1",
        y="pca_2",
        color="cluster",
        size="flights",
        hover_name="ROUTE",
        hover_data=[
            "flights",
            "avg_arrival_delay_positive",
            "delayed_15_pct",
            "delay_impact_score",
        ],
        title="Clusterizacao de rotas por perfil de atraso",
        template="plotly_dark",
        color_continuous_scale="Turbo",
        size_max=32,
    )
    fig.write_html(OUTPUT_DIR / "unsupervised_route_clusters_pca.html")

    summary_fig = px.bar(
        cluster_summary,
        x="cluster",
        y="avg_delay_impact_score",
        color="avg_delayed_15_pct",
        text="routes",
        title="Resumo dos clusters de rotas",
        labels={
            "avg_delay_impact_score": "Avg delay impact score",
            "avg_delayed_15_pct": "Avg delayed 15 pct",
            "routes": "Routes",
        },
        template="plotly_dark",
        color_continuous_scale="YlOrRd",
    )
    summary_fig.write_html(OUTPUT_DIR / "unsupervised_cluster_summary.html")

    return route_df, cluster_summary


def main() -> None:
    ensure_output_dir()
    supervised_metrics = run_supervised_modeling()
    _, cluster_summary = run_unsupervised_modeling()

    print("Supervised metrics")
    print(supervised_metrics.to_string(index=False))
    print("\nCluster summary")
    print(cluster_summary.to_string(index=False))
    print(f"\nOutputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
