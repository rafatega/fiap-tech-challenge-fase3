from __future__ import annotations

from pathlib import Path

import pandas as pd
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler


ROOT_DIR = Path(__file__).resolve().parents[2]
DATABASE_DIR = ROOT_DIR / "app" / "database"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs" / "elbow_analysis"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

ROUTE_PROFILE_PATH = DATABASE_DIR / "outputs" / "completed_route_delay_profile.csv"

RANDOM_STATE = 42
ROUTE_MIN_FLIGHTS = 500
K_MIN = 2
K_MAX = 10

CLUSTER_FEATURES = [
    "flights",
    "delayed_flights",
    "avg_arrival_delay_positive",
    "delayed_15_pct",
    "adjusted_delay_pct",
    "delay_impact_score",
]

REQUIRED_ROUTE_COLUMNS = [
    "ROUTE",
    "flights",
    "delayed_flights",
    "avg_arrival_delay",
    "delayed_15_pct",
    "adjusted_delay_pct",
    "delay_impact_score",
]


def load_route_features() -> tuple[pd.DataFrame, pd.DataFrame]:
    if not ROUTE_PROFILE_PATH.exists():
        raise FileNotFoundError(f"Missing route profile file: {ROUTE_PROFILE_PATH}")

    route_df = pd.read_csv(ROUTE_PROFILE_PATH)
    missing_columns = [
        col for col in REQUIRED_ROUTE_COLUMNS if col not in route_df.columns
    ]
    if missing_columns:
        raise ValueError(f"Missing route columns: {missing_columns}")

    route_df = route_df[route_df["flights"] >= ROUTE_MIN_FLIGHTS].copy()
    route_df["avg_arrival_delay_positive"] = route_df["avg_arrival_delay"].clip(
        lower=0
    )

    route_features = route_df[CLUSTER_FEATURES].apply(pd.to_numeric, errors="coerce")
    route_features = route_features.fillna(route_features.median(numeric_only=True))
    return route_df, route_features


def find_elbow_k(results: pd.DataFrame) -> int:
    """Estimate elbow as the point farthest from the line joining first and last k."""
    x = results["k"].astype(float)
    y = results["inertia"].astype(float)

    x_norm = (x - x.min()) / (x.max() - x.min())
    y_norm = (y - y.min()) / (y.max() - y.min())

    start = pd.Series({"x": x_norm.iloc[0], "y": y_norm.iloc[0]})
    end = pd.Series({"x": x_norm.iloc[-1], "y": y_norm.iloc[-1]})
    line = end - start
    line_length = (line["x"] ** 2 + line["y"] ** 2) ** 0.5

    distances = abs(
        line["x"] * (start["y"] - y_norm)
        - (start["x"] - x_norm) * line["y"]
    ) / line_length
    return int(results.loc[distances.idxmax(), "k"])


def create_elbow_chart(results: pd.DataFrame, recommended_k: int) -> go.Figure:
    fig = px.line(
        results,
        x="k",
        y="inertia",
        markers=True,
        title="Elbow analysis - KMeans route clusters",
        labels={
            "k": "Number of clusters (k)",
            "inertia": "Inertia",
        },
        template="plotly_dark",
    )
    recommended_inertia = results.loc[
        results["k"] == recommended_k, "inertia"
    ].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[recommended_k],
            y=[recommended_inertia],
            mode="markers+text",
            marker=dict(size=14, color="#ffcc00"),
            text=[f"recommended k={recommended_k}"],
            textposition="top center",
            name="recommended_k",
        )
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def create_silhouette_chart(results: pd.DataFrame, recommended_k: int) -> go.Figure:
    fig = px.line(
        results,
        x="k",
        y="silhouette_score",
        markers=True,
        title="Silhouette score by number of clusters",
        labels={
            "k": "Number of clusters (k)",
            "silhouette_score": "Silhouette score",
        },
        template="plotly_dark",
    )
    recommended_silhouette = results.loc[
        results["k"] == recommended_k, "silhouette_score"
    ].iloc[0]
    fig.add_trace(
        go.Scatter(
            x=[recommended_k],
            y=[recommended_silhouette],
            mode="markers+text",
            marker=dict(size=14, color="#ffcc00"),
            text=[f"elbow k={recommended_k}"],
            textposition="top center",
            name="elbow_k",
        )
    )
    fig.update_layout(height=560, margin=dict(l=10, r=10, t=60, b=10))
    return fig


def save_png_charts(results: pd.DataFrame, recommended_k: int) -> None:
    recommended = results[results["k"] == recommended_k].iloc[0]

    plt.style.use("dark_background")

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["k"], results["inertia"], marker="o", linewidth=2)
    ax.scatter(recommended_k, recommended["inertia"], s=140, color="#ffcc00")
    ax.annotate(
        f"recommended k={recommended_k}",
        (recommended_k, recommended["inertia"]),
        textcoords="offset points",
        xytext=(8, 12),
        color="#ffcc00",
    )
    ax.set_title("Elbow analysis - KMeans route clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Inertia")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "elbow_curve.png", dpi=160)
    plt.close(fig)

    fig, ax = plt.subplots(figsize=(10, 6))
    ax.plot(results["k"], results["silhouette_score"], marker="o", linewidth=2)
    ax.scatter(
        recommended_k,
        recommended["silhouette_score"],
        s=140,
        color="#ffcc00",
    )
    ax.annotate(
        f"elbow k={recommended_k}",
        (recommended_k, recommended["silhouette_score"]),
        textcoords="offset points",
        xytext=(8, 12),
        color="#ffcc00",
    )
    ax.set_title("Silhouette score by number of clusters")
    ax.set_xlabel("Number of clusters (k)")
    ax.set_ylabel("Silhouette score")
    ax.grid(alpha=0.25)
    fig.tight_layout()
    fig.savefig(OUTPUT_DIR / "silhouette_by_k.png", dpi=160)
    plt.close(fig)


def main() -> None:
    route_df, route_features = load_route_features()
    scaled_route_features = StandardScaler().fit_transform(route_features)

    rows = []
    for k in range(K_MIN, K_MAX + 1):
        kmeans = KMeans(n_clusters=k, n_init=20, random_state=RANDOM_STATE)
        labels = kmeans.fit_predict(scaled_route_features)
        rows.append(
            {
                "k": k,
                "inertia": kmeans.inertia_,
                "silhouette_score": silhouette_score(scaled_route_features, labels),
            }
        )

    results = pd.DataFrame(rows)
    recommended_k = find_elbow_k(results)
    results["recommended_by_elbow"] = results["k"] == recommended_k
    results["routes_used"] = len(route_df)
    results["route_min_flights"] = ROUTE_MIN_FLIGHTS
    results.to_csv(OUTPUT_DIR / "elbow_analysis_results.csv", index=False)

    elbow_fig = create_elbow_chart(results, recommended_k)
    elbow_fig.write_html(OUTPUT_DIR / "elbow_curve.html")

    silhouette_fig = create_silhouette_chart(results, recommended_k)
    silhouette_fig.write_html(OUTPUT_DIR / "silhouette_by_k.html")

    save_png_charts(results, recommended_k)

    print("Elbow analysis results")
    print(results.to_string(index=False))
    print(f"\nRecommended k by elbow heuristic: {recommended_k}")
    print(f"Outputs saved to: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
