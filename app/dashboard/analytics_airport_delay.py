from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st
import streamlit.components.v1 as components

try:
    import plotly.express as px
    import plotly.graph_objects as go

    HAS_PLOTLY = True
except Exception:
    HAS_PLOTLY = False
    px = None
    go = None


ROOT_DIR = Path(__file__).resolve().parents[2]
DATABASE_DIR = ROOT_DIR / "app" / "database"
DATABASE_OUTPUTS_DIR = DATABASE_DIR / "outputs"
EDA_OUTPUTS_DIR = ROOT_DIR / "app" / "eda" / "outputs"
ML_OUTPUTS_DIR = ROOT_DIR / "app" / "machine-learning" / "outputs"
OUTLIER_OUTPUTS_DIR = ML_OUTPUTS_DIR / "outlier_study"

RAW_TABLES = {
    "Airlines": DATABASE_DIR / "airlines.csv",
    "Airports": DATABASE_DIR / "airports.csv",
    "Flights": DATABASE_DIR / "flights.csv",
    "Airport ID to IATA": DATABASE_DIR / "airport_id_to_iata_2015_10.csv",
}

OUTPUT_FILES = {
    "airlines_profile_before": "completed_column_profile_airlines_before_treatment.csv",
    "airports_profile_before": "completed_column_profile_airports_before_treatment.csv",
    "flights_profile_before": "completed_column_profile_flights_before_treatment.csv",
    "completed_profile_before": "completed_column_profile_before_treatment.csv",
    "completed_profile_after": "completed_column_profile_after_treatment.csv",
    "numeric_profile": "completed_numeric_profile.csv",
    "origin_delay": "completed_origin_airport_delay_profile.csv",
    "destination_delay": "completed_destination_airport_delay_profile.csv",
    "route_delay": "completed_route_delay_profile.csv",
    "invalid_airport_codes": "flights_with_invalid_airport_codes.csv",
    "treatment_plan": "completed_treatment_plan.csv",
}


st.set_page_config(
    page_title="FIAP US Flight Analytics",
    layout="wide",
    initial_sidebar_state="expanded",
)


st.markdown(
    """
    <style>
    .stApp {
        background: #0f1420;
        color: #f4f7fb;
    }
    .main .block-container {padding-top: 1.5rem; padding-bottom: 2.5rem;}
    div[data-testid="stMetric"] {
        background: #171f2f;
        border: 1px solid #293449;
        border-radius: 8px;
        padding: 14px 16px;
        box-shadow: 0 1px 2px rgba(0, 0, 0, .24);
    }
    div[data-testid="stMetric"] label,
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        color: #f4f7fb;
    }
    .story-box {
        border-left: 4px solid #2f80ed;
        background: #f7fbff;
        color: #243044;
        padding: 0.85rem 1rem;
        border-radius: 6px;
        margin: 0.35rem 0 1rem 0;
    }
    .story-box strong, .story-box code, .story-box a {color: #172033;}
    .muted {color: #c8d2e3; font-size: .94rem;}
    div[data-testid="stDataFrame"] {
        border: 1px solid #293449;
        border-radius: 8px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


def first_existing_path(file_name: str) -> Path | None:
    """Find an EDA artifact, prioritizing the database output folder used by the app."""
    for folder in (DATABASE_OUTPUTS_DIR, EDA_OUTPUTS_DIR):
        path = folder / file_name
        if path.exists():
            return path
    return None


@st.cache_data(show_spinner=False)
def load_csv(path: str) -> pd.DataFrame:
    """Load one CSV artifact with lightweight type normalization."""
    df = pd.read_csv(path)
    for col in df.columns:
        if col.lower().endswith(("pct", "rate", "score", "delay", "flights", "count")):
            converted = pd.to_numeric(df[col], errors="coerce")
            if converted.notna().sum() > 0:
                df[col] = converted
    return df


@st.cache_data(show_spinner=False)
def load_head(path: str, nrows: int = 5) -> pd.DataFrame:
    """Read only a small preview from a raw table."""
    return pd.read_csv(path, nrows=nrows)


@st.cache_data(show_spinner=False)
def load_text(path: str) -> str:
    """Load a text artifact such as an exported Plotly HTML chart."""
    return Path(path).read_text(encoding="utf-8")


@st.cache_data(show_spinner=False)
def csv_row_count(path: str) -> int | None:
    """Count records without loading the full flights table in memory."""
    try:
        with open(path, "rb") as file:
            rows = sum(chunk.count(b"\n")
                       for chunk in iter(lambda: file.read(1024 * 1024), b""))
        return max(rows - 1, 0)
    except OSError:
        return None


def load_artifacts() -> dict[str, pd.DataFrame]:
    """Load all available EDA outputs and skip missing optional files."""
    artifacts: dict[str, pd.DataFrame] = {}
    for key, file_name in OUTPUT_FILES.items():
        path = first_existing_path(file_name)
        if path is not None:
            artifacts[key] = load_csv(str(path))
    return artifacts


def pick_col(df: pd.DataFrame | None, candidates: Iterable[str]) -> str | None:
    """Return the first matching column, accepting case-insensitive variations."""
    if df is None or df.empty:
        return None
    by_lower = {col.lower(): col for col in df.columns}
    for candidate in candidates:
        found = by_lower.get(candidate.lower())
        if found:
            return found
    return None


def safe_number(value: object, default: float = 0) -> float:
    return float(pd.to_numeric(value, errors="coerce")) if pd.notna(value) else default


def format_int(value: float | int | None) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{int(value):,}".replace(",", ".")


def format_float(value: float | int | None, suffix: str = "", digits: int = 1) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return f"{float(value):,.{digits}f}{suffix}".replace(",", "X").replace(".", ",").replace("X", ".")


def format_pct(value: float | int | None, digits: int = 2) -> str:
    if value is None or pd.isna(value):
        return "n/d"
    return format_float(float(value) * 100, "%", digits)


def section_note(text: str) -> None:
    st.markdown(f"<div class='story-box'>{text}</div>", unsafe_allow_html=True)


def friendly_table(df: pd.DataFrame, max_rows: int = 30) -> pd.DataFrame:
    """Return a polished table preview while preserving the original dataframe."""
    preview = df.head(max_rows).copy()
    numeric_cols = preview.select_dtypes(include="number").columns
    for col in numeric_cols:
        if "pct" in col.lower() or "score" in col.lower() or "delay" in col.lower():
            preview[col] = preview[col].round(2)
    return preview


def show_unavailable(title: str, detail: str = "A tabela necessária não foi encontrada nos outputs da EDA.") -> None:
    st.info(f"{title}: {detail}")


def show_html_artifact(path: Path, height: int = 560) -> None:
    if not path.exists():
        st.info(f"Arquivo não encontrado: {path.name}")
        return
    components.html(load_text(str(path)), height=height, scrolling=True)


def estimated_confusion_matrix_from_metrics(
    metrics: pd.DataFrame,
    profile_values: dict[object, object],
) -> pd.DataFrame | None:
    required = {"precision", "recall", "roc_auc", "model"}
    if not required.issubset(metrics.columns):
        return None

    best = metrics.copy()
    best["roc_auc"] = pd.to_numeric(best["roc_auc"], errors="coerce")
    best["precision"] = pd.to_numeric(best["precision"], errors="coerce")
    best["recall"] = pd.to_numeric(best["recall"], errors="coerce")
    best = best.sort_values("roc_auc", ascending=False).iloc[0]

    test_rows = safe_number(profile_values.get("test_rows"), np.nan)
    delayed_rate = safe_number(
        profile_values.get("delayed_15_rate_sample"), np.nan)
    precision = safe_number(best["precision"], np.nan)
    recall = safe_number(best["recall"], np.nan)
    if any(pd.isna(value) or value <= 0 for value in [test_rows, delayed_rate, precision, recall]):
        return None

    actual_delayed = int(round(test_rows * delayed_rate))
    actual_on_time = int(round(test_rows - actual_delayed))
    true_positive = int(round(recall * actual_delayed))
    false_negative = max(actual_delayed - true_positive, 0)
    false_positive = int(round(true_positive / precision - true_positive))
    true_negative = max(actual_on_time - false_positive, 0)

    return pd.DataFrame(
        [[true_negative, false_positive], [false_negative, true_positive]],
        index=["Real: sem atraso", "Real: atrasado 15+"],
        columns=["Previsto: sem atraso", "Previsto: atrasado 15+"],
    )


def show_confusion_matrix_with_percentages(
    metrics: pd.DataFrame,
    profile_values: dict[object, object],
) -> None:
    matrix = estimated_confusion_matrix_from_metrics(metrics, profile_values)
    if matrix is None:
        st.info("Não foi possível estimar os percentuais da matriz de confusão com os arquivos disponíveis.")
        return

    total = matrix.to_numpy().sum()
    percent = matrix / total
    labels = matrix.astype(int).astype(str) + "<br>" + \
        percent.map(lambda value: format_pct(value, 2))

    if HAS_PLOTLY:
        fig = px.imshow(
            matrix,
            text_auto=False,
            color_continuous_scale="Blues",
            title="Matriz de confusão com valores e percentuais do teste",
            labels=dict(x="Classe prevista", y="Classe real", color="Voos"),
        )
        fig.update_traces(text=labels.to_numpy(), texttemplate="%{text}")
        fig.update_layout(height=430, margin=dict(
            l=10, r=10, t=55, b=10), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(matrix, use_container_width=True)

    st.caption(
        "Matriz complementar estimada a partir das métricas e do perfil do conjunto de teste exportados pelo treinamento."
    )


def horizontal_bar(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    color: str | None = None,
    hover_data: list[str] | None = None,
    height: int = 430,
) -> None:
    """Render a horizontal ranking with Plotly when available, with a Streamlit fallback."""
    if df.empty:
        st.info(
            "Não há dados suficientes para esta visualização com os filtros atuais.")
        return
    chart_df = df.copy().sort_values(x, ascending=True)
    if HAS_PLOTLY:
        fig = px.bar(
            chart_df,
            x=x,
            y=y,
            orientation="h",
            color=color,
            hover_data=hover_data,
            title=title,
            color_continuous_scale="Blues",
        )
        fig.update_layout(height=height, margin=dict(
            l=10, r=10, t=55, b=10), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(title)
        st.bar_chart(chart_df.set_index(y)[x])


def scatter_chart(
    df: pd.DataFrame,
    x: str,
    y: str,
    size: str,
    color: str,
    hover_name: str,
    title: str,
) -> None:
    if df.empty:
        st.info(
            "Não há dados suficientes para esta visualização com os filtros atuais.")
        return
    if HAS_PLOTLY:
        fig = px.scatter(
            df,
            x=x,
            y=y,
            size=size,
            color=color,
            hover_name=hover_name,
            hover_data=[size, color],
            title=title,
            color_continuous_scale="Tealrose",
            size_max=34,
        )
        fig.update_xaxes(type="log")
        fig.update_layout(height=500, margin=dict(
            l=10, r=10, t=55, b=10), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.subheader(title)
        st.scatter_chart(df[[x, y]])


def route_matrix(route_df: pd.DataFrame, min_flights: int, top_n: int) -> None:
    route_col = pick_col(route_df, ["ROUTE"])
    origin_col = pick_col(route_df, ["ORIGIN_AIRPORT"])
    dest_col = pick_col(route_df, ["DESTINATION_AIRPORT"])
    score_col = pick_col(route_df, ["delay_impact_score", "score"])
    flights_col = pick_col(route_df, ["flights", "flight_count"])
    if not all([route_col, origin_col, dest_col, score_col, flights_col]):
        show_unavailable("Matriz origem x destino",
                         "Colunas de rota, origem, destino, score ou volume não estão disponíveis.")
        return

    filtered = route_df[pd.to_numeric(
        route_df[flights_col], errors="coerce") >= min_flights].copy()
    filtered[score_col] = pd.to_numeric(filtered[score_col], errors="coerce")
    top_routes = filtered.nlargest(top_n, score_col)
    if top_routes.empty:
        st.info("Nenhuma rota atende ao volume mínimo selecionado.")
        return

    matrix = top_routes.pivot_table(
        index=origin_col,
        columns=dest_col,
        values=score_col,
        aggfunc="max",
    )
    if HAS_PLOTLY:
        fig = px.imshow(
            matrix,
            color_continuous_scale="YlOrRd",
            aspect="auto",
            title="Matriz origem x destino: score de criticidade das rotas mais relevantes",
        )
        fig.update_layout(height=560, margin=dict(
            l=10, r=10, t=55, b=10), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.dataframe(matrix, use_container_width=True)


def geographic_routes_map(route_df: pd.DataFrame, min_flights: int, top_n: int = 120) -> None:
    route_col = pick_col(route_df, ["ROUTE"])
    score_col = pick_col(route_df, ["delay_impact_score"])
    flights_col = pick_col(route_df, ["flights"])
    delay_col = pick_col(
        route_df, ["avg_arrival_delay", "avg_arrival_delay_positive"])
    origin_lat_col = pick_col(route_df, ["ORIGIN_LATITUDE"])
    origin_lon_col = pick_col(route_df, ["ORIGIN_LONGITUDE"])
    dest_lat_col = pick_col(route_df, ["DESTINATION_LATITUDE"])
    dest_lon_col = pick_col(route_df, ["DESTINATION_LONGITUDE"])
    required = [
        route_col,
        score_col,
        flights_col,
        delay_col,
        origin_lat_col,
        origin_lon_col,
        dest_lat_col,
        dest_lon_col,
    ]
    if not all(required):
        show_unavailable(
            "Mapa de rotas geográficas",
            "Colunas de rota, coordenadas, volume, atraso ou score não estão completas.",
        )
        return

    filtered = route_df[pd.to_numeric(
        route_df[flights_col], errors="coerce") >= min_flights].copy()
    for col in [score_col, flights_col, delay_col, origin_lat_col, origin_lon_col, dest_lat_col, dest_lon_col]:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
    top_routes = filtered.dropna(subset=required).nlargest(top_n, score_col)
    if top_routes.empty:
        st.info("Nenhuma rota atende ao volume mínimo selecionado.")
        return

    if not HAS_PLOTLY:
        st.dataframe(friendly_table(top_routes, top_n),
                     use_container_width=True, hide_index=True)
        return

    fig = go.Figure()
    score_min = top_routes[score_col].min()
    score_range = max(top_routes[score_col].max() - score_min, 1)

    for _, row in top_routes.iterrows():
        normalized_score = (row[score_col] - score_min) / score_range
        line_width = 1.5 + normalized_score * 5
        fig.add_trace(
            go.Scattergeo(
                lon=[row[origin_lon_col], row[dest_lon_col]],
                lat=[row[origin_lat_col], row[dest_lat_col]],
                mode="lines",
                line=dict(width=line_width, color="rgba(255, 99, 71, 0.72)"),
                hoverinfo="text",
                text=(
                    f"Rota: {row[route_col]}<br>"
                    f"Score: {row[score_col]:.2f}<br>"
                    f"Flights: {row[flights_col]:,.0f}<br>"
                    f"Atraso médio positivo: {row[delay_col]:.2f} min"
                ),
                showlegend=False,
            )
        )

    airport_points = pd.concat(
        [
            top_routes[[route_col, origin_lat_col, origin_lon_col, score_col]].rename(
                columns={origin_lat_col: "lat", origin_lon_col: "lon"}
            ),
            top_routes[[route_col, dest_lat_col, dest_lon_col, score_col]].rename(
                columns={dest_lat_col: "lat", dest_lon_col: "lon"}
            ),
        ],
        ignore_index=True,
    )
    fig.add_trace(
        go.Scattergeo(
            lon=airport_points["lon"],
            lat=airport_points["lat"],
            mode="markers",
            marker=dict(
                size=7,
                color=airport_points[score_col],
                colorscale="YlOrRd",
                line=dict(width=0.6, color="#ffffff"),
                colorbar=dict(title="Score"),
            ),
            hoverinfo="text",
            text=airport_points[route_col],
            showlegend=False,
        )
    )
    fig.update_geos(
        scope="usa",
        projection_type="albers usa",
        showland=True,
        landcolor="#182132",
        showocean=True,
        oceancolor="#0f1420",
        showlakes=True,
        lakecolor="#0f1420",
        bgcolor="#0f1420",
        subunitcolor="#344054",
        countrycolor="#344054",
    )
    fig.update_layout(
        title="Top 120 rotas geográficas por score de atraso",
        height=620,
        margin=dict(l=10, r=10, t=55, b=10),
        title_x=0.02,
        paper_bgcolor="#0f1420",
        plot_bgcolor="#0f1420",
        font=dict(color="#f4f7fb"),
    )
    st.plotly_chart(fig, use_container_width=True)


def route_bubble_chart(route_df: pd.DataFrame, min_flights: int, top_n: int = 120) -> None:
    route_col = pick_col(route_df, ["ROUTE"])
    flights_col = pick_col(route_df, ["flights"])
    delay_col = pick_col(
        route_df, ["avg_arrival_delay_positive", "avg_arrival_delay"])
    score_col = pick_col(route_df, ["delay_impact_score"])
    pct_col = pick_col(route_df, ["delayed_15_pct"])
    if not all([route_col, flights_col, delay_col, score_col]):
        show_unavailable(
            "Gráfico de bolhas",
            "Colunas de rota, flights, avg_arrival_delay_positive ou score não estão completas.",
        )
        return

    filtered = route_df[pd.to_numeric(
        route_df[flights_col], errors="coerce") >= min_flights].copy()
    for col in [flights_col, delay_col, score_col]:
        filtered[col] = pd.to_numeric(filtered[col], errors="coerce")
    top_routes = filtered.dropna(
        subset=[route_col, flights_col, delay_col, score_col]).nlargest(top_n, score_col)
    if top_routes.empty:
        st.info("Nenhuma rota atende ao volume mínimo selecionado.")
        return

    if HAS_PLOTLY:
        hover_cols = [pct_col] if pct_col else None
        fig = px.scatter(
            top_routes,
            x=flights_col,
            y=delay_col,
            size=score_col,
            color=score_col,
            hover_name=route_col,
            hover_data=hover_cols,
            title="Top 120 rotas: volume, atraso médio positivo e score",
            color_continuous_scale="YlOrRd",
            size_max=44,
            template="plotly_dark",
        )
        fig.update_layout(height=520, margin=dict(
            l=10, r=10, t=55, b=10), title_x=0.02)
        fig.update_xaxes(title="Flights")
        fig.update_yaxes(title="Avg arrival delay positive (min)")
        st.plotly_chart(fig, use_container_width=True)
    else:
        st.scatter_chart(top_routes[[flights_col, delay_col]])


def profile_summary(profile: pd.DataFrame | None) -> dict[str, float]:
    if profile is None or profile.empty:
        return {"columns": 0, "missing_cols": 0, "total_missing": 0, "max_missing_pct": 0}
    missing_count_col = pick_col(profile, ["missing_count"])
    missing_pct_col = pick_col(profile, ["missing_pct"])
    total_missing = pd.to_numeric(
        profile[missing_count_col], errors="coerce").sum() if missing_count_col else 0
    missing_cols = (pd.to_numeric(
        profile[missing_count_col], errors="coerce") > 0).sum() if missing_count_col else 0
    max_missing_pct = pd.to_numeric(
        profile[missing_pct_col], errors="coerce").max() if missing_pct_col else 0
    return {
        "columns": len(profile),
        "missing_cols": missing_cols,
        "total_missing": total_missing,
        "max_missing_pct": max_missing_pct,
    }


def apply_sidebar_filters(route_df: pd.DataFrame | None, origin_df: pd.DataFrame | None, destination_df: pd.DataFrame | None):
    st.sidebar.header("Filtros")
    min_flights = st.sidebar.slider(
        "Volume mínimo de voos por rota",
        min_value=1,
        max_value=5000,
        value=500,
        step=100,
        help="Evita conclusões baseadas em rotas com pouco volume.",
    )
    top_n = st.sidebar.slider("Quantidade no ranking", 5, 30, 15, 5)

    airport_options = sorted(
        set(route_df[pick_col(route_df, ["ORIGIN_AIRPORT"])].dropna().astype(
            str)) if route_df is not None and pick_col(route_df, ["ORIGIN_AIRPORT"]) else []
    )
    selected_airports = st.sidebar.multiselect(
        "Filtrar aeroportos de origem", airport_options)

    if route_df is not None and selected_airports:
        origin_col = pick_col(route_df, ["ORIGIN_AIRPORT"])
        route_df = route_df[route_df[origin_col].astype(
            str).isin(selected_airports)]
    if origin_df is not None and selected_airports:
        origin_col = pick_col(origin_df, ["ORIGIN_AIRPORT"])
        if origin_col:
            origin_df = origin_df[origin_df[origin_col].astype(
                str).isin(selected_airports)]
    if destination_df is not None and selected_airports:
        dest_col = pick_col(destination_df, ["DESTINATION_AIRPORT"])
        if dest_col:
            destination_df = destination_df[destination_df[dest_col].astype(
                str).isin(selected_airports)]

    st.sidebar.caption(
        "Os filtros afetam principalmente rankings de rotas e aeroportos.")
    return min_flights, top_n, route_df, origin_df, destination_df


def data_understanding_tab(artifacts: dict[str, pd.DataFrame]) -> None:
    st.header("Qualidade dos dados antes da limpeza")
    section_note(
        "Esta etapa responde como os dados chegaram: tamanho das bases, tipos, nulos, cardinalidade e inconsistências de código de aeroporto."
    )

    rows = []
    for table_name, path in RAW_TABLES.items():
        if not path.exists():
            rows.append({"base": table_name, "linhas": None,
                        "colunas": None, "arquivo": "não encontrado"})
            continue
        head = load_head(str(path), 1)
        rows.append(
            {
                "base": table_name,
                "linhas": csv_row_count(str(path)),
                "colunas": len(head.columns),
                "arquivo": path.name,
            }
        )
    st.dataframe(pd.DataFrame(rows), use_container_width=True, hide_index=True)

    c1, c2, c3 = st.columns(3)
    invalid_df = artifacts.get("invalid_airport_codes")
    invalid_count = len(invalid_df) if invalid_df is not None else None
    c1.metric("Registros com código de aeroporto inválido",
              format_int(invalid_count))
    c2.metric("Bases cruas avaliadas", format_int(sum(path.exists()
              for path in RAW_TABLES.values())))
    c3.metric("Perfis de colunas disponíveis", format_int(
        sum(key.endswith("before") for key in artifacts)))

    st.subheader("Perfis antes do tratamento")
    profile_tabs = st.tabs(
        ["Airlines", "Airports", "Flights", "Join completo"])
    profile_map = [
        ("airlines_profile_before", "Perfil da tabela airlines"),
        ("airports_profile_before", "Perfil da tabela airports"),
        ("flights_profile_before", "Perfil da tabela flights"),
        ("completed_profile_before",
         "Perfil após join e filtro de voos completos, antes do tratamento"),
    ]
    for tab, (key, label) in zip(profile_tabs, profile_map):
        with tab:
            df = artifacts.get(key)
            if df is None:
                show_unavailable(label)
                continue
            summary = profile_summary(df)
            m1, m2, m3, m4 = st.columns(4)
            m1.metric("Colunas", format_int(summary["columns"]))
            m2.metric("Colunas com nulos", format_int(summary["missing_cols"]))
            m3.metric("Nulos totais", format_int(summary["total_missing"]))
            m4.metric("Maior % nulo", format_float(
                summary["max_missing_pct"], "%", 2))
            st.dataframe(friendly_table(df, 25),
                         use_container_width=True, hide_index=True)

    st.subheader("Inconsistencia especifica: IATA com mais de 3 caracteres")
    if invalid_df is not None:
        st.write(
            "Na base de voos, códigos de origem ou destino com mais de 3 caracteres indicavam `AIRPORT_ID` numérico em vez de código IATA. "
            "Esses registros foram isolados e depois corrigidos por cruzamento com uma tabela externa de mapeamento."
        )
        st.dataframe(friendly_table(invalid_df, 20),
                     use_container_width=True, hide_index=True)
    else:
        show_unavailable(
            "Códigos inválidos", "Não encontrei `flights_with_invalid_airport_codes.csv`.")


def prep_tab(artifacts: dict[str, pd.DataFrame]) -> None:
    st.header("Data Prep: transformando dados brutos em dados analíticos")
    section_note(
        "O notebook prepara uma base confiável para análise de atraso: corrige aeroportos, enriquece os voos com nomes e coordenadas, filtra voos concluídos e trata nulos de causas de atraso."
    )

    steps = [
        ("Coordenadas de aeroportos",
         "Preenchimento manual das coordenadas ausentes para ECP, PBG e UST."),
        ("Códigos IATA inválidos", "Identificação de origem/destino com mais de 3 caracteres e conversão de `AIRPORT_ID` para IATA com tabela BTS de outubro de 2015."),
        ("Join das tabelas", "Voos foram unidos a companhias aéreas e aeroportos de origem/destino para criar nomes, cidades, estados e coordenadas."),
        ("Colunas criadas", "`FLIGHT_DATE`, `SCHEDULED_DEPARTURE_HOUR`, `SCHEDULED_ARRIVAL_HOUR`, `ROUTE`, `IS_WEEKEND`, `ARRIVAL_DELAY_POSITIVE` e `IS_DELAYED_15`."),
        ("Recorte analítico", "A análise de atraso usa voos finalizados: `CANCELLED = 0`, `DIVERTED = 0` e `ARRIVAL_DELAY` não nulo."),
        ("Tratamento de nulos",
         "Causas de atraso nulas foram preenchidas com 0 e `TAIL_NUMBER` nulo recebeu `UNKNOWN`."),
        ("Outliers", "Atrasos extremos foram mantidos porque representam eventos reais e importantes para operação."),
    ]
    st.dataframe(pd.DataFrame(steps, columns=[
                 "Bloco", "Tratamento aplicado"]), use_container_width=True, hide_index=True)

    before = artifacts.get("completed_profile_before")
    after = artifacts.get("completed_profile_after")
    st.subheader("Antes x depois do tratamento")
    if before is None or after is None:
        show_unavailable("Comparação antes x depois",
                         "Perfis antes/depois do tratamento não estão completos.")
        return

    col_col = pick_col(before, ["column"])
    if not col_col:
        show_unavailable("Comparação antes x depois",
                         "Não encontrei a coluna de nome dos campos.")
        return

    after_col_col = pick_col(after, ["column"])
    missing_before = pick_col(before, ["missing_pct"])
    missing_after = pick_col(after, ["missing_pct"])
    count_before = pick_col(before, ["missing_count"])
    count_after = pick_col(after, ["missing_count"])
    required = [col_col, after_col_col, missing_before,
                missing_after, count_before, count_after]
    if not all(required):
        show_unavailable("Comparação antes x depois",
                         "Colunas de missing não estão completas nos perfis.")
        return

    before_compare = before[[col_col, missing_before, count_before]].rename(
        columns={
            col_col: "column",
            missing_before: "missing_pct_antes",
            count_before: "missing_count_antes",
        }
    )
    after_compare = after[[after_col_col, missing_after, count_after]].rename(
        columns={
            after_col_col: "column",
            missing_after: "missing_pct_depois",
            count_after: "missing_count_depois",
        }
    )
    merged = before_compare.merge(after_compare, on="column", how="outer")
    merged["reducao_missing_pct"] = (
        pd.to_numeric(merged["missing_pct_antes"], errors="coerce").fillna(0)
        - pd.to_numeric(merged["missing_pct_depois"],
                        errors="coerce").fillna(0)
    )
    merged = merged.sort_values("reducao_missing_pct", ascending=False)

    c1, c2, c3 = st.columns(3)
    c1.metric("Missing antes", format_int(pd.to_numeric(
        merged["missing_count_antes"], errors="coerce").sum()))
    c2.metric("Missing depois", format_int(pd.to_numeric(
        merged["missing_count_depois"], errors="coerce").sum()))
    c3.metric("Maior redução em p.p.", format_float(
        merged["reducao_missing_pct"].max(), " p.p.", 2))

    top_missing = merged.head(12).dropna(subset=["reducao_missing_pct"])
    horizontal_bar(
        top_missing,
        x="reducao_missing_pct",
        y="column",
        title="Maiores reduções de missing após o tratamento",
        height=430,
    )
    st.dataframe(friendly_table(merged, 30),
                 use_container_width=True, hide_index=True)


def overview_tab(artifacts: dict[str, pd.DataFrame]) -> None:
    st.header("US Flight Delay Analytics")
    st.markdown(
        """
        Este dashboard resume a EDA de atrasos em voos nos EUA. A pergunta central e:
        **quais aeroportos, rotas e características estão mais associados a atrasos em voos?**
        """
    )
    section_note(
        "A história segue quatro passos: entender a qualidade dos dados, explicar os tratamentos, analisar atrasos por rota/aeroporto e fechar com recomendações operacionais."
    )

    route_df = artifacts.get("route_delay")
    origin_df = artifacts.get("origin_delay")
    destination_df = artifacts.get("destination_delay")
    before = artifacts.get("completed_profile_before")
    after = artifacts.get("completed_profile_after")
    invalid_df = artifacts.get("invalid_airport_codes")

    total_routes = len(route_df) if route_df is not None else None
    total_airports = len(origin_df) if origin_df is not None else None
    invalid_count = len(invalid_df) if invalid_df is not None else None
    missing_before = profile_summary(
        before)["total_missing"] if before is not None else None
    missing_after = profile_summary(
        after)["total_missing"] if after is not None else None

    cols = st.columns(5)
    cols[0].metric("Rotas analisadas", format_int(total_routes))
    cols[1].metric("Aeroportos de origem", format_int(total_airports))
    cols[2].metric("Códigos inválidos corrigidos", format_int(invalid_count))
    cols[3].metric("Missing antes", format_int(missing_before))
    cols[4].metric("Missing depois", format_int(missing_after))

    st.subheader("Principais achados em uma tela")
    c1, c2 = st.columns(2)
    with c1:
        if origin_df is not None:
            score_col = pick_col(
                origin_df, ["origin_delay_impact_score", "delay_impact_score"])
            code_col = pick_col(origin_df, ["ORIGIN_AIRPORT"])
            name_col = pick_col(origin_df, ["ORIGIN_AIRPORT_NAME"])
            top = origin_df.sort_values(score_col, ascending=False).head(
                1) if score_col else pd.DataFrame()
            if not top.empty:
                st.metric("Origem mais crítica por score", str(
                    top.iloc[0][code_col]), format_float(top.iloc[0][score_col], " pts", 2))
                st.caption(str(top.iloc[0].get(name_col, "")))
        else:
            show_unavailable("Origem crítica")
    with c2:
        if route_df is not None:
            score_col = pick_col(route_df, ["delay_impact_score"])
            route_col = pick_col(route_df, ["ROUTE"])
            top = route_df.sort_values(score_col, ascending=False).head(
                1) if score_col else pd.DataFrame()
            if not top.empty:
                st.metric("Rota mais crítica por score", str(
                    top.iloc[0][route_col]), format_float(top.iloc[0][score_col], " pts", 2))
                st.caption(
                    "Score combina taxa ajustada de atraso, volume e atraso médio positivo.")
        else:
            show_unavailable("Rota crítica")

    if route_df is not None:
        score_col = pick_col(route_df, ["delay_impact_score"])
        route_col = pick_col(route_df, ["ROUTE"])
        flights_col = pick_col(route_df, ["flights"])
        delay_col = pick_col(route_df, ["avg_arrival_delay"])
        pct_col = pick_col(route_df, ["delayed_15_pct", "adjusted_delay_pct"])
        if all([score_col, route_col, flights_col, delay_col]):
            horizontal_bar(
                route_df.nlargest(12, score_col),
                x=score_col,
                y=route_col,
                title="Rotas com maior criticidade geral",
                color=pct_col,
                hover_data=[flights_col, delay_col, pct_col] if pct_col else [
                    flights_col, delay_col],
            )


def insights_tab(
    route_df: pd.DataFrame | None,
    origin_df: pd.DataFrame | None,
    destination_df: pd.DataFrame | None,
    min_flights: int,
    top_n: int,
) -> None:
    st.header("Insights de atrasos")
    section_note(
        "Aqui os rankings separam volume e atraso. O filtro de volume mínimo reduz o risco de uma rota rara parecer crítica apenas por poucos eventos extremos."
    )

    airport_tabs = st.tabs(["Origem", "Destino", "Rotas", "Volume x atraso"])

    with airport_tabs[0]:
        if origin_df is None:
            show_unavailable("Aeroportos de origem")
        else:
            score_col = pick_col(
                origin_df, ["origin_delay_impact_score", "delay_impact_score"])
            code_col = pick_col(origin_df, ["ORIGIN_AIRPORT"])
            flights_col = pick_col(origin_df, ["flights"])
            delay_col = pick_col(origin_df, ["avg_arrival_delay"])
            pct_col = pick_col(origin_df, ["delayed_15_pct"])
            if all([score_col, code_col, flights_col, delay_col, pct_col]):
                filtered = origin_df[pd.to_numeric(
                    origin_df[flights_col], errors="coerce") >= min_flights]
                horizontal_bar(
                    filtered.nlargest(top_n, score_col),
                    x=score_col,
                    y=code_col,
                    color=pct_col,
                    hover_data=[flights_col, delay_col, pct_col],
                    title="Aeroportos de origem com maior score de criticidade",
                )
                st.caption(
                    "Score = 60% taxa ajustada de atraso + 25% volume normalizado + 15% atraso médio positivo normalizado.")
                st.dataframe(friendly_table(filtered.nlargest(
                    top_n, score_col)), use_container_width=True, hide_index=True)
            else:
                show_unavailable(
                    "Aeroportos de origem", "Colunas de score, volume ou atraso não estão completas.")

    with airport_tabs[1]:
        if destination_df is None:
            show_unavailable("Aeroportos de destino")
        else:
            score_col = pick_col(
                destination_df, ["destination_delay_impact_score", "delay_impact_score"])
            code_col = pick_col(destination_df, ["DESTINATION_AIRPORT"])
            flights_col = pick_col(destination_df, ["flights"])
            delay_col = pick_col(destination_df, ["avg_arrival_delay"])
            pct_col = pick_col(destination_df, ["delayed_15_pct"])
            if all([score_col, code_col, flights_col, delay_col, pct_col]):
                filtered = destination_df[pd.to_numeric(
                    destination_df[flights_col], errors="coerce") >= min_flights]
                horizontal_bar(
                    filtered.nlargest(top_n, score_col),
                    x=score_col,
                    y=code_col,
                    color=pct_col,
                    hover_data=[flights_col, delay_col, pct_col],
                    title="Aeroportos de destino com maior score de criticidade",
                )
                st.dataframe(friendly_table(filtered.nlargest(
                    top_n, score_col)), use_container_width=True, hide_index=True)
            else:
                show_unavailable(
                    "Aeroportos de destino", "Colunas de score, volume ou atraso não estão completas.")

    with airport_tabs[2]:
        if route_df is None:
            show_unavailable("Rotas")
        else:
            score_col = pick_col(route_df, ["delay_impact_score"])
            route_col = pick_col(route_df, ["ROUTE"])
            flights_col = pick_col(route_df, ["flights"])
            delay_col = pick_col(route_df, ["avg_arrival_delay"])
            pct_col = pick_col(route_df, ["delayed_15_pct"])
            if all([score_col, route_col, flights_col, delay_col, pct_col]):
                filtered = route_df[pd.to_numeric(
                    route_df[flights_col], errors="coerce") >= min_flights]
                c1, c2 = st.columns(2)
                with c1:
                    horizontal_bar(
                        filtered.nlargest(top_n, score_col),
                        x=score_col,
                        y=route_col,
                        color=pct_col,
                        hover_data=[flights_col, delay_col, pct_col],
                        title="Rotas mais críticas por score",
                    )
                with c2:
                    horizontal_bar(
                        filtered.nlargest(top_n, delay_col),
                        x=delay_col,
                        y=route_col,
                        color=flights_col,
                        hover_data=[flights_col, score_col, pct_col],
                        title="Rotas com maior atraso médio positivo",
                    )
                st.dataframe(friendly_table(filtered.nlargest(
                    top_n, score_col)), use_container_width=True, hide_index=True)
            else:
                show_unavailable(
                    "Rotas", "Colunas de rota, score, volume ou atraso não estão completas.")

    with airport_tabs[3]:
        if route_df is None:
            show_unavailable("Volume x atraso")
        else:
            route_col = pick_col(route_df, ["ROUTE"])
            flights_col = pick_col(route_df, ["flights"])
            delay_col = pick_col(route_df, ["avg_arrival_delay"])
            score_col = pick_col(route_df, ["delay_impact_score"])
            pct_col = pick_col(route_df, ["delayed_15_pct"])
            if all([route_col, flights_col, delay_col, score_col]):
                filtered = route_df[pd.to_numeric(
                    route_df[flights_col], errors="coerce") >= min_flights]
                scatter_chart(
                    filtered,
                    x=flights_col,
                    y=delay_col,
                    size=score_col,
                    color=pct_col or score_col,
                    hover_name=route_col,
                    title="Relação entre volume de voos e atraso médio",
                )
                st.caption(
                    "Rotas no canto superior direito combinam alto volume e atraso médio relevante.")
            else:
                show_unavailable(
                    "Volume x atraso", "Colunas de volume, atraso ou score não estão completas.")


def routes_airports_tab(route_df: pd.DataFrame | None, min_flights: int, top_n: int) -> None:
    st.header("Rotas e aeroportos criticos")
    section_note(
        "Esta aba foca no mapa analítico origem x destino e em uma leitura operacional: quais conexões merecem investigação primeiro."
    )
    if route_df is None:
        show_unavailable("Rotas e aeroportos")
        return

    geographic_routes_map(route_df, min_flights=min_flights, top_n=120)
    route_bubble_chart(route_df, min_flights=min_flights, top_n=120)

    route_col = pick_col(route_df, ["ROUTE"])
    flights_col = pick_col(route_df, ["flights"])
    delay_col = pick_col(
        route_df, ["avg_arrival_delay_positive", "avg_arrival_delay"])
    pct_col = pick_col(route_df, ["delayed_15_pct"])
    score_col = pick_col(route_df, ["delay_impact_score"])
    if all([route_col, flights_col, delay_col, pct_col, score_col]):
        filtered = route_df[pd.to_numeric(
            route_df[flights_col], errors="coerce") >= min_flights].copy()
        filtered["operational_quadrant"] = np.select(
            [
                (filtered[flights_col] >= filtered[flights_col].median()) & (
                    filtered[delay_col] >= filtered[delay_col].median()),
                (filtered[flights_col] >= filtered[flights_col].median()),
                (filtered[delay_col] >= filtered[delay_col].median()),
            ],
            ["Alto volume e alto atraso", "Alto volume", "Alto atraso"],
            default="Monitorar",
        )
        st.subheader("Rotas de alto volume e alto atraso")
        priority = filtered[filtered["operational_quadrant"] ==
                            "Alto volume e alto atraso"].nlargest(top_n, score_col)
        st.dataframe(friendly_table(priority, top_n),
                     use_container_width=True, hide_index=True)


def distributions_tab(artifacts: dict[str, pd.DataFrame]) -> None:
    st.header("Distribuições e variáveis numéricas")
    section_note(
        "A EDA manteve atrasos extremos porque eles representam eventos reais. Para leitura visual, os histogramas também foram gerados com percentis p01 e p99."
    )

    numeric = artifacts.get("numeric_profile")
    if numeric is not None:
        col_col = pick_col(numeric, ["column"])
        outlier_col = pick_col(numeric, ["outlier_pct_iqr"])
        max_col = pick_col(numeric, ["max"])
        p99_col = pick_col(numeric, ["99%"])
        if all([col_col, outlier_col]):
            top_outliers = numeric.sort_values(
                outlier_col, ascending=False).head(12)
            horizontal_bar(
                top_outliers,
                x=outlier_col,
                y=col_col,
                title="Variáveis numéricas com maior proporção de outliers pelo IQR",
                hover_data=[max_col, p99_col] if max_col and p99_col else None,
            )
        st.dataframe(friendly_table(numeric, 25),
                     use_container_width=True, hide_index=True)
    else:
        show_unavailable("Perfil numerico")

    image_cols = st.columns(2)
    image_files = [
        ("Histogramas com p01-p99", "completed_numeric_distributions.png"),
        ("Histogramas sem percentis", "completed_numeric_distributions_raw.png"),
    ]
    for col, (caption, image_name) in zip(image_cols, image_files):
        path = first_existing_path(image_name)
        with col:
            if path:
                st.image(str(path), caption=caption, use_container_width=True)
            else:
                st.info(f"{caption}: imagem não encontrada.")


def ml_supervised_tab() -> None:
    st.header("Machine Learning: modelo supervisionado")
    section_note(
        "Esta etapa treina modelos de classificação para prever se um voo terá atraso de 15 minutos ou mais. O cenário modelado é após a decolagem, usando informações já conhecidas na operação."
    )

    metrics_path = ML_OUTPUTS_DIR / "supervised_classification_metrics.csv"
    profile_path = ML_OUTPUTS_DIR / "supervised_sample_profile.csv"
    if not metrics_path.exists() or not profile_path.exists():
        show_unavailable(
            "Resultados supervisionados",
            "Execute o script de treinamento para gerar supervised_classification_metrics.csv e supervised_sample_profile.csv.",
        )
        return

    metrics = load_csv(str(metrics_path))
    profile = load_csv(str(profile_path))
    profile_values = dict(zip(profile["metric"], profile["value"]))

    st.subheader("Base usada no treinamento")
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Usou amostra", str(profile_values.get("use_sample", "n/d")))
    c2.metric("Linhas usadas", format_int(
        safe_number(profile_values.get("supervised_rows"))))
    c3.metric("Treino", format_int(
        safe_number(profile_values.get("train_rows"))))
    c4.metric("Teste", format_int(
        safe_number(profile_values.get("test_rows"))))
    st.caption(
        f"Taxa de atraso 15+ na base usada: {format_pct(safe_number(profile_values.get('delayed_15_rate_sample')), 2)}."
    )

    st.subheader("Escopo e variáveis")
    st.markdown(
        """
        O target é `IS_DELAYED_15`, em que `1` indica atraso de 15 minutos ou mais.

        Foram comparados dois modelos: `LogisticRegression` e `RandomForestClassifier`.
        As variáveis numéricas e categóricas vieram da base tratada na etapa de Data Prep.
        O pipeline aplica imputação, escala numérica, one-hot encoding e agrupamento de categorias raras.

        Variáveis finais como `ARRIVAL_DELAY` e causas consolidadas de atraso não foram usadas para evitar vazamento direto do target.
        """
    )

    st.subheader("Comparação dos modelos")
    display_metrics = metrics.copy()
    metric_cols = ["accuracy", "precision", "recall", "f1", "roc_auc"]
    for col in metric_cols:
        display_metrics[col] = pd.to_numeric(display_metrics[col], errors="coerce")
    st.dataframe(
        display_metrics[["model", *metric_cols]].round(4),
        use_container_width=True,
        hide_index=True,
    )

    best = display_metrics.sort_values("roc_auc", ascending=False).iloc[0]
    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Melhor modelo", str(best["model"]))
    c2.metric("ROC AUC", format_float(best["roc_auc"], digits=4))
    c3.metric("F1", format_float(best["f1"], digits=4))
    c4.metric("Recall", format_float(best["recall"], digits=4))

    if HAS_PLOTLY:
        long_metrics = display_metrics.melt(
            id_vars="model",
            value_vars=metric_cols,
            var_name="metric",
            value_name="value",
        )
        fig = px.bar(
            long_metrics,
            x="metric",
            y="value",
            color="model",
            barmode="group",
            title="Métricas supervisionadas por modelo",
            template="plotly_dark",
        )
        fig.update_layout(height=430, margin=dict(
            l=10, r=10, t=55, b=10), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("Gráficos exportados")
    chart_tabs = st.tabs(["Curva ROC", "Matriz de confusão"])
    with chart_tabs[0]:
        st.markdown(
            "A curva ROC mostra a capacidade do modelo de separar voos atrasados e não atrasados em diferentes limiares."
        )
        show_html_artifact(ML_OUTPUTS_DIR / "supervised_roc_curve.html", 620)
    with chart_tabs[1]:
        st.markdown(
            "A matriz de confusão mostra acertos e erros da classificação, com valores absolutos e percentuais sobre o conjunto de teste."
        )
        show_confusion_matrix_with_percentages(metrics, profile_values)

    st.subheader("Interpretação")
    st.markdown(
        """
        A `LogisticRegression` foi selecionada porque apresentou o melhor desempenho geral.
        Ela superou a Random Forest em `accuracy`, `precision`, `recall`, `f1` e `roc_auc`.

        A leitura correta é que este modelo performa bem no cenário operacional após a decolagem.
        Para uma previsão antes da partida, seria necessário criar outro experimento removendo também
        `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF`.
        """
    )


def ml_unsupervised_tab() -> None:
    st.header("Machine Learning: modelo não supervisionado")
    section_note(
        "A clusterização agrupa rotas com perfis semelhantes de volume, atraso e impacto. Aqui não existe target: o algoritmo encontra estruturas internas nos dados."
    )

    summary_path = ML_OUTPUTS_DIR / "unsupervised_cluster_summary.csv"
    routes_path = ML_OUTPUTS_DIR / "unsupervised_route_clusters.csv"
    if not summary_path.exists() or not routes_path.exists():
        show_unavailable(
            "Resultados não supervisionados",
            "Execute o script de treinamento para gerar os arquivos de clusterização.",
        )
        return

    summary = load_csv(str(summary_path))
    routes = load_csv(str(routes_path))

    silhouette = pd.to_numeric(
        summary["silhouette_score"], errors="coerce").dropna()
    pca_variance = pd.to_numeric(
        summary["pca_explained_variance"], errors="coerce").dropna()

    st.subheader("Como a clusterização foi feita")
    st.markdown(
        """
        A unidade analisada foi a rota, não o voo individual. Antes do KMeans, foram consideradas apenas rotas com pelo menos 500 voos para reduzir ruído estatístico.

        O algoritmo usado foi `KMeans(n_clusters=4)`. As features foram padronizadas com `StandardScaler`, porque o KMeans calcula distância entre pontos.

        O PCA com 2 componentes foi usado para visualização dos clusters em um plano 2D.
        """
    )

    st.markdown(
        """
        **Como interpretar as métricas abaixo**

        - `Silhouette`: mede o quanto as rotas estão próximas das rotas do próprio cluster e distantes das rotas dos outros clusters. O valor vai de -1 a 1. Quanto mais perto de 1, melhor a separação. Valores perto de 0 indicam sobreposição. No projeto, o valor em torno de 0,35 indica separação moderada.
        - `Variância PCA 2D`: mostra quanto da informação original foi preservada quando reduzimos as features para dois eixos visuais. O valor de aproximadamente 96,70% indica que o gráfico em 2D representa bem a estrutura geral dos dados usados na clusterização.
        """
    )

    c1, c2, c3 = st.columns(3)
    c1.metric("Clusters", format_int(summary["cluster"].nunique()))
    c2.metric("Silhouette", format_float(
        silhouette.iloc[0] if not silhouette.empty else None, digits=4))
    c3.metric("Variância PCA 2D", format_pct(
        pca_variance.iloc[0] if not pca_variance.empty else None, 2))

    st.subheader("Resumo dos clusters")
    display_summary = summary.copy()
    numeric_cols = display_summary.select_dtypes(include="number").columns
    display_summary[numeric_cols] = display_summary[numeric_cols].round(4)
    st.dataframe(display_summary, use_container_width=True, hide_index=True)

    if HAS_PLOTLY:
        fig = px.bar(
            summary.sort_values("avg_delay_impact_score", ascending=False),
            x="cluster",
            y="avg_delay_impact_score",
            color="avg_delayed_15_pct",
            text="routes",
            title="Impacto médio de atraso por cluster",
            template="plotly_dark",
            color_continuous_scale="YlOrRd",
        )
        fig.update_layout(height=450, margin=dict(
            l=10, r=10, t=55, b=10), title_x=0.02)
        st.plotly_chart(fig, use_container_width=True)

    st.subheader("O que cada cluster representa")
    st.markdown(
        """
        - **Cluster 3:** rotas de alto volume e alto impacto. Tem a maior média de voos por rota e o maior score médio de impacto. É o grupo mais importante para priorização operacional.
        - **Cluster 1:** rotas com maior atraso médio positivo e maior percentual médio de atraso de 15+ minutos. Mesmo com volume menor que o cluster 3, concentra maior severidade relativa de atraso.
        - **Cluster 0:** rotas intermediárias. Apresenta volume, atraso e score em uma faixa média, servindo como grupo de comparação operacional.
        - **Cluster 2:** rotas menos críticas dentro do recorte analisado. Possui menor atraso médio positivo, menor percentual de atraso e menor score médio de impacto.
        """
    )

    st.subheader("Gráficos exportados")
    st.markdown(
        "Cada ponto representa uma rota. A cor indica o cluster e o tamanho indica o volume de voos."
    )
    show_html_artifact(
        ML_OUTPUTS_DIR / "unsupervised_route_clusters_pca.html", 650)

    st.subheader("Rotas por cluster")
    cluster_options = sorted(routes["cluster"].dropna().unique().tolist())
    selected_cluster = st.selectbox("Selecionar cluster", cluster_options)
    cluster_routes = routes[routes["cluster"] == selected_cluster].copy()
    score_col = pick_col(cluster_routes, ["delay_impact_score"])
    if score_col:
        cluster_routes = cluster_routes.sort_values(score_col, ascending=False)
    st.dataframe(
        friendly_table(cluster_routes, 20),
        use_container_width=True,
        hide_index=True,
    )

    st.subheader("Interpretação")
    st.markdown(
        """
        O `silhouette_score` de aproximadamente 0,35 indica separação moderada.
        Isso significa que os clusters não estão perfeitamente isolados, mas possuem estrutura útil para segmentação.

        A clusterização é mais adequada como ferramenta de leitura operacional:

        - identificar rotas de alto volume e alto impacto;
        - separar rotas com maior severidade relativa de atraso;
        - priorizar investigações em grupos de rotas semelhantes;
        - apoiar comunicação executiva sobre perfis de risco.
        """
    )


def ml_critical_tab() -> None:
    st.header("Machine Learning: conclusões, limites e próximos passos")
    section_note(
        "Esta aba resume a leitura crítica dos modelos. O objetivo é separar conclusões fortes, cuidados de interpretação e próximas evoluções do projeto."
    )

    st.subheader("Principais conclusões")
    st.markdown(
        """
        - A `LogisticRegression` foi o melhor modelo supervisionado entre os algoritmos comparados.
        - O resultado supervisionado foi forte, com alto `ROC AUC` e bom equilíbrio entre `precision` e `recall`.
        - O desempenho deve ser interpretado dentro do escopo correto: predição após a decolagem.
        - A clusterização encontrou grupos de rotas interpretáveis por volume, atraso e impacto.
        - O `silhouette_score` indica separação moderada, suficiente para segmentação exploratória, mas não para afirmar que os grupos são perfeitamente separados.
        """
    )

    st.subheader("Estudo de outliers")
    outlier_metrics_path = OUTLIER_OUTPUTS_DIR / "outlier_strategy_metrics.csv"
    outlier_tail_path = OUTLIER_OUTPUTS_DIR / "outlier_tail_diagnostics.csv"
    if outlier_metrics_path.exists() and outlier_tail_path.exists():
        outlier_metrics = load_csv(str(outlier_metrics_path))
        outlier_tail = load_csv(str(outlier_tail_path))
        st.markdown(
            "O estudo de outliers comparou manter extremos, winsorizar e remover linhas extremas. A melhor estratégia foi manter os valores extremos."
        )
        st.dataframe(
            outlier_metrics.round(4),
            use_container_width=True,
            hide_index=True,
        )

        important_tail = outlier_tail[
            outlier_tail["feature"].isin(["DEPARTURE_DELAY", "TAXI_OUT", "WHEELS_OFF", "DISTANCE"])
        ].copy()
        st.markdown(
            "A leitura das caudas mostrou que extremos de `DEPARTURE_DELAY` e `TAXI_OUT` carregam forte sinal preditivo."
        )
        st.dataframe(
            important_tail.round(4),
            use_container_width=True,
            hide_index=True,
        )
    else:
        st.info("Arquivos do estudo de outliers não encontrados.")

    st.subheader("Limitações")
    st.markdown(
        """
        - O modelo supervisionado atual não é uma previsão pré-voo.
        - Variáveis como `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF` aumentam bastante o poder preditivo porque já descrevem a operação em andamento.
        - A base não inclui clima histórico detalhado por aeroporto e horário.
        - Não há dados de manutenção, tripulação, conexões, capacidade aeroportuária ou malha operacional.
        - A divisão supervisionada é aleatória; uma validação temporal seria mais rigorosa para simular uso futuro.
        - Na clusterização, o número de clusters foi fixado em 4; outros valores de `k` ainda podem ser avaliados.
        """
    )

    st.subheader("Melhorias e próximos passos")
    st.markdown(
        """
        **Modelo supervisionado**

        - Criar uma versão pré-voo removendo variáveis conhecidas somente após o início da operação.
        - Testar validação temporal, treinando em meses anteriores e testando em meses posteriores.
        - Avaliar modelos adicionais, como Gradient Boosting, XGBoost, LightGBM ou HistGradientBoosting.
        - Calibrar o limiar de classificação conforme o objetivo: maior recall ou maior precision.
        - Usar interpretabilidade com permutation importance ou SHAP.

        **Modelo não supervisionado**

        - Comparar diferentes valores de `k`.
        - Avaliar estabilidade dos clusters ao longo do tempo.
        - Testar clusters por aeroporto, companhia, estado ou período do dia.
        - Validar os grupos com conhecimento operacional do domínio.
        """
    )


def conclusions_tab() -> None:
    st.header("Conclusões")
    st.markdown(
        """
        **O que aprendemos**

        1. A qualidade dos dados precisava ser entendida antes dos rankings: cancelamentos, desvios, causas de atraso nulas e codigos de aeroporto numericos poderiam distorcer a leitura.
        2. O recorte analítico mais confiável usa voos finalizados, com `ARRIVAL_DELAY` conhecido e enriquecimento de aeroporto/companhia.
        3. Ranking por atraso médio isolado não basta. O score combina volume, taxa ajustada de atraso e atraso médio positivo para priorizar rotas e aeroportos operacionalmente relevantes.
        4. Outliers de atraso não foram descartados, pois no contexto aéreo eles são eventos reais e importantes para a gestão.

        **Como usar o dashboard**

        Ajuste o volume mínimo na barra lateral e observe quais rotas continuam aparecendo no topo. Se uma rota permanece crítica mesmo com alto volume mínimo, ela merece investigação operacional mais forte.
        """
    )


def main() -> None:
    artifacts = load_artifacts()
    route_df = artifacts.get("route_delay")
    origin_df = artifacts.get("origin_delay")
    destination_df = artifacts.get("destination_delay")
    min_flights, top_n, route_df, origin_df, destination_df = apply_sidebar_filters(
        route_df, origin_df, destination_df)

    st.sidebar.markdown("---")
    st.sidebar.caption(f"Fonte principal: `{DATABASE_DIR}`")
    st.sidebar.caption("Artefatos carregados: " + str(len(artifacts)))

    tabs = st.tabs(
        [
            "Visão Geral",
            "Qualidade dos Dados",
            "Tratamento / Data Prep",
            "Insights",
            "Rotas e Aeroportos",
            "Distribuições",
            "ML Supervisionado",
            "ML Não Supervisionado",
            "ML Conclusões",
        ]
    )

    with tabs[0]:
        overview_tab(artifacts)
    with tabs[1]:
        data_understanding_tab(artifacts)
    with tabs[2]:
        prep_tab(artifacts)
    with tabs[3]:
        insights_tab(route_df, origin_df, destination_df, min_flights, top_n)
    with tabs[4]:
        routes_airports_tab(route_df, min_flights, top_n)
    with tabs[5]:
        distributions_tab(artifacts)
    with tabs[6]:
        ml_supervised_tab()
    with tabs[7]:
        ml_unsupervised_tab()
    with tabs[8]:
        ml_critical_tab()


if __name__ == "__main__":
    main()
