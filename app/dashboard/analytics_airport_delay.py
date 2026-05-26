from __future__ import annotations

from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd
import streamlit as st

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


def show_unavailable(title: str, detail: str = "A tabela necessaria nao foi encontrada nos outputs da EDA.") -> None:
    st.info(f"{title}: {detail}")


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
            "Nao ha dados suficientes para esta visualizacao com os filtros atuais.")
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
            "Nao ha dados suficientes para esta visualizacao com os filtros atuais.")
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
                         "Colunas de rota, origem, destino, score ou volume nao estao disponiveis.")
        return

    filtered = route_df[pd.to_numeric(
        route_df[flights_col], errors="coerce") >= min_flights].copy()
    filtered[score_col] = pd.to_numeric(filtered[score_col], errors="coerce")
    top_routes = filtered.nlargest(top_n, score_col)
    if top_routes.empty:
        st.info("Nenhuma rota atende ao volume minimo selecionado.")
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
        "Volume minimo de voos por rota",
        min_value=1,
        max_value=5000,
        value=500,
        step=100,
        help="Evita conclusoes baseadas em rotas com pouco volume.",
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
        "Esta etapa responde como os dados chegaram: tamanho das bases, tipos, nulos, cardinalidade e inconsistencias de codigo de aeroporto."
    )

    rows = []
    for table_name, path in RAW_TABLES.items():
        if not path.exists():
            rows.append({"base": table_name, "linhas": None,
                        "colunas": None, "arquivo": "nao encontrado"})
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
    c1.metric("Registros com codigo de aeroporto invalido",
              format_int(invalid_count))
    c2.metric("Bases cruas avaliadas", format_int(sum(path.exists()
              for path in RAW_TABLES.values())))
    c3.metric("Perfis de colunas disponiveis", format_int(
        sum(key.endswith("before") for key in artifacts)))

    st.subheader("Perfis antes do tratamento")
    profile_tabs = st.tabs(
        ["Airlines", "Airports", "Flights", "Join completo"])
    profile_map = [
        ("airlines_profile_before", "Perfil da tabela airlines"),
        ("airports_profile_before", "Perfil da tabela airports"),
        ("flights_profile_before", "Perfil da tabela flights"),
        ("completed_profile_before",
         "Perfil apos join e filtro de voos completos, antes do tratamento"),
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
            "Na base de voos, codigos de origem ou destino com mais de 3 caracteres indicavam `AIRPORT_ID` numerico em vez de codigo IATA. "
            "Esses registros foram isolados e depois corrigidos por cruzamento com uma tabela externa de mapeamento."
        )
        st.dataframe(friendly_table(invalid_df, 20),
                     use_container_width=True, hide_index=True)
    else:
        show_unavailable(
            "Codigos invalidos", "Nao encontrei `flights_with_invalid_airport_codes.csv`.")


def prep_tab(artifacts: dict[str, pd.DataFrame]) -> None:
    st.header("Data Prep: transformando dados brutos em dados analiticos")
    section_note(
        "O notebook prepara uma base confiavel para analise de atraso: corrige aeroportos, enriquece os voos com nomes e coordenadas, filtra voos concluidos e trata nulos de causas de atraso."
    )

    steps = [
        ("Coordenadas de aeroportos",
         "Preenchimento manual das coordenadas ausentes para ECP, PBG e UST."),
        ("Codigos IATA invalidos", "Identificacao de origem/destino com mais de 3 caracteres e conversao de `AIRPORT_ID` para IATA com tabela BTS de outubro de 2015."),
        ("Join das tabelas", "Voos foram unidos a companhias aereas e aeroportos de origem/destino para criar nomes, cidades, estados e coordenadas."),
        ("Colunas criadas", "`FLIGHT_DATE`, `SCHEDULED_DEPARTURE_HOUR`, `SCHEDULED_ARRIVAL_HOUR`, `ROUTE`, `IS_WEEKEND`, `ARRIVAL_DELAY_POSITIVE` e `IS_DELAYED_15`."),
        ("Recorte analitico", "A analise de atraso usa voos finalizados: `CANCELLED = 0`, `DIVERTED = 0` e `ARRIVAL_DELAY` nao nulo."),
        ("Tratamento de nulos",
         "Causas de atraso nulas foram preenchidas com 0 e `TAIL_NUMBER` nulo recebeu `UNKNOWN`."),
        ("Outliers", "Atrasos extremos foram mantidos porque representam eventos reais e importantes para operacao."),
    ]
    st.dataframe(pd.DataFrame(steps, columns=[
                 "Bloco", "Tratamento aplicado"]), use_container_width=True, hide_index=True)

    before = artifacts.get("completed_profile_before")
    after = artifacts.get("completed_profile_after")
    st.subheader("Antes x depois do tratamento")
    if before is None or after is None:
        show_unavailable("Comparacao antes x depois",
                         "Perfis antes/depois do tratamento nao estao completos.")
        return

    col_col = pick_col(before, ["column"])
    if not col_col:
        show_unavailable("Comparacao antes x depois",
                         "Nao encontrei a coluna de nome dos campos.")
        return

    after_col_col = pick_col(after, ["column"])
    missing_before = pick_col(before, ["missing_pct"])
    missing_after = pick_col(after, ["missing_pct"])
    count_before = pick_col(before, ["missing_count"])
    count_after = pick_col(after, ["missing_count"])
    required = [col_col, after_col_col, missing_before,
                missing_after, count_before, count_after]
    if not all(required):
        show_unavailable("Comparacao antes x depois",
                         "Colunas de missing nao estao completas nos perfis.")
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
    c3.metric("Maior reducao em p.p.", format_float(
        merged["reducao_missing_pct"].max(), " p.p.", 2))

    top_missing = merged.head(12).dropna(subset=["reducao_missing_pct"])
    horizontal_bar(
        top_missing,
        x="reducao_missing_pct",
        y="column",
        title="Maiores reducoes de missing apos o tratamento",
        height=430,
    )
    st.dataframe(friendly_table(merged, 30),
                 use_container_width=True, hide_index=True)


def overview_tab(artifacts: dict[str, pd.DataFrame]) -> None:
    st.header("US Flight Delay Analytics")
    st.markdown(
        """
        Este dashboard resume a EDA de atrasos em voos nos EUA. A pergunta central e:
        **quais aeroportos, rotas e caracteristicas estao mais associados a atrasos em voos?**
        """
    )
    section_note(
        "A historia segue quatro passos: entender a qualidade dos dados, explicar os tratamentos, analisar atrasos por rota/aeroporto e fechar com recomendacoes operacionais."
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
    cols[2].metric("Codigos invalidos corrigidos", format_int(invalid_count))
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
                st.metric("Origem mais critica por score", str(
                    top.iloc[0][code_col]), format_float(top.iloc[0][score_col], " pts", 2))
                st.caption(str(top.iloc[0].get(name_col, "")))
        else:
            show_unavailable("Origem critica")
    with c2:
        if route_df is not None:
            score_col = pick_col(route_df, ["delay_impact_score"])
            route_col = pick_col(route_df, ["ROUTE"])
            top = route_df.sort_values(score_col, ascending=False).head(
                1) if score_col else pd.DataFrame()
            if not top.empty:
                st.metric("Rota mais critica por score", str(
                    top.iloc[0][route_col]), format_float(top.iloc[0][score_col], " pts", 2))
                st.caption(
                    "Score combina taxa ajustada de atraso, volume e atraso medio positivo.")
        else:
            show_unavailable("Rota critica")

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
        "Aqui os rankings separam volume e atraso. O filtro de volume minimo reduz o risco de uma rota rara parecer critica apenas por poucos eventos extremos."
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
                    "Score = 60% taxa ajustada de atraso + 25% volume normalizado + 15% atraso medio positivo normalizado.")
                st.dataframe(friendly_table(filtered.nlargest(
                    top_n, score_col)), use_container_width=True, hide_index=True)
            else:
                show_unavailable(
                    "Aeroportos de origem", "Colunas de score, volume ou atraso nao estao completas.")

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
                    "Aeroportos de destino", "Colunas de score, volume ou atraso nao estao completas.")

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
                        title="Rotas mais criticas por score",
                    )
                with c2:
                    horizontal_bar(
                        filtered.nlargest(top_n, delay_col),
                        x=delay_col,
                        y=route_col,
                        color=flights_col,
                        hover_data=[flights_col, score_col, pct_col],
                        title="Rotas com maior atraso medio positivo",
                    )
                st.dataframe(friendly_table(filtered.nlargest(
                    top_n, score_col)), use_container_width=True, hide_index=True)
            else:
                show_unavailable(
                    "Rotas", "Colunas de rota, score, volume ou atraso nao estao completas.")

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
                    title="Relacao entre volume de voos e atraso medio",
                )
                st.caption(
                    "Rotas no canto superior direito combinam alto volume e atraso medio relevante.")
            else:
                show_unavailable(
                    "Volume x atraso", "Colunas de volume, atraso ou score nao estao completas.")


def routes_airports_tab(route_df: pd.DataFrame | None, min_flights: int, top_n: int) -> None:
    st.header("Rotas e aeroportos criticos")
    section_note(
        "Esta aba foca no mapa analitico origem x destino e em uma leitura operacional: quais conexoes merecem investigacao primeiro."
    )
    if route_df is None:
        show_unavailable("Rotas e aeroportos")
        return

    route_matrix(route_df, min_flights=min_flights, top_n=max(top_n, 15))

    route_col = pick_col(route_df, ["ROUTE"])
    flights_col = pick_col(route_df, ["flights"])
    delay_col = pick_col(route_df, ["avg_arrival_delay"])
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
    st.header("Distribuicoes e variaveis numericas")
    section_note(
        "A EDA manteve atrasos extremos porque eles representam eventos reais. Para leitura visual, os histogramas tambem foram gerados com percentis p01 e p99."
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
                title="Variaveis numericas com maior proporcao de outliers pelo IQR",
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
                st.info(f"{caption}: imagem nao encontrada.")


def conclusions_tab() -> None:
    st.header("Conclusoes")
    st.markdown(
        """
        **O que aprendemos**

        1. A qualidade dos dados precisava ser entendida antes dos rankings: cancelamentos, desvios, causas de atraso nulas e codigos de aeroporto numericos poderiam distorcer a leitura.
        2. O recorte analitico mais confiavel usa voos finalizados, com `ARRIVAL_DELAY` conhecido e enriquecimento de aeroporto/companhia.
        3. Ranking por atraso medio isolado nao basta. O score combina volume, taxa ajustada de atraso e atraso medio positivo para priorizar rotas e aeroportos operacionalmente relevantes.
        4. Outliers de atraso nao foram descartados, pois no contexto aereo eles sao eventos reais e importantes para a gestao.

        **Como usar o dashboard**

        Ajuste o volume minimo na barra lateral e observe quais rotas continuam aparecendo no topo. Se uma rota permanece critica mesmo com alto volume minimo, ela merece investigacao operacional mais forte.
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
            "Visao Geral",
            "Qualidade dos Dados",
            "Tratamento / Data Prep",
            "Insights",
            "Rotas e Aeroportos",
            "Distribuicoes",
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


if __name__ == "__main__":
    main()
