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


# =============================================================================
# Script monolitico para treinamento e avaliacao de modelos de Machine Learning
# =============================================================================
#
# Este arquivo e um "gemeo comentado" de train_ml_models.py. A diferenca principal
# e estrutural: aqui o fluxo esta organizado em blocos sequenciais, como em um
# notebook ou roteiro de treinamento. Isso facilita acompanhar a jornada completa:
# leitura de dados, preparacao, treino, avaliacao, persistencia e visualizacoes.
#
# Em um ambiente produtivo, a versao modular costuma ser preferivel porque facilita
# testes, reuso e manutencao. A versao monolitica e excelente para aprendizado,
# auditoria didatica e apresentacoes tecnicas em que queremos enxergar o pipeline
# inteiro sem navegar entre funcoes.


# =============================================================================
# 1. Configuracoes globais do experimento
# =============================================================================

# ROOT_DIR aponta para a raiz do projeto. Como este arquivo esta em:
# app/machine-learning/train_ml_models_monolith_commented.py
# parents[2] sobe para a pasta raiz do repositorio.
ROOT_DIR = Path(__file__).resolve().parents[2]

# Diretórios de entrada e saida. Centralizar caminhos reduz risco de salvar
# artefatos em locais diferentes ao longo do experimento.
DATABASE_DIR = ROOT_DIR / "app" / "database"
OUTPUT_DIR = Path(__file__).resolve().parent / "outputs"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

FLIGHTS_TREATED_PATH = DATABASE_DIR / "flights_treated.csv"
ROUTE_PROFILE_PATH = DATABASE_DIR / "outputs" / \
    "completed_route_delay_profile.csv"

# RANDOM_STATE fixa a aleatoriedade para que amostragem, split e modelos sejam
# reproduziveis. Em ML engineering, reproduzibilidade e essencial para comparar
# experimentos de forma justa.
RANDOM_STATE = 42

# Como flights_treated.csv pode ser grande, treinamos com uma amostra controlada.
# Isso reduz custo computacional e ainda permite manter uma avaliacao consistente.
SUPERVISED_SAMPLE_SIZE = 180_000
CHUNK_SIZE = 250_000

# No estudo nao supervisionado, rotas com volume muito baixo podem produzir
# estatisticas instaveis. O filtro abaixo ajuda a trabalhar com perfis mais
# confiaveis do ponto de vista estatistico.
ROUTE_MIN_FLIGHTS = 500

TARGET = "IS_DELAYED_15"

# Features numericas usadas para prever atraso superior a 15 minutos.
#
# O cenario assumido e "apos a decolagem": por isso mantemos informacoes ja
# conhecidas nesse momento, como DEPARTURE_DELAY, TAXI_OUT e WHEELS_OFF. Ao mesmo
# tempo, evitamos variaveis de chegada ou atraso final, pois elas vazariam a
# resposta do problema para dentro do modelo. Esse cuidado e chamado de evitar
# data leakage.
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

# Features categoricas representam entidades ou estados discretos. Elas serao
# transformadas com OneHotEncoder dentro do pipeline para que os algoritmos
# consigam consumir esses valores.
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


# =============================================================================
# 2. Carregamento amostral do dataset supervisionado
# =============================================================================

if not FLIGHTS_TREATED_PATH.exists():
    raise FileNotFoundError(
        f"Missing treated flights file: {FLIGHTS_TREATED_PATH}")

supervised_samples: list[pd.DataFrame] = []
rng = np.random.default_rng(RANDOM_STATE)

# A leitura em chunks e uma estrategia comum quando o dataset nao cabe
# confortavelmente em memoria. Em vez de carregar tudo, processamos blocos e
# coletamos amostras de cada parte do arquivo.
for chunk in pd.read_csv(
    FLIGHTS_TREATED_PATH,
    usecols=lambda col: col in SUPERVISED_USECOLS,
    chunksize=CHUNK_SIZE,
):
    if TARGET not in chunk.columns:
        raise ValueError(f"Target column {TARGET} was not found.")

    # Normalizacao do target:
    # CSVs podem guardar booleanos como True/False, 1/0, sim/nao ou texto.
    # Padronizar para inteiro 0/1 evita comportamento ambiguo nas metricas e
    # nos modelos.
    if chunk[TARGET].dtype == bool:
        chunk[TARGET] = chunk[TARGET].astype(int)
    else:
        chunk[TARGET] = (
            chunk[TARGET]
            .astype(str)
            .str.strip()
            .str.lower()
            .isin(["true", "1", "sim", "yes"])
            .astype(int)
        )

    chunk = chunk.dropna(subset=[TARGET])

    # A fracao dinamica tenta manter memoria sob controle sem ignorar regioes do
    # arquivo. O calculo usa uma estimativa simples do numero de chunks a partir
    # do tamanho fisico do CSV.
    expected_chunks = max(1, FLIGHTS_TREATED_PATH.stat().st_size // 85_000_000)
    frac = min(0.25, max(0.01, SUPERVISED_SAMPLE_SIZE /
               (expected_chunks * CHUNK_SIZE)))

    # Cada chunk recebe um random_state derivado do gerador principal. Assim, a
    # amostragem e reprodutivel, mas nao identica entre chunks.
    sampled_chunk = chunk.sample(
        frac=frac, random_state=int(rng.integers(0, 1_000_000)))
    supervised_samples.append(sampled_chunk)

supervised_data = pd.concat(supervised_samples, ignore_index=True)

# Caso a amostragem por chunks exceda o tamanho desejado, fazemos um corte final
# tambem reprodutivel.
if len(supervised_data) > SUPERVISED_SAMPLE_SIZE:
    supervised_data = supervised_data.sample(
        SUPERVISED_SAMPLE_SIZE, random_state=RANDOM_STATE)

supervised_data = supervised_data.reset_index(drop=True)


# =============================================================================
# 3. Preparacao de features e separacao treino/teste
# =============================================================================

# Para categoricas, "UNKNOWN" e uma escolha pragmatica: preserva a linha e cria
# uma categoria explicita para ausencia de informacao.
supervised_data[CATEGORICAL_FEATURES] = (
    supervised_data[CATEGORICAL_FEATURES].fillna("UNKNOWN").astype(str)
)

# Para numericas, valores invalidos viram NaN. A imputacao sera feita dentro do
# pipeline, evitando que estatisticas do conjunto de teste contaminem o treino.
for col in NUMERIC_FEATURES:
    supervised_data[col] = pd.to_numeric(supervised_data[col], errors="coerce")

X = supervised_data[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
y = supervised_data[TARGET].astype(int)

# stratify=y mantem proporcao de atrasados e nao atrasados nos conjuntos de
# treino e teste. Isso e especialmente importante em problemas com classes
# desbalanceadas.
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.25,
    stratify=y,
    random_state=RANDOM_STATE,
)


# =============================================================================
# 4. Pipelines de preprocessamento e modelos supervisionados
# =============================================================================

# Pipeline numerico:
# - SimpleImputer(strategy="median") e robusto contra outliers.
# - StandardScaler coloca variaveis numericas em escala comparavel, o que e
#   importante para modelos lineares como LogisticRegression.
numeric_pipeline = Pipeline(
    steps=[
        ("imputer", SimpleImputer(strategy="median")),
        ("scaler", StandardScaler()),
    ]
)

# Pipeline categorico:
# - OneHotEncoder transforma categorias em colunas binarias.
# - handle_unknown="infrequent_if_exist" melhora robustez quando aparecem
#   categorias raras ou nao vistas.
# - min_frequency=50 agrupa categorias pouco frequentes, reduzindo dimensionalidade
#   e risco de overfitting em valores raros.
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

logistic_preprocessor = ColumnTransformer(
    transformers=[
        ("num", numeric_pipeline, NUMERIC_FEATURES),
        ("cat", categorical_pipeline, CATEGORICAL_FEATURES),
    ],
    remainder="drop",
)

# Criamos preprocessadores separados para cada modelo. Em scikit-learn, cada
# Pipeline guarda estado depois do fit; compartilhar o mesmo objeto entre modelos
# pode confundir auditoria e persistencia.
forest_preprocessor = ColumnTransformer(
    transformers=[
        (
            "num",
            Pipeline(
                steps=[
                    ("imputer", SimpleImputer(strategy="median")),
                    ("scaler", StandardScaler()),
                ]
            ),
            NUMERIC_FEATURES,
        ),
        (
            "cat",
            Pipeline(
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
            ),
            CATEGORICAL_FEATURES,
        ),
    ],
    remainder="drop",
)

models: dict[str, Pipeline] = {
    # Logistic Regression e um bom baseline: simples, rapido e interpretavel.
    # class_weight="balanced" compensa desbalanceamento ajustando o peso das
    # classes na funcao de perda.
    "logistic_regression": Pipeline(
        steps=[
            ("preprocess", logistic_preprocessor),
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
    # Random Forest captura relacoes nao lineares e interacoes entre variaveis.
    # Os limites de profundidade e tamanho minimo de folha controlam overfitting
    # e tornam o treinamento mais previsivel em dados grandes.
    "random_forest": Pipeline(
        steps=[
            ("preprocess", forest_preprocessor),
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


# =============================================================================
# 5. Treinamento, avaliacao e persistencia dos modelos supervisionados
# =============================================================================

supervised_metric_rows = []
trained_models: dict[str, Pipeline] = {}

for model_name, model in models.items():
    print(f"Training supervised model: {model_name}")
    model.fit(X_train, y_train)

    prediction = model.predict(X_test)
    probability = model.predict_proba(X_test)[:, 1]

    # Em classificacao binaria, olhar apenas accuracy pode esconder problemas em
    # classes minoritarias. Por isso registramos precision, recall, f1 e ROC AUC.
    metrics = {
        "accuracy": accuracy_score(y_test, prediction),
        "precision": precision_score(y_test, prediction, zero_division=0),
        "recall": recall_score(y_test, prediction, zero_division=0),
        "f1": f1_score(y_test, prediction, zero_division=0),
        "roc_auc": roc_auc_score(y_test, probability),
        "model": model_name,
    }

    supervised_metric_rows.append(metrics)
    trained_models[model_name] = model

    # Persistimos o pipeline inteiro, nao apenas o estimador. Isso garante que o
    # mesmo preprocessamento usado no treino sera aplicado na inferencia.
    joblib.dump(model, OUTPUT_DIR / f"{model_name}_classifier.joblib")

supervised_metrics = pd.DataFrame(
    supervised_metric_rows).sort_values("roc_auc", ascending=False)
supervised_metrics.to_csv(
    OUTPUT_DIR / "supervised_classification_metrics.csv", index=False)

best_model_name = supervised_metrics.iloc[0]["model"]
best_supervised_model = trained_models[best_model_name]


# =============================================================================
# 6. Visualizacoes do melhor modelo supervisionado
# =============================================================================

best_probability = best_supervised_model.predict_proba(X_test)[:, 1]
best_prediction = best_supervised_model.predict(X_test)

# A curva ROC mostra a relacao entre taxa de verdadeiros positivos e falsos positivos para diferentes limiares de classificacao.
# Quanto mais a curva se aproxima do canto superior esquerdo do grafico, melhor o desempenho do modelo.
# A linha diagonal tracejada representa um modelo aleatorio (baseline).
# Interpretação da curva ROC:
# - Se a curva estiver muito proxima da linha diagonal, o modelo tem pouco poder de discriminacao entre as classes.
# - Se a curva estiver mais proxima do canto superior esquerdo, o modelo tem melhor desempenho,
# indicando que consegue identificar bem os casos positivos (atrasados) sem classificar muitos casos negativos (nao atrasados) como positivos.
# A curva pode ser lida como: "Para um dado limiar de classificacao, qual a taxa de verdadeiros positivos e falsos positivos que o modelo produz?"
# O limiar da classificacao (threshold) pode ser ajustado para balancear precision e recall, dependendo do custo de falsos positivos vs falsos negativos
# no contexto de negocio.

fpr, tpr, _ = roc_curve(y_test, best_probability)
roc_fig = go.Figure()
roc_fig.add_trace(go.Scatter(x=fpr, y=tpr, mode="lines", name=best_model_name))
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
    title=f"ROC Curve - {best_model_name}",
    xaxis_title="False positive rate",
    yaxis_title="True positive rate",
    template="plotly_dark",
)
roc_fig.write_html(OUTPUT_DIR / "supervised_roc_curve.html")

matrix = confusion_matrix(y_test, best_prediction)
matrix_fig = px.imshow(
    matrix,
    text_auto=True,
    labels=dict(x="Predicted", y="Actual", color="Flights"),
    x=["on_time", "delayed_15"],
    y=["on_time", "delayed_15"],
    title=f"Confusion Matrix - {best_model_name}",
    color_continuous_scale="Blues",
)
matrix_fig.update_layout(template="plotly_dark")
matrix_fig.write_html(OUTPUT_DIR / "supervised_confusion_matrix.html")

# Perfil da amostra: pequeno artefato de governanca experimental. Ele ajuda a
# responder perguntas como "quantas linhas foram treinadas?" e "qual era a taxa
# de atraso na amostra?".
sample_profile = pd.DataFrame(
    [
        {"metric": "sample_rows", "value": len(supervised_data)},
        {"metric": "train_rows", "value": len(X_train)},
        {"metric": "test_rows", "value": len(X_test)},
        {"metric": "delayed_15_rate_sample", "value": y.mean()},
    ]
)
sample_profile.to_csv(
    OUTPUT_DIR / "supervised_sample_profile.csv", index=False)


# =============================================================================
# 7. Carregamento e validacao do dataset para clusterizacao de rotas (Modelo nao supervisionado)
# =============================================================================

if not ROUTE_PROFILE_PATH.exists():
    raise FileNotFoundError(
        f"Missing route profile file: {ROUTE_PROFILE_PATH}")

route_df = pd.read_csv(ROUTE_PROFILE_PATH)

required_route_columns = [
    "ROUTE",
    "flights",
    "delayed_flights",
    "avg_arrival_delay",
    "delayed_15_pct",
    "adjusted_delay_pct",
    "delay_impact_score",
]
missing_route_columns = [
    col for col in required_route_columns if col not in route_df.columns]
if missing_route_columns:
    raise ValueError(f"Missing route columns: {missing_route_columns}")

# Removemos rotas pouco frequentes para evitar clusters guiados por ruido
# amostral. Esse tipo de criterio costuma ser uma decisao de produto/negocio e
# estatistica, nao apenas tecnica.
route_df = route_df[route_df["flights"] >= ROUTE_MIN_FLIGHTS].copy()

# Atrasos medios negativos indicam adiantamento. Para o perfil de impacto de
# atraso, transformamos a medida em uma versao positiva, focada no tamanho do
# problema operacional.
route_df["avg_arrival_delay_positive"] = route_df["avg_arrival_delay"].clip(
    lower=0)

cluster_features = [
    "flights",
    "delayed_flights",
    "avg_arrival_delay_positive",
    "delayed_15_pct",
    "adjusted_delay_pct",
    "delay_impact_score",
]

route_features = route_df[cluster_features].apply(
    pd.to_numeric, errors="coerce")
route_features = route_features.fillna(
    route_features.median(numeric_only=True))

# KMeans usa distancias euclidianas; por isso a escala das variaveis importa
# muito. Sem StandardScaler, uma coluna com numeros grandes, como flights,
# dominaria artificialmente a formacao dos clusters.
scaled_route_features = StandardScaler().fit_transform(route_features)


# =============================================================================
# 8. PCA, KMeans e avaliacao nao supervisionada
# =============================================================================

# PCA aqui e usado para visualizacao em 2D, nao como etapa obrigatoria do KMeans.
# Ele comprime a informacao das features padronizadas em dois eixos principais.
pca = PCA(n_components=2, random_state=RANDOM_STATE)
pca_components = pca.fit_transform(scaled_route_features)

# KMeans agrupa rotas com perfis semelhantes de volume e atraso.
kmeans = KMeans(n_clusters=4, n_init=20, random_state=RANDOM_STATE)
route_df["cluster"] = kmeans.fit_predict(scaled_route_features)
route_df["pca_1"] = pca_components[:, 0]
route_df["pca_2"] = pca_components[:, 1]

route_df.sort_values(["cluster", "delay_impact_score"], ascending=[True, False]).to_csv(
    OUTPUT_DIR / "unsupervised_route_clusters.csv",
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

# Silhouette score mede o quanto os clusters estao separados e coesos. Nao e uma
# verdade absoluta de negocio, mas e um bom indicador tecnico para comparar
# configuracoes de clusterizacao.
# Entendo as colunas salvas em unsupervised_cluster_summary:
# - cluster: identificador do cluster
# - routes: numero de rotas no cluster
# - avg_flights: numero medio de voos por rota no cluster
# - avg_arrival_delay_positive: atraso medio positivo de chegada por rota no cluster
# - avg_delayed_15_pct: porcentagem media de voos atrasados mais de 15 minutos por rota no cluster
# - avg_delay_impact_score: pontuacao media de impacto de atraso por rota no cluster, combinando volume e severidade do atraso
# - top_route: a rota mais impactante do cluster, servindo como exemplo representativo do perfil de atraso encontrado naquele grupo
# - silhouette_score: medida de qualidade da clusterizacao, indicando o quao bem separados e coesos os clusters estao.
# Valores mais altos indicam clusters mais distintos e bem formados.
# - pca_explained_variance: porcentagem da variancia total dos dados originais que e capturada pelos dois componentes principais usados para visualizacao.
# Valores mais altos indicam que a visualizacao em 2D representa melhor a estrutura dos dados.
cluster_summary["silhouette_score"] = silhouette_score(
    scaled_route_features,
    route_df["cluster"],
)
cluster_summary["pca_explained_variance"] = pca.explained_variance_ratio_.sum()
cluster_summary.to_csv(
    OUTPUT_DIR / "unsupervised_cluster_summary.csv", index=False)


# =============================================================================
# 9. Visualizacoes do estudo nao supervisionado
# =============================================================================

route_cluster_fig = px.scatter(
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
route_cluster_fig.write_html(
    OUTPUT_DIR / "unsupervised_route_clusters_pca.html")

cluster_summary_fig = px.bar(
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
cluster_summary_fig.write_html(
    OUTPUT_DIR / "unsupervised_cluster_summary.html")


# =============================================================================
# 10. Saida final no terminal
# =============================================================================

print("Supervised metrics")
print(supervised_metrics.to_string(index=False))
print("\nCluster summary")
print(cluster_summary.to_string(index=False))
print(f"\nOutputs saved to: {OUTPUT_DIR}")
