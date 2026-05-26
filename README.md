# FIAP Tech Challenge - Fase 3

**Tema: Análise e modelagem de atrasos em voos nos Estados Unidos**
**Grupo: 51**
**Integrante: Rafael Tegazzini**
**Vídeo de apresentação:** 

Projeto de análise e modelagem de atrasos em voos nos Estados Unidos, desenvolvido para o Tech Challenge da FIAP.

O trabalho cobre a jornada completa de dados: entendimento das bases, tratamento, análise exploratória, estudo de outliers, modelagem supervisionada, clusterização de rotas e dashboard em Streamlit para apresentação dos resultados.

## Objetivo

Responder, com dados, duas perguntas principais:

1. Quais aeroportos, rotas e características estão mais associados a atrasos em voos?
2. É possível criar modelos de Machine Learning para prever atrasos e agrupar rotas com perfis operacionais semelhantes?

## Bases utilizadas

As bases principais ficam em `app/database`:

- `flights.csv`: base principal de voos.
- `airlines.csv`: de/para das companhias aéreas.
- `airports.csv`: de/para dos aeroportos, com cidade, estado, país e coordenadas.
- `airport_id_to_iata_2015_10.csv`: tabela externa usada para converter códigos numéricos de aeroportos para IATA.

A base tratada final usada na modelagem foi:

- `app/database/flights_treated.csv`

## O que foi feito

### 1. Entendimento e tratamento dos dados

A primeira etapa foi entender a qualidade das bases, mapear nulos, inconsistências e preparar um dataframe analítico confiável.

Principais ações:

- análise de perfil das bases cruas;
- identificação de códigos de aeroportos inválidos;
- conversão de `AIRPORT_ID` numérico para código IATA;
- correção de coordenadas faltantes em aeroportos;
- join entre voos, companhias aéreas e aeroportos;
- criação de colunas auxiliares, como `ROUTE`, `FLIGHT_DATE`, `IS_WEEKEND` e `IS_DELAYED_15`;
- filtro para voos finalizados;
- tratamento de nulos em causas de atraso e `TAIL_NUMBER`;
- criação de rankings de rotas e aeroportos por criticidade.

Documentação detalhada:

- `app/eda/eda.md`

### 2. Score de criticidade

Para evitar conclusões baseadas apenas em atraso médio, foi criado um score de criticidade combinando:

- taxa ajustada de atraso;
- volume de voos;
- atraso médio positivo.

Esse score foi usado para priorizar rotas e aeroportos mais relevantes operacionalmente.

### 3. Estudo de outliers

Os outliers foram avaliados com cuidado porque, em dados de aviação, valores extremos podem representar eventos reais.

Foram comparadas estratégias como:

- manter valores extremos;
- usar `RobustScaler`;
- winsorizar p01-p99;
- winsorizar p05-p95;
- remover linhas fora de p01-p99;
- remover linhas fora de p05-p95.

Conclusão:

> A melhor estratégia foi manter os valores extremos, pois eles carregavam sinal preditivo importante, principalmente em `DEPARTURE_DELAY` e `TAXI_OUT`.

Documentação detalhada:

- `app/machine-learning/outlier-study.md`

### 4. Machine Learning supervisionado

Foi criado um modelo de classificação para prever se um voo teria atraso de 15 minutos ou mais.

Target:

- `IS_DELAYED_15`

Modelos testados:

- `LogisticRegression`
- `RandomForestClassifier`

Resultado principal:

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,9255 | 0,7502 | 0,8987 | 0,8178 | 0,9673 |
| Random Forest | 0,9027 | 0,7025 | 0,8279 | 0,7601 | 0,9227 |

A Regressão Logística foi o melhor modelo geral nas métricas avaliadas.

Observação importante:

> O modelo supervisionado atual representa um cenário após a decolagem, pois usa variáveis como `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF`. Ele não deve ser interpretado como previsão pré-voo.

Documentação detalhada:

- `app/machine-learning/machine-learning.md`

### 5. Machine Learning não supervisionado

Foi criada uma clusterização de rotas para agrupar perfis semelhantes de volume e atraso.

Algoritmo:

- `KMeans`

Features usadas:

- `flights`
- `delayed_flights`
- `avg_arrival_delay_positive`
- `delayed_15_pct`
- `adjusted_delay_pct`
- `delay_impact_score`

Resultados:

- `silhouette_score`: 0,3484
- variância explicada pelo PCA 2D: 96,70%

Interpretação:

- os clusters apresentam separação moderada;
- a clusterização é útil para segmentação e priorização operacional;
- não deve ser interpretada como separação perfeita entre grupos.

Documentação detalhada:

- `app/machine-learning/machine-learning.md`

### 6. Dashboard Streamlit

Foi criado um dashboard em Streamlit para apresentar:

- qualidade dos dados;
- tratamentos aplicados;
- insights de atraso;
- rotas e aeroportos críticos;
- distribuições numéricas;
- resultados supervisionados;
- clusterização de rotas;
- conclusões e próximos passos.

Arquivo principal:

- `app/dashboard/analytics_airport_delay.py`

Documentação detalhada:

- `app/dashboard/doc_analytics_airport_delay.md`

## Estrutura do projeto

```text
.
├── app
│   ├── dashboard
│   │   ├── analytics_airport_delay.py
│   │   └── doc_analytics_airport_delay.md
│   ├── database
│   │   ├── airlines.csv
│   │   ├── airports.csv
│   │   ├── flights.csv
│   │   ├── airport_id_to_iata_2015_10.csv
│   │   └── outputs
│   ├── eda
│   │   ├── eda.ipynb
│   │   ├── eda.md
│   │   └── outputs
│   └── machine-learning
│       ├── train_ml_models.py
│       ├── outlier_sensitivity_study.py
│       ├── machine-learning.md
│       ├── outlier-study.md
│       └── outputs
├── project_scope
├── pyproject.toml
├── uv.lock
└── README.md
```

## Como executar

### 1. Criar ambiente

Este projeto usa Python e dependências declaradas no `pyproject.toml`.

Com `uv`:

```powershell
uv sync
```

Ou, usando o ambiente virtual já criado no projeto:

```powershell
.\.venv\Scripts\python.exe --version
```

### 2. Executar treinamento de Machine Learning

```powershell
.\.venv\Scripts\python.exe app\machine-learning\train_ml_models.py
```

Esse script:

- carrega a base tratada;
- treina os modelos supervisionados;
- avalia as métricas;
- salva modelos `.joblib`;
- executa a clusterização;
- gera tabelas e gráficos HTML.

### 3. Executar estudo de outliers

```powershell
.\.venv\Scripts\python.exe app\machine-learning\outlier_sensitivity_study.py
```

### 4. Executar dashboard

```powershell
.\.venv\Scripts\streamlit.exe run app\dashboard\analytics_airport_delay.py
```

## Configurações úteis

O treinamento permite usar amostra ou base inteira.

No arquivo `app/machine-learning/train_ml_models.py`:

```python
USE_SAMPLE = True
SUPERVISED_SAMPLE_SIZE = 180_000
CHUNK_SIZE = 250_000
```

Para treinar com a base inteira:

```python
USE_SAMPLE = False
```

Observação:

- `CHUNK_SIZE` controla quantas linhas são lidas por vez.
- `SUPERVISED_SAMPLE_SIZE` controla o tamanho final da amostra quando `USE_SAMPLE = True`.
- Com `USE_SAMPLE = False`, a base completa é usada, o que exige mais memória.

## Principais outputs

EDA:

- `app/eda/outputs/*.csv`
- `app/eda/outputs/*.png`
- `app/database/outputs/*.csv`

Machine Learning:

- `app/machine-learning/outputs/supervised_classification_metrics.csv`
- `app/machine-learning/outputs/supervised_sample_profile.csv`
- `app/machine-learning/outputs/supervised_roc_curve.html`
- `app/machine-learning/outputs/supervised_confusion_matrix.html`
- `app/machine-learning/outputs/unsupervised_cluster_summary.csv`
- `app/machine-learning/outputs/unsupervised_route_clusters.csv`
- `app/machine-learning/outputs/*.joblib`

Outliers:

- `app/machine-learning/outputs/outlier_study/outlier_strategy_metrics.csv`
- `app/machine-learning/outputs/outlier_study/outlier_tail_diagnostics.csv`

## Desafios encontrados

- A base de voos é grande, com mais de 5,7 milhões de registros.
- Havia códigos de aeroportos em formato numérico misturados com códigos IATA.
- Algumas colunas tinham muitos nulos, mas parte deles representava ausência de atraso e não erro.
- Outliers precisaram ser avaliados com contexto de negócio, pois atrasos extremos podem ser eventos reais.
- O target é desbalanceado: cerca de 18,61% dos voos tiveram atraso de 15 minutos ou mais.
- A modelagem supervisionada atual é operacional após a decolagem, não pré-voo.
- A clusterização (KMeans, silhouette) tem separação moderada, não perfeita.

## Limitações

- Não foram usados dados climáticos detalhados por aeroporto e horário.
- Não há dados de manutenção, tripulação, conexões ou capacidade aeroportuária.
- A validação supervisionada atual usa divisão aleatória, não validação temporal.
- O número de clusters foi fixado em 4.

## Próximos passos

- Criar uma versão 'antes do embarque' do modelo supervisionado.
- Fazer validação temporal, treinando em meses anteriores e testando em meses posteriores.
- Testar modelos adicionais, como Gradient Boosting, XGBoost, LightGBM ou HistGradientBoosting.
- Calibrar o limiar de classificação conforme o objetivo operacional.
- Enriquecer a base com clima histórico, feriados e sazonalidade.
- Testar diferentes valores de `k` na clusterização.
- Avaliar clusterizações por aeroporto, companhia aérea, estado ou período do dia.

## Documentações detalhadas

- EDA e tratamento: `app/eda/eda.md`
- Estudo de outliers: `app/machine-learning/outlier-study.md`
- Machine Learning: `app/machine-learning/machine-learning.md`
- Dashboard: `app/dashboard/doc_analytics_airport_delay.md`
