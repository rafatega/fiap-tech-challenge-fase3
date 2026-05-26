# Machine Learning

Esta etapa atende aos requisitos de modelagem supervisionada e nao supervisionada do Tech Challenge.

Script principal:

- `app/machine-learning/train_ml_models.py`

Outputs:

- `app/machine-learning/outputs`

## Objetivo

Construir modelos para responder duas perguntas:

1. **Supervisionado:** e possivel prever se um voo vai atrasar 15 minutos ou mais?
2. **Nao supervisionado:** e possivel agrupar rotas com perfis semelhantes de atraso?

## Base usada

Base principal:

- `app/database/flights_treated.csv`

Essa base ja vem da etapa de EDA/tratamento, com:

- voos finalizados;
- aeroportos corrigidos;
- joins com companhias e aeroportos;
- colunas auxiliares como `ROUTE`, `IS_WEEKEND`, `SCHEDULED_DEPARTURE_HOUR`;
- target `IS_DELAYED_15`.

Para clusterizacao, foi usada a tabela agregada:

- `app/database/outputs/completed_route_delay_profile.csv`

## Escopo das variaveis supervisionadas

O modelo supervisionado considera um cenario de predicao **apos a decolagem**.

Por isso, foram mantidas variaveis operacionais ja conhecidas quando o aviao sai do chao:

- `DEPARTURE_DELAY`
- `TAXI_OUT`
- `WHEELS_OFF`

E foram removidas variaveis conhecidas apenas no fim da operacao ou diretamente ligadas ao target:

- `ARRIVAL_DELAY`
- `AIR_SYSTEM_DELAY`
- `SECURITY_DELAY`
- `AIRLINE_DELAY`
- `LATE_AIRCRAFT_DELAY`
- `WEATHER_DELAY`

Motivo:

- `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF` ja existem depois da decolagem;
- `ARRIVAL_DELAY` define diretamente o target `IS_DELAYED_15`;
- as causas finais de atraso so aparecem depois da chegada ou consolidacao operacional.

Observacao importante:

- este nao e um modelo pre-voo;
- ele simula uma predicao durante a operacao, depois que o aviao ja saiu do chao.

## Modelagem supervisionada

Problema escolhido:

- classificacao binaria.

Target:

- `IS_DELAYED_15`

Interpretacao:

- `1`: voo atrasou 15 minutos ou mais;
- `0`: voo nao atrasou 15 minutos ou mais.

### Features usadas

Numericas:

- `MONTH`
- `DAY`
- `DAY_OF_WEEK`
- `SCHEDULED_DEPARTURE`
- `SCHEDULED_ARRIVAL`
- `SCHEDULED_DEPARTURE_HOUR`
- `SCHEDULED_ARRIVAL_HOUR`
- `SCHEDULED_TIME`
- `DISTANCE`
- `ORIGIN_LATITUDE`
- `ORIGIN_LONGITUDE`
- `DESTINATION_LATITUDE`
- `DESTINATION_LONGITUDE`
- `DEPARTURE_DELAY`
- `TAXI_OUT`
- `WHEELS_OFF`

Categoricas:

- `AIRLINE`
- `ORIGIN_AIRPORT`
- `DESTINATION_AIRPORT`
- `ORIGIN_STATE`
- `DESTINATION_STATE`
- `ROUTE`
- `IS_WEEKEND`

### Pre-processamento

Numericas:

- missing preenchido com mediana;
- padronizacao com `StandardScaler`.

Categoricas:

- missing preenchido com `UNKNOWN`;
- codificacao com `OneHotEncoder`;
- categorias raras agrupadas com `min_frequency=50`.

### Algoritmos comparados

Foram comparados dois modelos:

1. `LogisticRegression`
2. `RandomForestClassifier`

Ambos usam `class_weight` para reduzir o impacto do desbalanceamento da classe de atraso.

### Amostragem

A base `flights_treated.csv` e grande, entao o script le o CSV em chunks e cria uma amostra reproduzivel.

Perfil da amostra usada no ultimo treinamento:

- linhas na amostra: `178.871`
- treino: `134.153`
- teste: `44.718`
- taxa de voos atrasados na amostra: `18,56%`

## Resultados supervisionados

Metricas do ultimo treinamento:

| modelo | accuracy | precision | recall | f1 | roc_auc |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,9233 | 0,7478 | 0,8855 | 0,8108 | 0,9629 |
| Random Forest | 0,9126 | 0,7350 | 0,8273 | 0,7784 | 0,9280 |

Melhor modelo por `roc_auc`:

- `LogisticRegression`

Interpretacao:

- a performance ficou alta porque o modelo ja sabe se o voo saiu atrasado e quanto tempo ficou taxiando;
- `LogisticRegression` superou `RandomForest` neste recorte, com maior ROC AUC e F1;
- o resultado e mais realista que usar `ARRIVAL_DELAY`, mas ainda nao representa previsao antes da partida;
- para um modelo pre-voo, seria preciso remover tambem `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF`.

Arquivos gerados:

- `supervised_classification_metrics.csv`
- `supervised_sample_profile.csv`
- `supervised_roc_curve.html`
- `supervised_confusion_matrix.html`
- `logistic_regression_classifier.joblib`
- `random_forest_classifier.joblib`

## Modelagem nao supervisionada

Abordagem escolhida:

- clusterizacao de rotas.

Algoritmo:

- `KMeans`

Redimensionamento para visualizacao:

- `PCA` com 2 componentes.

### Base usada

Arquivo:

- `completed_route_delay_profile.csv`

Filtro:

- apenas rotas com pelo menos `500` voos.

Motivo:

- evitar clusters influenciados por rotas com pouco volume.

### Features usadas na clusterizacao

- `flights`
- `delayed_flights`
- `avg_arrival_delay_positive`
- `delayed_15_pct`
- `adjusted_delay_pct`
- `delay_impact_score`

### Configuracao

- numero de clusters: `4`
- random state: `42`
- dados padronizados com `StandardScaler`

## Resultados nao supervisionados

Metricas:

- silhouette score: `0,3484`
- variancia explicada pelo PCA 2D: `96,70%`

Resumo dos clusters:

| cluster | rotas | media flights | atraso medio positivo | delayed 15 pct medio | score medio | rota exemplo |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 249 | 5.818,82 | 13,48 | 20,56 | 22,70 | SFO-LAX |
| 1 | 564 | 1.464,86 | 16,46 | 25,03 | 17,82 | DFW-HNL |
| 0 | 1.292 | 1.567,08 | 11,93 | 18,54 | 14,24 | CLT-MIA |
| 2 | 823 | 1.282,88 | 8,07 | 12,30 | 10,23 | LIH-HNL |

Interpretacao dos clusters:

- **Cluster 3:** rotas de alto volume e alto impacto. Sao as rotas mais importantes para priorizacao operacional.
- **Cluster 1:** rotas com maior atraso medio e maior percentual de atraso, mas volume menor que o cluster 3.
- **Cluster 0:** rotas intermediarias, com atraso e score medios.
- **Cluster 2:** rotas menos criticas, com menor atraso medio e menor percentual de atraso.

Arquivos gerados:

- `unsupervised_route_clusters.csv`
- `unsupervised_cluster_summary.csv`
- `unsupervised_route_clusters_pca.html`
- `unsupervised_cluster_summary.html`

## Como executar

No terminal, a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe app\machine-learning\train_ml_models.py
```

O script vai:

1. criar uma amostra da base tratada;
2. treinar os dois modelos supervisionados;
3. avaliar os modelos;
4. salvar metricas e modelos;
5. clusterizar as rotas;
6. salvar tabelas e graficos HTML.

## Limitações

- O modelo supervisionado considera predicao apos decolagem, nao antes da partida.
- `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF` aumentam bastante o poder preditivo.
- Variaveis finais como `ARRIVAL_DELAY` e causas consolidadas de atraso foram removidas para evitar vazamento direto do target.
- Nao ha dados climaticos detalhados por aeroporto e horario.
- Nao ha informacoes operacionais de malha, conexoes, manutencao ou tripulacao.
- O target `IS_DELAYED_15` e desbalanceado.
- A amostragem foi usada por eficiencia; para uma versao final, pode-se treinar com mais dados ou fazer validacao temporal.
- A clusterizacao depende das features agregadas por rota; outros agrupamentos podem ser feitos por aeroporto, companhia ou estado.

## Proximos passos recomendados

- Testar validacao temporal: treinar em meses anteriores e testar em meses posteriores.
- Criar features de periodo do dia.
- Adicionar feriados e estacoes do ano.
- Enriquecer com clima historico.
- Testar modelos adicionais, como gradient boosting.
- Calibrar o limiar de classificacao conforme objetivo: mais recall ou mais precisao.
- Criar interpretabilidade com importancia de variaveis ou permutation importance.
