# Machine Learning

Esta etapa atende aos requisitos de modelagem supervisionada e nao supervisionada do Tech Challenge.

Script principal:

- `app/machine-learning/train_ml_models.py`

Diretorio de saida:

- `app/machine-learning/outputs`

## Objetivo

A etapa de Machine Learning foi dividida em dois problemas:

1. **Modelagem supervisionada:** prever se um voo tera atraso de 15 minutos ou mais.
2. **Modelagem nao supervisionada:** agrupar rotas com perfis semelhantes de volume e atraso.

Essas duas abordagens respondem perguntas diferentes:

- O modelo supervisionado ajuda a estimar risco de atraso.
- A clusterizacao ajuda a segmentar rotas e apoiar priorizacao operacional.

## Bases utilizadas

Base principal:

- `app/database/flights_treated.csv`

Essa base ja vem da etapa de tratamento, contendo:

- apenas voos finalizados;
- aeroportos corrigidos;
- joins com companhias aereas e aeroportos;
- colunas auxiliares como `ROUTE`, `IS_WEEKEND`, `SCHEDULED_DEPARTURE_HOUR`;
- target `IS_DELAYED_15`.

Para a clusterizacao, foi usada uma base agregada por rota:

- `app/database/outputs/completed_route_delay_profile.csv`

## Perfil da base supervisionada

O treinamento supervisionado atual foi executado com a base completa.

| Indicador | Valor |
|---|---:|
| Uso de amostra | `False` |
| Linhas usadas | 5.714.008 |
| Linhas de treino | 4.285.506 |
| Linhas de teste | 1.428.502 |
| Taxa de atraso de 15+ minutos | 18,61% |

A divisao treino/teste foi feita com:

```python
test_size=0.25
stratify=y
random_state=42
```

Isso significa que 25% dos dados foram separados para teste e a proporcao do target foi preservada no treino e no teste.

## Escopo da predicao supervisionada

O modelo supervisionado simula um cenario de predicao **apos a decolagem**.

Foram mantidas variaveis operacionais ja conhecidas quando o aviao saiu do chao:

- `DEPARTURE_DELAY`
- `TAXI_OUT`
- `WHEELS_OFF`

Foram evitadas variaveis conhecidas apenas no final da operacao ou diretamente ligadas ao target:

- `ARRIVAL_DELAY`
- `AIR_SYSTEM_DELAY`
- `SECURITY_DELAY`
- `AIRLINE_DELAY`
- `LATE_AIRCRAFT_DELAY`
- `WEATHER_DELAY`

Motivo:

- `ARRIVAL_DELAY` define diretamente se o voo atrasou ou nao.
- As causas finais de atraso so sao conhecidas depois da chegada ou consolidacao operacional.
- Usar essas variaveis criaria vazamento de dados, tambem conhecido como `data leakage`.

Observacao importante:

> Este nao e um modelo pre-voo. Ele representa um modelo operacional apos a decolagem. Para previsao antes da partida, seria necessario remover tambem `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF`.

## Modelagem supervisionada

### Problema

Foi treinado um modelo de classificacao binaria.

Target:

- `IS_DELAYED_15`

Interpretacao:

- `1`: voo atrasou 15 minutos ou mais.
- `0`: voo nao atrasou 15 minutos ou mais.

### Features numericas

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

### Features categoricas

- `AIRLINE`
- `ORIGIN_AIRPORT`
- `DESTINATION_AIRPORT`
- `ORIGIN_STATE`
- `DESTINATION_STATE`
- `ROUTE`
- `IS_WEEKEND`

### Pre-processamento

Para variaveis numericas:

- conversao para numerico;
- imputacao de valores faltantes com mediana;
- padronizacao com `StandardScaler`.

Para variaveis categoricas:

- preenchimento de nulos com `UNKNOWN`;
- codificacao com `OneHotEncoder`;
- agrupamento de categorias raras com `min_frequency=50`.

### Algoritmos testados

Foram comparados dois algoritmos:

| Algoritmo | Papel no estudo |
|---|---|
| `LogisticRegression` | Baseline linear, rapido, interpretavel e forte para comparacao inicial. |
| `RandomForestClassifier` | Modelo de arvores capaz de capturar relacoes nao lineares e interacoes. |

Ambos foram treinados com `class_weight`, pois a classe de atraso e desbalanceada: cerca de 18,6% dos voos possuem `IS_DELAYED_15 = 1`.

## Metricas supervisionadas

As metricas usadas foram:

| Metrica | Como interpretar |
|---|---|
| `accuracy` | Percentual geral de acertos. Pode ser enganosa em bases desbalanceadas. |
| `precision` | Entre os voos previstos como atrasados, quantos realmente atrasaram. |
| `recall` | Entre todos os voos que atrasaram, quantos o modelo conseguiu capturar. |
| `f1` | Equilibrio entre precision e recall. |
| `roc_auc` | Capacidade do modelo de separar voos atrasados de nao atrasados em diferentes limiares. |

Neste problema, `f1`, `recall` e `roc_auc` sao mais informativas que apenas `accuracy`, porque existe desbalanceamento entre voos atrasados e nao atrasados.

## Resultados supervisionados

Arquivo:

- `app/machine-learning/outputs/supervised_classification_metrics.csv`

| Modelo | Accuracy | Precision | Recall | F1 | ROC AUC |
|---|---:|---:|---:|---:|---:|
| Logistic Regression | 0,9255 | 0,7502 | 0,8987 | 0,8178 | 0,9673 |
| Random Forest | 0,9027 | 0,7025 | 0,8279 | 0,7601 | 0,9227 |

Melhor modelo:

- `LogisticRegression`

A regressao logistica superou a Random Forest em todas as metricas avaliadas.

Interpretacao:

- O modelo capturou aproximadamente 89,87% dos voos atrasados.
- Quando o modelo previu atraso, acertou aproximadamente 75,02% das vezes.
- O `ROC AUC` de 0,9673 indica forte capacidade de separacao entre voos atrasados e nao atrasados.
- O resultado alto e explicado principalmente porque o modelo ja conhece informacoes operacionais relevantes, como atraso de partida e tempo de taxiamento.

## Graficos supervisionados

Arquivos gerados:

- `app/machine-learning/outputs/supervised_roc_curve.html`
- `app/machine-learning/outputs/supervised_confusion_matrix.html`

### Curva ROC

A curva ROC compara a taxa de verdadeiros positivos contra a taxa de falsos positivos para diferentes limiares de classificacao.

Como interpretar:

- Quanto mais a curva se aproxima do canto superior esquerdo, melhor.
- A linha diagonal representa um modelo aleatorio.
- O `ROC AUC` resume a qualidade da separacao.

No resultado atual, a regressao logistica obteve `ROC AUC = 0,9673`, indicando desempenho muito forte na separacao entre voos com e sem atraso.

### Matriz de confusao

A matriz de confusao mostra quatro grupos:

- voos nao atrasados previstos corretamente;
- voos atrasados previstos corretamente;
- voos nao atrasados previstos como atrasados;
- voos atrasados previstos como nao atrasados.

Ela deve ser lida junto com `precision` e `recall`.

No contexto de atraso de voos, um alto `recall` e relevante porque significa que o modelo deixa escapar menos voos que realmente atrasaram. Por outro lado, a `precision` mostra o custo de alertas falsos.

## Modelagem nao supervisionada

### Objetivo

A modelagem nao supervisionada buscou agrupar rotas com perfis semelhantes de atraso e volume.

Neste caso, o modelo nao usa um target. Ele procura padroes internos nos dados.

A pergunta respondida foi:

> Quais rotas possuem comportamento operacional parecido em termos de volume, atraso e impacto?

### Unidade analisada

A clusterizacao foi feita por rota, nao por voo individual.

Exemplos de rotas:

- `SFO-LAX`
- `DFW-HNL`
- `CLT-MIA`

### Filtro aplicado

Foram consideradas apenas rotas com pelo menos 500 voos:

```python
ROUTE_MIN_FLIGHTS = 500
```

Motivo:

- rotas com poucos voos podem gerar percentuais instaveis;
- o filtro reduz ruido estatistico;
- os clusters passam a representar rotas com maior confiabilidade operacional.

### Features usadas

- `flights`
- `delayed_flights`
- `avg_arrival_delay_positive`
- `delayed_15_pct`
- `adjusted_delay_pct`
- `delay_impact_score`

### Algoritmo

Foi usado:

- `KMeans`

Configuracao:

```python
KMeans(n_clusters=4, n_init=20, random_state=42)
```

O KMeans agrupa pontos com base em distancia. Por isso, antes do treino, as variaveis foram padronizadas com `StandardScaler`.

Sem padronizacao, colunas com valores maiores, como `flights`, dominariam artificialmente a formacao dos clusters.

### PCA para visualizacao

Tambem foi usado:

```python
PCA(n_components=2)
```

O PCA reduziu as features para dois eixos principais, permitindo visualizar os clusters em grafico 2D.

Importante:

> O PCA foi usado para visualizacao. A interpretacao de negocio deve considerar tambem o resumo numerico dos clusters.

## Metricas nao supervisionadas

Como nao existe uma resposta correta previa, nao ha `accuracy`, `precision` ou `recall` na clusterizacao.

Foram usadas duas medidas principais:

| Metrica | Valor | Como interpretar |
|---|---:|---|
| `silhouette_score` | 0,3484 | Mede separacao e coesao dos clusters. |
| `pca_explained_variance` | 96,70% | Mede quanta informacao foi preservada na visualizacao 2D do PCA. |

### Silhouette score

O `silhouette_score` varia de -1 a 1.

| Faixa | Interpretacao |
|---:|---|
| Proximo de 1 | Clusters bem separados e coesos. |
| Proximo de 0 | Clusters parcialmente sobrepostos. |
| Negativo | Muitos pontos podem estar no cluster errado. |

O valor encontrado foi:

```text
0,3484
```

Interpretacao:

> A clusterizacao apresenta separacao moderada. Os grupos nao sao perfeitamente separados, mas existe estrutura util nos dados.

Para dados operacionais reais, essa leitura e aceitavel. Rotas aereas tendem a formar perfis continuos, nao blocos perfeitamente isolados.

### Variancia explicada pelo PCA

O PCA explicou:

```text
96,70%
```

Isso indica que a visualizacao em duas dimensoes preserva boa parte da estrutura das variaveis originais.

Porem, PCA alto nao significa automaticamente que a clusterizacao e excelente. Ele apenas indica que o grafico 2D e uma boa representacao visual dos dados usados.

## Resultados nao supervisionados

Arquivo:

- `app/machine-learning/outputs/unsupervised_cluster_summary.csv`

| Cluster | Rotas | Media de voos | Atraso medio positivo | % medio de atraso 15+ | Score medio de impacto | Rota exemplo |
|---:|---:|---:|---:|---:|---:|---|
| 3 | 249 | 5.818,82 | 13,48 | 20,56 | 22,70 | SFO-LAX |
| 1 | 564 | 1.464,86 | 16,46 | 25,03 | 17,82 | DFW-HNL |
| 0 | 1.292 | 1.567,08 | 11,93 | 18,54 | 14,24 | CLT-MIA |
| 2 | 823 | 1.282,88 | 8,07 | 12,30 | 10,23 | LIH-HNL |

### Interpretacao dos clusters

**Cluster 3**

Rotas de alto volume e alto impacto. Possui a maior media de voos por rota e o maior score medio de impacto. Deve ser tratado como grupo prioritario para acompanhamento operacional.

**Cluster 1**

Rotas com maior atraso medio positivo e maior percentual medio de atraso de 15+ minutos. Apesar de ter volume menor que o cluster 3, concentra rotas com maior severidade relativa de atraso.

**Cluster 0**

Rotas intermediarias. Possuem atraso e score de impacto medios, funcionando como grupo de comportamento operacional medio.

**Cluster 2**

Rotas menos criticas dentro do recorte analisado. Possuem menor atraso medio positivo, menor percentual de atraso e menor score medio de impacto.

## Graficos nao supervisionados

Arquivos gerados:

- `app/machine-learning/outputs/unsupervised_route_clusters_pca.html`
- `app/machine-learning/outputs/unsupervised_cluster_summary.html`

### Grafico PCA dos clusters

O grafico `unsupervised_route_clusters_pca.html` mostra as rotas em duas dimensoes.

Como interpretar:

- cada ponto representa uma rota;
- a cor representa o cluster;
- o tamanho representa o volume de voos;
- pontos proximos possuem perfil semelhante nas features usadas.

Com `pca_explained_variance = 96,70%`, essa visualizacao preserva boa parte da informacao original e pode ser usada para entender a distribuicao geral dos grupos.

### Grafico de resumo dos clusters

O grafico `unsupervised_cluster_summary.html` compara os clusters pelo `avg_delay_impact_score`.

Como interpretar:

- barras mais altas indicam clusters com maior impacto medio de atraso;
- a cor representa o percentual medio de voos atrasados;
- o texto indica a quantidade de rotas em cada cluster.

Esse grafico ajuda a identificar rapidamente quais grupos devem receber prioridade operacional.

## Apresentacao critica dos resultados

### Principais conclusoes

- A regressao logistica foi o melhor modelo supervisionado entre os algoritmos testados.
- O desempenho supervisionado foi alto, com `ROC AUC = 0,9673` e `F1 = 0,8178`.
- O modelo tem forte capacidade de identificar voos atrasados, com `recall = 0,8987`.
- A alta performance deve ser interpretada dentro do escopo correto: predicao apos a decolagem.
- A clusterizacao encontrou grupos interpretaveis de rotas, especialmente separando rotas de alto volume e alto impacto.
- O `silhouette_score = 0,3484` indica separacao moderada, nao perfeita.
- A clusterizacao e util como ferramenta de segmentacao e priorizacao, nao como prova de que existem grupos naturalmente isolados.

### Limitacoes

- O modelo supervisionado nao representa uma previsao antes da partida.
- `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF` aumentam muito o poder preditivo porque ja descrevem parte da operacao em andamento.
- Nao foram usados dados climaticos historicos detalhados.
- Nao ha informacoes sobre conexoes, tripulacao, manutencao, malha operacional ou capacidade aeroportuaria.
- O target e desbalanceado, pois apenas 18,61% dos voos atrasaram 15 minutos ou mais.
- A avaliacao supervisionada usa divisao aleatoria, nao validacao temporal.
- A clusterizacao depende das features agregadas por rota; outros recortes poderiam gerar grupos diferentes.
- O numero de clusters foi fixado em 4, sem comparacao sistematica com outros valores de `k`.

### Riscos de interpretacao

- Nao se deve afirmar que o modelo preve atraso antes do voo.
- Nao se deve interpretar `accuracy` isoladamente, pois a base e desbalanceada.
- Nao se deve tratar o `silhouette_score` como uma verdade absoluta de negocio.
- Clusters devem ser avaliados pela combinacao de metrica tecnica e interpretabilidade operacional.

## Melhorias e proximos passos

### Para o modelo supervisionado

- Criar uma versao pre-voo removendo `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF`.
- Fazer validacao temporal, treinando em meses anteriores e testando em meses posteriores.
- Testar modelos adicionais, como Gradient Boosting, XGBoost, LightGBM ou HistGradientBoosting.
- Calibrar o limiar de classificacao conforme o objetivo operacional.
- Avaliar `precision-recall curve`, especialmente por causa do desbalanceamento do target.
- Adicionar interpretabilidade com permutation importance ou SHAP.
- Criar features de periodo do dia, feriados, estacao do ano e sazonalidade.
- Enriquecer com dados climaticos historicos por aeroporto e horario.

### Para a clusterizacao

- Testar diferentes valores de `k` e comparar `silhouette_score`, inercia e interpretabilidade.
- Avaliar outros algoritmos, como DBSCAN, HDBSCAN ou Gaussian Mixture Models.
- Criar clusterizacoes por aeroporto, companhia aerea, estado ou periodo do dia.
- Comparar clusters por estabilidade temporal.
- Validar os clusters com conhecimento de negocio, por exemplo verificando se as rotas agrupadas fazem sentido operacionalmente.

## Como executar

No terminal, a partir da raiz do projeto:

```powershell
.\.venv\Scripts\python.exe app\machine-learning\train_ml_models.py
```

O script executa:

1. leitura da base tratada;
2. preparacao das features supervisionadas;
3. divisao treino/teste;
4. treinamento da regressao logistica e Random Forest;
5. avaliacao dos modelos supervisionados;
6. persistencia dos modelos treinados;
7. clusterizacao das rotas;
8. geracao de tabelas e graficos HTML.

## Configuracao de amostragem

O script permite treinar com amostra ou com a base completa:

```python
USE_SAMPLE = True
SUPERVISED_SAMPLE_SIZE = 180_000
CHUNK_SIZE = 250_000
```

Para usar a base inteira:

```python
USE_SAMPLE = False
```

`CHUNK_SIZE` controla quantas linhas sao lidas por vez do CSV. Ele nao define o tamanho final do treino. O tamanho final depende de `USE_SAMPLE` e `SUPERVISED_SAMPLE_SIZE`.

## Arquivos gerados

Supervisionado:

- `supervised_classification_metrics.csv`
- `supervised_sample_profile.csv`
- `supervised_roc_curve.html`
- `supervised_confusion_matrix.html`
- `logistic_regression_classifier.joblib`
- `random_forest_classifier.joblib`

Nao supervisionado:

- `unsupervised_route_clusters.csv`
- `unsupervised_cluster_summary.csv`
- `unsupervised_route_clusters_pca.html`
- `unsupervised_cluster_summary.html`
