# Estudo de Outliers

## Objetivo

Este estudo avalia se os valores extremos da base de voos devem ser removidos, limitados ou mantidos no treinamento do modelo supervisionado de previsao de atraso.

A duvida principal era:

> Valores muito altos em colunas como `DISTANCE`, `DEPARTURE_DELAY`, `TAXI_OUT` e `WHEELS_OFF` sao erros de dados ou eventos reais que ajudam o modelo a prever atraso?

A decisao foi baseada em experimento, nao apenas em regra estatistica.

## Contexto da base

A base tratada possui:

| Indicador | Valor |
|---|---:|
| Total de voos | 5.714.008 |
| Voos com atraso de 15 minutos ou mais | 1.063.439 |
| Voos sem atraso de 15 minutos ou mais | 4.650.569 |
| Taxa de atraso | 18,61% |

O target usado no modelo foi:

- `IS_DELAYED_15 = 1`: voo atrasou 15 minutos ou mais.
- `IS_DELAYED_15 = 0`: voo nao atrasou 15 minutos ou mais.

Como a classe positiva representa cerca de 18,6% da base, a divisao entre treino e teste foi feita com `stratify=y`, preservando a proporcao de atrasos nos dois conjuntos.

Com `test_size=0.25`, a divisao aproximada foi:

| Conjunto | Linhas |
|---|---:|
| Treino | 4.285.506 |
| Teste | 1.428.502 |

## Por que outliers exigem cuidado neste problema

Nem todo valor extremo e erro.

Em uma base de voos, alguns valores altos podem representar situacoes reais:

- `DISTANCE` alta pode ser apenas uma rota longa.
- `DEPARTURE_DELAY` alto pode ser um atraso operacional real.
- `TAXI_OUT` alto pode indicar congestionamento, fila de decolagem ou condicao operacional especifica.
- `WHEELS_OFF` alto pode estar associado a voos em horarios mais tarde.

Se esses casos forem removidos sem analise, o modelo pode perder informacao justamente sobre os voos mais problematicos.

Por isso, a pergunta correta nao e apenas:

> Este valor e extremo?

Mas sim:

> Este valor extremo e impossivel ou ele representa um evento raro, mas real?

## Estrategias testadas

Foram comparadas seis estrategias:

| Estrategia | Descricao |
|---|---|
| `baseline_keep_extremes` | Mantem todos os valores extremos sem tratamento adicional. |
| `robust_scaler_keep_extremes` | Mantem os extremos, mas troca `StandardScaler` por `RobustScaler`. |
| `winsorize_p01_p99` | Mantem todas as linhas, mas limita valores abaixo do p01 e acima do p99. |
| `winsorize_p05_p95` | Mantem todas as linhas, mas limita valores abaixo do p05 e acima do p95. |
| `remove_train_outside_p01_p99` | Remove do treino linhas fora do intervalo p01-p99. O teste permanece original. |
| `remove_train_outside_p05_p95` | Remove do treino linhas fora do intervalo p05-p95. O teste permanece original. |

Uma diferenca importante:

- Winsorizacao nao remove linhas. Ela apenas substitui valores extremos pelos limites definidos.
- Remocao elimina a linha inteira do treino quando alguma coluna analisada fica fora do intervalo.

O conjunto de teste foi mantido com a distribuicao real da base. Isso evita medir o modelo em um mundo artificialmente limpo.

## Resultados do modelo

Resultados salvos em:

- `app/machine-learning/outputs/outlier_study/outlier_strategy_metrics.csv`

| Estrategia | Accuracy | Precision | Recall | F1 | ROC AUC | Linhas removidas do treino |
|---|---:|---:|---:|---:|---:|---:|
| `baseline_keep_extremes` | 0,9255 | 0,7502 | 0,8987 | 0,8178 | 0,9673 | 0 |
| `robust_scaler_keep_extremes` | 0,9254 | 0,7499 | 0,8986 | 0,8175 | 0,9672 | 0 |
| `remove_train_outside_p01_p99` | 0,9203 | 0,7311 | 0,9047 | 0,8087 | 0,9672 | 285.936 |
| `winsorize_p01_p99` | 0,9246 | 0,7470 | 0,8996 | 0,8162 | 0,9672 | 0 |
| `remove_train_outside_p05_p95` | 0,9058 | 0,6859 | 0,9113 | 0,7827 | 0,9646 | 1.483.921 |
| `winsorize_p05_p95` | 0,9167 | 0,7245 | 0,8914 | 0,7993 | 0,9643 | 0 |

O melhor resultado geral foi obtido com `baseline_keep_extremes`, ou seja, mantendo os valores extremos.

Essa estrategia apresentou:

- maior `accuracy`;
- maior `precision`;
- maior `f1`;
- maior `roc_auc`;
- sem remocao de linhas do treino.

## Leitura dos resultados

A estrategia baseline ficou levemente acima de `RobustScaler` e `winsorize_p01_p99`. Isso indica que os extremos nao estavam prejudicando o modelo.

A winsorizacao p01-p99 teve resultado muito proximo, mas inferior ao baseline. Isso sugere que limitar moderadamente os extremos nao destruiu completamente o sinal, mas tambem nao trouxe ganho.

A winsorizacao p05-p95 foi pior. Esse corte e mais agressivo, porque limita os 5% menores e os 5% maiores valores. Na pratica, ele achatou informacoes que eram uteis para o modelo.

A remocao p05-p95 foi a pior estrategia entre as principais metricas. Ela removeu 1.483.921 linhas do treino, aproximadamente 34,6% dos dados de treinamento. Isso e agressivo demais para este problema.

## Analise das caudas

Resultados salvos em:

- `app/machine-learning/outputs/outlier_study/outlier_tail_diagnostics.csv`

Essa analise compara os voos nos maiores valores de cada coluna contra o restante da base.

As principais colunas sao:

- `threshold`: valor a partir do qual comeca a cauda.
- `tail_rows`: quantidade de voos na cauda.
- `tail_pct`: percentual da base naquela cauda.
- `target_rate_tail`: taxa de atraso dentro da cauda.
- `target_rate_rest`: taxa de atraso fora da cauda.
- `target_rate_lift`: quantas vezes a cauda atrasa mais que o restante.

Principais achados:

| Feature | Corte | Interpretacao |
|---|---|---|
| `DEPARTURE_DELAY` | p95 >= 67 min | 99,99% dos voos nessa cauda atrasaram. Lift de 6,98x. |
| `DEPARTURE_DELAY` | p99 >= 167 min | Praticamente 100% dos voos nessa cauda atrasaram. Lift de 5,62x. |
| `TAXI_OUT` | p95 >= 31 min | 56,21% dos voos nessa cauda atrasaram. Lift de 3,42x. |
| `TAXI_OUT` | p99 >= 50 min | 91,41% dos voos nessa cauda atrasaram. Lift de 5,12x. |
| `WHEELS_OFF` | p99 >= 2307 | 45,64% dos voos nessa cauda atrasaram. Lift de 2,49x. |
| `DISTANCE` | p99 >= 2588 | 19,16% dos voos nessa cauda atrasaram. Lift de 1,03x. |

Essa leitura mostra que `DEPARTURE_DELAY` e `TAXI_OUT` possuem extremos altamente informativos.

Em linguagem simples:

> Quando o atraso de partida ou o tempo de taxiamento esta muito alto, a chance de o voo terminar atrasado aumenta muito.

Portanto, esses extremos nao parecem ser sujeira nos dados. Eles representam eventos operacionais reais e relevantes para a previsao.

Ja `DISTANCE` teve lift proximo de 1. Isso indica que voos muito longos nao atrasam muito mais que o restante da base. Mesmo assim, distancia alta tambem nao deve ser considerada erro automaticamente, porque representa uma caracteristica geografica real da rota.

## Conclusao

A estrategia escolhida foi manter os valores extremos no treinamento.

Motivos:

- O baseline apresentou o melhor desempenho geral.
- Os extremos de `DEPARTURE_DELAY` e `TAXI_OUT` carregam forte sinal preditivo.
- A remocao de outliers reduziu informacao relevante do treino.
- A winsorizacao agressiva p05-p95 piorou as metricas.
- Em dados de voos, valores extremos podem representar eventos operacionais reais, nao erros.

Conclusao final:

> Para este problema, os valores extremos foram mantidos porque representam eventos reais da operacao aerea e melhoram a capacidade preditiva do modelo. O tratamento classico de outliers por percentil foi avaliado, mas nao foi adotado porque reduziu o desempenho e poderia remover ou limitar informacoes importantes sobre atrasos severos.

## Como executar o estudo

Na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe app\machine-learning\outlier_sensitivity_study.py
```

Por padrao, o script pode ser configurado com amostra ou base inteira:

```python
USE_SAMPLE = True
SAMPLE_SIZE = 120_000
```

Para rodar com a base inteira:

```python
USE_SAMPLE = False
```

Mesmo com `USE_SAMPLE = False`, o arquivo continua sendo lido em chunks. A diferenca e que todos os chunks sao mantidos para o estudo.

Configuracoes principais:

| Parametro | Funcao |
|---|---|
| `CHUNK_SIZE` | Controla quantas linhas sao lidas por vez do CSV. |
| `SAMPLE_SIZE` | Controla quantas linhas serao usadas quando `USE_SAMPLE = True`. |
| `USE_SAMPLE` | Liga ou desliga a amostragem. |
| `test_size=0.25` | Separa 25% dos dados para teste. |
| `stratify=y` | Mantem a proporcao do target no treino e no teste. |

## Arquivos gerados

- `app/machine-learning/outputs/outlier_study/outlier_strategy_metrics.csv`
- `app/machine-learning/outputs/outlier_study/outlier_tail_diagnostics.csv`
