# Estudo de outliers

## Ideia principal

Nem todo valor extremo é erro. Em dados de voos, distancia muito alta, tempo de taxi maior ou atraso severo podem ser eventos raros, mas reais. Remover esses casos sem criterio pode deixar o modelo artificialmente limpo e piorar sua capacidade de lidar com operacoes reais.

Por isso, o tratamento de outliers deve ser decidido por experimento e por contexto de negocio, nao apenas por regra estatistica.

## Separar erro de evento raro

Use esta regra pratica:

- Erro de dado: valor impossivel ou inconsistente com a definicao da coluna.
- Evento raro real: valor extremo, mas plausivel para uma rota, aeroporto, horario ou evento operacional.

Exemplos:

- `DISTANCE` alta pode ser normal para rotas longas.
- `DEPARTURE_DELAY` muito alto pode ser um atraso real.
- `TAXI_OUT` muito alto pode ocorrer em aeroportos congestionados ou em situacoes operacionais especificas.
- Horarios codificados fora de 0-2400 seriam candidatos mais fortes a erro.

## Risco de remover outliers reais

Remover extremos pode gerar tres problemas:

- Perda de sinal: os casos raros podem ser justamente os mais informativos para prever atraso.
- Modelo otimista demais: a metrica melhora em um teste tambem limpo, mas o modelo falha quando encontra extremos reais.
- Mudanca de distribuicao: o treino deixa de representar o mundo real.

O overfitting normalmente nao acontece porque voce manteve outliers reais. Ele acontece quando o modelo aprende padroes especificos demais do treino. Remover dados reais pode reduzir variancia, mas tambem pode aumentar vies e piorar generalizacao em casos importantes.

## Estrategia recomendada

Para este projeto, a abordagem mais defensavel e comparar cenarios:

1. Manter extremos como baseline.
2. Usar `RobustScaler`, mantendo extremos.
3. Winsorizar p1-p99, aprendendo limites apenas no treino.
4. Winsorizar p5-p95, como cenario agressivo.
5. Remover linhas extremas apenas do treino e avaliar no teste original.

A avaliacao deve preservar o teste com distribuicao real. Se voce remove extremos do teste, mede desempenho em um mundo artificialmente facil.

## Como executar

Na raiz do projeto:

```powershell
.\.venv\Scripts\python.exe app\machine-learning\outlier_sensitivity_study.py
```

Por padrao, o script roda com amostra para acelerar os testes:

```python
USE_SAMPLE = True
SAMPLE_SIZE = 120_000
```

Para rodar com a base inteira, altere no script:

```python
USE_SAMPLE = False
```

O script continuara lendo o CSV em chunks, mas vai concatenar todos os chunks antes do treino. Isso pode consumir bastante memoria, porque `flights_treated.csv` tem milhoes de linhas.

Saidas:

- `app/machine-learning/outputs/outlier_study/outlier_strategy_metrics.csv`
- `app/machine-learning/outputs/outlier_study/outlier_tail_diagnostics.csv`

## Como interpretar

`outlier_strategy_metrics.csv` compara metricas de classificacao:

- `roc_auc`: capacidade geral de ordenar risco de atraso.
- `f1`: equilibrio entre precision e recall.
- `recall`: quanto o modelo captura dos voos atrasados.
- `precision`: quanto das previsoes de atraso realmente atrasam.
- `train_rows_removed_pct`: percentual removido do treino quando a estrategia remove linhas.

`outlier_tail_diagnostics.csv` mostra se os extremos carregam sinal:

- `tail_pct`: percentual de linhas acima do p95 ou p99.
- `target_rate_tail`: taxa de atraso nos extremos.
- `target_rate_rest`: taxa de atraso fora dos extremos.
- `target_rate_lift`: quanto a cauda aumenta ou reduz a chance de atraso.

Se `target_rate_lift` for muito maior que 1, a cauda contem sinal preditivo. Nesse caso, remover os extremos tende a apagar informacao util.

## Decisao sugerida

Use esta matriz:

- Baseline vence ou empata: mantenha extremos.
- `RobustScaler` vence: mantenha extremos e use escala robusta.
- Winsor p1-p99 vence levemente: use clipping moderado, documentando que eventos extremos continuam representados ate o limite p99.
- Winsor p5-p95 vence: desconfie; pode estar limpando demais e perdendo eventos reais.
- Remocao vence no teste original: investigue quais linhas sairam antes de adotar, porque pode estar removendo segmentos especificos de rotas, aeroportos ou atrasos severos.

Para `DISTANCE`, a recomendacao inicial e nao remover por percentil. Distancia representa a realidade geografica da rota. Para `DEPARTURE_DELAY` e `TAXI_OUT`, prefira manter ou winsorizar moderadamente, porque valores extremos podem ser raros, mas operacionalmente reais.
