# Dashboard - Analytics Airport Delay

## Objetivo

O arquivo `analytics_airport_delay.py` implementa um dashboard em Streamlit para apresentar a jornada analítica do projeto: entendimento dos dados, tratamento, EDA, insights operacionais de atraso e resultados de Machine Learning.

## Tecnologias

- Linguagem: Python
- Interface: Streamlit
- Manipulação de dados: pandas e numpy
- Visualizações: Plotly e componentes HTML do Streamlit

## Dados utilizados

O dashboard consome artefatos já gerados nas etapas anteriores do projeto.

Principais entradas:

- `app/database/airlines.csv`
- `app/database/airports.csv`
- `app/database/flights.csv`
- `app/database/airport_id_to_iata_2015_10.csv`
- `app/database/outputs/*.csv`
- `app/eda/outputs/*.csv`
- `app/eda/outputs/*.png`
- `app/machine-learning/outputs/*.csv`
- `app/machine-learning/outputs/*.html`

## Conteúdo do dashboard

Abas principais:

- `Visão Geral`: resumo executivo dos achados.
- `Qualidade dos Dados`: perfil das bases cruas e inconsistências.
- `Tratamento / Data Prep`: principais decisões de preparação dos dados.
- `Insights`: rankings de aeroportos, rotas e relações entre volume e atraso.
- `Rotas e Aeroportos`: mapas, bolhas e rotas críticas.
- `Distribuições`: perfis numéricos e análise de outliers.
- `ML Supervisionado`: resultados dos modelos de classificação.
- `ML Não Supervisionado`: clusterização de rotas com KMeans.
- `ML Conclusões`: limitações, conclusões e próximos passos.

## Como executar

Na raiz do projeto:

```powershell
.\.venv\Scripts\streamlit.exe run app\dashboard\analytics_airport_delay.py
```

## Observações de manutenção

- O dashboard não refaz o tratamento nem o treinamento dos modelos; ele lê arquivos já exportados.
- Se alguma tabela, imagem ou HTML não existir, a respectiva seção exibe uma mensagem informativa.
- Para atualizar os números do dashboard, execute novamente os scripts de EDA ou Machine Learning que geram os artefatos consumidos.
