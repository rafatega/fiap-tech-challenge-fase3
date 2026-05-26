# EDA e Tratamento dos Dados

Este documento resume o que foi feito no notebook `app/eda/eda.ipynb` e serve como apoio para quem continuar o projeto.

## Objetivo da EDA

Analisar atrasos de voos nos EUA, com foco em:

- qualidade inicial dos dados;
- aeroportos de origem e destino;
- rotas;
- volume de voos;
- atraso medio;
- percentual de voos atrasados;
- criacao de rankings de criticidade.

A pergunta central da analise foi:

> Quais aeroportos, rotas e caracteristicas estao mais associados a atrasos em voos?

## Bases utilizadas

As bases principais estao em `app/database`:

- `airlines.csv`: cadastro das companhias aereas.
- `airports.csv`: cadastro dos aeroportos, com IATA, nome, cidade, estado, pais, latitude e longitude.
- `flights.csv`: base principal de voos.
- `airport_id_to_iata_2015_10.csv`: tabela externa usada para converter `AIRPORT_ID` numerico para codigo IATA.

Os outputs da EDA usados pelo dashboard ficam principalmente em:

- `app/database/outputs`
- `app/eda/outputs`

## Perfil dos dados antes do tratamento

Foram criados perfis de colunas para entender os dados antes de limpar:

- tipo de dado;
- quantidade de nulos;
- percentual de nulos;
- quantidade de valores unicos;
- percentual de valores unicos.

Arquivos gerados:

- `completed_column_profile_airlines_before_treatment.csv`
- `completed_column_profile_airports_before_treatment.csv`
- `completed_column_profile_flights_before_treatment.csv`
- `completed_column_profile_before_treatment.csv`

Principais pontos encontrados:

- `airlines.csv` nao apresentou problemas relevantes de missing.
- `airports.csv` tinha 3 aeroportos sem latitude/longitude.
- `flights.csv` tinha muitos nulos em colunas de cancelamento, horarios reais, atrasos e causas de atraso.
- As causas de atraso so aparecem para voos com atraso relevante; por isso muitos nulos nessas colunas nao significam necessariamente erro.
- Alguns aeroportos em `flights.csv` estavam com codigo numerico, e nao com codigo IATA de 3 letras.

## Inconsistencia de aeroportos

Foi identificado que alguns registros da base `flights.csv` tinham `ORIGIN_AIRPORT` ou `DESTINATION_AIRPORT` com mais de 3 caracteres.

Regra usada:

- codigo IATA valido deve ter 3 caracteres;
- valores com mais de 3 caracteres foram tratados como `AIRPORT_ID` numerico.

Esses casos foram exportados em:

- `flights_with_invalid_airport_codes.csv`

Observacao importante:

- os casos ocorreram em outubro de 2015;
- para corrigir, foi usada uma tabela externa do BTS com mapeamento entre `AIRPORT_ID` e IATA;
- o arquivo final usado no projeto foi `airport_id_to_iata_2015_10.csv`.

Durante o tratamento, foram preservadas as colunas originais:

- `ORIGIN_AIRPORT_ORIGINAL`
- `DESTINATION_AIRPORT_ORIGINAL`

Tambem foi criada uma flag:

- `AIRPORT_CODE_WAS_FIXED`

Essa flag indica quais linhas tiveram codigo de aeroporto corrigido.

## Tratamento da tabela de aeroportos

Na base `airports.csv`, 3 aeroportos tinham coordenadas faltantes:

- `ECP`
- `PBG`
- `UST`

As coordenadas foram preenchidas manualmente a partir de consulta externa.

Depois disso, a base de aeroportos foi usada duas vezes no join:

- como aeroporto de origem;
- como aeroporto de destino.

Para evitar conflito de nomes, as colunas foram renomeadas com prefixos.

Exemplos:

- `AIRPORT` virou `ORIGIN_AIRPORT_NAME` ou `DESTINATION_AIRPORT_NAME`;
- `CITY` virou `ORIGIN_CITY` ou `DESTINATION_CITY`;
- `LATITUDE` virou `ORIGIN_LATITUDE` ou `DESTINATION_LATITUDE`;
- `LONGITUDE` virou `ORIGIN_LONGITUDE` ou `DESTINATION_LONGITUDE`.

## Join das bases

A base analitica foi criada a partir do join entre:

1. `flights.csv`
2. `airlines.csv`
3. `airports.csv` como origem
4. `airports.csv` como destino

Resultado esperado:

- cada voo passa a ter nome da companhia;
- nome, cidade, estado, pais e coordenadas do aeroporto de origem;
- nome, cidade, estado, pais e coordenadas do aeroporto de destino.

O join foi feito com `how="left"` para preservar os voos.

## Colunas criadas

Depois do join, foram criadas colunas auxiliares para analise:

- `FLIGHT_DATE`: data do voo criada a partir de `YEAR`, `MONTH` e `DAY`.
- `SCHEDULED_DEPARTURE_HOUR`: hora programada de partida.
- `SCHEDULED_ARRIVAL_HOUR`: hora programada de chegada.
- `ROUTE`: rota no formato `ORIGIN_AIRPORT-DESTINATION_AIRPORT`.
- `IS_WEEKEND`: indica se o voo ocorreu no fim de semana.
- `ARRIVAL_DELAY_POSITIVE`: atraso de chegada com valores negativos convertidos para zero.
- `IS_DELAYED_15`: indica se o voo atrasou 15 minutos ou mais.

## Recorte usado para analise de atraso

Para analisar atrasos, foi criada uma base apenas com voos finalizados.

Filtro aplicado:

- `CANCELLED = 0`
- `DIVERTED = 0`
- `ARRIVAL_DELAY` nao nulo

Essa base foi chamada no notebook de `df_completed_raw`.

Depois foram removidas colunas que deixam de ser uteis nesse recorte:

- `CANCELLED`
- `DIVERTED`
- `CANCELLATION_REASON`

A base tratada final foi chamada de `df_completed`.

## Tratamento de nulos

Colunas de motivo de atraso:

- `AIR_SYSTEM_DELAY`
- `SECURITY_DELAY`
- `AIRLINE_DELAY`
- `LATE_AIRCRAFT_DELAY`
- `WEATHER_DELAY`

Tratamento:

- nulos foram preenchidos com `0`.

Motivo:

- quando o voo nao tem atraso relevante, essas colunas ficam nulas;
- para EDA, `0` representa ausencia de atraso atribuido a aquele motivo.

Coluna `TAIL_NUMBER`:

- nulos foram preenchidos com `UNKNOWN`.

Motivo:

- evita perder registros por ausencia de identificador da aeronave.

## Comparacao antes e depois

Foram gerados dois perfis para comparar o efeito do tratamento:

- `completed_column_profile_before_treatment.csv`
- `completed_column_profile_after_treatment.csv`

Esses arquivos mostram:

- missing antes;
- missing depois;
- colunas impactadas pelo tratamento;
- reducao de nulos.

No dashboard, esses perfis sao usados na aba de tratamento.

## Variaveis numericas e outliers

Foi gerado um perfil numerico em:

- `completed_numeric_profile.csv`

Esse arquivo contem:

- media;
- desvio padrao;
- minimo;
- percentis;
- maximo;
- IQR;
- limites por IQR;
- quantidade e percentual de outliers.

Tambem foram gerados histogramas:

- `completed_numeric_distributions_raw.png`: distribuicoes sem aplicar recorte visual por percentis.
- `completed_numeric_distributions.png`: distribuicoes com visualizacao limitada por p01 e p99.

Decisao tomada:

- os outliers de atraso foram mantidos.

Motivo:

- atrasos extremos podem ser eventos reais e relevantes operacionalmente;
- remover esses valores poderia apagar justamente casos importantes para a analise.

## Analises de atraso geradas

Foram criadas tres tabelas principais de ranking.

### Rotas

Arquivo:

- `completed_route_delay_profile.csv`

Granularidade:

- rota origem-destino.

Principais colunas:

- `ROUTE`
- `ORIGIN_AIRPORT`
- `DESTINATION_AIRPORT`
- `flights`
- `delayed_flights`
- `avg_arrival_delay`
- `delayed_15_rate`
- `delayed_15_pct`
- `adjusted_delay_pct`
- `delay_impact_score`

### Aeroportos de origem

Arquivo:

- `completed_origin_airport_delay_profile.csv`

Granularidade:

- aeroporto de origem.

Score:

- `origin_delay_impact_score`

### Aeroportos de destino

Arquivo:

- `completed_destination_airport_delay_profile.csv`

Granularidade:

- aeroporto de destino.

Score:

- `destination_delay_impact_score`

## Como o score de criticidade foi calculado

O score foi criado para evitar uma conclusao ruim baseada apenas no maior atraso medio.

Problema evitado:

- uma rota com poucos voos pode ter atraso medio alto por acaso;
- isso nao significa necessariamente que ela seja operacionalmente mais critica.

Componentes usados:

- taxa de voos atrasados em 15 minutos ou mais;
- volume de voos;
- atraso medio positivo.

Foi aplicada suavizacao estatistica na taxa de atraso:

- rotas ou aeroportos com poucos voos sao puxados para a media global;
- isso reduz distorcao em grupos pequenos.

Peso usado no score:

- 60% taxa ajustada de atraso;
- 25% volume normalizado;
- 15% atraso medio positivo normalizado.

Interpretacao:

- score alto indica maior prioridade operacional;
- nao significa apenas maior atraso medio;
- combina frequencia, volume e severidade.

## Cuidados de interpretacao

Evite concluir apenas pelo maior atraso medio.

Sempre considerar:

- quantidade de voos;
- percentual de voos atrasados;
- atraso medio;
- score de criticidade;
- contexto da rota ou aeroporto.

Rotas com poucos voos podem aparecer com atrasos medios extremos, mas isso pode nao representar um problema recorrente.

Por isso o dashboard possui filtro de volume minimo de voos.

## Outputs mais importantes para o dashboard

O dashboard `app/dashboard/analytics_airport_delay.py` usa principalmente:

- `completed_column_profile_airlines_before_treatment.csv`
- `completed_column_profile_airports_before_treatment.csv`
- `completed_column_profile_flights_before_treatment.csv`
- `completed_column_profile_before_treatment.csv`
- `completed_column_profile_after_treatment.csv`
- `completed_numeric_profile.csv`
- `completed_route_delay_profile.csv`
- `completed_origin_airport_delay_profile.csv`
- `completed_destination_airport_delay_profile.csv`
- `flights_with_invalid_airport_codes.csv`
- `completed_numeric_distributions.png`
- `completed_numeric_distributions_raw.png`

## Resumo do fluxo

1. Ler bases cruas.
2. Criar perfil inicial de colunas.
3. Corrigir coordenadas faltantes em aeroportos.
4. Identificar codigos de aeroporto invalidos na base de voos.
5. Converter `AIRPORT_ID` numerico para IATA.
6. Fazer join entre voos, companhias e aeroportos.
7. Criar colunas auxiliares de data, hora, rota e atraso.
8. Filtrar voos finalizados.
9. Tratar nulos em causas de atraso e `TAIL_NUMBER`.
10. Gerar perfis antes/depois.
11. Analisar variaveis numericas e outliers.
12. Criar rankings de rotas, aeroportos de origem e aeroportos de destino.
13. Calcular score de criticidade.
14. Alimentar o dashboard Streamlit com os outputs gerados.