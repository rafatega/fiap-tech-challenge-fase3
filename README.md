# Tech Challenge Fase 3 - FIAP

## Próximos passos

### Modelo supervisionado
* Estudar como realizar o tratamento de outliers da base:
- Quais colunas que possuem outliers devem ser tratadas?
- Quais colunas não devem ser tratados pois os outliers representam casos reais relevantes para o modelo?
- Qual técnica de tratamento de outliers deve ser aplicada para cada coluna (remoção, winsorização, transformação, etc)?

* Entender quais features serão usadas no modelo supervisionado:
- Quais features tem maior potencial preditivo para o modelo de atraso?
- Quais features tem alta correlação com a variável alvo (atraso)?

### Modelo não supervisionado
* A base usada para o modelo atualmente é diferente da base usada para o modelo supervisionado. Avaliar se é possível usar a mesma base para ambos os modelos, ou se é necessário realizar algum tipo de transformação ou engenharia de features para alinhar as bases.
* As métricas de desempenho não parecem muito boas. Avaliar se é possível melhorar o desempenho do modelo não supervisionado, seja ajustando os hiperparâmetros, usando uma técnica de clusterização diferente, ou realizando algum tipo de pré-processamento ou engenharia de features.
