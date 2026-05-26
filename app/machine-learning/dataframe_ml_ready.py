# Essa etapa será reponsável por preparar os dados para o dashboard, realizando as seguintes tarefas:
# 1. Carregar os dados de voos filtrada apenas finalizados, já tratada os nulos e inconsistencias.
# 2. Tratar outliers que possam distorcer as análises.
# 3. Salvar os dados tratados em um novo arquivo CSV para ser utilizado no treinamento do modelo.

import pandas as pd
