from datetime import datetime
from meteostat import Point, Daily
import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf

# Definindo a localização e período
location = Point(40.7128, -74.0060)  # Latitude e Longitude para Nova York
start = '2023-01-01'
end = '2024-01-01'

# Convertendo 'start' e 'end' para objetos datetime
start_date = datetime.strptime(start, '%Y-%m-%d')
end_date = datetime.strptime(end, '%Y-%m-%d')

# Obtendo os dados diários de temperatura e precipitação
data = Daily(location, start_date, end_date)
data = data.fetch()

# Criando um DataFrame com os dados de temperatura e precipitação
df = pd.DataFrame({
    'date': data.index,
    'temperature': data['tavg'],
    'precipitation': data['prcp']
})
df.set_index('date', inplace=True)

# Plotando as temperaturas e precipitações
df.plot(subplots=True, figsize=(10, 6), title='Temperatura e Precipitação Diárias - Nova York')
plt.xlabel('Date')
plt.show()

# Calculando a Correlação Cruzada (CCF) entre Temperatura e Precipitação
lags = range(-20, 21)
ccf_values = ccf(df['temperature'], df['precipitation'])[:len(lags)]

# Encontrando o lag mais importante
max_ccf_index = abs(ccf_values).argmax()
most_important_lag = lags[max_ccf_index]
most_important_ccf = ccf_values[max_ccf_index]

# Exibindo o lag mais importante
print(f"Lag mais importante: {most_important_lag}, Valor da correlação cruzada: {most_important_ccf:.3f}")

# Plotando a Correlação Cruzada
plt.figure(figsize=(10, 6))
plt.bar(lags, ccf_values)
plt.xlabel('Lag')
plt.ylabel('Correlação Cruzada')
plt.title('Correlação Cruzada entre Temperatura e Precipitação - Nova York')
plt.axvline(most_important_lag, color='red', linestyle='--')
plt.show()
