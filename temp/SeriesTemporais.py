import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.stattools import ccf

# Criando um DataFrame de exemplo com duas variáveis temporais
data = {
    'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'input_signal': pd.Series(range(100)) + pd.Series(range(100)).apply(lambda x: x % 10),
    'output_signal': pd.Series(range(100)) + pd.Series(range(100)).apply(lambda x: (x % 10) - 5),
}

# Criando DataFrame
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Plotando os sinais
df.plot(subplots=True, figsize=(10, 6))
plt.show()

# 1. Analisando a Autocorrelação (ACF) e a Parcial (PACF)
# Autocorrelação da variável de entrada
plot_acf(df['input_signal'], lags=20, title="ACF - Input Signal")
plt.show()

# Autocorrelação parcial da variável de entrada
plot_pacf(df['input_signal'], lags=20, title="PACF - Input Signal")
plt.show()

# 2. Correlação Cruzada (CCF)
# Correlação cruzada entre input_signal e output_signal
lags = range(-20, 21)
ccf_values = ccf(df['input_signal'], df['output_signal'])[:len(lags)]

plt.bar(lags, ccf_values)
plt.xlabel('Lag')
plt.ylabel('Correlação Cruzada')
plt.title('Correlação Cruzada entre Input e Output')
plt.show()

# 3. Aplicando Lags para Criar um Modelo de Soft Sensor
# Usando lags identificados para prever output_signal
df['input_lag_1'] = df['input_signal'].shift(1)
df['input_lag_2'] = df['input_signal'].shift(2)

# Exemplo de como usar os lags para prever output_signal
print(df.head(10))

