import pandas as pd
import matplotlib.pyplot as plt
from statsmodels.tsa.stattools import ccf
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# URL do conjunto de dados Tennessee Eastman Process Fault-Free Testing
url = 'https://github.com/ruoqiu/TEP/raw/master/TEP_FaultFree_Testing.csv'
df = pd.read_csv(url)

# Renomeando as colunas para refletir os nomes das variáveis TEP
tep_columns = [
    'XMEAS1', 'XMEAS2', 'XMEAS3', 'XMEAS4', 'XMEAS5', 'XMEAS6', 'XMEAS7', 'XMEAS8', 'XMEAS9', 'XMEAS10',
    'XMEAS11', 'XMEAS12', 'XMEAS13', 'XMEAS14', 'XMEAS15', 'XMEAS16', 'XMEAS17', 'XMEAS18', 'XMEAS19', 'XMEAS20',
    'XMEAS21', 'XMEAS22', 'XMEAS23', 'XMEAS24', 'XMEAS25', 'XMEAS26', 'XMEAS27', 'XMEAS28', 'XMEAS29', 'XMEAS30',
    'XMV1', 'XMV2', 'XMV3', 'XMV4', 'XMV5', 'XMV6', 'XMV7', 'XMV8', 'XMV9', 'XMV10', 'XMV11'
]
df.columns = tep_columns

# Selecionando variáveis para análise
# Aqui, vamos prever 'XMEAS9' (Pressão no tanque C) com base em 'XMV5' (Válvula de fluxo de alimentação)
df = df[['XMV5', 'XMEAS9']]

# Calculando a Correlação Cruzada (CCF) entre XMV5 e XMEAS9
lags = range(-20, 21)
ccf_values = ccf(df['XMV5'], df['XMEAS9'])[:len(lags)]

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
plt.title('Correlação Cruzada entre XMV5 e XMEAS9 no Tennessee Eastman Process')
plt.axvline(most_important_lag, color='red', linestyle='--')
plt.show()

# Criando variáveis de lag com o lag mais importante
df[f'XMV5_lag_{abs(most_important_lag)}'] = df['XMV5'].shift(abs(most_important_lag))

# Removendo linhas com valores nulos (por causa do shift)
df.dropna(inplace=True)

# Dividindo os dados em variáveis independentes (X) e dependentes (y)
X = df[[f'XMV5_lag_{abs(most_important_lag)}']]
y = df['XMEAS9']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando o Modelo de Regressão Linear
model = LinearRegression()
model.fit(X_train, y_train)

# Prevendo a saída com o modelo
y_pred = model.predict(X_test)

# Avaliando o Modelo
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)
print(f'Mean Squared Error: {mse:.3f}')
print(f'R² Score: {r2:.3f}')

# Plotando os resultados reais e previstos
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='XMEAS9 Real', marker='o')
plt.plot(y_test.index, y_pred, label='XMEAS9 Previsto', marker='x')
plt.xlabel('Índice')
plt.ylabel('XMEAS9 (Pressão no tanque C)')
plt.legend()
plt.title(f'Previsão de XMEAS9 usando Lag de {most_important_lag} Períodos')
plt.show()



