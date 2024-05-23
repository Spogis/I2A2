import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score

# Criando um DataFrame de exemplo com duas variáveis temporais
data = {
    'date': pd.date_range(start='2023-01-01', periods=100, freq='D'),
    'input_signal': pd.Series(range(100)) + pd.Series(range(100)).apply(lambda x: x % 10),
    'output_signal': pd.Series(range(100)) + pd.Series(range(100)).apply(lambda x: (x % 10) - 5),
}

# Criando DataFrame
df = pd.DataFrame(data)
df.set_index('date', inplace=True)

# Aplicando lags na variável input_signal
df['input_lag_1'] = df['input_signal'].shift(1)
df['input_lag_2'] = df['input_signal'].shift(2)
df['input_lag_3'] = df['input_signal'].shift(3)

# Removendo linhas com valores nulos (por causa dos shifts)
df.dropna(inplace=True)

# Dividindo os dados em variáveis independentes (X) e dependentes (y)
X = df[['input_signal', 'input_lag_1', 'input_lag_2', 'input_lag_3']]
y = df['output_signal']

# Dividindo os dados em conjuntos de treino e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Treinando um modelo de Regressão Linear
linear_model = LinearRegression()
linear_model.fit(X_train, y_train)

# Treinando um modelo de Random Forest
rf_model = RandomForestRegressor(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Previsões
y_pred_linear = linear_model.predict(X_test)
y_pred_rf = rf_model.predict(X_test)

# Avaliando os Modelos
def evaluate_model(y_true, y_pred, model_name):
    mse = mean_squared_error(y_true, y_pred)
    r2 = r2_score(y_true, y_pred)
    print(f"{model_name} - Mean Squared Error: {mse:.3f}, R²: {r2:.3f}")

evaluate_model(y_test, y_pred_linear, "Linear Regression")
evaluate_model(y_test, y_pred_rf, "Random Forest")

# Plotando os resultados
plt.figure(figsize=(10, 6))
plt.plot(y_test.index, y_test, label='True Output', marker='o')
plt.plot(y_test.index, y_pred_linear, label='Predicted (Linear Regression)', marker='x')
plt.plot(y_test.index, y_pred_rf, label='Predicted (Random Forest)', marker='x')
plt.xlabel('Date')
plt.ylabel('Output Signal')
plt.legend()
plt.show()
