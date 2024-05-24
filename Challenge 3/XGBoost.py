import pandas as pd
import numpy as np
import xgboost as xgb
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import matplotlib.pyplot as plt
import holidays

# Definir a semente para reprodutibilidade
seed = 42
np.random.seed(seed)

# Carregar os dados de treinamento
train_data = pd.read_csv('datasets/train.csv')

# Carregar os dados de teste
test_data = pd.read_csv('datasets/test.csv')

# Converter a coluna de data para datetime nos dois datasets
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Função para criar características de data
def create_date_features(df):
    df['year'] = df['date'].dt.year
    df['month'] = df['date'].dt.month
    df['day'] = df['date'].dt.day
    df['dayofweek'] = df['date'].dt.dayofweek
    df['weekofyear'] = df['date'].dt.isocalendar().week
    df['is_weekend'] = df['date'].dt.dayofweek >= 5  # 5 e 6 correspondem a sábado e domingo
    return df

# Aplicar as características de data aos datasets de treino e teste
train_data = create_date_features(train_data)
test_data = create_date_features(test_data)

# Obter os feriados dos EUA para todos os anos presentes nos dados de treinamento
years = train_data['date'].dt.year.unique()
us_holidays = holidays.US(years=years)

# Criar uma coluna indicando se a data é um feriado
train_data['holiday'] = train_data['date'].isin(us_holidays)
test_data['holiday'] = test_data['date'].isin(us_holidays)

# Calcular a data de início da validação (80% do intervalo de datas)
total_days = (train_data['date'].max() - train_data['date'].min()).days
calculated_validation_start_date = train_data['date'].min() + pd.Timedelta(days=int(total_days * 0.8))

# Garantir que a data de início da validação existe no conjunto de dados
validation_start_date = train_data[train_data['date'] >= calculated_validation_start_date]['date'].min()

# Dividir o conjunto de dados em treino e validação
train_part = train_data[train_data['date'] < validation_start_date]
validation_part = train_data[train_data['date'] >= validation_start_date]

# Preparar os dados para XGBoost
features = ['year', 'month', 'day', 'dayofweek', 'weekofyear', 'store', 'item', 'holiday', 'is_weekend']
target = 'sales'

X_train_part = train_part[features]
y_train_part = train_part[target]
X_validation_part = validation_part[features]
y_validation_part = validation_part[target]

# Treinar o modelo XGBoost
model = xgb.XGBRegressor(
    objective='reg:squarederror',
    n_estimators=1000,
    learning_rate=0.01,
    max_depth=6,
    subsample=0.8,
    colsample_bytree=0.8,
    gamma=0,
    random_state=seed
)

model.fit(
    X_train_part, y_train_part,
    eval_set=[(X_train_part, y_train_part), (X_validation_part, y_validation_part)],
    early_stopping_rounds=50,
    verbose=True
)

# Fazer previsões no conjunto de validação
y_validation_pred = model.predict(X_validation_part)

# Combinar as previsões e os valores reais
validation_part['predicted_sales'] = y_validation_pred
comparison_df = validation_part[['date', 'store', 'item', 'sales', 'predicted_sales']]

# Agrupar por data para obter as vendas totais diárias
daily_comparison_df = comparison_df.groupby('date').sum().reset_index()

# Calcular as métricas de erro
mae = mean_absolute_error(daily_comparison_df['sales'], daily_comparison_df['predicted_sales'])
rmse = np.sqrt(mean_squared_error(daily_comparison_df['sales'], daily_comparison_df['predicted_sales']))
r2 = r2_score(daily_comparison_df['sales'], daily_comparison_df['predicted_sales'])

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R^2: {r2}')

# Gráfico de comparação de previsões e valores reais
plt.figure(figsize=(12, 6))
plt.plot(daily_comparison_df['date'], daily_comparison_df['sales'], label='Actual Sales', color='blue')
plt.plot(daily_comparison_df['date'], daily_comparison_df['predicted_sales'], label='Predicted Sales', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual Sales vs Predicted Sales (Daily Total)')
plt.legend()

# Salvar a figura
plt.savefig('forecast_vs_actuals.png')

# Mostrar a figura
plt.show()

# Preparar os dados de teste para previsão
X_test = test_data[features]

# Fazer previsões no conjunto de teste
test_data['sales'] = model.predict(X_test)

# Arredondar as previsões para o número inteiro mais próximo
test_data['sales'] = test_data['sales'].round().astype(int)

# Manter apenas as colunas id e sales
submission_output = test_data[['id', 'sales']]

# Salvar o arquivo de submissão atualizado
output_file_path = 'submission.csv'
submission_output.to_csv(output_file_path, index=False)

print(f'Submission file saved to {output_file_path}')
