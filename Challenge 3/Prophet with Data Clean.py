import pandas as pd
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import matplotlib.pyplot as plt
import holidays

# Carregar os dados de treinamento
train_data = pd.read_csv('datasets/train.csv')

# Carregar os dados de teste
test_data = pd.read_csv('datasets/test.csv')

# Converter a coluna de data para datetime nos dois datasets
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Agrupar os dados de treinamento por data, loja e item para obter as vendas diárias totais
daily_sales = train_data.groupby(['date', 'store', 'item']).sum().reset_index()

# Função para remover outliers usando o método IQR
def remove_outliers(df):
    Q1 = df['sales'].quantile(0.25)
    Q3 = df['sales'].quantile(0.75)
    IQR = Q3 - Q1
    lower_bound = Q1 - 1.5 * IQR
    upper_bound = Q3 + 1.5 * IQR
    return df[(df['sales'] >= lower_bound) & (df['sales'] <= upper_bound)]

# Remover outliers
daily_sales = remove_outliers(daily_sales)

# Calcular a data de início da validação (80% do intervalo de datas)
total_days = (daily_sales['date'].max() - daily_sales['date'].min()).days
calculated_validation_start_date = daily_sales['date'].min() + pd.Timedelta(days=int(total_days * 0.8))

# Garantir que a data de início da validação existe no conjunto de dados
validation_start_date = daily_sales[daily_sales['date'] >= calculated_validation_start_date]['date'].min()

# Dividir o conjunto de dados em treino e validação
train_part = daily_sales[daily_sales['date'] < validation_start_date]
validation_part = daily_sales[daily_sales['date'] >= validation_start_date]

# Preparar o DataFrame para o Prophet
prophet_train_data = train_part.rename(columns={'date': 'ds', 'sales': 'y'})

# Obter os feriados dos EUA para todos os anos presentes nos dados de treinamento
years = prophet_train_data['ds'].dt.year.unique()
us_holidays = holidays.US(years=years)

# Criar um DataFrame de feriados
holiday_data = pd.DataFrame(list(us_holidays.items()), columns=['ds', 'holiday'])
holiday_data['ds'] = pd.to_datetime(holiday_data['ds'])

# Lista para armazenar previsões e valores reais para validação
validation_forecasts = []
validation_actuals = []

# Treinar e prever para cada combinação de loja e item no conjunto de validação
for (store, item), group in prophet_train_data.groupby(['store', 'item']):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        changepoint_prior_scale=0.05,
        n_changepoints=25,
        holidays=holiday_data
    )
    model.fit(group[['ds', 'y']])

    # Filtrar as datas relevantes para este grupo específico
    future_dates = validation_part[(validation_part['store'] == store) & (validation_part['item'] == item)]['date'].unique()
    future = pd.DataFrame(future_dates, columns=['ds'])

    forecast = model.predict(future)
    forecast['store'] = store
    forecast['item'] = item

    actuals = validation_part[(validation_part['store'] == store) & (validation_part['item'] == item)]
    validation_forecasts.append(forecast[['ds', 'store', 'item', 'yhat']])
    validation_actuals.append(actuals[['date', 'store', 'item', 'sales']])

# Combinar todas as previsões e valores reais
validation_forecasts_df = pd.concat(validation_forecasts).rename(columns={'ds': 'date', 'yhat': 'predicted_sales'})
validation_actuals_df = pd.concat(validation_actuals).rename(columns={'date': 'ds', 'sales': 'actual_sales'})

# Renomear as colunas para garantir que a junção funcione
validation_actuals_df = validation_actuals_df.rename(columns={'ds': 'date'})

# Agrupar por data para obter as vendas totais diárias
comparison_df = pd.merge(validation_forecasts_df, validation_actuals_df, on=['date', 'store', 'item'])
daily_comparison_df = comparison_df.groupby('date').sum().reset_index()

# Calcular as métricas de erro
mae = mean_absolute_error(daily_comparison_df['actual_sales'], daily_comparison_df['predicted_sales'])
rmse = np.sqrt(mean_squared_error(daily_comparison_df['actual_sales'], daily_comparison_df['predicted_sales']))
r2 = r2_score(daily_comparison_df['actual_sales'], daily_comparison_df['predicted_sales'])

# Gráfico de comparação de previsões e valores reais
plt.figure(figsize=(12, 6))
plt.plot(daily_comparison_df['date'], daily_comparison_df['actual_sales'], label='Actual Sales', color='blue')
plt.plot(daily_comparison_df['date'], daily_comparison_df['predicted_sales'], label='Predicted Sales', color='red', linestyle='--')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.title('Actual Sales vs Predicted Sales (Daily Total)')
plt.legend()

# Salvar a figura
plt.savefig('forecast_vs_actuals.png')

# Mostrar a figura
plt.show()

# Preparar o DataFrame completo para o Prophet
prophet_data = daily_sales.rename(columns={'date': 'ds', 'sales': 'y'})

# Lista para armazenar previsões
all_forecasts = []

# Treinar e prever para cada combinação de loja e item no conjunto de teste
for (store, item), group in prophet_data.groupby(['store', 'item']):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        changepoint_prior_scale=0.05,
        n_changepoints=25,
        holidays=holiday_data
    )
    model.fit(group[['ds', 'y']])

    # Filtrar as datas relevantes para este grupo específico
    future_dates = test_data[(test_data['store'] == store) & (test_data['item'] == item)]['date'].unique()
    future = pd.DataFrame(future_dates, columns=['ds'])

    forecast = model.predict(future)
    forecast['store'] = store
    forecast['item'] = item

    all_forecasts.append(forecast[['ds', 'store', 'item', 'yhat']])

# Combinar todas as previsões
all_forecasts_df = pd.concat(all_forecasts)

# Unir previsões com o conjunto de dados de teste original
all_forecasts_df = all_forecasts_df.rename(columns={'ds': 'date', 'yhat': 'sales'})
submission_data = test_data.merge(all_forecasts_df, on=['date', 'store', 'item'], how='left')

# Arredondar as previsões para o número inteiro mais próximo
submission_data['sales'] = submission_data['sales'].round().astype(int)

# Manter apenas as colunas id e sales
submission_output = submission_data[['id', 'sales']]

# Salvar o arquivo de submissão atualizado
output_file_path = 'submission.csv'
submission_output.to_csv(output_file_path, index=False)

print(f'Submission file saved to {output_file_path}')

print(f'Mean Absolute Error (MAE): {mae}')
print(f'Root Mean Squared Error (RMSE): {rmse}')
print(f'R^2: {r2}')
