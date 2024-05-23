# Importar bibliotecas necessárias
import pandas as pd
from prophet import Prophet

# Carregar os dados de treinamento
train_data = pd.read_csv('datasets/train.csv')

# Carregar os dados de teste
test_data = pd.read_csv('datasets/test.csv')

# Converter a coluna de data para datetime nos dois datasets
train_data['date'] = pd.to_datetime(train_data['date'])
test_data['date'] = pd.to_datetime(test_data['date'])

# Agrupar os dados de treinamento por data, loja e item para obter as vendas diárias totais
daily_sales = train_data.groupby(['date', 'store', 'item']).sum().reset_index()


# Preparar o DataFrame para o Prophet
prophet_data = daily_sales.rename(columns={'date': 'ds', 'sales': 'y'})

# Lista para armazenar previsões
all_forecasts = []

# Treinar e prever para cada combinação de loja e item
for (store, item), group in prophet_data.groupby(['store', 'item']):
    model = Prophet(
        yearly_seasonality=True,
        weekly_seasonality=True,
        daily_seasonality=False,
        seasonality_mode='additive',
        seasonality_prior_scale=10,
        holidays_prior_scale=10,
        changepoint_prior_scale=0.05,
        n_changepoints=25
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
