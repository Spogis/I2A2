# Store_Item_Demand_Forecasting_EDA_with_Daily_Seasonality.ipynb

# Importar bibliotecas necessárias
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from statsmodels.tsa.seasonal import seasonal_decompose

# Configurar visualizações
sns.set(style="whitegrid")

# Carregar os dados de treinamento
train_data = pd.read_csv('datasets/train.csv')

# Converter a coluna de data para datetime
train_data['date'] = pd.to_datetime(train_data['date'])

# Visão geral dos dados
print("Train Data Info:")
print(train_data.info())

# Estatísticas descritivas
print("Train Data Descriptive Statistics:")
print(train_data.describe())

# Verificar valores ausentes
print("Missing Values in Train Data:")
print(train_data.isnull().sum())

# Visualizar as primeiras linhas dos dados de treinamento
print("First 5 rows of Train Data:")
print(train_data.head())

# Visualizar a distribuição das vendas
plt.figure(figsize=(12, 6))
sns.histplot(train_data['sales'], bins=50, kde=True)
plt.title('Distribution of Sales')
plt.xlabel('Sales')
plt.ylabel('Frequency')
plt.show()

# Visualizar a série temporal das vendas diárias agregadas
daily_sales = train_data.groupby('date').sum().reset_index()
plt.figure(figsize=(15, 6))
plt.plot(daily_sales['date'], daily_sales['sales'])
plt.title('Daily Sales Over Time')
plt.xlabel('Date')
plt.ylabel('Sales')
plt.grid(True)
plt.show()

# Análise de sazonalidade mensal
train_data['month'] = train_data['date'].dt.month
monthly_sales_avg = train_data.groupby('month')['sales'].mean()

plt.figure(figsize=(12, 6))
sns.barplot(x=monthly_sales_avg.index, y=monthly_sales_avg.values, palette="viridis")
plt.title('Average Sales per Month')
plt.xlabel('Month')
plt.ylabel('Average Sales')
plt.show()

# Análise de sazonalidade semanal com nomes dos dias da semana
train_data['day_of_week'] = train_data['date'].dt.dayofweek
train_data['day_name'] = train_data['date'].dt.day_name()
weekly_sales_avg = train_data.groupby('day_name')['sales'].mean()
weekly_sales_avg = weekly_sales_avg.reindex(['Monday', 'Tuesday', 'Wednesday', 'Thursday', 'Friday', 'Saturday', 'Sunday'])

plt.figure(figsize=(12, 6))
sns.barplot(x=weekly_sales_avg.index, y=weekly_sales_avg.values, palette="viridis")
plt.title('Average Sales per Day of Week')
plt.xlabel('Day of Week')
plt.ylabel('Average Sales')
plt.show()

# Análise de vendas por loja
store_sales = train_data.groupby('store')['sales'].sum()

plt.figure(figsize=(12, 6))
store_sales.plot(kind='bar', color='skyblue')
plt.title('Total Sales by Store')
plt.xlabel('Store')
plt.ylabel('Total Sales')
plt.grid(axis='y')
plt.show()

# Análise de vendas por item
item_sales = train_data.groupby('item')['sales'].sum()

plt.figure(figsize=(12, 6))
item_sales.plot(kind='bar', color='lightgreen')
plt.title('Total Sales by Item')
plt.xlabel('Item')
plt.ylabel('Total Sales')
plt.grid(axis='y')
plt.show()

# Decomposição sazonal para análise detalhada de sazonalidade

# Vamos usar apenas as vendas totais agregadas para a decomposição
decomposition = seasonal_decompose(daily_sales['sales'], model='additive', period=365)

# Plotar decomposição
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
ax1.plot(decomposition.observed)
ax1.set_title('Observed')
ax2.plot(decomposition.trend)
ax2.set_title('Trend')
ax3.plot(decomposition.seasonal)
ax3.set_title('Seasonal')
ax4.plot(decomposition.resid)
ax4.set_title('Residual')
plt.tight_layout()
plt.show()

# Calcular variância total e variância sazonal
total_variance = np.var(decomposition.observed)
seasonal_variance = np.var(decomposition.seasonal)
print(f'Total Variance: {total_variance}')
print(f'Seasonal Variance (Yearly): {seasonal_variance}')

# Calcular proporção da variância explicada pela componente sazonal
yearly_seasonality_ratio = seasonal_variance / total_variance
print(f'Yearly Seasonality Ratio: {yearly_seasonality_ratio}')

# Decomposição sazonal semanal para análise detalhada de sazonalidade semanal
decomposition_weekly = seasonal_decompose(daily_sales['sales'], model='additive', period=7)

# Plotar decomposição semanal
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
ax1.plot(decomposition_weekly.observed)
ax1.set_title('Observed (Weekly)')
ax2.plot(decomposition_weekly.trend)
ax2.set_title('Trend (Weekly)')
ax3.plot(decomposition_weekly.seasonal)
ax3.set_title('Seasonal (Weekly)')
ax4.plot(decomposition_weekly.resid)
ax4.set_title('Residual (Weekly)')
plt.tight_layout()
plt.show()

# Calcular variância sazonal semanal
weekly_seasonal_variance = np.var(decomposition_weekly.seasonal)
print(f'Seasonal Variance (Weekly): {weekly_seasonal_variance}')

# Calcular proporção da variância explicada pela componente sazonal semanal
weekly_seasonality_ratio = weekly_seasonal_variance / total_variance
print(f'Weekly Seasonality Ratio: {weekly_seasonality_ratio}')

# Decomposição sazonal diária para análise detalhada de sazonalidade diária
decomposition_daily = seasonal_decompose(daily_sales['sales'], model='additive', period=1)

# Plotar decomposição diária
fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(15, 12))
ax1.plot(decomposition_daily.observed)
ax1.set_title('Observed (Daily)')
ax2.plot(decomposition_daily.trend)
ax2.set_title('Trend (Daily)')
ax3.plot(decomposition_daily.seasonal)
ax3.set_title('Seasonal (Daily)')
ax4.plot(decomposition_daily.resid)
ax4.set_title('Residual (Daily)')
plt.tight_layout()
plt.show()

# Calcular variância sazonal diária
daily_seasonal_variance = np.var(decomposition_daily.seasonal)
print(f'Seasonal Variance (Daily): {daily_seasonal_variance}')

# Calcular proporção da variância explicada pela componente sazonal diária
daily_seasonality_ratio = daily_seasonal_variance / total_variance
print(f'Daily Seasonality Ratio: {daily_seasonality_ratio}')

# Definir limiares para proporções de sazonalidade
seasonality_threshold = 0.05  # Por exemplo, 5% da variância total

# Conclusões baseadas na análise
if yearly_seasonality_ratio > seasonality_threshold:
    print("Incluir sazonalidade anual: True")
else:
    print("Incluir sazonalidade anual: False")

if weekly_seasonality_ratio > seasonality_threshold:
    print("Incluir sazonalidade semanal: True")
else:
    print("Incluir sazonalidade semanal: False")

if daily_seasonality_ratio > seasonality_threshold:
    print("Incluir sazonalidade diária: True")
else:
    print("Incluir sazonalidade diária: False")
