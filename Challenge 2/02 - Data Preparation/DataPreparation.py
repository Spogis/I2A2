import pandas as pd
import numpy as np
import os


directory = '../titanic_datasets/'
filename = 'train.csv'
file_path = directory + filename

train_dataset = pd.read_csv(file_path, index_col='PassengerId')
train_dataset['DatasetName'] = 'train'

filename = 'test.csv'
file_path = directory + filename
test_dataset = pd.read_csv(file_path, index_col='PassengerId')
test_dataset['DatasetName'] = 'test'

dataset = pd.concat([train_dataset, test_dataset])


duplicates = dataset.duplicated().sum()
print(f"Duplicatas no dataset: {duplicates}")

########################################################################################################################
#   Adicionando a Coluna Title
########################################################################################################################
dataset['Title'] = dataset['Name'].str.extract(' ([A-Za-z]+)\.', expand=False)
print("Lista de Títulos:", dataset['Title'].unique())
# Contagem de ocorrências de cada título
titulos_contagem = dataset['Title'].value_counts()
print("Contagem de cada Título:\n", titulos_contagem)

title_to_sex = {
    "Mr": "male",
    "Master": "male",
    "Miss": "female",
    "Mrs": "female",
    "Ms": "female",
    "Mme": "female",        # Madame, equivalente francês de "Mrs"
    "Mlle": "female",       # Mademoiselle, equivalente francês de "Miss"
    "Don": "male",          # Título espanhol para homens
    "Dona": "female",       # Título espanhol para mulheres
    "Rev": "male",          # Reverendo, geralmente masculino, mas poderia ser feminino em casos modernos
    "Dr": "male",           # Usei como Masculino pois na época existia uma pequena formação de Doutores mulheres
    "Major": "male",        # Major, tipicamente masculino, mas não exclusivamente
    "Lady": "female",       # Título nobre para mulheres
    "Sir": "male",          # Título nobre para homens
    "Col": "male",          # Coronel, predominantemente masculino
    "Capt": "male",         # Capitão, predominantemente masculino
    "Countess": "female",   # Condessa, feminino
    "Jonkheer": "male"      # Título nobre holandês, masculino
}

dataset['Title'] = dataset['Title'].replace('Mlle', 'Miss')
dataset['Title'] = dataset['Title'].replace('Ms', 'Miss')
dataset['Title'] = dataset['Title'].replace('Mme', 'Mrs')

dataset['Title'] = dataset['Title'].replace('Don', 'Mrs')
dataset['Title'] = dataset['Title'].replace('Dona', 'Miss')

########################################################################################################################
#   Ajustando a Idade
########################################################################################################################
print("Dados faltantes na coluna Age =", dataset['Age'].isnull().sum())

age_median_global = dataset['Age'].median()
# Calculando a mediana de 'Age' por 'Title', 'Sex', e 'Pclass', e imputando valores faltantes
for (title, sex, pclass), subgroup in dataset.groupby(['Title', 'Sex', 'Pclass']):
    age_median = subgroup['Age'].median()
    # Verifica se age_median é NaN
    if pd.isna(age_median):
        # Se for NaN, usa a mediana global como fallback
        age_median = age_median_global
    dataset.loc[(dataset['Age'].isnull()) & (dataset['Title'] == title) & (dataset['Sex'] == sex) & (dataset['Pclass'] == pclass), 'Age'] = age_median

print("Dados faltantes na coluna Age após ajuste =", dataset['Age'].isnull().sum())

########################################################################################################################
# Ajustando o Fare para os valores iguais a zero (possivelmente ganharam a passagem ou erro no input
########################################################################################################################

fare_mediana_por_pclass = dataset[dataset['Fare'] > 0].groupby('Pclass')['Fare'].median()

# Imputação dos valores de Fare iguais a zero
for pclass, mediana in fare_mediana_por_pclass.items():
    dataset.loc[(dataset['Fare'] == 0) & (dataset['Pclass'] == pclass), 'Fare'] = mediana

valor_mais_comum = dataset['Fare'].mode()[0]
# Substituir os valores faltantes na coluna 'Fare' pelo valor mais comum
dataset['Fare'] = dataset['Fare'].fillna(valor_mais_comum)

########################################################################################################################
# Ajustando oos valores faltantes da coluna Embarked pelo valor mais comum
########################################################################################################################
# Encontrar o valor mais comum (moda) na coluna 'Embarked'
valor_mais_comum = dataset['Embarked'].mode()[0]
# Substituir os valores faltantes na coluna 'Embarked' pelo valor mais comum
dataset['Embarked'] = dataset['Embarked'].fillna(valor_mais_comum)


########################################################################################################################
# New Features
########################################################################################################################
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1
dataset['IsAlone'] = 0  # Inicializa com 0 (não está viajando sozinho)
dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = 1  # Atualiza para 1 se a pessoa está viajando sozinha

bins = [0, 12, 19, 40, 60, np.inf]
labels = ['Child', 'Teenager', 'Adult', 'MiddleAge', 'Senior']
dataset['AgeGroup'] = pd.cut(dataset['Age'], bins=bins, labels=labels, right=False)

Numero_Quartis = 4
dataset['FarePerPerson'] = dataset['Fare'] / (dataset['SibSp'] + dataset['Parch'] + 1)
dataset['FareQuartile'] = pd.qcut(dataset['Fare'], Numero_Quartis, labels=False)
dataset['FarePerPersonQuartile'] = pd.qcut(dataset['FarePerPerson'], Numero_Quartis, labels=False)

dataset['CabinPrefix'] = dataset['Cabin'].str[0]
fare_bins = pd.qcut(dataset['Fare'], Numero_Quartis, labels=False)  # Divide em quartis

# Encontrando o prefixo de cabine mais comum por Pclass e FareBin
most_common_prefix = dataset.groupby(['Pclass', fare_bins])['CabinPrefix'].apply(lambda x: x.mode()[0] if not x.mode().empty else np.nan)

# Imputação
for pclass in dataset['Pclass'].unique():
    for fare_bin in range(Numero_Quartis):  # Número de quartis definidos anteriormente
        prefix = most_common_prefix.get((pclass, fare_bin))
        if pd.notnull(prefix):
            mask = (dataset['CabinPrefix'].isnull()) & (dataset['Pclass'] == pclass) & (fare_bins == fare_bin)
            dataset.loc[mask, 'CabinPrefix'] = prefix

valor_mais_comum = dataset['CabinPrefix'].mode()[0]
# Substituir os valores faltantes na coluna 'CabinPrefix' pelo valor mais comum
dataset['CabinPrefix'] = dataset['CabinPrefix'].fillna(valor_mais_comum)

valores_vazios_por_coluna = dataset.isna().sum()
print("Valores Vazios")
print(valores_vazios_por_coluna)

zeros_por_coluna = (dataset == 0).sum()
print("Valores Zero")
print(zeros_por_coluna)

filename = 'train'
filtered_df = dataset[dataset['DatasetName'] == 'train']
filtered_df.to_csv('new_titanic_datasets/New' + filename + 'Data.csv')
#filtered_df.to_excel('new_titanic_datasets/New' + filename + 'Data.xlsx')

filename = 'test'
filtered_df = dataset[dataset['DatasetName'] == 'test']
filtered_df.to_csv('new_titanic_datasets/New' + filename + 'Data.csv')
#filtered_df.to_excel('new_titanic_datasets/New' + filename + 'Data.xlsx')


########################################################################################################################
#   Verificando se temos duplicatas entre os datasets
########################################################################################################################
test_data = pd.read_csv('new_titanic_datasets/NewtestData.csv')
train_data = pd.read_csv('new_titanic_datasets/NewtrainData.csv')

common_columns = test_data.columns.intersection(train_data.columns)
combined_data = pd.concat([test_data[common_columns], train_data[common_columns]], ignore_index=True)

# Verificando duplicatas no dataset combinado
duplicates_across_datasets = combined_data.duplicated().sum()

print(f"Duplicatas entre os datasets: {duplicates_across_datasets}")

# Verificando duplicatas na coluna 'Name' dentro de cada dataset individualmente
duplicates_in_test_name = test_data['Name'].duplicated().sum()
duplicates_in_train_name = train_data['Name'].duplicated().sum()
print(f"Duplicatas na coluna 'Name' no dataset de teste: {duplicates_in_test_name}")
print(f"Duplicatas na coluna 'Name' no dataset de treino: {duplicates_in_train_name}")

# Para verificar duplicatas entre os datasets na coluna 'Name'
combined_names = pd.concat([test_data['Name'], train_data['Name']], ignore_index=True)
duplicates_across_datasets_name = combined_names.duplicated().sum()
print(f"Duplicatas na coluna 'Name' entre os datasets: {duplicates_across_datasets_name}")