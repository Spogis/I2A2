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

dataset['Title'] = dataset['Title'].replace('Lady', 'Miss')
dataset['Title'] = dataset['Title'].replace('Countess', 'Miss')

dataset['Title'] = dataset['Title'].replace('Jonkheer', 'Mr')

dataset['Title'] = dataset['Title'].replace('Rev', 'Mr')

dataset.loc[(dataset['Title'] == 'Dr') & (dataset['Sex'] == 'male'), 'Title'] = 'Mr'
dataset.loc[(dataset['Title'] == 'Dr') & (dataset['Sex'] == 'female'), 'Title'] = 'Mrs'

dataset.loc[(dataset['Title'] == 'Mrs') & (dataset['Sex'] == 'male'), 'Title'] = 'Mr'

dataset['Title'] = dataset['Title'].replace('Col', 'Mr')
dataset['Title'] = dataset['Title'].replace('Capt', 'Mr')
dataset['Title'] = dataset['Title'].replace('Major', 'Mr')
dataset['Title'] = dataset['Title'].replace('Sir', 'Mr')

# dataset['Title'] = dataset['Title'].replace('Col', 'Other')
# dataset['Title'] = dataset['Title'].replace('Capt', 'Other')
# dataset['Title'] = dataset['Title'].replace('Major', 'Other')
# dataset['Title'] = dataset['Title'].replace('Sir', 'Other')

########################################################################################################################
Numero_Quartis = 4
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


########################################################################################################################
dataset['FamilySize'] = dataset['SibSp'] + dataset['Parch'] + 1

########################################################################################################################
def categorize_family_size(size):
    if size >= 5:
        return 'Big Group'
    elif size >= 2:
        return 'Small Group'
    else:
        return 'Single'

# Aplicando a função a cada valor na coluna 'FamilySize'
dataset['TicketAppearances'] = dataset['FamilySize'].apply(categorize_family_size)

########################################################################################################################
dataset['IsAlone'] = "With Family"  # Inicializa com não está viajando sozinho
dataset.loc[dataset['FamilySize'] == 1, 'IsAlone'] = "Alone"  # Atualiza para Alone se a pessoa está viajando sozinha


########################################################################################################################
filename = 'train'
filtered_df = dataset[dataset['DatasetName'] == 'train']
filtered_df.to_csv('new_titanic_datasets/New' + filename + 'Data.csv')

filename = 'test'
filtered_df = dataset[dataset['DatasetName'] == 'test']
filtered_df.to_csv('new_titanic_datasets/New' + filename + 'Data.csv')