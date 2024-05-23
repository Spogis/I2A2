import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
from xgboost import XGBClassifier  # Importando o XGBClassifier
import matplotlib.pyplot as plt
import seaborn as sns

# Carregando os dados
train_data = pd.read_csv('../02 - Data Preparation/new_titanic_datasets/NewtrainData.csv')
validation_data = pd.read_csv('../02 - Data Preparation/new_titanic_datasets/NewtestData.csv')

# Especificando as colunas categóricas e numéricas
categorical_features = ['Title', 'Sex', 'TicketAppearances', 'CabinPrefix', 'IsAlone', 'Embarked']
numerical_features = ['Pclass', 'Fare', 'FamilySize', 'SibSp', 'Parch']

# Codificação das variáveis categóricas
label_encoder = LabelEncoder()
for feature in categorical_features:
    combined_data = pd.concat([train_data[feature], validation_data[feature]], axis=0)
    combined_data_encoded = label_encoder.fit_transform(combined_data)
    train_data[feature] = combined_data_encoded[:len(train_data)]
    validation_data[feature] = combined_data_encoded[len(train_data):]

# Padronização das variáveis numéricas
scaler = StandardScaler()
scaler.fit(train_data[numerical_features])
train_data[numerical_features] = scaler.transform(train_data[numerical_features])
validation_data[numerical_features] = scaler.transform(validation_data[numerical_features])

# Separando as variáveis independentes da variável alvo
y = train_data['Survived']
X = train_data[categorical_features + numerical_features]

# Dividindo os dados em conjuntos de treinamento e teste
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Definindo o espaço de hiperparâmetros para o Grid Search com XGBoost
param_grid = {
    'n_estimators': [100, 200, 300, 700, 1000],  # Número de árvores
    'max_depth': [3, 4, 5, 6, 7, 8],  # Máxima profundidade da árvore
    'learning_rate': [0.01, 0.05, 0.1],  # Taxa de aprendizado
    'subsample': [0.6, 0.7, 0.8, 0.9],  # Fração de amostras a serem usadas para ajustar cada árvore
    'colsample_bytree': [0.6, 0.7, 0.8, 0.9],  # Fração de colunas a serem usadas para cada árvore
    'min_child_weight': [1, 2, 3, 4]  # Soma mínima dos pesos das instâncias necessárias em um filho
}

# Criando o modelo XGBClassifier para o Grid Search
xgb = XGBClassifier(use_label_encoder=False, eval_metric='logloss')

# Criando o objeto GridSearchCV
grid_search = GridSearchCV(estimator=xgb, param_grid=param_grid, cv=3, n_jobs=-1, verbose=2, scoring='accuracy')

# Realizando o Grid Search com os dados de treinamento
grid_search.fit(X_train, y_train)

# Imprimindo os melhores parâmetros
print("Melhores parâmetros:", grid_search.best_params_)

# Avaliando o modelo no conjunto de teste com os melhores hiperparâmetros
best_model = grid_search.best_estimator_
test_accuracy = best_model.score(X_test, y_test)
print(f"Acurácia no conjunto de teste com os melhores hiperparâmetros: {test_accuracy*100:.2f}%")

# Fazendo previsões com o melhor modelo
predictions = best_model.predict(X)

# Calculando e plotando a matriz de confusão
conf_matrix = confusion_matrix(y, predictions)
conf_matrix_percentage_per_class = conf_matrix / np.sum(conf_matrix, axis=1, keepdims=True) * 100
annot = np.array([["{:.1f}%".format(val) for val in row] for row in conf_matrix_percentage_per_class])

plt.figure(figsize=(10, 8))
sns.heatmap(conf_matrix_percentage_per_class, annot=annot, fmt="", cmap="Blues",
            xticklabels=['Not Survived', 'Survived'],
            yticklabels=['Not Survived', 'Survived'],
            square=True,
            cbar_kws={"shrink": 0.75},
            annot_kws={"size": 14}, linewidth=.5)
plt.ylabel('True Label')
plt.xlabel('Predicted Label')
plt.title('Confusion Matrix')
plt.tight_layout()
plt.show()

# A parte sobre importância das características é mantida, pois o XGBoost também fornece essa funcionalidade
feature_importances = best_model.feature_importances_
features = categorical_features + numerical_features
importances_df = pd.DataFrame({'Features': features, 'Importance': feature_importances}).sort_values(by='Importance', ascending=False)

plt.figure(figsize=(10, 6))
sns.barplot(x='Importance', y='Features', data=importances_df)
plt.title('Feature Importance')
plt.xlabel('Importance')
plt.ylabel('Features')
plt.show()

X_Validation = validation_data[categorical_features + numerical_features]
predictions = best_model.predict(X_Validation)
output = pd.DataFrame({'PassengerId': validation_data.PassengerId, 'Survived': predictions})
output.to_csv('submission_temp.csv', index=False)
print("Your submission was successfully saved!")
