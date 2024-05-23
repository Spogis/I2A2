import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

from tensorflow import keras
from keras.models import Sequential
from keras.layers import Dense
from keras import initializers
from keras.regularizers import l2

train_data = pd.read_csv('../02 - Data Preparation/new_titanic_datasets/NewtrainData.csv')
validation_data = pd.read_csv('../02 - Data Preparation/new_titanic_datasets/NewtestData.csv')

# Especificando as colunas categóricas e numéricas
# categorical_features = ['Sex', 'Embarked', 'Title', 'AgeGroup', 'CabinPrefix', 'IsAlone']
# numerical_features = ['Pclass', 'Age', 'SibSp', 'Parch', 'Fare', 'FamilySize', 'FarePerPerson']

categorical_features = ['Title', 'Sex', 'TicketAppearances', 'CabinPrefix', 'IsAlone', 'Embarked']
numerical_features = ['Pclass', 'Fare', 'FamilySize', 'SibSp', 'Parch']

label_encoder = LabelEncoder()
for feature in categorical_features:
    # Combina os dados de treino e validação para garantir consistência na codificação
    combined_data = pd.concat([train_data[feature], validation_data[feature]], axis=0)
    combined_data_encoded = label_encoder.fit_transform(combined_data)

    # Divide os dados codificados de volta entre treino e validação
    train_data[feature] = combined_data_encoded[:len(train_data)]
    validation_data[feature] = combined_data_encoded[len(train_data):]

scaler = StandardScaler()
# Fit no treino e transforma em treino e teste
scaler.fit(train_data[numerical_features])
train_data[numerical_features] = scaler.transform(train_data[numerical_features])
validation_data[numerical_features] = scaler.transform(validation_data[numerical_features])

# Supondo que 'Survived' é a coluna target
y = train_data['Survived']
X = train_data[categorical_features + numerical_features]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)


def Create_Keras_Model(in_features, n_camadas_ocultas):
    # MLP Regression
    model = Sequential()
    initializer = initializers.GlorotNormal(seed=42)

    # Adicionando a camada de entrada
    model.add(Dense(64, input_dim=in_features, activation="relu", kernel_initializer=initializer))

    # Adicionando camadas ocultas
    for i in range(n_camadas_ocultas):
        model.add(Dense(64, activation="relu", kernel_initializer=initializer))

    # Adicionando a camada de saída
    model.add(Dense(1, activation='sigmoid', kernel_initializer=initializer))

    opt = keras.optimizers.Adam(learning_rate=0.0004)

    model.compile(optimizer=opt, loss='binary_crossentropy',
                  metrics=['accuracy'])

    model.summary()

    return model

n_camadas_ocultas = 3
in_features = X_train.shape[1]
model = Create_Keras_Model(in_features, n_camadas_ocultas)


# Treinamento do modelo
early_stopping = keras.callbacks.EarlyStopping(
        monitor='loss',
        patience=10,
        min_delta=0.001,
        restore_best_weights=True,
    )

history = model.fit(
    X_train, y_train,
    validation_data=(X_test, y_test),
    batch_size=512,
    epochs=2000,
    callbacks=[early_stopping],
    verbose=1,
)



# Avaliação do modelo no conjunto de teste
test_loss, test_accuracy = model.evaluate(X_test, y_test, verbose=2)
print(f"Acurácia no conjunto de teste: {test_accuracy*100:.2f}%")

# Previsões (probabilidades) do modelo
predictions = model.predict(X)
# Convertendo probabilidades em classificações binárias (0 ou 1)
predictions = (predictions > 0.5).astype(int).flatten()

# Calculando a matriz de confusão
conf_matrix = confusion_matrix(y, predictions)
print(conf_matrix)
# Convertendo para porcentagens
sum_per_class = np.sum(conf_matrix, axis=1, keepdims=True)
# Evitar divisão por zero
with np.errstate(divide='ignore', invalid='ignore'):
    conf_matrix_percentage_per_class = np.nan_to_num(conf_matrix / sum_per_class * 100)

# Criando strings de anotação com o símbolo de porcentagem
annot = np.array([["{:.1f}%".format(val) for val in row] for row in conf_matrix_percentage_per_class])

# Plotando a matriz de confusão
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


# Plot da acurácia de treinamento e validação
plt.plot(history.history['accuracy'], label='Acurácia de treinamento')
plt.plot(history.history['val_accuracy'], label='Acurácia de validação')
plt.title('Gráfico de Convergência - Acurácia')
plt.ylabel('Acurácia')
plt.xlabel('Época')
plt.legend(loc='upper left')
plt.show()

# Plot da perda de treinamento e validação
plt.plot(history.history['loss'], label='Perda de treinamento')
plt.plot(history.history['val_loss'], label='Perda de validação')
plt.title('Gráfico de Convergência - Perda')
plt.ylabel('Perda')
plt.xlabel('Época')
plt.legend(loc='upper right')
plt.show()


X_Validation = validation_data[categorical_features + numerical_features]
# Previsões (probabilidades) do modelo
predictions = model.predict(X_Validation)
# Convertendo probabilidades em classificações binárias (0 ou 1)
predictions = (predictions > 0.5).astype(int).flatten()

output = pd.DataFrame({'PassengerId': validation_data.PassengerId, 'Survived': predictions})
output.to_csv('submission_temp.csv', index=False)
print("Your submission was successfully saved!")