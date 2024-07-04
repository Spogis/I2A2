import matplotlib
matplotlib.use('Agg')  # Usar backend não interativo

import os
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Flatten, Dense, Dropout
from tensorflow.keras.applications import VGG16
from sklearn.metrics import confusion_matrix, classification_report
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import keras_tuner as kt
import random
from tensorflow.keras.preprocessing import image

# Diretórios do dataset
train_dir = 'datasets/training_set'
test_dir = 'datasets/test_set'

# Parâmetros
img_width, img_height = 150, 150
batch_size = 32
epochs = 10  # Menos epochs já que estamos usando transfer learning
validation_split = 0.2  # 20% para validação

# Data augmentation e geração de dados
datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True,
    validation_split=validation_split)

train_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='training')

validation_generator = datagen.flow_from_directory(
    train_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    subset='validation')

test_datagen = ImageDataGenerator(rescale=1. / 255)

test_generator = test_datagen.flow_from_directory(
    test_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='binary',
    shuffle=False)


# Definindo a função de criação do modelo para o Keras Tuner
def build_model(hp):
    base_model = VGG16(weights='imagenet', include_top=False, input_shape=(img_width, img_height, 3))
    for layer in base_model.layers:
        layer.trainable = False

    model = Sequential()
    model.add(base_model)
    model.add(Flatten())
    model.add(Dense(units=hp.Int('units', min_value=128, max_value=512, step=32), activation='relu'))
    model.add(Dropout(rate=hp.Float('dropout', min_value=0.2, max_value=0.5, step=0.1)))
    model.add(Dense(1, activation='sigmoid'))

    model.compile(
        optimizer=tf.keras.optimizers.Adam(hp.Float('learning_rate', min_value=1e-4, max_value=1e-2, sampling='LOG')),
        loss='binary_crossentropy',
        metrics=['accuracy'])

    return model


# Inicializando o Keras Tuner Hyperband
tuner = kt.Hyperband(build_model,
                     objective='val_accuracy',
                     max_epochs=10,
                     factor=3,
                     directory='keras_tuner',
                     project_name='cat_dog_classifier')

# Callbacks para Early Stopping
stop_early = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=5)

# Busca dos melhores hiperparâmetros
tuner.search(train_generator,
             steps_per_epoch=train_generator.samples // batch_size,
             epochs=epochs,
             validation_data=validation_generator,
             validation_steps=validation_generator.samples // batch_size,
             callbacks=[stop_early])

# Obtendo os melhores hiperparâmetros
best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
print(f"Melhor número de unidades na camada densa: {best_hps.get('units')}")
print(f"Melhor taxa de dropout: {best_hps.get('dropout')}")
print(f"Melhor taxa de aprendizado: {best_hps.get('learning_rate')}")

# Construir o modelo com os melhores hiperparâmetros
model = tuner.hypermodel.build(best_hps)

# Treinamento do modelo com os melhores hiperparâmetros
history = model.fit(
    train_generator,
    steps_per_epoch=train_generator.samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=validation_generator.samples // batch_size)

# Salvando o modelo
model.save('cat_dog_classifier_vgg16_tuned.h5')

# Avaliação do modelo no conjunto de teste
test_generator.reset()
predictions = model.predict(test_generator, steps=test_generator.samples // batch_size + 1)
predicted_classes = np.where(predictions > 0.5, 1, 0).flatten()

# Obtendo as classes verdadeiras
true_classes = test_generator.classes

# Matriz de confusão
conf_matrix = confusion_matrix(true_classes, predicted_classes)

# Convertendo a matriz de confusão para porcentagens
conf_matrix_percent = conf_matrix.astype('float') / conf_matrix.sum(axis=1)[:, np.newaxis] * 100
print('Matriz de Confusão (em %):')
print(conf_matrix_percent)

# Relatório de classificação
class_report = classification_report(true_classes, predicted_classes, target_names=['cats', 'dogs'])
print('Relatório de Classificação:')
print(class_report)

# Salvando o relatório de classificação em um arquivo de texto
report_path = 'classification_report_tuned.txt'
with open(report_path, 'w') as f:
    f.write('Relatório de Classificação:\n')
    f.write(class_report)

# Plotando a matriz de confusão (em %)
plt.figure(figsize=(8, 6))
sns.heatmap(conf_matrix_percent, annot=True, fmt=".2f", cmap="Blues", xticklabels=['cats', 'dogs'],
            yticklabels=['cats', 'dogs'])
plt.xlabel('Predicted Label')
plt.ylabel('True Label')
plt.title('Confusion Matrix (in %)')
conf_matrix_path = 'confusion_matrix_tuned_percent.png'
plt.savefig(conf_matrix_path)
plt.show()

# Plotando o gráfico de convergência e salvando as figuras
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.plot(history.history['loss'], label='Loss')
plt.plot(history.history['val_loss'], label='Val Loss')
plt.title('Model Loss')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
loss_fig_path = 'model_loss_tuned.png'
plt.savefig(loss_fig_path)

plt.subplot(1, 2, 2)
plt.plot(history.history['accuracy'], label='Accuracy')
plt.plot(history.history['val_accuracy'], label='Val Accuracy')
plt.title('Model Accuracy')
plt.xlabel('Epochs')
plt.ylabel('Accuracy')
plt.legend()
accuracy_fig_path = 'model_accuracy_tuned.png'
plt.savefig(accuracy_fig_path)

plt.tight_layout()
plt.show()

print(f'Relatório de classificação salvo em {report_path}')
print(f'Matriz de confusão salva em {conf_matrix_path}')
print(f'Gráficos de perda e precisão salvos em {loss_fig_path} e {accuracy_fig_path}, respectivamente')

# Testando o modelo em 20 imagens aleatórias do conjunto de teste
random_indices = random.sample(range(len(test_generator.filenames)), 20)
plt.figure(figsize=(20, 10))

for i, idx in enumerate(random_indices):
    img_path = os.path.join(test_dir, test_generator.filenames[idx])
    img = image.load_img(img_path, target_size=(img_width, img_height))
    img_array = image.img_to_array(img) / 255.0
    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)
    predicted_class = 'dog' if prediction > 0.5 else 'cat'

    plt.subplot(4, 5, i + 1)
    plt.imshow(img)
    plt.title(f'Predicted: {predicted_class}')
    plt.axis('off')

test_images_path = 'test_images_predictions.png'
plt.savefig(test_images_path)
plt.show()

print(f'Predições de imagens de teste salvas em {test_images_path}')
