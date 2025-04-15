import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from tensorflow import keras
from tensorflow.keras.datasets import cifar10
import matplotlib.pyplot as plt
from random import sample
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam, SGD, RMSprop
from tensorflow.keras.layers import LeakyReLU
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout

from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import train_test_split
import pandas as pd

import os
import numpy as np
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt

# Parámetros globales
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 20
ORIGIN_DIR = 'images_original/'

def crear_generadores(data_dir=ORIGIN_DIR):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    train_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='training'
    )

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return train_gen, val_gen

def construir_modelo(num_clases):
    model = Sequential([
        Conv2D(32, (3, 3), activation='relu', input_shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_clases, activation='softmax')
    ])

    model.compile(optimizer='adam',
                  loss='categorical_crossentropy',
                  metrics=['accuracy'])
    return model

def entrenar_modelo(model, train_gen, val_gen):
    history = model.fit(
        train_gen, 
        validation_data=val_gen, 
        epochs=EPOCHS)
    return history

def guardar_modelo(model, filename="modelo_gtzan_cnn.h5"):
    model.save(filename)

def cargar_modelo(filename="modelo_gtzan_cnn.h5"):
    return load_model(filename)

def predecir_genero(img_path_or_array, model, class_indices):
    if isinstance(img_path_or_array, str):
        img = load_img(img_path_or_array, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
    else:
        img_array = img_path_or_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)
    prediction = model.predict(img_array)
    predicted_class = np.argmax(prediction)

    labels = {v: k for k, v in class_indices.items()}
    return labels[predicted_class]

def graficar_resultados(history):
    acc = history.history['accuracy']
    val_acc = history.history['val_accuracy']
    loss = history.history['loss']
    val_loss = history.history['val_loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Entrenamiento')
    plt.plot(epochs_range, val_acc, label='Validación')
    plt.title('Precisión')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Validación')
    plt.title('Pérdida')
    plt.legend()

    plt.savefig('grafica_CNN.png')
    plt.show()

if __name__ == "__main__":
    train_gen, val_gen = crear_generadores()
    model = construir_modelo(num_clases=len(train_gen.class_indices))
    history = entrenar_modelo(model, train_gen, val_gen)
    guardar_modelo(model)
    graficar_resultados(history)
    """
    # Ejemplo de predicción
    ejemplo = 'GTZAN_spectrograms/rock/rock.00000.png'
    modelo_cargado = cargar_modelo()
    genero = predecir_genero(ejemplo, modelo_cargado, train_gen.class_indices)
    print("Género predicho:", genero)
    """
