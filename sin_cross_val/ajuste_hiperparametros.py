import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report
from random import sample
from PIL import Image
from io import BytesIO
from itertools import product

# Parámetros globales
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 50
ORIGIN_DIR = 'images_GTZAN/'
AUDIO_DIR = 'audios_GTZAN/'
FILENAME_SAVED_MODEL = "modelo_gtzan_cnn.h5"

def cargar_y_preparar_dataset(data_dir=ORIGIN_DIR):
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

def crear_y_compilar_modelo(num_clases, filters=32, dropout=0.3, lr=0.0005, optimizer_name="adam"):
    if optimizer_name == "adam":
        opt = Adam(learning_rate=lr)
    else:
        opt = RMSprop(learning_rate=lr)

    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),

        Conv2D(filters, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(filters*2, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(filters*4, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        tf.keras.layers.GlobalAveragePooling2D(),

        Dense(128, activation='relu'),
        Dropout(dropout),
        Dense(num_clases, activation='softmax')
    ])

    model.compile(
        optimizer=opt,
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )

    return model

def graficar_entrenamiento(history, filename="grafica_entrenamiento_ajustado.png"):
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

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"✅ Gráfica de entrenamiento guardada como {filename}")

def ajustar_arquitectura(param_grid, output_file="resultados_grid_search.csv"):
    results = []
    best_accuracy = 0
    best_model = None
    best_history = None

    for i, params in enumerate(param_grid):
        print(f"\n🔎 Ejecutando experimento {i+1} / {len(param_grid)}: {params}")

        global IMG_HEIGHT, IMG_WIDTH, BATCH_SIZE
        IMG_HEIGHT = IMG_WIDTH = params["img_size"]
        BATCH_SIZE = params["batch_size"]

        train_gen, val_gen = cargar_y_preparar_dataset()

        model = crear_y_compilar_modelo(
            num_clases=len(train_gen.class_indices),
            filters=params["filters"],
            dropout=params["dropout"],
            lr=params["lr"],
            optimizer_name=params["optimizer"]
        )

        early_stopping = EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True)
        history = model.fit(train_gen, validation_data=val_gen, epochs=EPOCHS,
                            callbacks=[early_stopping], verbose=1)

        loss, acc, prec, rec = model.evaluate(val_gen)
        print(f"Resultado experimento {i+1}: Accuracy={acc:.4f} Precision={prec:.4f} Recall={rec:.4f}")

        if acc > best_accuracy:
            best_accuracy = acc
            model.save("mejor_modelo_ajustado.h5")
            best_history = history
            print(f"✅ Nuevo mejor modelo guardado con Accuracy={acc:.4f}")

        results.append({
            "img_size": params["img_size"],
            "filters": params["filters"],
            "dropout": params["dropout"],
            "batch_size": params["batch_size"],
            "lr": params["lr"],
            "optimizer": params["optimizer"],
            "accuracy": acc,
            "precision": prec,
            "recall": rec
        })

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\n✅ Resultados guardados en {output_file}")

    if best_history:
        graficar_entrenamiento(best_history, filename="grafica_entrenamiento_mejor_modelo.png")

def generar_param_grid():
    param_grid = []

    for (dropout, batch_size, lr, optimizer) in product(
        [0.2, 0.3],
        [16, 32],
        [0.001, 0.0005],
        ["adam", "rmsprop"]
    ):
        param_grid.append({
            "img_size": 128,
            "filters": 16,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": optimizer
        })

    for (filters, dropout, batch_size, lr, optimizer) in product(
        [16, 32],
        [0.2, 0.3],
        [16, 32],
        [0.001, 0.0005],
        ["adam", "rmsprop"]
    ):
        param_grid.append({
            "img_size": 256,
            "filters": filters,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": optimizer
        })

    return param_grid

if __name__ == "__main__":
    # Generar las combinaciones
    param_grid = generar_param_grid()

    # Ajustar hiperparámetros de la arquitectura
    ajustar_arquitectura(param_grid)
