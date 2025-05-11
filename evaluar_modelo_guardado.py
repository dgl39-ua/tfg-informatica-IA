import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall

# Configuración
MODEL_FILE = "modelo_gtzan_cnn.h5"   # "mejor_modelo_ajustado.h5" "modelo_gtzan_cnn.h5"
DATASET_DIR = "images_GTZAN/"
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32

def cargar_dataset(data_dir):
    datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

    val_gen = datagen.flow_from_directory(
        data_dir,
        target_size=(IMG_HEIGHT, IMG_WIDTH),
        batch_size=BATCH_SIZE,
        class_mode='categorical',
        subset='validation'
    )

    return val_gen

def buscar_y_evaluar_modelo(nombre_modelo):
    if os.path.exists(nombre_modelo):
        print(f"✅ Modelo encontrado: {nombre_modelo}")
        modelo_guardado = load_model(nombre_modelo, compile=False)

        # Añadir métricas manualmente (importante para modelos sin compile)
        modelo_guardado.compile(
            optimizer='adam',
            loss='categorical_crossentropy',
            metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
        )

        val_gen = cargar_dataset(DATASET_DIR)
        results = modelo_guardado.evaluate(val_gen, verbose=1)

        print(f"\n📋 Resultados del modelo '{nombre_modelo}':")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")
    else:
        print(f"❌ Modelo '{nombre_modelo}' no encontrado.")

if __name__ == "__main__":
    buscar_y_evaluar_modelo(MODEL_FILE)
