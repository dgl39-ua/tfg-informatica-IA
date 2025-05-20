import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import confusion_matrix, classification_report

# Configuración
MODEL_FILE = "mejor_modelo_ajustado.h5"   # "mejor_modelo_ajustado.h5" // "modelo_gtzan_cnn.h5" // "modelo_gtzan_cnn_arq2.h5"
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
        subset='validation',
        shuffle=False
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

        # Mostrar un resumen de la estructura del modelo
        modelo_guardado.summary()
        print()

        val_gen = cargar_dataset(DATASET_DIR)
        results = modelo_guardado.evaluate(val_gen, verbose=1)

        print(f"\n📋 Resultados del modelo '{nombre_modelo}':")
        print(f"Accuracy: {results[1]:.4f}")
        print(f"Precision: {results[2]:.4f}")
        print(f"Recall: {results[3]:.4f}")

        # Mostrar matriz de confusión
        mostrar_matriz_confusion(modelo_guardado, val_gen)
        print("✅ Matriz de confusión guardada como 'matriz_confusion_modelo.png'")
    else:
        print(f"❌ Modelo '{nombre_modelo}' no encontrado.")

# Función para mostrar la matriz de confusión
def mostrar_matriz_confusion(modelo, val_gen):
    # Obtener nombres de clases
    etiquetas = {v: k for k, v in val_gen.class_indices.items()}
    nombres_clases = [etiquetas[i] for i in range(len(etiquetas))]

    # Obtener predicciones y etiquetas reales
    y_true = []
    y_pred = []

    val_gen.reset()  # Reiniciamos el generador

    for _ in range(len(val_gen)):
        X_batch, y_batch = next(val_gen)
        pred_batch = modelo.predict(X_batch)
        y_true.extend(np.argmax(y_batch, axis=1))
        y_pred.extend(np.argmax(pred_batch, axis=1))

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.xlabel('Género predicho')
    plt.ylabel('Género verdadero')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig('matriz_confusion.png')
    plt.show()

    # Reporte detallado
    report = classification_report(y_true, y_pred, target_names=nombres_clases)
    print(report)
    with open("classification_report.txt", "w") as f:
        f.write(report)

if __name__ == "__main__":
    buscar_y_evaluar_modelo(MODEL_FILE)
