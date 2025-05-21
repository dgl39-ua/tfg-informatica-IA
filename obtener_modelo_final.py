import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
from sklearn.metrics import classification_report, confusion_matrix
import seaborn as sns
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 50
LEARNING_RATE = 0.0005
DROPOUT_RATE = 0.3
FILTERS = 16
IMAGES_DIR = 'images_GTZAN_256/'
FILENAME_SAVED_MODEL = "modelo_definitivo_entrenado_completo.keras"

def crear_y_compilar_modelo(num_clases):
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(FILTERS, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(FILTERS*2, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(FILTERS*4, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),

        Dense(128, activation='relu'),
        Dropout(DROPOUT_RATE),
        Dense(num_clases, activation='softmax') # Capa de salida
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=LEARNING_RATE),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )

    return model

# Función para mostrar la evolución de la pérdida y la precisión durante el entrenamiento
def graficar_resultados(history):
    epochs_range = range(len(history.history['accuracy']))

    plt.figure(figsize=(12, 8))

    # Accuracy
    plt.plot(epochs_range, history.history['accuracy'], label='Accuracy', marker='o')

    # Loss
    plt.plot(epochs_range, history.history['loss'], label='Loss', marker='o')

    # Precision (si está disponible)
    if 'precision' in history.history:
        plt.plot(epochs_range, history.history['precision'], label='Precision', marker='o')

    # Recall (si está disponible)
    if 'recall' in history.history:
        plt.plot(epochs_range, history.history['recall'], label='Recall', marker='o')

    plt.title('Evolución de las métricas durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grafica_metricas_entrenamiento.png')
    plt.show()

def graficar_metricas(history):
    epochs_range = range(len(history.history['accuracy']))

    plt.figure(figsize=(12, 8))

    # Accuracy
    plt.plot(epochs_range, history.history['accuracy'], label='Accuracy', marker='o')

    # Loss
    plt.plot(epochs_range, history.history['loss'], label='Loss', marker='o')

    # Precision (si está disponible)
    if 'precision' in history.history:
        plt.plot(epochs_range, history.history['precision'], label='Precision', marker='o')

    # Recall (si está disponible)
    if 'recall' in history.history:
        plt.plot(epochs_range, history.history['recall'], label='Recall', marker='o')

    plt.title('Evolución de las métricas durante el entrenamiento')
    plt.xlabel('Épocas')
    plt.ylabel('Valor')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig('grafica_metricas_entrenamiento.png')
    print("Gráfica guardada como grafica_metricas_entrenamiento.png")
    plt.show()

# Función para mostrar la matriz de confusión
def mostrar_matriz_confusion(y_true, y_pred, class_indices):
    etiquetas = {v: k for k, v in class_indices.items()}
    nombres_clases = [etiquetas[i] for i in range(len(etiquetas))]

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.xlabel('Género predicho')
    plt.ylabel('Género real')
    plt.title('Matriz de Confusión Modelo Final')
    plt.tight_layout()
    plt.savefig('matriz_confusion_modelo_final.png')
    plt.show()

    report = classification_report(y_true, y_pred, target_names=nombres_clases)
    print("\nClassification report del modelo final:\n")
    print(report)
    with open("classification_report_modelo_final.txt", "w") as f:
        f.write(report)
    print("Classification report guardado como 'classification_report_modelo_final.txt'")

# Función que permite predecir el género musical utilizando el modelo
def predecir_genero(img_path_or_array, model, class_indices):
    if isinstance(img_path_or_array, str):
        img = load_img(img_path_or_array, target_size=(IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
    elif isinstance(img_path_or_array, Image.Image):
        img = img_path_or_array.resize((IMG_SIZE, IMG_SIZE))
        img_array = img_to_array(img) / 255.0
    else:
        img_array = img_path_or_array / 255.0

    img_array = np.expand_dims(img_array, axis=0)

    prediction = model.predict(img_array)[0]
    predicted_class = np.argmax(prediction)
    probabilidad = np.max(prediction)

    labels = {v: k for k, v in class_indices.items()}
    
    # Obtener la probabilidad de cada uno de los 10 géneros
    probabilidades = {labels[i]: float(prediction[i]) for i in range(len(prediction))}

    return labels[predicted_class], probabilidad, probabilidades

# Función que genera una gráfica con las probabilidades de predicción
def graficar_probabilidades(probabilidades):
    plt.figure(figsize=(10, 5))
    plt.bar(probabilidades.keys(), probabilidades.values(), color='mediumslateblue')
    plt.xticks(rotation=45)
    plt.ylabel("Probabilidad")
    plt.title("Probabilidad por Género")
    plt.tight_layout()
    plt.savefig('probabilidades_prediccion.png')
    plt.show()


if __name__ == "__main__":
    print("\n========================================")
    print("ENTRENAMIENTO FINAL CON TODO EL DATASET")
    print("========================================")

    # Cargar el dataset GTZAN
    filepaths = []
    labels = []

    for class_dir in os.listdir(IMAGES_DIR):
        class_path = os.path.join(IMAGES_DIR, class_dir)
        if os.path.isdir(class_path):
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(class_path, fname))
                    labels.append(class_dir)

    df = pd.DataFrame({"filename": filepaths, "class": labels})

    datagen = ImageDataGenerator(rescale=1./255)

    # Dataset completo
    full_gen = datagen.flow_from_dataframe(
        df,
        x_col="filename",
        y_col="class",
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=True
    )
    
    # Crear y compilar el modelo final
    model = crear_y_compilar_modelo(num_clases=len(full_gen.class_indices))

    # Mostrar un resumen de la estructura del modelo final
    model.summary()
    print()

    # Incluir callbacks
    checkpoint_path = "checkpoint_modelo_definitivo.keras"
    callbacks = [
        EarlyStopping(monitor='accuracy', patience=10, restore_best_weights=True),
        ModelCheckpoint(filepath=checkpoint_path, monitor='accuracy', save_best_only=True, verbose=0)
    ]

    # Entrenar el modelo
    start_time = time.time()
    history = model.fit(
        full_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    training_time = time.time() - start_time
    print(f"\nTiempo de entrenamiento en el dataset completo: {training_time:.2f} segundos")

    model.save(FILENAME_SAVED_MODEL)
    print(f"Modelo final guardado como {FILENAME_SAVED_MODEL}")

    # Evaluación final
    results = model.evaluate(full_gen, verbose=0)
    loss, acc, precision, recall = results
    print(f"\nEvaluación final sobre todo el dataset:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")

    y_true = full_gen.classes
    y_pred_probs = model.predict(full_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    f1 = f1_score(y_true, y_pred, average='macro')
    print(f"F1-score: {f1:.4f}")
   
    # Gráfica de entrenamiento
    graficar_resultados(history)
    graficar_metricas(history)

    # Mostrar la matriz de confusión
    mostrar_matriz_confusion(y_true, y_pred, full_gen.class_indices)

    # Ejemplo de predicción
    ejemplo = 'images_GTZAN/pop/pop.00047.png'
    genero, probabilidad, probabilidades = predecir_genero(ejemplo, model, full_gen.class_indices)
    print("Género predicho:", genero)
    print(f"Confianza: {probabilidad:.2%}")

    # Mostrar una gráfica de barras con la probabilidad de cada uno de los 10 géneros
    graficar_probabilidades(probabilidades)
