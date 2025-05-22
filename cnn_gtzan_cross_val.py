import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
from tensorflow import keras
import matplotlib.pyplot as plt
from random import sample
import time
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
import librosa
import librosa.display
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score
import pandas as pd
from io import BytesIO

IMG_HEIGHT = 256
IMG_WIDTH = 256
BATCH_SIZE = 16
EPOCHS = 50
IMAGES_DIR = 'images_GTZAN/'
AUDIOS_DIR = 'audios_GTZAN/'

# Función que genera los espectrogramas de Mel a partir de los audios del dataset
def generar_espectrogramas(origen_audio_dir=AUDIOS_DIR, destino_img_dir=IMAGES_DIR):
    hl = 512
    hi = 128
    target_sr = 22050
    duration = 30

    generos = os.listdir(origen_audio_dir)
    for genero in generos:
        ruta_audio = os.path.join(origen_audio_dir, genero)
        ruta_destino = os.path.join(destino_img_dir, genero)

        if not os.path.isdir(ruta_audio):
            continue

        os.makedirs(ruta_destino, exist_ok=True)

        for archivo in os.listdir(ruta_audio):
            if archivo.lower().endswith(".wav"):
                audio_path = os.path.join(ruta_audio, archivo)
                img_name = os.path.splitext(archivo)[0] + ".png"
                img_path = os.path.join(ruta_destino, img_name)

                if os.path.exists(img_path):
                    continue

                try:
                    y, sr = librosa.load(audio_path, sr=target_sr, duration=duration)
                    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hi, hop_length=hl)
                    S_dB = librosa.power_to_db(S, ref=np.max)

                    plt.rcParams["figure.figsize"] = [12.92, 5.12]
                    plt.rcParams["figure.autolayout"] = True
                    fig, ax = plt.subplots()
                    librosa.display.specshow(S_dB, sr=sr, hop_length=hl, y_axis='mel', x_axis=None, ax=ax)
                    plt.axis('off')

                    buf = BytesIO()
                    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
                    plt.close(fig)
                    buf.seek(0)
                    imagen = Image.open(buf).convert("RGB")
                    imagen = imagen.resize((IMG_WIDTH, IMG_HEIGHT))
                    imagen.save(img_path)

                    print(f"Generado: {img_path}")

                except Exception as e:
                    print(f"Error procesando {audio_path}: {e}")

# Función que busca y evalúa la precisión del modelo guardado
def buscar_modelo_guardado(best_f1):
    if os.path.exists("modelo_cross_val.keras"):
        print("Se ha encontrado un modelo previo. Evaluando...")
        previous_model = load_model("modelo_cross_val.keras")

        datagen = ImageDataGenerator(rescale=1./255)
        val_gen_prev = datagen.flow_from_dataframe(
            df,
            x_col="filename",
            y_col="class",
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False
        )

        y_true_prev = val_gen_prev.classes
        y_pred_probs_prev = previous_model.predict(val_gen_prev)
        y_pred_prev = np.argmax(y_pred_probs_prev, axis=1)
        best_f1 = f1_score(y_true_prev, y_pred_prev, average='macro')
        print(f"F1-score del modelo previo: {best_f1:.4f}")
    
    return best_f1

# Función que comprueba y grafica los conjuntos divididos
def comprobar_equilibrio_conjuntos(train_gen, val_gen):
    # Obtener los nombres de las clases
    genres = list(train_gen.class_indices.keys())

    # Contar las instancias por clase
    train_class_counts = pd.Series(train_gen.classes).value_counts().sort_index()
    val_class_counts = pd.Series(val_gen.classes).value_counts().sort_index()

    # Visualización en un gráfico de barras
    plt.figure(figsize=(10, 5))
    x = np.arange(len(genres))
    width = 0.35

    plt.bar(x - width/2, train_class_counts, width, label='Entrenamiento')
    plt.bar(x + width/2, val_class_counts, width, label='Validación')

    plt.xticks(ticks=x, labels=genres, rotation=45)
    plt.xlabel("Géneros musicales")
    plt.ylabel("Número de imágenes")
    plt.title("Distribución de clases en el conjunto de entrenamiento y validación")
    plt.legend()
    plt.tight_layout()
    plt.savefig('equilibrio_conjuntos.png')
    plt.show()

# Función que carga el dataset y lo divide en los conjuntos de entrenamiento y test
def cargar_y_preparar_dataset(data_dir):
    # Generar listado de imágenes y etiquetas
    filepaths = []
    labels = []

    for class_dir in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_dir)
        if os.path.isdir(class_path):
            for fname in os.listdir(class_path):
                if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                    filepaths.append(os.path.join(class_path, fname))
                    labels.append(class_dir)
    
    df = pd.DataFrame({"filename": filepaths, "class": labels})

    # Stratified K-Fold
    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    for train_idx, val_idx in skf.split(df["filename"], df["class"]):
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        # Paso 3: Generadores
        datagen = ImageDataGenerator(rescale=1./255)

        train_gen = datagen.flow_from_dataframe(
            train_df,
            x_col="filename",
            y_col="class",
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            class_mode="categorical",
            batch_size=BATCH_SIZE
        )

        val_gen = datagen.flow_from_dataframe(
            val_df,
            x_col="filename",
            y_col="class",
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            class_mode="categorical",
            batch_size=BATCH_SIZE
        )

    comprobar_equilibrio_conjuntos(train_gen, val_gen)

    return train_gen, val_gen

# Función para construir el modelo
def crear_y_compilar_modelo_1_preliminar(num_clases):
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),  # Aplanamiento de las salidas para conectarlas a la capa Dense
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_clases, activation='softmax') # Capa de salida
    ])
    
    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    return model

def crear_y_compilar_modelo(num_clases):
    model = Sequential([
        Input(shape=(IMG_HEIGHT, IMG_WIDTH, 3)),
        Conv2D(32, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(64, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Conv2D(128, (3, 3), activation='relu'),
        MaxPooling2D((2, 2)),
        Flatten(),  # Aplanamiento de las salidas para conectarlas a la capa Dense
        Dense(128, activation='relu'),
        Dropout(0.3),
        Dense(num_clases, activation='softmax') # Capa de salida
    ])
    
    model.compile(
        optimizer=Adam(learning_rate=0.0005),
        loss='categorical_crossentropy',
        metrics=['accuracy', Precision(name='precision'), Recall(name='recall')]
    )

    return model

# Entrenar modelo
def entrenar_modelo(model, train_gen, val_gen, callbacks):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return history

# Función para evaluar y mostrar los resultados
def evaluar_resultados(model, val_gen):
    test_loss, test_acc = model.evaluate(val_gen)

    print(f"Precisión en test: {test_acc:.4f}")
    print(f"Pérdida en test: {test_loss:.4f}")

    results = model.evaluate(val_gen)
    print(f"Fold results: {results}")

    return test_loss, test_acc

# Función para mostrar la evolución de la pérdida y la precisión durante el entrenamiento
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
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Validación')
    plt.title('Pérdida')
    plt.legend()

    plt.savefig('grafica_CNN_cross_val.png')
    plt.show()

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

# Función para mostrar la matriz de confusión
def mostrar_matriz_confusion(best_model, best_val_gen):
    # Obtener nombres de clases
    etiquetas = {v: k for k, v in best_val_gen.class_indices.items()}
    nombres_clases = [etiquetas[i] for i in range(len(etiquetas))]

    # Obtener predicciones y etiquetas reales
    y_true = best_val_gen.classes
    y_pred_probs = best_model.predict(best_val_gen)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # Matriz de confusión
    cm = confusion_matrix(y_true, y_pred)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples', xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.xlabel('Género predicho')
    plt.ylabel('Género real')
    plt.title('Matriz de Confusión')
    plt.tight_layout()
    plt.savefig('matriz_confusion_cross_val.png')
    plt.show()

    # Reporte detallado
    report = classification_report(y_true, y_pred, target_names=nombres_clases)
    print("\nClassification report del mejor modelo:\n")
    print(report)
    with open("classification_report_modelo_cross_val.txt", "w") as f:
        f.write(report)
    print("Classification report guardado como 'classification_report_modelo_cross_val.txt'")

def predecir_genero(img_path_or_array, model, class_indices):
    if isinstance(img_path_or_array, str):
        img = load_img(img_path_or_array, target_size=(IMG_HEIGHT, IMG_WIDTH))
        img_array = img_to_array(img) / 255.0
    elif isinstance(img_path_or_array, Image.Image):
        img = img_path_or_array.resize((IMG_WIDTH, IMG_HEIGHT))
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


if __name__ == "__main__":
    # Generar espectogramas a partir de los audios
    #generar_espectrogramas(origen_audio_dir=AUDIOS_DIR, destino_img_dir=IMAGES_DIR)

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

    skf = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

    best_f1 = 0

    # Si ya existe un modelo guardado, cargarlo y evaluarlo
    #best_f1 = buscar_modelo_guardado(best_f1)
    #print(best_f1)
    #print()

    best_model = None
    best_history = None
    best_val_gen = None

    best_loss = 0
    best_acc = 0
    best_precision = 0
    best_recall = 0

    for fold, (train_idx, val_idx) in enumerate(skf.split(df["filename"], df["class"])):
        print(f"\n============================")
        print(f"Fold {fold+1}/5")
        print("============================")
    
        train_df = df.iloc[train_idx]
        val_df = df.iloc[val_idx]

        datagen = ImageDataGenerator(rescale=1./255)

        train_gen = datagen.flow_from_dataframe(
            train_df,
            x_col="filename",
            y_col="class",
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical"
        )

        val_gen = datagen.flow_from_dataframe(
            val_df,
            x_col="filename",
            y_col="class",
            target_size=(IMG_HEIGHT, IMG_WIDTH),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False
        )

        comprobar_equilibrio_conjuntos(train_gen, val_gen)

        # Crear el modelo y compilar
        model = crear_y_compilar_modelo(num_clases=len(train_gen.class_indices))

        # Mostrar un resumen de la estructura del modelo
        model.summary()
        print()

        # Incluir callbacks
        checkpoint_path = f"checkpoint_fold{fold+1}.keras"
        callbacks = [
            EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
            ModelCheckpoint(filepath=checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=0)
        ]

        # Entrenar el modelo
        start_time = time.time()
        history = entrenar_modelo(model, train_gen, val_gen, callbacks)
        training_time = time.time() - start_time

        print(f"Tiempo de entrenamiento fold {fold+1}: {training_time:.2f} s")

        # Calcular f1-score sobre val
        y_true = val_gen.classes
        y_pred_probs = model.predict(val_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)

        f1 = f1_score(y_true, y_pred, average='macro')
        print(f"F1-score fold {fold+1}: {f1:.4f}")

        # Y evaluar el resto de métricas
        results = model.evaluate(val_gen, verbose=0)
        loss, acc, precision, recall = results
        print(f"Métricas adicionales: Accuracy={acc:.4f} Precision={precision:.4f} Recall={recall:.4f}")

        # Guardar el mejor modelo si supera f1
        if f1 > best_f1:
            best_f1 = f1
            best_model = model
            best_history = history
            best_val_gen = val_gen
            best_loss = loss
            best_acc = acc
            best_precision = precision
            best_recall = recall
            model.save("modelo_cross_val.keras")
            print("Mejor modelo actualizado y guardado.")

    print("\nResultados finales del mejor modelo seleccionado:")
    print(f"Accuracy={best_acc:.4f} Precision={best_precision:.4f} Recall={best_recall:.4f}")
    print(f"F1-score={best_f1:.4f}")

    # Gráfica del entrenamiento
    graficar_resultados(best_history)

    # Mostrar la matriz de confusión
    mostrar_matriz_confusion(best_model, best_val_gen)

    """
    # Ejemplo de predicción
    ejemplo = 'images_GTZAN/pop/pop.00047.png'
    genero, probabilidad, probabilidades = predecir_genero(ejemplo, modelo_guardado, train_gen.class_indices)
    print("Género predicho:", genero)
    print(f"Confianza: {probabilidad:.2%}")

    # Mostrar una gráfica de barras con la probabilidad de cada uno de los 10 géneros
    graficar_probabilidades(probabilidades)
    """