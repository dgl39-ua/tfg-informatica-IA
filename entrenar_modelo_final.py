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
import librosa
import librosa.display
from PIL import Image
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score
from io import BytesIO

IMG_SIZE = 256
BATCH_SIZE = 16
EPOCHS = 50
IMAGES_DIR = 'images_GTZAN_256/'
FILENAME_SAVED_MODEL = "modelo_definitivo_entrenado_completo.keras"
AUDIOS_IA_DIR = 'audios_eval_sist_generativos/'
IMAGES_IA_DIR = 'images_sist_generativos/'

# Función que carga el dataset
def cargar_y_preparar_dataset():
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
        shuffle=False
    )

    return full_gen

def crear_y_compilar_modelo(num_clases, filters=16):
    model = Sequential([
        Input(shape=(IMG_SIZE, IMG_SIZE, 3)),
        Conv2D(filters, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(filters*2, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        Conv2D(filters*4, (3, 3), activation='relu'),
        BatchNormalization(),
        MaxPooling2D((2, 2)),

        GlobalAveragePooling2D(),

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

# Función para mostrar la evolución de la pérdida y la precisión durante el entrenamiento
def graficar_resultados(history):
    acc = history.history['accuracy']
    loss = history.history['loss']
    epochs_range = range(len(acc))

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs_range, acc, label='Accuracy de entrenamiento')
    plt.title('Evolución de la Accuracy')
    plt.xlabel('Épocas')
    plt.ylabel('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Pérdida de entrenamiento', color='orange')
    plt.title('Evolución de la Pérdida')
    plt.xlabel('Épocas')
    plt.ylabel('Loss')
    plt.legend()

    plt.tight_layout()
    plt.savefig('grafica_entrenamiento_modelo_final.png')
    plt.show()

# Función para mostrar la matriz de confusión
def mostrar_matriz_confusion(y_true, y_pred, class_indices, matrix_filename, report_filename):
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
    plt.savefig(f'{matrix_filename}.png')
    plt.show()

    report = classification_report(y_true, y_pred, target_names=nombres_clases)
    print("\nClassification report del modelo final:\n")
    print(report)
    with open(report_filename, "w") as f:
        f.write(report)
    print(f"Classification report guardado como '{report_filename}'")

# Función que carga el modelo guardado
def cargar_modelo(filename=FILENAME_SAVED_MODEL):
    return load_model(filename)

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


# Función que genera los espectrogramas de Mel a partir de los audios del dataset
def generar_espectrogramas_IA(origen_audio_dir, destino_img_dir):
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
                    imagen = imagen.resize((IMG_SIZE, IMG_SIZE))
                    imagen.save(img_path)

                    print(f"Generado: {img_path}")

                except Exception as e:
                    print(f"Error procesando {audio_path}: {e}")

def evaluar_modelo_en_ia(model, test_dir, matrix_filemame, report_filename):
    os.makedirs("resultados_IA", exist_ok=True)

    # Preparar generador para los espectrogramas IA
    datagen = ImageDataGenerator(rescale=1./255)
    ia_gen = datagen.flow_from_directory(
        test_dir,
        target_size=(IMG_SIZE, IMG_SIZE),
        batch_size=BATCH_SIZE,
        class_mode="categorical",
        shuffle=False
    )

    # Usar las clases detectadas en el conjunto IA
    class_indices_ia = ia_gen.class_indices
    etiquetas = {v: k for k, v in class_indices_ia.items()}
    nombres_clases = [etiquetas[i] for i in range(len(etiquetas))]

    y_true = ia_gen.classes
    y_pred = []
    predicciones_data = []

    # Predicciones individuales
    for i, file_path in enumerate(ia_gen.filepaths):
        genero_predicho, probabilidad, _ = predecir_genero(file_path, model, class_indices_ia)
        clase_real = etiquetas[y_true[i]]
        pred_idx = list(etiquetas.values()).index(genero_predicho)

        y_pred.append(pred_idx)

        predicciones_data.append({
            "archivo": os.path.basename(file_path),
            "clase_real": clase_real,
            "clase_predicha": genero_predicho,
            "probabilidad": round(probabilidad, 4)
        })

    # Calcular métricas
    f1 = f1_score(y_true, y_pred, average='macro')
    loss, acc, precision, recall = model.evaluate(ia_gen, verbose=0)

    # Guardar resultados generales
    with open("resultados_IA/resultados_IA.txt", "w") as f:
        f.write("Evaluación del modelo sobre los audios generados por IA:\n")
        f.write(f"Loss: {loss:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")

    print("\nEvaluación del modelo sobre los audios generados por IA:")
    print(f"Loss: {loss:.4f}")
    print(f"Accuracy: {acc:.4f}")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1-score: {f1:.4f}")

    # Guardar predicciones por archivo
    df_predicciones = pd.DataFrame(predicciones_data)
    df_predicciones.to_csv("resultados_IA/predicciones_IA.csv", index=False)
    print("Predicciones guardadas en 'predicciones_IA.csv'")

    # Matriz de confusión y classification report
    mostrar_matriz_confusion(y_true, y_pred, class_indices_ia, matrix_filemame, report_filename)

    # Predicción de ejemplo (primera imagen generada por IA)
    ejemplo = ia_gen.filepaths[0]
    genero, probabilidad, probabilidades = predecir_genero(ejemplo, model, class_indices_ia)
    print("\nEjemplo de predicción con imagen de IA:")
    print("Género predicho:", genero)
    print(f"Confianza: {probabilidad:.2%}")

    # Gráfica de probabilidades
    graficar_probabilidades(probabilidades)


def predecir_y_guardar_todas_las_predicciones(directorio, modelo, class_indices, output_csv="predicciones.csv"):
    resultados = []

    # Invertir class_indices para recuperar nombre de clase desde índice
    etiquetas = {v: k for k, v in class_indices.items()}

    for clase in sorted(os.listdir(directorio)):
        clase_path = os.path.join(directorio, clase)
        if not os.path.isdir(clase_path):
            continue

        for fname in sorted(os.listdir(clase_path)):
            if not fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                continue

            ruta_img = os.path.join(clase_path, fname)

            # Cargar y preprocesar la imagen
            img = load_img(ruta_img, target_size=(IMG_SIZE, IMG_SIZE))
            img_array = img_to_array(img) / 255.0
            img_array = np.expand_dims(img_array, axis=0)

            # Realizar la predicción
            pred = modelo.predict(img_array, verbose=0)[0]
            idx_predicho = np.argmax(pred)
            confianza = float(np.max(pred))
            genero_predicho = etiquetas.get(idx_predicho, f"Clase_{idx_predicho}")

            resultados.append({
                "archivo": fname,
                "clase_real": clase,
                "clase_predicha": genero_predicho,
                "probabilidad": round(confianza, 4)
            })

    # Guardar resultados en CSV
    df_resultados = pd.DataFrame(resultados)
    df_resultados.to_csv(output_csv, index=False)
    print(f"✅ Predicciones guardadas en: {output_csv}")

if __name__ == "__main__":
    print("\n========================================")
    print("ENTRENAMIENTO FINAL CON TODO EL DATASET")
    print("========================================")

    # Cargar el dataset GTZAN
    full_gen = cargar_y_preparar_dataset()
    
    # Crear y compilar el modelo final
    model = crear_y_compilar_modelo(num_clases=len(full_gen.class_indices), filters=16)

    # Mostrar un resumen de la estructura del modelo final
    model.summary()
    print()

    # Incluir callbacks
    checkpoint_path = f"checkpoint_modelo_definitivo.keras"
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

    with open("resultados_finales.txt", "w") as f:
        f.write(f"Loss: {loss:.4f}\n")
        f.write(f"Accuracy: {acc:.4f}\n")
        f.write(f"Precision: {precision:.4f}\n")
        f.write(f"Recall: {recall:.4f}\n")
        f.write(f"F1-score: {f1:.4f}\n")
   
    # Gráfica de entrenamiento
    graficar_resultados(history)

    # Mostrar la matriz de confusión
    mostrar_matriz_confusion(y_true, y_pred, full_gen.class_indices, matrix_filename="matriz_confusion_modelo_final", report_filename="classification_report_modelo_final")

    # Generar los espectrogramas a partir de los audios de IA
    #generar_espectrogramas_IA(origen_audio_dir=AUDIOS_IA_DIR, destino_img_dir=IMAGES_IA_DIR)

    # Evaluación con los espectrogramas generados a partir de los audios de IA
    """
    test_dir = IMAGES_IA_DIR
    matrix_filename="matriz_confusion_IA"
    report_filename="classification_report_IA"
    model = cargar_modelo(filename=FILENAME_SAVED_MODEL)
    evaluar_modelo_en_ia(model, test_dir, matrix_filename, report_filename)
    """
    """
    # Ejemplo de predicción
    modelo_guardado = cargar_modelo(FILENAME_SAVED_MODEL)
    ejemplo = 'images_GTZAN/pop/pop.00047.png'
    genero, probabilidad, probabilidades = predecir_genero(ejemplo, modelo_guardado, full_gen.class_indices)
    print("Género predicho:", genero)
    print(f"Confianza: {probabilidad:.2%}")

    # Mostrar una gráfica de barras con la probabilidad de cada uno de los 10 géneros
    graficar_probabilidades(probabilidades)
    """
    
    directorio_test = IMAGES_IA_DIR
    modelo_guardado = cargar_modelo(FILENAME_SAVED_MODEL)
    full_gen = cargar_y_preparar_dataset()
    predecir_y_guardar_todas_las_predicciones(directorio_test, modelo_guardado, full_gen.class_indices)
