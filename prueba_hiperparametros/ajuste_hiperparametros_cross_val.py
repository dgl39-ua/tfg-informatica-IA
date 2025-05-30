import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

os.environ["CUDA_VISIBLE_DEVICES"] = "0"
os.environ["TF_FORCE_GPU_ALLOW_GROWTH"] = "true"

os.environ.pop('TF_GPU_ALLOCATOR', None)

import banner

banner.banner(' GPU disponibles para TF ')

import tensorflow as tf

print(tf.config.list_physical_devices('GPU'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras import Input
from tensorflow.keras.models import Sequential, load_model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam, RMSprop
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.metrics import Precision, Recall
from tensorflow.keras.preprocessing.image import ImageDataGenerator, load_img, img_to_array
from sklearn.metrics import confusion_matrix, classification_report, f1_score
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.optimizers import Adam, RMSprop
from random import sample
from PIL import Image
from io import BytesIO
from itertools import product
import librosa
import librosa.display

# Parámetros globales
IMG_HEIGHT = 128
IMG_WIDTH = 128
BATCH_SIZE = 32
EPOCHS = 50
AUDIO_DIR = 'audios_GTZAN/'

# Función que genera los espectrogramas de Mel a partir de los audios del dataset
# permitiendo elegir el tamaño de las imágenes (ej. 128x128 o 256x256)
def generar_espectrogramas(origen_audio_dir, destino_img_dir, img_size=(128, 128)):
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
                    imagen = imagen.resize(img_size)
                    imagen.save(img_path)

                    print(f"Generado: {img_path}")

                except Exception as e:
                    print(f"Error procesando {audio_path}: {e}")

def crear_y_compilar_modelo(num_clases, image_size=128, filters=32, dropout=0.3, lr=0.0005, optimizer_name="adam"):
    if optimizer_name == "adam":
        opt = Adam(learning_rate=lr)
    else:
        opt = RMSprop(learning_rate=lr)

    model = Sequential([
        Input(shape=(image_size, image_size, 3)),

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
    plt.title('Accuracy')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs_range, loss, label='Entrenamiento')
    plt.plot(epochs_range, val_loss, label='Validación')
    plt.title('Pérdida')
    plt.legend()

    plt.tight_layout()
    plt.savefig(filename)
    plt.close()
    print(f"Gráfica de entrenamiento guardada como {filename}")

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
    plt.title('Matriz de confusión modelo ajustado')
    plt.tight_layout()
    plt.savefig('matriz_confusion_mejor_modelo_cross_val.png')
    plt.show()
    print("Matriz de confusión guardada como 'matriz_confusion_mejor_modelo_cross_val.png'")

    # Reporte detallado
    report = classification_report(y_true, y_pred, target_names=nombres_clases)
    print("\nClassification report del mejor modelo:\n")
    print(report)

    with open("classification_report_mejor_modelo_cross_val.txt", "w") as f:
        f.write(report)

    print("Classification report guardado como 'classification_report_mejor_modelo_cross_val.txt'")

# Ajusta hiperparámetros usando Stratified K-Fold Cross Validation
def ajustar_arquitectura_cross_validation(param_grid, output_file="resultados_crossval_grid_search.csv", n_splits=5):
    best_f1 = 0
    best_model = None
    best_history = None
    best_val_gen = None

    results = []

    for i, params in enumerate(param_grid):
        image_size = params["img_size"]
        batch_size = params["batch_size"]
        origin_dir = f"images_GTZAN_{image_size}"

        print(f"\nEjecutando experimento {i+1}/{len(param_grid)}: {params}")
        print()
        print(f"Obteniendo imágenes del directorio {origin_dir}")

        # Cargar imágenes para el tamaño a probar
        filepaths = []
        labels = []

        for class_dir in os.listdir(origin_dir):
            class_path = os.path.join(origin_dir, class_dir)
            if os.path.isdir(class_path):
                for fname in os.listdir(class_path):
                    if fname.lower().endswith(('.png', '.jpg', '.jpeg')):
                        filepaths.append(os.path.join(class_path, fname))
                        labels.append(class_dir)

        df = pd.DataFrame({"filename": filepaths, "class": labels})

        skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
        fold_metrics = []

        for fold_idx, (train_idx, val_idx) in enumerate(skf.split(df["filename"], df["class"])):
            print(f"\n============================")
            print(f"Fold {fold_idx+1}/5")
            print("============================")
            
            train_df = df.iloc[train_idx]
            val_df = df.iloc[val_idx]

            datagen = ImageDataGenerator(rescale=1./255)

            train_gen = datagen.flow_from_dataframe(
                train_df,
                x_col="filename",
                y_col="class",
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode="categorical"
            )

            val_gen = datagen.flow_from_dataframe(
                val_df,
                x_col="filename",
                y_col="class",
                target_size=(image_size, image_size),
                batch_size=batch_size,
                class_mode="categorical",
                shuffle=False
            )

            # Crear el modelo y compilar
            model = crear_y_compilar_modelo(
                num_clases=len(train_gen.class_indices),
                image_size=image_size,
                filters=params["filters"],
                dropout=params["dropout"],
                lr=params["lr"],
                optimizer_name=params["optimizer"]
            )

            # Mostrar un resumen de la estructura del modelo
            model.summary()
            print()

            # Incluir callbacks
            checkpoint_path = f"checkpoint_fold{fold_idx+1}.keras"
            callbacks = [
                EarlyStopping(monitor='val_accuracy', patience=10, restore_best_weights=True),
                ModelCheckpoint(checkpoint_path, monitor='val_accuracy', save_best_only=True, verbose=0)
            ]

            # Entrenar el modelo
            history = model.fit(
                train_gen,
                validation_data=val_gen,
                epochs=EPOCHS,
                callbacks=callbacks,
                verbose=1
            )

            # Calcular f1 sobre este fold
            y_true = val_gen.classes
            y_pred_probs = model.predict(val_gen)
            y_pred = np.argmax(y_pred_probs, axis=1)
            
            f1 = f1_score(y_true, y_pred, average='macro')
            print(f"F1-score fold {fold_idx+1}: {f1:.4f}")

            results_fold = model.evaluate(val_gen, verbose=0)  # [loss, acc, precision, recall]
            fold_metrics.append(np.append(results_fold, f1))

        fold_metrics = np.array(fold_metrics)
        avg_results = np.mean(fold_metrics, axis=0)  # [loss, acc, precision, recall, f1]
        std_results = np.std(fold_metrics, axis=0)

        results.append({
            "img_size": params["img_size"],
            "filters": params["filters"],
            "dropout": params["dropout"],
            "batch_size": params["batch_size"],
            "lr": params["lr"],
            "optimizer": params["optimizer"],
            "accuracy_mean": avg_results[1],
            "accuracy_std": std_results[1],
            "precision_mean": avg_results[2],
            "precision_std": std_results[2],
            "recall_mean": avg_results[3],
            "recall_std": std_results[3],
            "f1_score_mean": avg_results[4],
            "f1_score_std": std_results[4]
        })

        if avg_results[4] > best_f1:
            best_f1 = avg_results[4]
            best_model = model
            best_history = history
            best_val_gen = val_gen
            model.save("mejor_modelo_ajustado.keras")
            print(f"Nuevo mejor modelo global guardado (F1-score={best_f1:.4f})")

        print(f"Media cross-validation: Accuracy={avg_results[1]:.4f} Precision={avg_results[2]:.4f} Recall={avg_results[3]:.4f} F1-score={avg_results[4]:.4f}")

    df_results = pd.DataFrame(results)
    df_results.to_csv(output_file, index=False)
    print(f"\nResultados cross-validation guardados en {output_file}")

    if best_model and best_val_gen:
        print("\nGenerando resultados finales del mejor modelo...")

        # Gráfica del entrenamiento del modelo ajustado
        graficar_entrenamiento(best_history, filename="grafica_entrenamiento_mejor_modelo_cross_val.png")

        # Mostrar la matriz de confusión y el classification report
        mostrar_matriz_confusion(best_model, best_val_gen)
        
def generar_param_grid():
    param_grid = []

    for (img_size, filters, dropout, batch_size, lr, optimizer) in product(
        [128, 256],                # Ambos tamaños de imagen
        [16, 32],                  # Ambos valores de filtros
        [0.2, 0.3],                # Dropout
        [16, 32],                  # Batch size
        [0.001, 0.0005],           # Learning rate
        ["adam", "rmsprop"]        # Optimizador
    ):
        param_grid.append({
            "img_size": img_size,
            "filters": filters,
            "dropout": dropout,
            "batch_size": batch_size,
            "lr": lr,
            "optimizer": optimizer
        })

    return param_grid

if __name__ == "__main__":
    # Generar espectrogramas para 128x128
    generar_espectrogramas(AUDIO_DIR, "images_GTZAN_128", img_size=(128, 128))

    # Generar espectrogramas para 256x256
    generar_espectrogramas(AUDIO_DIR, "images_GTZAN_256", img_size=(256, 256))
    
    # Generar las combinaciones
    param_grid = generar_param_grid()

    # Ajustar hiperparámetros de la arquitectura
    ajustar_arquitectura_cross_validation(param_grid)
