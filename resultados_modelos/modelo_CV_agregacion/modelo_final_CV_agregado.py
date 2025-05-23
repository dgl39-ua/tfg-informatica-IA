import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import tensorflow as tf
import numpy as np
import pandas as pd
import time
import matplotlib.pyplot as plt
import seaborn as sns
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import Sequential
from tensorflow.keras import Input
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, BatchNormalization, GlobalAveragePooling2D
from tensorflow.keras.metrics import Precision, Recall
from sklearn.metrics import f1_score
from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report

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

# Entrenar modelo
def entrenar_modelo(model, train_gen, val_gen, callbacks):
    history = model.fit(
        train_gen,
        validation_data=val_gen,
        epochs=EPOCHS,
        callbacks=callbacks
    )
    
    return history

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

if __name__ == "__main__":
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

    all_y_true = []
    all_y_pred = []

    best_f1 = 0

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
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical"
        )

        val_gen = datagen.flow_from_dataframe(
            val_df,
            x_col="filename",
            y_col="class",
            target_size=(IMG_SIZE, IMG_SIZE),
            batch_size=BATCH_SIZE,
            class_mode="categorical",
            shuffle=False
        )

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

        # Obtener etiquetas y predicciones de este fold
        y_true = val_gen.classes
        y_pred_probs = model.predict(val_gen)
        y_pred = np.argmax(y_pred_probs, axis=1)

        # Acumular
        all_y_true.extend(y_true)
        all_y_pred.extend(y_pred)

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

    etiquetas = {v: k for k, v in best_val_gen.class_indices.items()}
    nombres_clases = [etiquetas[i] for i in range(len(etiquetas))]

    # Matriz de confusión
    cm = confusion_matrix(all_y_true, all_y_pred)
    print("Matriz de confusión agregada:\n", cm)

    # Métricas agregadas
    acc_global   = accuracy_score(all_y_true, all_y_pred)
    prec_global  = precision_score(all_y_true, all_y_pred, average='macro')
    rec_global   = recall_score(all_y_true, all_y_pred, average='macro')
    f1_global    = f1_score(all_y_true, all_y_pred, average='macro')

    print(f"\nMétricas globales:")
    print(f"  Accuracy:  {acc_global:.4f}")
    print(f"  Precision: {prec_global:.4f}")
    print(f"  Recall:    {rec_global:.4f}")
    print(f"  F1-score:  {f1_global:.4f}")

    # Reporte detallado
    report = classification_report(all_y_true, all_y_pred, target_names=nombres_clases)
    print("\nClassification Report global:\n")
    print(report)

    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Purples',
                xticklabels=nombres_clases, yticklabels=nombres_clases)
    plt.xlabel('Género predicho')
    plt.ylabel('Género real')
    plt.title('Matriz de Confusión Agregada (5 folds)')
    plt.tight_layout()
    plt.savefig('matriz_confusion_global.png')
    plt.show()

    with open("classification_report_global.txt", "w") as f:
        f.write(report)
    print("Reporte global guardado en 'classification_report_global.txt'")

    # Gráfica del entrenamiento
    graficar_resultados(best_history)

    # Mostrar la matriz de confusión
    #mostrar_matriz_confusion(best_model, best_val_gen)