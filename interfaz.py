import logging, os
logging.disable(logging.WARNING)
os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"

import streamlit as st
import numpy as np
from io import BytesIO
import tempfile
import librosa
import librosa.display
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import matplotlib.pyplot as plt
from PIL import Image, UnidentifiedImageError
from librosa.util.exceptions import ParameterError
import audioread
from audioread.exceptions import NoBackendError
from tensorflow.errors import ResourceExhaustedError

IMG_SIZE = 256
BATCH_SIZE = 16
IMAGES_DIR = 'images_GTZAN_256/'
FILENAME_SAVED_MODEL = 'modelo_cross_val.keras'     # 'modelo_reentrenado.keras'
AUDIOS_IA_DIR = 'audios_eval_sist_generativos/'
IMAGES_IA_DIR = 'images_sist_generativos/'

# Función que convierte el audio en espectrograma
def audio_a_espectrograma(wav_file):
    # Guardar temporalmente el archivo subido
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmpfile:
        tmpfile.write(wav_file.read())
        tmpfile_path = tmpfile.name

    # Parámetros
    hl = 512  # hop_length, number of samples per time-step in spectrogram
    hi = 128  # número de bandas mel, Height of image
    duration = 30  # segundos
    target_sr = 22050  # frecuencia de muestreo estándar

    # Cargar el audio
    y, sr = librosa.load(tmpfile_path, sr=target_sr, duration=duration)

    # Generar el espectrograma de Mel
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=hi, hop_length=hl)
    S_dB = librosa.power_to_db(S, ref=np.max)

    # Ajustar tamaño de la figura
    plt.rcParams["figure.figsize"] = [12.92, 5.12]  # para 1292 time bins (30s @ 22050Hz, hl=512)
    plt.rcParams["figure.autolayout"] = True

    fig, ax = plt.subplots()
    img = librosa.display.specshow(S_dB, x_axis=None, y_axis='mel', sr=sr, hop_length=hl, ax=ax)

    plt.axis('off')  # Ocultar los ejes

    # Convertir figura en imagen
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    imagen = Image.open(buf).convert("RGB")

    # Redimensionar para la CNN
    imagen = imagen.resize((IMG_SIZE, IMG_SIZE))

    return imagen

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

# -------------------------------
# Interfaz
# -------------------------------

# Cargar el modelo y las clases
st.title("🎵 Clasificador de Géneros Musicales")
st.markdown("---")

with st.spinner("Cargando modelo..."):
    if not os.path.isfile(FILENAME_SAVED_MODEL):
        st.error(f"🚫 No se ha encontrado el modelo: `{FILENAME_SAVED_MODEL}`")
        st.stop()
    
    try:
        modelo = load_model(FILENAME_SAVED_MODEL)
    except Exception as e:
        st.error(f"🚫 Ha ocurrido un error al cargar el modelo:\n{e}")
        st.stop()

    class_indices = {'blues': 0, 'classical': 1, 'country': 2, 'disco': 3, 'hiphop': 4, 'jazz': 5, 'metal': 6, 'pop': 7, 'reggae': 8, 'rock': 9}    # Se cargan las clases
    etiquetas = {v: k for k, v in class_indices.items()}
st.success("Modelo cargado correctamente ✅")

# Selector de tipo de archivo
tipo = st.radio("Selecciona el tipo de archivo a analizar:", ["🎧 Audio (.wav)", "🖼️ Imagen de espectrograma (.png, .jpg)"])

# Subir el archivo
archivo = st.file_uploader("Sube tu archivo:", type=["wav", "png", "jpg", "jpeg"])

# Procesar el archivo
if archivo is not None:
    if tipo == "🎧 Audio (.wav)":
        try:
            st.audio(archivo, format='audio/wav')
            with st.spinner("Generando espectrograma..."):
                espectrograma_img = audio_a_espectrograma(archivo)
        except NoBackendError:
            st.error("🚫 No se ha podido procesar el archivo como audio. Por favor, asegúrate de seleccionar un audio en formato .wav.")
            st.stop()
        except ParameterError as e:
            st.error("🚫 El audio está corrupto o no es un WAV válido.")
            st.stop()
        except Exception as e:
            st.error(f"🚫 Error inesperado al procesar audio: {e}")
            st.stop()

        # Mostrar espectrograma
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.image(espectrograma_img, caption="Espectrograma generado", width=400)
    else:
        try:
            espectrograma_img = Image.open(archivo).convert("RGB")
            espectrograma_img = espectrograma_img.resize((IMG_SIZE, IMG_SIZE))
        except UnidentifiedImageError:
            st.error("🚫 El archivo que has subido no es una imagen válida. Por favor, asegúrate de seleccionar una imagen en formato .png o .jpg.")
            st.stop()
        
        # Mostrar espectrograma
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.image(espectrograma_img, caption="Espectrograma cargado", width=400)

    # Clasificar el género
    if st.button("🎼 Clasificar género"):
        try:
            with st.spinner("Clasificando género..."):
                genero, probabilidad, probabilidades = predecir_genero(espectrograma_img, modelo, class_indices)
        except (ValueError, ResourceExhaustedError) as e:
            st.error(f"🚫 Error al hacer la predicción: {e}")
            st.stop()

        st.markdown(f"🎤 **Género predicho:** `{genero}`")
        st.markdown(f"📊 **Confianza:** `{probabilidad:.2%}`")

        try:
            graficar_probabilidades(probabilidades)
            st.pyplot(plt)
        except Exception as e:
            st.error(f"🚫 No se pudo dibujar la gráfica de probabilidades: {e}")

st.markdown("---")
st.caption("CNN y GTZAN - Diego García López - TFG INFORMÁTICA")