import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import tempfile
import librosa
import librosa.display
import matplotlib.pyplot as plt
from entrenar_modelo_final import IMG_SIZE, IMAGES_DIR, FILENAME_SAVED_MODEL, cargar_modelo, cargar_y_preparar_dataset, predecir_genero, graficar_probabilidades

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

# -------------------------------
# Interfaz
# -------------------------------

# Cargar el modelo y las clases
st.title("🎵 Clasificador de Géneros Musicales")
st.markdown("---")

with st.spinner("Cargando modelo..."):
    modelo = cargar_modelo(FILENAME_SAVED_MODEL)
    full_gen, _ = cargar_y_preparar_dataset(IMAGES_DIR)
    class_indices = full_gen.class_indices # Se cargan las clases
    etiquetas = {v: k for k, v in class_indices.items()}
st.success("Modelo cargado correctamente ✅")

# Selector de tipo de archivo
tipo = st.radio("Selecciona el tipo de archivo a analizar:", ["🎧 Audio (.wav)", "🖼️ Imagen de espectrograma (.png, .jpg)"])

# Subir el archivo
archivo = st.file_uploader("Sube tu archivo:", type=["wav", "png", "jpg", "jpeg"])

# Procesar el archivo
if archivo is not None:
    if tipo == "🎧 Audio (.wav)":
        st.audio(archivo, format='audio/wav')
        with st.spinner("Generando espectrograma..."):
            espectrograma_img = audio_a_espectrograma(archivo)
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.image(espectrograma_img, caption="Espectrograma generado", width=400)
    else:
        espectrograma_img = Image.open(archivo).convert("RGB")
        espectrograma_img = espectrograma_img.resize((IMG_SIZE, IMG_SIZE))
        cols = st.columns([1, 2, 1])
        with cols[1]:
            st.image(espectrograma_img, caption="Espectrograma cargado", width=400)

    # Clasificar el género
    if st.button("🎼 Clasificar género"):
        genero, probabilidad, probabilidades = predecir_genero(espectrograma_img, modelo, full_gen.class_indices)

        st.markdown(f"🎤 **Género predicho:** `{genero}`")
        st.markdown(f"📊 **Confianza:** `{probabilidad:.2%}`")

        graficar_probabilidades(probabilidades)
        st.pyplot(plt)

st.markdown("---")
st.caption("CNN y GTZAN - Diego García López - TFG INFORMÁTICA")