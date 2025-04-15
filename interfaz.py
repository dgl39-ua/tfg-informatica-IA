import streamlit as st
import numpy as np
from PIL import Image
from io import BytesIO
import librosa
import librosa.display
import matplotlib.pyplot as plt
from cnn_gtzan import cargar_modelo, predecir_genero, IMG_HEIGHT, IMG_WIDTH, crear_generadores

# --- Cargar modelo y clases ---
st.title("🎵 Clasificador de Géneros Musicales")

with st.spinner("Cargando modelo..."):
    modelo = cargar_modelo("modelo_gtzan_cnn.h5")
    train_gen, _ = crear_generadores()
    class_indices = train_gen.class_indices
    etiquetas = {v: k for k, v in class_indices.items()}
st.success("Modelo cargado correctamente ✅")

# --- Selector de tipo de archivo ---
tipo = st.radio("¿Qué quieres subir?", ["🎧 Audio (.wav)", "🖼️ Imagen de espectrograma (.png, .jpg)"])

# --- Subida de archivo ---
archivo = st.file_uploader("Sube tu archivo:", type=["wav", "png", "jpg", "jpeg"])

def audio_a_spectrograma(wav_bytes):
    # Cargar audio
    y, sr = librosa.load(wav_bytes, sr=None)

    # Obtener mel-espectrograma
    S = librosa.feature.melspectrogram(y=y, sr=sr, n_mels=128)
    S_DB = librosa.power_to_db(S, ref=np.max)

    # Convertir en imagen RGB
    fig, ax = plt.subplots(figsize=(2, 2), dpi=64)
    librosa.display.specshow(S_DB, sr=sr, x_axis=None, y_axis=None, cmap='viridis')
    plt.axis('off')

    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', pad_inches=0)
    plt.close(fig)
    buf.seek(0)
    imagen = Image.open(buf).convert("RGB")
    return imagen

if archivo is not None:
    if tipo == "🎧 Audio (.wav)":
        st.audio(archivo, format='audio/wav')
        with st.spinner("Generando espectrograma..."):
            espectrograma_img = audio_a_spectrograma(archivo)
        st.image(espectrograma_img, caption="Espectrograma generado", use_container_width=True)
    else:
        espectrograma_img = Image.open(archivo).convert("RGB")
        espectrograma_img = espectrograma_img.resize((IMG_WIDTH, IMG_HEIGHT))
        st.image(espectrograma_img, caption="Espectrograma cargado", use_container_width=True)

    # Clasificar
    if st.button("🎼 Clasificar género"):
        img_array = np.array(espectrograma_img.resize((IMG_WIDTH, IMG_HEIGHT))) / 255.0
        img_array = np.expand_dims(img_array, axis=0)

        predicciones = modelo.predict(img_array)
        indice_predicho = np.argmax(predicciones)
        probabilidad = np.max(predicciones)

        st.markdown(f"🎤 **Género predicho:** `{etiquetas[indice_predicho]}`")
        st.markdown(f"📊 **Confianza:** `{probabilidad:.2%}`")

st.text("CNN y GTZAN - Diego García López - 5º I2ADE")