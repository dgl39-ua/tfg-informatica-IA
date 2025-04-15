# 🎵 Clasificación de Géneros Musicales con CNN y GTZAN

Este proyecto entrena una **red neuronal convolucional (CNN)** para clasificar espectrogramas generados a partir del dataset **GTZAN**, un conjunto popular de datos de audio que contiene 100 ejemplos de 10 géneros musicales distintos.

Forma parte del Trabajo de Fin de Grado en Ingeniería Informática, en el que se exploran técnicas de inteligencia artificial aplicadas a la música, en este caso para la clasificación automática de géneros musicales.

Diego García López - Doble Grado en Ingeniería Informática y Administración y Dirección de Empresas - Curso 2024/2025

## 📂 Estructura del Proyecto

. ├── images_GTZAN/ # Carpeta que contiene los espectrogramas organizados por género 
│ ├── blues/ 
│ ├── classical/ 
│ ├── country/
│ ├── disco/
│ ├── hiphop/
│ ├── jazz/
│ ├── metal/
│ ├── pop/
│ ├── reggae/
│ ├── rock/
├── modelo_cnn_entrenado.keras # Mejor modelo guardado (se genera después de entrenar) 
├── optimizacion_CNN.png # Gráfica de evolución de precisión en validación (se genera) 
├── cnn_gtzan.py # Script principal del proyecto
├── interfaz.py # Aplicación en Streamlit
├── README.md


## 📚 Requisitos

- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- PIL (Pillow)
- pandas

Se pueden instalar los requisitos con:
pip install tensorflow numpy scikit-learn matplotlib seaborn pillow pandas

## 🧪 Entrenamiento y guardado de un modelo
Para entrenar y probar varias configuraciones de la red neuronal, además de guardar el modelo:
python3 cnn_gtzan.py

## 🌐 Aplicación Web con Streamlit

Este proyecto incluye una aplicación desarrollada con **Streamlit** que permite clasificar géneros musicales a partir de espectrogramas de forma interactiva, a través de una interfaz web simple e intuitiva.

¿Qué se puede hacer?
- Subir un audio (formato '.wav')
- Subir un espectrograma (imagen '.png' o '.jpg')
- Predecir el género musical utilizando el modelo CNN entrenado
- Visualizar la predicción y la probabilidad de acierto del género

Se puede ejecutar con el siguiente comando:
streamlit run interfaz.py

