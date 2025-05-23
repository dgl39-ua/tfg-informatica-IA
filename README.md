# 🎵 Clasificación de Géneros Musicales con CNN y GTZAN

Este proyecto entrena una **red neuronal convolucional (CNN)** para clasificar espectrogramas generados a partir del dataset **GTZAN**, un conjunto popular de datos de audio que contiene 100 ejemplos de 10 géneros musicales distintos.

Forma parte del Trabajo de Fin de Grado en Ingeniería Informática, en el que se exploran técnicas de inteligencia artificial aplicadas a la música, en este caso para la generación automática de música y clasificación de géneros musicales.

Diego García López - Doble Grado en Ingeniería Informática y ADE - Curso 2024/2025 - Universidad de Alicante

## 📂 Estructura del Proyecto

. ├── audios_GTZAN/ &nbsp; &nbsp; &nbsp; *# Carpeta que contiene los audios de GTZAN organizados por género*  
├── images_sist_generativos/ &nbsp; &nbsp; &nbsp; *# Carpeta con los audios generados por los sistemas de IA probados*  
├── images_GTZAN_128/ &nbsp; &nbsp; &nbsp; *# Carpeta que contiene los espectrogramas 128x128 organizados por género*  
├── images_GTZAN_256/ &nbsp; &nbsp; &nbsp; *# Carpeta que contiene los espectrogramas 256x256 organizados por género*  
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
├── images_sist_generativos/ &nbsp; &nbsp; &nbsp; *# Carpeta con los espectrogramas generados a partir de los audios de IA*  
├── prueba_hiperparametros/ &nbsp; &nbsp; &nbsp; *# Carpeta con el fichero y los resultados del ajuste de hiperparámetros*  
├── reentrenamiento_modelo/ &nbsp; &nbsp; &nbsp; *# Carpeta con los ficheros del reentrenamiento con el dataset completo*  
├── resultados_modelos/ &nbsp; &nbsp; &nbsp; *# Carpeta con los resultados y predicciones de los modelos finales*  
├── sin_cross_val/ &nbsp; &nbsp; &nbsp; *# Carpeta con los resultados iniciales sin usar validación cruzada*  
├── modelo_cross_val.keras &nbsp; &nbsp; &nbsp; *# Mejor modelo de la validación cruzada*  
├── modelo_reentrenado.keras &nbsp; &nbsp; &nbsp; *# Mejor modelo del reentrenamiento con todo el dataset*  
├── interfaz.py &nbsp; &nbsp; &nbsp; *# Aplicación web en Streamlit*  
├── README.md &nbsp; &nbsp; &nbsp; *# Descripción del proyecto*  
├── realizar_predicciones.py &nbsp; &nbsp; &nbsp; *# Descripción del proyecto*  

## 📚 Requisitos

- Python 3.8+
- TensorFlow 2.x
- NumPy
- scikit-learn
- Matplotlib
- Seaborn
- PIL (Pillow)
- pandas
- librosa
- audioread
- streamlit

Se pueden instalar los requisitos con:
```bash
pip install tensorflow numpy scikit-learn matplotlib seaborn pillow pandas librosa audioread streamlit
```

## 🧪 Entrenamiento y guardado de un modelo
Para entrenar y probar varias configuraciones de la red neuronal, además de guardar el modelo:
```bash
source ~/tensorflow/bin/activate
```
```bash
python3 cnn_gtzan_cross_val.py
```

Para realizar predicciones sobre los espectrogramas generados a partir de las piezas compuestas por los sistemas generativos evaluados:
```bash
python3 realizar_predicciones.py
```

Para realizar predicciones sobre el género de cualquier audio o imagen se puede acudir a la interfaz (ver el siguiente apartado).

## 🌐 Aplicación web básica con Streamlit

Este proyecto incluye una aplicación desarrollada con **Streamlit** que permite clasificar géneros musicales a partir de espectrogramas de forma interactiva, a través de una interfaz web simple e intuitiva.

**¿Qué se puede hacer?**
- Subir un audio (formato '.wav')
- Subir un espectrograma (imagen '.png' o '.jpg')
- Predecir el género musical utilizando el modelo CNN entrenado
- Visualizar la predicción y la probabilidad de acierto del género

Se puede ejecutar con el siguiente comando:
```bash
streamlit run interfaz.py
```

