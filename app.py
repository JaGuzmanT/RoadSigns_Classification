#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoadSigns Classification - Aplicación Web

Este módulo implementa una interfaz web para el modelo de clasificación
de señales viales utilizando Streamlit.

Autor: Tu Nombre
Fecha: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import cv2
import tensorflow as tf
from tensorflow.keras.models import load_model
import time
from PIL import Image
import io
import glob
from pathlib import Path

# Importar módulos propios
from config import MODELS_DIR, RESULTS_DIR
from utils import enhance_image, correct_perspective, detect_traffic_signs
from visualize import plot_attention_maps

# Configuración de la página
st.set_page_config(
    page_title="Clasificador de Señales Viales 🚦",
    page_icon="🚦",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Estilos CSS personalizados
st.markdown("""
<style>
    .main-header {
        font-size: 2.5rem;
        color: #FF4B4B;
        text-align: center;
        margin-bottom: 1rem;
    }
    .sub-header {
        font-size: 1.5rem;
        color: #4B77BE;
        margin-bottom: 1rem;
    }
    .info-box {
        background-color: #F0F2F6;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .success-box {
        background-color: #D5F5E3;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .warning-box {
        background-color: #FDEBD0;
        padding: 1rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
    }
    .prediction-box {
        background-color: #E8F8F5;
        padding: 1.5rem;
        border-radius: 0.5rem;
        margin-bottom: 1rem;
        text-align: center;
    }
    .footer {
        text-align: center;
        margin-top: 2rem;
        color: #777;
        font-size: 0.8rem;
    }
</style>
""", unsafe_allow_html=True)

# Función para cargar el modelo
@st.cache_resource
def load_trained_model(model_path=None):
    """Carga un modelo entrenado.
    
    Args:
        model_path: Ruta al modelo guardado
        
    Returns:
        Modelo cargado
    """
    if model_path is None:
        # Buscar el modelo más reciente
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
        if not model_files:
            st.error(f"No se encontraron modelos en {MODELS_DIR}")
            return None
        
        # Ordenar por fecha de modificación (más reciente primero)
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
        model_path = os.path.join(MODELS_DIR, model_files[0])
    
    try:
        model = load_model(model_path)
        return model
    except Exception as e:
        st.error(f"Error al cargar el modelo: {e}")
        return None

# Función para cargar el mapeo de clases
@st.cache_data
def load_class_mapping(filepath=None):
    """Carga el mapeo de índices a nombres de clases.
    
    Args:
        filepath: Ruta al archivo de mapeo
        
    Returns:
        Diccionario de nombres de clases
    """
    if filepath is None:
        filepath = os.path.join(MODELS_DIR, 'class_names.csv')
    
    if not os.path.isfile(filepath):
        st.warning(f"No se encontró el archivo de mapeo de clases en {filepath}")
        return {}
    
    try:
        df = pd.read_csv(filepath)
        class_names = dict(zip(df['ClassId'], df['SignName']))
        return class_names
    except Exception as e:
        st.error(f"Error al cargar el mapeo de clases: {e}")
        return {}

# Función para preprocesar la imagen
def preprocess_image(img, target_size=(32, 32), enhance=True, correct_persp=True):
    """Preprocesa una imagen para la predicción.
    
    Args:
        img: Imagen a preprocesar (array de numpy o PIL.Image)
        target_size: Tamaño objetivo para redimensionar
        enhance: Si se debe mejorar la imagen
        correct_persp: Si se debe corregir la perspectiva
        
    Returns:
        Imagen preprocesada como array de numpy
    """
    # Convertir a array de numpy si es PIL.Image
    if isinstance(img, Image.Image):
        img = np.array(img)
    
    # Convertir a RGB si es en escala de grises
    if len(img.shape) == 2:
        img = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)
    elif img.shape[2] == 4:  # Si tiene canal alfa (RGBA)
        img = img[:, :, :3]  # Tomar solo RGB
    
    # Guardar imagen original para visualización
    original_img = img.copy()
    
    # Corregir perspectiva si se solicita
    if correct_persp:
        img = correct_perspective(img)
    
    # Mejorar imagen si se solicita
    if enhance:
        img = enhance_image(img)
    
    # Redimensionar
    img_resized = cv2.resize(img, target_size)
    
    # Normalizar a [0,1]
    img_normalized = img_resized.astype('float32') / 255.0
    
    # Expandir dimensiones para el modelo (batch_size=1)
    img_batch = np.expand_dims(img_normalized, axis=0)
    
    return img_batch, original_img, img

# Función para realizar la predicción
def predict_sign(model, img, class_names, confidence_threshold=0.7):
    """Realiza la predicción de la clase de señal vial.
    
    Args:
        model: Modelo entrenado
        img: Imagen preprocesada (con batch_size=1)
        class_names: Diccionario de nombres de clases
        confidence_threshold: Umbral de confianza para aceptar predicciones
        
    Returns:
        Clase predicha, probabilidad y top-k predicciones
    """
    # Realizar predicción
    start_time = time.time()
    predictions = model.predict(img)
    inference_time = (time.time() - start_time) * 1000  # ms
    
    # Obtener top-k predicciones
    top_k_indices = np.argsort(predictions[0])[::-1][:5]  # Top 5
    top_k_probabilities = predictions[0][top_k_indices]
    
    # Obtener clase con mayor probabilidad
    predicted_class_idx = top_k_indices[0]
    probability = top_k_probabilities[0]
    
    # Verificar umbral de confianza
    if probability < confidence_threshold:
        predicted_class = "Desconocida (baja confianza)"
        is_confident = False
    else:
        predicted_class = class_names.get(predicted_class_idx, f"Clase {predicted_class_idx}")
        is_confident = True
    
    # Preparar resultados top-k
    top_k_results = []
    for i, idx in enumerate(top_k_indices):
        class_name = class_names.get(idx, f"Clase {idx}")
        prob = top_k_probabilities[i]
        top_k_results.append((class_name, prob, idx))
    
    return predicted_class, probability, top_k_results, inference_time, is_confident, predicted_class_idx

# Función para generar mapa de atención
def generate_attention_map(model, img, class_idx):
    """Genera un mapa de atención para la imagen usando Grad-CAM.
    
    Args:
        model: Modelo entrenado
        img: Imagen preprocesada (con batch_size=1)
        class_idx: Índice de la clase a visualizar
        
    Returns:
        Figura con mapa de atención
    """
    try:
        # Buscar última capa convolucional
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
        else:
            return None
        
        # Crear modelo para Grad-CAM
        grad_model = tf.keras.models.Model(
            inputs=[model.inputs],
            outputs=[model.get_layer(layer_name).output, model.output]
        )
        
        # Calcular gradientes
        with tf.GradientTape() as tape:
            conv_outputs, predictions = grad_model(img)
            loss = predictions[:, class_idx]
        
        # Extraer características y calcular ponderaciones
        output = conv_outputs[0]
        grads = tape.gradient(loss, conv_outputs)[0]
        
        # Promediar gradientes espacialmente
        weights = tf.reduce_mean(grads, axis=(0, 1))
        
        # Crear mapa de calor ponderado
        cam = np.zeros(output.shape[0:2], dtype=np.float32)
        for i, w in enumerate(weights):
            cam += w * output[:, :, i]
        
        # Aplicar ReLU al mapa de calor
        cam = np.maximum(cam, 0)
        
        # Normalizar mapa de calor
        cam = (cam - cam.min()) / (cam.max() - cam.min() + 1e-7)
        
        # Redimensionar al tamaño de la imagen original
        cam = cv2.resize(cam, (img.shape[2], img.shape[1]))
        
        # Convertir a mapa de calor
        heatmap = cv2.applyColorMap(np.uint8(255 * cam), cv2.COLORMAP_JET)
        
        # Convertir de BGR a RGB para matplotlib
        heatmap = cv2.cvtColor(heatmap, cv2.COLOR_BGR2RGB)
        
        # Superponer mapa de calor en la imagen original
        img_rgb = (img[0] * 255).astype(np.uint8)
        if img_rgb.shape[-1] == 1:  # Imagen en escala de grises
            img_rgb = cv2.cvtColor(img_rgb, cv2.COLOR_GRAY2RGB)
        
        superimposed_img = heatmap * 0.4 + img_rgb
        superimposed_img = superimposed_img / superimposed_img.max() * 255
        superimposed_img = superimposed_img.astype(np.uint8)
        
        # Crear figura
        fig, ax = plt.subplots(1, 3, figsize=(12, 4))
        
        # Imagen original
        ax[0].imshow(img[0])
        ax[0].set_title('Imagen Original')
        ax[0].axis('off')
        
        # Mapa de calor
        ax[1].imshow(heatmap)
        ax[1].set_title('Mapa de Atención')
        ax[1].axis('off')
        
        # Superposición
        ax[2].imshow(superimposed_img)
        ax[2].set_title('Superposición')
        ax[2].axis('off')
        
        plt.tight_layout()
        
        return fig
    except Exception as e:
        st.error(f"Error al generar mapa de atención: {e}")
        return None

# Función para detectar y clasificar múltiples señales en una imagen
def detect_and_classify(model, img, class_names, confidence_threshold=0.5):
    """Detecta y clasifica múltiples señales en una imagen.
    
    Args:
        model: Modelo entrenado
        img: Imagen original
        class_names: Diccionario de nombres de clases
        confidence_threshold: Umbral de confianza para aceptar predicciones
        
    Returns:
        Imagen con detecciones y lista de resultados
    """
    # Detectar posibles señales
    sign_regions = detect_traffic_signs(img)
    
    if not sign_regions:
        return img, []
    
    # Imagen para visualización
    result_img = img.copy()
    
    # Lista para almacenar resultados
    results = []
    
    # Clasificar cada región
    for i, region in enumerate(sign_regions):
        # Preprocesar región
        region_processed, _, _ = preprocess_image(region, enhance=True, correct_persp=False)
        
        # Predecir
        predictions = model.predict(region_processed)
        predicted_class_idx = np.argmax(predictions[0])
        probability = predictions[0][predicted_class_idx]
        
        # Verificar umbral de confianza
        if probability >= confidence_threshold:
            # Obtener nombre de clase
            class_name = class_names.get(predicted_class_idx, f"Clase {predicted_class_idx}")
            
            # Añadir a resultados
            results.append({
                'class_name': class_name,
                'probability': float(probability),
                'region': region
            })
            
            # Dibujar rectángulo y etiqueta en la imagen original
            h, w = region.shape[:2]
            y, x = np.where(np.all(result_img == region[0, 0], axis=-1))
            if len(x) > 0 and len(y) > 0:
                x1, y1 = x[0], y[0]
                cv2.rectangle(result_img, (x1, y1), (x1 + w, y1 + h), (0, 255, 0), 2)
                cv2.putText(result_img, f"{class_name} ({probability:.2f})", 
                           (x1, y1 - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
    
    return result_img, results

# Función principal para la interfaz de usuario
def main():
    """Función principal para la interfaz de usuario."""
    # Título y descripción
    st.markdown('<h1 class="main-header">Clasificador de Señales Viales 🚦</h1>', unsafe_allow_html=True)
    st.markdown("""
    <div class="info-box">
        Esta aplicación utiliza un modelo de redes neuronales convolucionales para clasificar 
        señales viales. Puedes cargar una imagen de una señal y el modelo te dirá qué tipo de señal es.
    </div>
    """, unsafe_allow_html=True)
    
    # Barra lateral
    with st.sidebar:
        st.markdown('<h2 class="sub-header">Configuración ⚙️</h2>', unsafe_allow_html=True)
        
        # Selección de modelo
        model_files = [f for f in os.listdir(MODELS_DIR) if f.endswith('.h5')]
        if model_files:
            selected_model = st.selectbox(
                "Seleccionar modelo",
                options=model_files,
                index=0
            )
            model_path = os.path.join(MODELS_DIR, selected_model)
        else:
            st.error("No se encontraron modelos en el directorio de modelos.")
            model_path = None
        
        # Opciones de preprocesamiento
        st.markdown("### Opciones de preprocesamiento")
        enhance_img = st.checkbox("Mejorar imagen", value=True)
        correct_persp_img = st.checkbox("Corregir perspectiva", value=True)
        
        # Umbral de confianza
        confidence_threshold = st.slider(
            "Umbral de confianza",
            min_value=0.0,
            max_value=1.0,
            value=0.7,
            step=0.05,
            help="Umbral mínimo de confianza para aceptar una predicción"
        )
        
        # Modo de detección múltiple
        detection_mode = st.checkbox(
            "Modo de detección múltiple", 
            value=False,
            help="Detectar y clasificar múltiples señales en una imagen"
        )
        
        # Mostrar mapa de atención
        show_attention = st.checkbox(
            "Mostrar mapa de atención", 
            value=True,
            help="Visualizar en qué partes de la imagen se enfoca el modelo"
        )
        
        # Información del modelo
        st.markdown("### Información del modelo")
        if model_path and os.path.isfile(model_path):
            model_size = os.path.getsize(model_path) / (1024 * 1024)  # MB
            model_date = time.ctime(os.path.getmtime(model_path))
            st.info(f"Tamaño: {model_size:.2f} MB\nFecha: {model_date}")
    
    # Cargar modelo y mapeo de clases
    model = load_trained_model(model_path if 'model_path' in locals() else None)
    class_names = load_class_mapping()
    
    if model is None:
        st.error("No se pudo cargar el modelo. Por favor, verifica que exista un modelo válido.")
        return
    
    if not class_names:
        st.warning("No se pudo cargar el mapeo de clases. Las predicciones mostrarán índices en lugar de nombres.")
    
    # Pestañas para diferentes modos de entrada
    tab1, tab2, tab3 = st.tabs(["📷 Subir imagen", "📁 Procesar directorio", "📊 Estadísticas"])
    
    # Pestaña 1: Subir imagen
    with tab1:
        st.markdown('<h2 class="sub-header">Subir imagen de señal vial</h2>', unsafe_allow_html=True)
        
        # Subir imagen
        uploaded_file = st.file_uploader("Selecciona una imagen", type=["jpg", "jpeg", "png", "bmp"])
        
        # Capturar imagen desde cámara
        use_camera = st.checkbox("Usar cámara", value=False)
        if use_camera:
            camera_img = st.camera_input("Tomar foto de señal vial")
            if camera_img is not None:
                uploaded_file = camera_img
        
        if uploaded_file is not None:
            # Leer imagen
            image = Image.open(uploaded_file)
            img_array = np.array(image)
            
            # Mostrar imagen original
            st.image(image, caption="Imagen original", use_column_width=True)
            
            # Botón para procesar
            if st.button("Clasificar señal"):
                with st.spinner("Procesando imagen..."):
                    if detection_mode:
                        # Modo de detección múltiple
                        result_img, detections = detect_and_classify(
                            model, img_array, class_names, confidence_threshold
                        )
                        
                        if detections:
                            # Mostrar imagen con detecciones
                            st.image(result_img, caption="Detecciones", use_column_width=True)
                            
                            # Mostrar resultados
                            st.markdown('<h3 class="sub-header">Señales detectadas</h3>', unsafe_allow_html=True)
                            
                            # Crear columnas para cada detección
                            cols = st.columns(min(len(detections), 4))
                            
                            for i, detection in enumerate(detections):
                                with cols[i % 4]:
                                    st.image(detection['region'], caption=detection['class_name'])
                                    st.markdown(f"**Confianza:** {detection['probability']:.2%}")
                        else:
                            st.warning("No se detectaron señales viales en la imagen.")
                    else:
                        # Modo de clasificación única
                        # Preprocesar imagen
                        img_processed, original_img, enhanced_img = preprocess_image(
                            img_array, enhance=enhance_img, correct_persp=correct_persp_img
                        )
                        
                        # Realizar predicción
                        predicted_class, probability, top_k_results, inference_time, is_confident, predicted_class_idx = predict_sign(
                            model, img_processed, class_names, confidence_threshold
                        )
                        
                        # Mostrar resultado principal
                        if is_confident:
                            st.markdown(f"""
                            <div class="success-box prediction-box">
                                <h2>Señal detectada: {predicted_class}</h2>
                                <h3>Confianza: {probability:.2%}</h3>
                                <p>Tiempo de inferencia: {inference_time:.2f} ms</p>
                            </div>
                            """, unsafe_allow_html=True)
                        else:
                            st.markdown(f"""
                            <div class="warning-box prediction-box">
                                <h2>Resultado: {predicted_class}</h2>
                                <h3>Confianza: {probability:.2%}</h3>
                                <p>La confianza está por debajo del umbral ({confidence_threshold:.2%})</p>
                            </div>
                            """, unsafe_allow_html=True)
                        
                        # Mostrar top-k predicciones
                        st.markdown('<h3 class="sub-header">Top 5 predicciones</h3>', unsafe_allow_html=True)
                        
                        # Crear gráfico de barras para top-k
                        fig, ax = plt.subplots(figsize=(10, 5))
                        classes = [r[0] for r in top_k_results]
                        probs = [r[1] for r in top_k_results]
                        
                        bars = ax.barh(classes, probs, color='skyblue')
                        ax.set_xlim(0, 1)
                        ax.set_xlabel('Probabilidad')
                        ax.set_title('Top 5 predicciones')
                        
                        # Añadir valores a las barras
                        for bar in bars:
                            width = bar.get_width()
                            ax.text(width + 0.01, bar.get_y() + bar.get_height()/2, f'{width:.4f}', 
                                   va='center')
                        
                        st.pyplot(fig)
                        
                        # Mostrar mapa de atención si se solicita
                        if show_attention and is_confident:
                            st.markdown('<h3 class="sub-header">Mapa de atención</h3>', unsafe_allow_html=True)
                            attention_fig = generate_attention_map(model, img_processed, predicted_class_idx)
                            if attention_fig:
                                st.pyplot(attention_fig)
                            else:
                                st.warning("No se pudo generar el mapa de atención.")
                        
                        # Mostrar imágenes de preprocesamiento
                        if enhance_img or correct_persp_img:
                            st.markdown('<h3 class="sub-header">Preprocesamiento</h3>', unsafe_allow_html=True)
                            preproc_cols = st.columns(3)
                            with preproc_cols[0]:
                                st.image(original_img, caption="Original")
                            with preproc_cols[1]:
                                st.image(enhanced_img, caption="Preprocesada")
                            with preproc_cols[2]:
                                st.image(img_processed[0], caption="Entrada al modelo")
    
    # Pestaña 2: Procesar directorio
    with tab2:
        st.markdown('<h2 class="sub-header">Procesar directorio de imágenes</h2>', unsafe_allow_html=True)
        
        # Entrada de directorio
        dir_path = st.text_input("Ruta del directorio con imágenes")
        
        if dir_path and os.path.isdir(dir_path):
            # Listar archivos de imagen
            image_extensions = [".jpg", ".jpeg", ".png", ".bmp"]
            image_files = []
            for ext in image_extensions:
                image_files.extend(glob.glob(os.path.join(dir_path, f"*{ext}")))
                image_files.extend(glob.glob(os.path.join(dir_path, f"*{ext.upper()}")))
            
            st.info(f"Se encontraron {len(image_files)} imágenes en el directorio.")
            
            # Botón para procesar directorio
            if st.button("Procesar directorio"):
                if not image_files:
                    st.warning("No se encontraron imágenes en el directorio especificado.")
                else:
                    # Crear directorio para resultados
                    results_dir = os.path.join(dir_path, "resultados_clasificacion")
                    os.makedirs(results_dir, exist_ok=True)
                    
                    # Barra de progreso
                    progress_bar = st.progress(0)
                    status_text = st.empty()
                    
                    # Resultados
                    results = []
                    
                    # Procesar cada imagen
                    for i, img_path in enumerate(image_files):
                        try:
                            # Actualizar progreso
                            progress = (i + 1) / len(image_files)
                            progress_bar.progress(progress)
                            status_text.text(f"Procesando {i+1}/{len(image_files)}: {os.path.basename(img_path)}")
                            
                            # Leer imagen
                            img = cv2.imread(img_path)
                            img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                            
                            if detection_mode:
                                # Modo de detección múltiple
                                result_img, detections = detect_and_classify(
                                    model, img, class_names, confidence_threshold
                                )
                                
                                # Guardar imagen con detecciones
                                output_path = os.path.join(results_dir, f"detected_{os.path.basename(img_path)}")
                                cv2.imwrite(output_path, cv2.cvtColor(result_img, cv2.COLOR_RGB2BGR))
                                
                                # Añadir a resultados
                                for detection in detections:
                                    results.append({
                                        'image': os.path.basename(img_path),
                                        'class': detection['class_name'],
                                        'confidence': detection['probability']
                                    })
                            else:
                                # Modo de clasificación única
                                # Preprocesar imagen
                                img_processed, _, _ = preprocess_image(
                                    img, enhance=enhance_img, correct_persp=correct_persp_img
                                )
                                
                                # Realizar predicción
                                predicted_class, probability, _, _, is_confident, _ = predict_sign(
                                    model, img_processed, class_names, confidence_threshold
                                )
                                
                                # Añadir a resultados
                                results.append({
                                    'image': os.path.basename(img_path),
                                    'class': predicted_class,
                                    'confidence': float(probability),
                                    'is_confident': is_confident
                                })
                        except Exception as e:
                            st.error(f"Error al procesar {img_path}: {e}")
                    
                    # Completar progreso
                    progress_bar.progress(1.0)
                    status_text.text("Procesamiento completado")
                    
                    # Crear DataFrame con resultados
                    results_df = pd.DataFrame(results)
                    
                    # Guardar resultados
                    csv_path = os.path.join(results_dir, "resultados_clasificacion.csv")
                    results_df.to_csv(csv_path, index=False)
                    
                    # Mostrar resultados
                    st.markdown('<h3 class="sub-header">Resultados</h3>', unsafe_allow_html=True)
                    st.dataframe(results_df)
                    
                    # Estadísticas
                    if not detection_mode and len(results_df) > 0:
                        st.markdown('<h3 class="sub-header">Estadísticas</h3>', unsafe_allow_html=True)
                        
                        # Contar clases
                        class_counts = results_df['class'].value_counts()
                        
                        # Gráfico de distribución de clases
                        fig, ax = plt.subplots(figsize=(10, 6))
                        class_counts.plot(kind='bar', ax=ax)
                        ax.set_title('Distribución de clases detectadas')
                        ax.set_ylabel('Cantidad')
                        ax.set_xlabel('Clase')
                        plt.xticks(rotation=45, ha='right')
                        plt.tight_layout()
                        st.pyplot(fig)
                        
                        # Histograma de confianza
                        fig, ax = plt.subplots(figsize=(10, 6))
                        results_df['confidence'].hist(bins=20, ax=ax)
                        ax.set_title('Distribución de confianza')
                        ax.set_xlabel('Confianza')
                        ax.set_ylabel('Frecuencia')
                        plt.tight_layout()
                        st.pyplot(fig)
                    
                    # Enlace para descargar resultados
                    with open(csv_path, 'rb') as f:
                        csv_data = f.read()
                    st.download_button(
                        label="Descargar resultados CSV",
                        data=csv_data,
                        file_name="resultados_clasificacion.csv",
                        mime="text/csv"
                    )
                    
                    st.success(f"Procesamiento completado. Resultados guardados en {results_dir}")
        else:
            if dir_path:
                st.error(f"El directorio {dir_path} no existe.")
    
    # Pestaña 3: Estadísticas
    with tab3:
        st.markdown('<h2 class="sub-header">Estadísticas del modelo</h2>', unsafe_allow_html=True)
        
        # Verificar si existen resultados de evaluación
        eval_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
        if os.path.isfile(eval_path):
            try:
                with open(eval_path, 'r') as f:
                    eval_results = json.load(f)
                
                # Mostrar métricas principales
                metrics_cols = st.columns(4)
                with metrics_cols[0]:
                    st.metric("Precisión", f"{eval_results['accuracy']:.4f}")
                with metrics_cols[1]:
                    st.metric("Recall", f"{eval_results['recall']:.4f}")
                with metrics_cols[2]:
                    st.metric("F1-Score", f"{eval_results['f1_score']:.4f}")
                with metrics_cols[3]:
                    st.metric("Tiempo de inferencia", f"{eval_results['inference_time_ms']:.2f} ms")
                
                # Mostrar matriz de confusión
                st.markdown('<h3 class="sub-header">Matriz de confusión</h3>', unsafe_allow_html=True)
                cm_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
                if os.path.isfile(cm_path):
                    st.image(cm_path)
                
                # Mostrar métricas por clase
                st.markdown('<h3 class="sub-header">Métricas por clase</h3>', unsafe_allow_html=True)
                metrics_path = os.path.join(RESULTS_DIR, 'metrics_by_class.png')
                if os.path.isfile(metrics_path):
                    st.image(metrics_path)
                
                # Mostrar ejemplos de errores
                st.markdown('<h3 class="sub-header">Ejemplos de errores</h3>', unsafe_allow_html=True)
                errors_path = os.path.join(RESULTS_DIR, 'classification_errors.png')
                if os.path.isfile(errors_path):
                    st.image(errors_path)
                
                # Mostrar resultados de robustez
                robustness_dir = os.path.join(RESULTS_DIR, 'robustness')
                if os.path.isdir(robustness_dir):
                    st.markdown('<h3 class="sub-header">Evaluación de robustez</h3>', unsafe_allow_html=True)
                    
                    # Mostrar gráficos de robustez
                    for metric in ['accuracy', 'f1_score']:
                        metric_path = os.path.join(robustness_dir, f"robustness_{metric}.png")
                        if os.path.isfile(metric_path):
                            st.image(metric_path, caption=f"Robustez - {metric}")
            except Exception as e:
                st.error(f"Error al cargar resultados de evaluación: {e}")
        else:
            st.info("No se encontraron resultados de evaluación. Ejecuta el script evaluate.py para generar estadísticas.")
            
            # Mostrar información del modelo
            st.markdown('<h3 class="sub-header">Información del modelo</h3>', unsafe_allow_html=True)
            
            # Mostrar resumen del modelo
            model_summary = []
            model.summary(print_fn=lambda x: model_summary.append(x))
            st.code('\n'.join(model_summary))
    
    # Pie de página
    st.markdown("""
    <div class="footer">
        <p>Desarrollado con ❤️ para el proyecto de Clasificación de Señales Viales</p>
        <p>© 2024 - Todos los derechos reservados</p>
    </div>
    """, unsafe_allow_html=True)

# Punto de entrada
if __name__ == "__main__":
    main()