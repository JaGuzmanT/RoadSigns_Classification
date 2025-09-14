#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoadSigns Classification - Predicción con modelo entrenado

Este script permite utilizar un modelo entrenado para clasificar
imágenes de señales de tráfico. Puede procesar imágenes individuales
o directorios completos de imágenes.

Autor: Tu Nombre
Fecha: 2024
"""

import os
import argparse
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import cv2
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import img_to_array
from tqdm import tqdm

# Diccionario de clases (ejemplo basado en GTSRB)
# En un caso real, esto se cargaría desde un archivo
CLASS_NAMES = {
    0: 'Límite de velocidad (20km/h)',
    1: 'Límite de velocidad (30km/h)',
    2: 'Límite de velocidad (50km/h)',
    3: 'Límite de velocidad (60km/h)',
    4: 'Límite de velocidad (70km/h)',
    5: 'Límite de velocidad (80km/h)',
    6: 'Fin de límite de velocidad (80km/h)',
    7: 'Límite de velocidad (100km/h)',
    8: 'Límite de velocidad (120km/h)',
    9: 'Prohibido adelantar',
    10: 'Prohibido adelantar para vehículos pesados',
    11: 'Intersección con prioridad',
    12: 'Carretera con prioridad',
    13: 'Ceda el paso',
    14: 'Stop',
    15: 'Prohibido el paso',
    16: 'Prohibido vehículos pesados',
    17: 'Prohibido el paso',
    18: 'Peligro',
    19: 'Curva peligrosa a la izquierda',
    20: 'Curva peligrosa a la derecha',
    21: 'Curva doble',
    22: 'Firme irregular',
    23: 'Firme deslizante',
    24: 'Estrechamiento de calzada',
    25: 'Obras',
    26: 'Semáforos',
    27: 'Peatones',
    28: 'Niños',
    29: 'Ciclistas',
    30: 'Nieve',
    31: 'Animales',
    32: 'Fin de prohibiciones',
    33: 'Giro obligatorio a la derecha',
    34: 'Giro obligatorio a la izquierda',
    35: 'Dirección obligatoria recto',
    36: 'Dirección obligatoria recto o derecha',
    37: 'Dirección obligatoria recto o izquierda',
    38: 'Paso obligatorio por la derecha',
    39: 'Paso obligatorio por la izquierda',
    40: 'Intersección de sentido giratorio obligatorio',
    41: 'Fin de prohibición de adelantamiento',
    42: 'Fin de prohibición de adelantamiento para vehículos pesados'
}

# Función para preprocesar una imagen
def preprocess_image(image_path, target_size=(32, 32)):
    """Preprocesa una imagen para la predicción.
    
    Args:
        image_path: Ruta a la imagen
        target_size: Tamaño objetivo para redimensionar
        
    Returns:
        Imagen preprocesada lista para el modelo
    """
    # Cargar imagen
    img = cv2.imread(image_path)
    if img is None:
        raise ValueError(f"No se pudo cargar la imagen: {image_path}")
    
    # Convertir de BGR a RGB (OpenCV carga en BGR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Redimensionar
    img = cv2.resize(img, target_size)
    
    # Normalizar
    img = img.astype('float32') / 255.0
    
    # Expandir dimensiones para el batch
    img = np.expand_dims(img, axis=0)
    
    return img

# Función para predecir una imagen
def predict_image(model, image_path, target_size=(32, 32), show=True):
    """Predice la clase de una imagen de señal de tráfico.
    
    Args:
        model: Modelo cargado
        image_path: Ruta a la imagen
        target_size: Tamaño objetivo para redimensionar
        show: Si se debe mostrar la imagen con la predicción
        
    Returns:
        Clase predicha y confianza
    """
    # Preprocesar imagen
    try:
        img = preprocess_image(image_path, target_size)
    except Exception as e:
        print(f"Error al procesar la imagen {image_path}: {e}")
        return None, 0
    
    # Realizar predicción
    predictions = model.predict(img, verbose=0)[0]
    predicted_class = np.argmax(predictions)
    confidence = predictions[predicted_class]
    
    # Obtener nombre de la clase
    class_name = CLASS_NAMES.get(predicted_class, f"Clase {predicted_class}")
    
    # Mostrar resultados
    print(f"Predicción para {os.path.basename(image_path)}:")
    print(f"  Clase: {class_name}")
    print(f"  Confianza: {confidence:.4f}")
    
    # Mostrar imagen con predicción
    if show:
        # Cargar imagen original para visualización
        display_img = cv2.imread(image_path)
        display_img = cv2.cvtColor(display_img, cv2.COLOR_BGR2RGB)
        
        plt.figure(figsize=(8, 6))
        plt.imshow(display_img)
        plt.title(f"Predicción: {class_name}\nConfianza: {confidence:.4f}")
        plt.axis('off')
        plt.tight_layout()
        
        # Crear directorio de resultados si no existe
        os.makedirs('results', exist_ok=True)
        output_path = f"results/prediction_{os.path.basename(image_path)}"
        plt.savefig(output_path)
        print(f"  Imagen con predicción guardada en: {output_path}")
        
        if show:
            plt.show()
        plt.close()
    
    return predicted_class, confidence

# Función para procesar un directorio de imágenes
def process_directory(model, directory, target_size=(32, 32)):
    """Procesa todas las imágenes en un directorio.
    
    Args:
        model: Modelo cargado
        directory: Directorio con imágenes
        target_size: Tamaño objetivo para redimensionar
    """
    # Verificar que el directorio existe
    if not os.path.isdir(directory):
        print(f"Error: El directorio {directory} no existe")
        return
    
    # Obtener todas las imágenes
    image_extensions = ('.jpg', '.jpeg', '.png', '.bmp')
    image_files = [os.path.join(directory, f) for f in os.listdir(directory) 
                  if f.lower().endswith(image_extensions)]
    
    if not image_files:
        print(f"No se encontraron imágenes en {directory}")
        return
    
    print(f"Procesando {len(image_files)} imágenes de {directory}...")
    
    # Crear un DataFrame para almacenar resultados
    results = []
    
    # Procesar cada imagen
    for img_path in tqdm(image_files):
        predicted_class, confidence = predict_image(model, img_path, target_size, show=False)
        if predicted_class is not None:
            results.append({
                'imagen': os.path.basename(img_path),
                'clase': predicted_class,
                'nombre_clase': CLASS_NAMES.get(predicted_class, f"Clase {predicted_class}"),
                'confianza': confidence
            })
    
    # Mostrar resumen
    print("\nResumen de predicciones:")
    for i, result in enumerate(results, 1):
        print(f"{i}. {result['imagen']}: {result['nombre_clase']} (confianza: {result['confianza']:.4f})")
    
    # Crear visualización de resultados
    create_results_visualization(image_files, results)

# Función para crear visualización de resultados
def create_results_visualization(image_files, results):
    """Crea una visualización de los resultados de predicción.
    
    Args:
        image_files: Lista de rutas a imágenes
        results: Lista de resultados de predicción
    """
    # Limitar a 20 imágenes para la visualización
    max_images = min(20, len(image_files))
    image_files = image_files[:max_images]
    results = results[:max_images]
    
    # Calcular filas y columnas para el grid
    n_cols = min(5, max_images)
    n_rows = (max_images + n_cols - 1) // n_cols
    
    # Crear figura
    plt.figure(figsize=(15, 3 * n_rows))
    
    # Mostrar cada imagen con su predicción
    for i, (img_path, result) in enumerate(zip(image_files, results)):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Cargar y mostrar imagen
        img = cv2.imread(img_path)
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        plt.imshow(img)
        
        # Añadir título con predicción
        plt.title(f"{result['nombre_clase']}\n{result['confianza']:.2f}", fontsize=10)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Guardar visualización
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/batch_predictions.png')
    print("\nVisualización de resultados guardada en 'results/batch_predictions.png'")
    plt.close()

# Función para generar mapa de atención
def generate_attention_map(model, image_path, target_size=(32, 32)):
    """Genera un mapa de atención para visualizar qué partes de la imagen
    son más importantes para la predicción del modelo.
    
    Args:
        model: Modelo cargado
        image_path: Ruta a la imagen
        target_size: Tamaño objetivo para redimensionar
    """
    # Esta es una implementación simplificada de Grad-CAM
    # En un caso real, se utilizaría una biblioteca como tf-keras-vis
    print("\n🔍 Generando mapa de atención (esta función es un placeholder)...")
    print("  Para implementar Grad-CAM completo, se recomienda usar tf-keras-vis u otra biblioteca especializada.")
    
    # Cargar imagen original para visualización
    original_img = cv2.imread(image_path)
    original_img = cv2.cvtColor(original_img, cv2.COLOR_BGR2RGB)
    
    # Crear un mapa de atención simulado (esto es solo para demostración)
    attention_map = np.random.rand(target_size[0], target_size[1])
    attention_map = cv2.resize(attention_map, (original_img.shape[1], original_img.shape[0]))
    
    # Visualizar
    plt.figure(figsize=(12, 4))
    
    plt.subplot(1, 3, 1)
    plt.imshow(original_img)
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1, 3, 2)
    plt.imshow(attention_map, cmap='jet')
    plt.title('Mapa de Atención')
    plt.axis('off')
    
    plt.subplot(1, 3, 3)
    plt.imshow(original_img)
    plt.imshow(attention_map, cmap='jet', alpha=0.5)
    plt.title('Superposición')
    plt.axis('off')
    
    plt.tight_layout()
    
    # Guardar visualización
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/attention_map.png')
    print("  Mapa de atención guardado en 'results/attention_map.png'")
    plt.close()

# Función principal
def main():
    """Función principal que ejecuta la predicción."""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Predicción de señales de tráfico con modelo entrenado')
    parser.add_argument('--model', type=str, default='Models/model_best.h5', help='Ruta al modelo entrenado')
    parser.add_argument('--image', type=str, help='Ruta a la imagen para predecir')
    parser.add_argument('--dir', type=str, help='Directorio con imágenes para predecir')
    parser.add_argument('--size', type=int, default=32, help='Tamaño de imagen para el modelo')
    parser.add_argument('--attention', action='store_true', help='Generar mapa de atención')
    args = parser.parse_args()
    
    # Verificar que se proporcionó una imagen o directorio
    if not args.image and not args.dir:
        parser.error("Debe proporcionar --image o --dir")
    
    # Verificar que el modelo existe
    if not os.path.isfile(args.model):
        print(f"Error: El modelo {args.model} no existe")
        return
    
    # Cargar modelo
    print(f"\n🔄 Cargando modelo desde {args.model}...")
    try:
        model = load_model(args.model)
        print("✅ Modelo cargado correctamente")
    except Exception as e:
        print(f"❌ Error al cargar el modelo: {e}")
        return
    
    # Procesar imagen individual
    if args.image:
        if not os.path.isfile(args.image):
            print(f"Error: La imagen {args.image} no existe")
            return
        
        print(f"\n🖼️ Procesando imagen: {args.image}")
        predict_image(model, args.image, target_size=(args.size, args.size))
        
        # Generar mapa de atención si se solicita
        if args.attention:
            generate_attention_map(model, args.image, target_size=(args.size, args.size))
    
    # Procesar directorio
    if args.dir:
        process_directory(model, args.dir, target_size=(args.size, args.size))
    
    print("\n🎉 Proceso de predicción completado")

# Punto de entrada
if __name__ == "__main__":
    print("\n🚀 RoadSigns Classification - Predicción con modelo entrenado")
    print("=" * 70)
    
    main()
    
    print("=" * 70)