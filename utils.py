#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoadSigns Classification - Utilidades

Este módulo contiene funciones de utilidad para el proyecto de clasificación
de señales viales, incluyendo preprocesamiento de imágenes, carga de datos,
y otras funciones auxiliares.

Autor: Tu Nombre
Fecha: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import cv2
from sklearn.model_selection import train_test_split
from tensorflow.keras.utils import to_categorical
from tqdm import tqdm

# Función para cargar imágenes desde directorios
def load_images_from_directory(directory, img_size=(32, 32), grayscale=False):
    """Carga imágenes desde un directorio organizado por clases.
    
    Args:
        directory: Directorio raíz que contiene subdirectorios para cada clase
        img_size: Tamaño al que se redimensionarán las imágenes
        grayscale: Si se deben cargar las imágenes en escala de grises
        
    Returns:
        X: Array de imágenes
        y: Array de etiquetas
        class_names: Diccionario de nombres de clases
    """
    X = []
    y = []
    class_names = {}
    
    # Verificar que el directorio existe
    if not os.path.isdir(directory):
        raise ValueError(f"El directorio {directory} no existe")
    
    # Obtener subdirectorios (clases)
    subdirs = [d for d in os.listdir(directory) if os.path.isdir(os.path.join(directory, d))]
    
    if not subdirs:
        raise ValueError(f"No se encontraron subdirectorios en {directory}")
    
    print(f"Cargando imágenes desde {len(subdirs)} clases...")
    
    # Procesar cada subdirectorio (clase)
    for class_idx, subdir in enumerate(sorted(subdirs)):
        class_dir = os.path.join(directory, subdir)
        class_names[class_idx] = subdir
        
        # Obtener archivos de imagen
        image_extensions = ('.jpg', '.jpeg', '.png', '.bmp', '.ppm')
        image_files = [f for f in os.listdir(class_dir) if f.lower().endswith(image_extensions)]
        
        if not image_files:
            print(f"Advertencia: No se encontraron imágenes en {class_dir}")
            continue
        
        print(f"  Clase {class_idx} ({subdir}): {len(image_files)} imágenes")
        
        # Cargar cada imagen
        for img_file in tqdm(image_files, desc=f"Clase {subdir}", leave=False):
            img_path = os.path.join(class_dir, img_file)
            
            try:
                # Cargar imagen
                if grayscale:
                    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
                    # Añadir canal para mantener consistencia dimensional
                    img = np.expand_dims(img, axis=-1)
                else:
                    img = cv2.imread(img_path)
                    # Convertir de BGR a RGB
                    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                
                # Redimensionar
                img = cv2.resize(img, img_size)
                
                # Añadir a los arrays
                X.append(img)
                y.append(class_idx)
                
            except Exception as e:
                print(f"Error al cargar {img_path}: {e}")
    
    # Convertir a arrays numpy
    X = np.array(X, dtype='float32') / 255.0  # Normalizar a [0,1]
    y = np.array(y)
    
    print(f"Carga completada: {len(X)} imágenes, {len(class_names)} clases")
    
    return X, y, class_names

# Función para dividir datos en conjuntos de entrenamiento, validación y prueba
def split_data(X, y, test_size=0.2, val_size=0.1, random_state=42):
    """Divide los datos en conjuntos de entrenamiento, validación y prueba.
    
    Args:
        X: Array de imágenes
        y: Array de etiquetas
        test_size: Proporción de datos para prueba
        val_size: Proporción de datos para validación
        random_state: Semilla para reproducibilidad
        
    Returns:
        Conjuntos de datos divididos
    """
    # Primera división: separar datos de prueba
    X_train_val, X_test, y_train_val, y_test = train_test_split(
        X, y, test_size=test_size, random_state=random_state, stratify=y
    )
    
    # Segunda división: separar datos de entrenamiento y validación
    # val_size relativo al tamaño de train_val
    relative_val_size = val_size / (1 - test_size)
    
    X_train, X_val, y_train, y_val = train_test_split(
        X_train_val, y_train_val, 
        test_size=relative_val_size, 
        random_state=random_state,
        stratify=y_train_val
    )
    
    print(f"División de datos:")
    print(f"  Entrenamiento: {X_train.shape[0]} imágenes")
    print(f"  Validación: {X_val.shape[0]} imágenes")
    print(f"  Prueba: {X_test.shape[0]} imágenes")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Función para aplicar aumento de datos a una imagen
def augment_image(img, rotation_range=15, shift_range=0.1, shear_range=0.1, zoom_range=0.1):
    """Aplica transformaciones aleatorias a una imagen para aumento de datos.
    
    Args:
        img: Imagen a transformar
        rotation_range: Rango de rotación en grados
        shift_range: Rango de desplazamiento
        shear_range: Rango de cizallamiento
        zoom_range: Rango de zoom
        
    Returns:
        Imagen transformada
    """
    height, width = img.shape[:2]
    
    # Rotación
    if rotation_range > 0:
        angle = np.random.uniform(-rotation_range, rotation_range)
        M = cv2.getRotationMatrix2D((width/2, height/2), angle, 1)
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    
    # Desplazamiento
    if shift_range > 0:
        tx = np.random.uniform(-shift_range, shift_range) * width
        ty = np.random.uniform(-shift_range, shift_range) * height
        M = np.float32([[1, 0, tx], [0, 1, ty]])
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    
    # Cizallamiento
    if shear_range > 0:
        shear = np.random.uniform(-shear_range, shear_range)
        M = np.float32([[1, shear, 0], [0, 1, 0]])
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    
    # Zoom
    if zoom_range > 0:
        zoom = np.random.uniform(1 - zoom_range, 1 + zoom_range)
        M = cv2.getRotationMatrix2D((width/2, height/2), 0, zoom)
        img = cv2.warpAffine(img, M, (width, height), borderMode=cv2.BORDER_REPLICATE)
    
    return img

# Función para visualizar ejemplos de imágenes
def visualize_samples(X, y, class_names, n_samples=10, random_seed=None):
    """Visualiza ejemplos de imágenes del conjunto de datos.
    
    Args:
        X: Array de imágenes
        y: Array de etiquetas
        class_names: Diccionario de nombres de clases
        n_samples: Número de muestras a visualizar
        random_seed: Semilla para selección aleatoria
    """
    if random_seed is not None:
        np.random.seed(random_seed)
    
    # Seleccionar índices aleatorios
    indices = np.random.choice(len(X), min(n_samples, len(X)), replace=False)
    
    # Calcular filas y columnas para el grid
    n_cols = min(5, n_samples)
    n_rows = (n_samples + n_cols - 1) // n_cols
    
    # Crear figura
    plt.figure(figsize=(15, 3 * n_rows))
    
    # Mostrar cada imagen
    for i, idx in enumerate(indices):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Mostrar imagen
        if X[idx].shape[-1] == 1:  # Imagen en escala de grises
            plt.imshow(X[idx].squeeze(), cmap='gray')
        else:  # Imagen RGB
            plt.imshow(X[idx])
        
        # Añadir título con clase
        class_idx = y[idx]
        class_name = class_names.get(class_idx, f"Clase {class_idx}")
        plt.title(class_name)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Guardar visualización
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/data_samples.png')
    print("Visualización de muestras guardada en 'results/data_samples.png'")
    
    return plt.gcf()

# Función para visualizar ejemplos de aumento de datos
def visualize_augmentation(img, n_samples=5):
    """Visualiza ejemplos de aumento de datos aplicados a una imagen.
    
    Args:
        img: Imagen original
        n_samples: Número de ejemplos a generar
    """
    plt.figure(figsize=(15, 3))
    
    # Mostrar imagen original
    plt.subplot(1, n_samples + 1, 1)
    if img.shape[-1] == 1:  # Imagen en escala de grises
        plt.imshow(img.squeeze(), cmap='gray')
    else:  # Imagen RGB
        plt.imshow(img)
    plt.title('Original')
    plt.axis('off')
    
    # Mostrar versiones aumentadas
    for i in range(n_samples):
        plt.subplot(1, n_samples + 1, i + 2)
        augmented = augment_image(img.copy())
        
        if augmented.shape[-1] == 1:  # Imagen en escala de grises
            plt.imshow(augmented.squeeze(), cmap='gray')
        else:  # Imagen RGB
            plt.imshow(augmented)
            
        plt.title(f'Aumentada {i+1}')
        plt.axis('off')
    
    plt.tight_layout()
    
    # Guardar visualización
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/augmentation_examples.png')
    print("Visualización de aumento de datos guardada en 'results/augmentation_examples.png'")
    
    return plt.gcf()

# Función para guardar y cargar el mapeo de clases
def save_class_mapping(class_names, filepath='Models/class_names.csv'):
    """Guarda el mapeo de índices a nombres de clases.
    
    Args:
        class_names: Diccionario de nombres de clases
        filepath: Ruta donde guardar el mapeo
    """
    # Crear directorio si no existe
    os.makedirs(os.path.dirname(filepath), exist_ok=True)
    
    # Crear DataFrame y guardar
    df = pd.DataFrame(list(class_names.items()), columns=['ClassId', 'SignName'])
    df.to_csv(filepath, index=False)
    print(f"Mapeo de clases guardado en {filepath}")

def load_class_mapping(filepath='Models/class_names.csv'):
    """Carga el mapeo de índices a nombres de clases.
    
    Args:
        filepath: Ruta desde donde cargar el mapeo
        
    Returns:
        Diccionario de nombres de clases
    """
    if not os.path.isfile(filepath):
        print(f"Advertencia: No se encontró el archivo de mapeo de clases en {filepath}")
        return {}
    
    df = pd.read_csv(filepath)
    class_names = dict(zip(df['ClassId'], df['SignName']))
    print(f"Mapeo de clases cargado desde {filepath}")
    return class_names

# Función para aplicar mejoras de contraste a imágenes
def enhance_image(img):
    """Aplica mejoras de contraste y nitidez a una imagen.
    
    Args:
        img: Imagen a mejorar
        
    Returns:
        Imagen mejorada
    """
    # Convertir a escala de grises si es RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Ecualización de histograma
    equalized = cv2.equalizeHist(gray)
    
    # Filtro de nitidez
    kernel = np.array([[-1, -1, -1],
                       [-1,  9, -1],
                       [-1, -1, -1]])
    sharpened = cv2.filter2D(equalized, -1, kernel)
    
    # Si la imagen original era RGB, convertir de nuevo
    if len(img.shape) == 3 and img.shape[2] == 3:
        # Crear imagen RGB con el canal mejorado
        enhanced = img.copy()
        # Aplicar la mejora solo al canal de luminancia
        hsv = cv2.cvtColor(enhanced, cv2.COLOR_RGB2HSV)
        hsv[:,:,2] = sharpened
        enhanced = cv2.cvtColor(hsv, cv2.COLOR_HSV2RGB)
        return enhanced
    else:
        return sharpened

# Función para corregir perspectiva de una imagen
def correct_perspective(img, corners=None):
    """Corrige la perspectiva de una imagen de señal de tráfico.
    
    Args:
        img: Imagen a corregir
        corners: Esquinas de la señal (si se conocen)
        
    Returns:
        Imagen con perspectiva corregida
    """
    # Esta es una implementación simplificada
    # En un caso real, se detectarían los bordes de la señal
    
    height, width = img.shape[:2]
    
    if corners is None:
        # Si no se proporcionan esquinas, usar valores predeterminados
        # que simulan una ligera distorsión de perspectiva
        src_points = np.float32([
            [0.1*width, 0.1*height],
            [0.9*width, 0.1*height],
            [0.9*width, 0.9*height],
            [0.1*width, 0.9*height]
        ])
    else:
        src_points = np.float32(corners)
    
    # Puntos de destino (rectángulo)
    dst_points = np.float32([
        [0, 0],
        [width, 0],
        [width, height],
        [0, height]
    ])
    
    # Calcular matriz de transformación y aplicar
    M = cv2.getPerspectiveTransform(src_points, dst_points)
    corrected = cv2.warpPerspective(img, M, (width, height))
    
    return corrected

# Función para detectar señales en una imagen
def detect_traffic_signs(img, min_area=500):
    """Detecta posibles señales de tráfico en una imagen.
    
    Args:
        img: Imagen donde buscar señales
        min_area: Área mínima para considerar un contorno como señal
        
    Returns:
        Lista de recortes de posibles señales
    """
    # Convertir a escala de grises si es RGB
    if len(img.shape) == 3 and img.shape[2] == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
    else:
        gray = img.copy()
    
    # Aplicar desenfoque para reducir ruido
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    
    # Detectar bordes
    edges = cv2.Canny(blurred, 50, 150)
    
    # Encontrar contornos
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    # Filtrar contornos por área y forma
    sign_regions = []
    for contour in contours:
        area = cv2.contourArea(contour)
        if area < min_area:
            continue
        
        # Aproximar contorno a polígono
        epsilon = 0.02 * cv2.arcLength(contour, True)
        approx = cv2.approxPolyDP(contour, epsilon, True)
        
        # Filtrar por número de vértices (3-8 para señales comunes)
        if 3 <= len(approx) <= 8:
            # Obtener rectángulo delimitador
            x, y, w, h = cv2.boundingRect(contour)
            
            # Extraer región
            sign_region = img[y:y+h, x:x+w]
            sign_regions.append(sign_region)
    
    return sign_regions

# Función principal para probar las utilidades
def test_utils():
    """Función para probar las utilidades del módulo."""
    print("Probando funciones de utilidad...")
    
    # Crear imagen de prueba
    test_img = np.zeros((100, 100, 3), dtype=np.uint8)
    cv2.rectangle(test_img, (20, 20), (80, 80), (255, 0, 0), -1)  # Rectángulo azul
    cv2.circle(test_img, (50, 50), 30, (0, 255, 0), -1)  # Círculo verde
    
    # Probar aumento de datos
    print("\nProbando aumento de datos:")
    augmented = augment_image(test_img)
    
    # Mostrar original y aumentada
    plt.figure(figsize=(10, 5))
    plt.subplot(1, 2, 1)
    plt.imshow(test_img)
    plt.title('Imagen Original')
    plt.axis('off')
    
    plt.subplot(1, 2, 2)
    plt.imshow(augmented)
    plt.title('Imagen Aumentada')
    plt.axis('off')
    
    plt.tight_layout()
    plt.show()
    
    print("Pruebas completadas")

# Punto de entrada para pruebas
if __name__ == "__main__":
    test_utils()