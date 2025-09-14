#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoadSigns Classification - Configuración

Este módulo contiene la configuración centralizada para el proyecto de clasificación
de señales viales, incluyendo rutas de directorios, hiperparámetros del modelo,
y otras configuraciones.

Autor: Tu Nombre
Fecha: 2024
"""

import os
from pathlib import Path

# Rutas de directorios
BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = os.path.join(BASE_DIR, 'data')
MODELS_DIR = os.path.join(BASE_DIR, 'Models')
RESULTS_DIR = os.path.join(BASE_DIR, 'results')

# Crear directorios si no existen
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Configuración de datos
DATA_CONFIG = {
    'img_size': (32, 32),  # Tamaño de las imágenes (ancho, alto)
    'grayscale': False,     # Si se deben usar imágenes en escala de grises
    'test_size': 0.2,       # Proporción de datos para prueba
    'val_size': 0.1,        # Proporción de datos para validación
    'random_state': 42      # Semilla para reproducibilidad
}

# Configuración de aumento de datos
AUGMENTATION_CONFIG = {
    'rotation_range': 15,    # Rango de rotación en grados
    'shift_range': 0.1,     # Rango de desplazamiento
    'shear_range': 0.1,     # Rango de cizallamiento
    'zoom_range': 0.1,      # Rango de zoom
    'horizontal_flip': True, # Si se permite volteo horizontal
    'vertical_flip': False   # Si se permite volteo vertical (generalmente False para señales)
}

# Hiperparámetros del modelo
MODEL_CONFIG = {
    'architecture': 'cnn',   # Tipo de arquitectura ('cnn', 'resnet', etc.)
    'input_shape': (32, 32, 3),  # Forma de entrada (alto, ancho, canales)
    'num_classes': None,     # Número de clases (se establecerá automáticamente)
    'learning_rate': 0.001,  # Tasa de aprendizaje
    'batch_size': 32,        # Tamaño del lote
    'epochs': 50,            # Número máximo de épocas
    'early_stopping': True,  # Si se debe usar early stopping
    'patience': 10,          # Paciencia para early stopping
    'dropout_rate': 0.5      # Tasa de dropout
}

# Configuración de entrenamiento
TRAINING_CONFIG = {
    'use_augmentation': True,  # Si se debe usar aumento de datos
    'use_class_weights': True, # Si se deben usar pesos de clase para desbalance
    'save_best_only': True,    # Guardar solo el mejor modelo
    'monitor_metric': 'val_accuracy'  # Métrica a monitorear
}

# Configuración de evaluación
EVAL_CONFIG = {
    'metrics': ['accuracy', 'precision', 'recall', 'f1'],  # Métricas a evaluar
    'confusion_matrix': True,  # Si se debe generar matriz de confusión
    'classification_report': True,  # Si se debe generar reporte de clasificación
    'visualize_errors': True   # Si se deben visualizar errores de clasificación
}

# Configuración de predicción
PREDICT_CONFIG = {
    'confidence_threshold': 0.7,  # Umbral de confianza para aceptar predicciones
    'top_k': 3,                  # Número de predicciones principales a mostrar
    'use_enhancement': True,     # Si se deben mejorar las imágenes antes de predecir
    'batch_prediction': True     # Si se deben hacer predicciones por lotes
}

# Configuración de visualización
VISUALIZATION_CONFIG = {
    'plot_history': True,      # Si se debe visualizar el historial de entrenamiento
    'plot_confusion_matrix': True,  # Si se debe visualizar la matriz de confusión
    'plot_examples': True,     # Si se deben visualizar ejemplos de datos
    'plot_augmentation': True, # Si se deben visualizar ejemplos de aumento de datos
    'plot_attention': True     # Si se deben visualizar mapas de atención
}

# Configuración de logging
LOGGING_CONFIG = {
    'log_level': 'INFO',      # Nivel de logging
    'save_logs': True,        # Si se deben guardar logs
    'log_dir': os.path.join(BASE_DIR, 'logs'),  # Directorio de logs
    'tensorboard': True       # Si se debe usar TensorBoard
}

# Crear directorio de logs si no existe
if LOGGING_CONFIG['save_logs']:
    os.makedirs(LOGGING_CONFIG['log_dir'], exist_ok=True)

# Configuración de hardware
HARDWARE_CONFIG = {
    'use_gpu': True,          # Si se debe usar GPU
    'multi_gpu': False,       # Si se deben usar múltiples GPUs
    'mixed_precision': True,  # Si se debe usar precisión mixta
    'memory_growth': True     # Si se debe usar crecimiento de memoria
}

# Función para obtener la configuración completa
def get_config():
    """Obtiene la configuración completa del proyecto.
    
    Returns:
        Diccionario con toda la configuración
    """
    return {
        'data': DATA_CONFIG,
        'augmentation': AUGMENTATION_CONFIG,
        'model': MODEL_CONFIG,
        'training': TRAINING_CONFIG,
        'evaluation': EVAL_CONFIG,
        'prediction': PREDICT_CONFIG,
        'visualization': VISUALIZATION_CONFIG,
        'logging': LOGGING_CONFIG,
        'hardware': HARDWARE_CONFIG,
        'paths': {
            'base_dir': BASE_DIR,
            'data_dir': DATA_DIR,
            'models_dir': MODELS_DIR,
            'results_dir': RESULTS_DIR,
            'log_dir': LOGGING_CONFIG['log_dir']
        }
    }

# Función para actualizar la configuración
def update_config(section, key, value):
    """Actualiza un valor específico en la configuración.
    
    Args:
        section: Sección de la configuración ('data', 'model', etc.)
        key: Clave a actualizar
        value: Nuevo valor
    """
    if section == 'data':
        DATA_CONFIG[key] = value
    elif section == 'augmentation':
        AUGMENTATION_CONFIG[key] = value
    elif section == 'model':
        MODEL_CONFIG[key] = value
    elif section == 'training':
        TRAINING_CONFIG[key] = value
    elif section == 'evaluation':
        EVAL_CONFIG[key] = value
    elif section == 'prediction':
        PREDICT_CONFIG[key] = value
    elif section == 'visualization':
        VISUALIZATION_CONFIG[key] = value
    elif section == 'logging':
        LOGGING_CONFIG[key] = value
    elif section == 'hardware':
        HARDWARE_CONFIG[key] = value
    else:
        raise ValueError(f"Sección de configuración desconocida: {section}")

# Punto de entrada para pruebas
if __name__ == "__main__":
    # Mostrar configuración actual
    import json
    print("Configuración actual:")
    print(json.dumps(get_config(), indent=2, default=str))