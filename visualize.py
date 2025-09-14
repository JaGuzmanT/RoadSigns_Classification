#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoadSigns Classification - Visualizaciones

Este módulo contiene funciones para visualizar los resultados del entrenamiento
y las predicciones del modelo de clasificación de señales viales.

Autor: Tu Nombre
Fecha: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import cv2
from sklearn.metrics import confusion_matrix, classification_report
import tensorflow as tf
from tensorflow.keras.models import load_model
import itertools
from pathlib import Path

# Importar configuración
from config import RESULTS_DIR, MODELS_DIR

# Crear directorio de resultados si no existe
os.makedirs(RESULTS_DIR, exist_ok=True)

# Función para visualizar el historial de entrenamiento
def plot_training_history(history, save_path=None):
    """Visualiza el historial de entrenamiento del modelo.
    
    Args:
        history: Objeto History de Keras o diccionario con historial
        save_path: Ruta donde guardar la visualización
    """
    # Convertir a diccionario si es un objeto History
    if not isinstance(history, dict):
        history = history.history
    
    # Crear figura con subplots
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(15, 5))
    
    # Graficar precisión
    ax1.plot(history['accuracy'], label='Entrenamiento')
    ax1.plot(history['val_accuracy'], label='Validación')
    ax1.set_title('Precisión del Modelo 🎯', fontsize=14)
    ax1.set_ylabel('Precisión')
    ax1.set_xlabel('Época')
    ax1.legend()
    ax1.grid(True, linestyle='--', alpha=0.6)
    
    # Graficar pérdida
    ax2.plot(history['loss'], label='Entrenamiento')
    ax2.plot(history['val_loss'], label='Validación')
    ax2.set_title('Pérdida del Modelo 📉', fontsize=14)
    ax2.set_ylabel('Pérdida')
    ax2.set_xlabel('Época')
    ax2.legend()
    ax2.grid(True, linestyle='--', alpha=0.6)
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'training_history.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Historial de entrenamiento guardado en {save_path}")
    
    return fig

# Función para visualizar la matriz de confusión
def plot_confusion_matrix(y_true, y_pred, class_names, normalize=False, save_path=None):
    """Visualiza la matriz de confusión del modelo.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Diccionario de nombres de clases
        normalize: Si se debe normalizar la matriz
        save_path: Ruta donde guardar la visualización
    """
    # Calcular matriz de confusión
    cm = confusion_matrix(y_true, y_pred)
    
    # Normalizar si se especifica
    if normalize:
        cm = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
        title = 'Matriz de Confusión Normalizada 🧩'
        fmt = '.2f'
    else:
        title = 'Matriz de Confusión 🧩'
        fmt = 'd'
    
    # Crear figura
    plt.figure(figsize=(12, 10))
    
    # Usar seaborn para mejorar la visualización
    sns.heatmap(cm, annot=True, fmt=fmt, cmap='Blues',
                xticklabels=[class_names.get(i, f"Clase {i}") for i in range(len(class_names))],
                yticklabels=[class_names.get(i, f"Clase {i}") for i in range(len(class_names))])
    
    # Configurar etiquetas y título
    plt.ylabel('Etiqueta Verdadera', fontsize=12)
    plt.xlabel('Etiqueta Predicha', fontsize=12)
    plt.title(title, fontsize=16)
    
    # Rotar etiquetas para mejor visualización
    plt.xticks(rotation=45, ha='right')
    plt.yticks(rotation=45)
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'confusion_matrix.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Matriz de confusión guardada en {save_path}")
    
    return plt.gcf()

# Función para visualizar ejemplos de predicciones
def plot_predictions(X, y_true, y_pred, class_names, n_samples=10, random_seed=None, save_path=None):
    """Visualiza ejemplos de predicciones del modelo.
    
    Args:
        X: Imágenes
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Diccionario de nombres de clases
        n_samples: Número de muestras a visualizar
        random_seed: Semilla para selección aleatoria
        save_path: Ruta donde guardar la visualización
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
        
        # Obtener nombres de clases
        true_class = class_names.get(y_true[idx], f"Clase {y_true[idx]}")
        pred_class = class_names.get(y_pred[idx], f"Clase {y_pred[idx]}")
        
        # Añadir título con clases
        if y_true[idx] == y_pred[idx]:
            color = 'green'
        else:
            color = 'red'
        
        plt.title(f"Real: {true_class}\nPred: {pred_class}", color=color)
        plt.axis('off')
    
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'prediction_examples.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Ejemplos de predicciones guardados en {save_path}")
    
    return plt.gcf()

# Función para visualizar errores de clasificación
def plot_classification_errors(X, y_true, y_pred, class_names, n_samples=10, save_path=None):
    """Visualiza ejemplos de errores de clasificación.
    
    Args:
        X: Imágenes
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Diccionario de nombres de clases
        n_samples: Número máximo de errores a visualizar
        save_path: Ruta donde guardar la visualización
    """
    # Encontrar índices donde la predicción es incorrecta
    error_indices = np.where(y_true != y_pred)[0]
    
    if len(error_indices) == 0:
        print("¡No se encontraron errores de clasificación!")
        return None
    
    # Limitar a n_samples
    if len(error_indices) > n_samples:
        error_indices = np.random.choice(error_indices, n_samples, replace=False)
    
    # Calcular filas y columnas para el grid
    n_cols = min(5, len(error_indices))
    n_rows = (len(error_indices) + n_cols - 1) // n_cols
    
    # Crear figura
    plt.figure(figsize=(15, 3 * n_rows))
    
    # Mostrar cada error
    for i, idx in enumerate(error_indices):
        plt.subplot(n_rows, n_cols, i + 1)
        
        # Mostrar imagen
        if X[idx].shape[-1] == 1:  # Imagen en escala de grises
            plt.imshow(X[idx].squeeze(), cmap='gray')
        else:  # Imagen RGB
            plt.imshow(X[idx])
        
        # Obtener nombres de clases
        true_class = class_names.get(y_true[idx], f"Clase {y_true[idx]}")
        pred_class = class_names.get(y_pred[idx], f"Clase {y_pred[idx]}")
        
        # Añadir título con clases
        plt.title(f"Real: {true_class}\nPred: {pred_class}", color='red')
        plt.axis('off')
    
    plt.suptitle(f"Errores de Clasificación ❌ ({len(error_indices)} de {len(y_true)} muestras)", 
                 fontsize=16, y=1.02)
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'classification_errors.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Errores de clasificación guardados en {save_path}")
    
    return plt.gcf()

# Función para visualizar mapas de atención (Grad-CAM)
def plot_attention_maps(model, img, class_idx, layer_name=None, save_path=None):
    """Visualiza mapas de atención para una imagen usando Grad-CAM.
    
    Args:
        model: Modelo de Keras
        img: Imagen a visualizar (debe ser preprocesada para el modelo)
        class_idx: Índice de la clase a visualizar
        layer_name: Nombre de la capa para Grad-CAM (si es None, se usa la última capa convolucional)
        save_path: Ruta donde guardar la visualización
    """
    # Asegurarse de que la imagen tenga la forma correcta
    if len(img.shape) == 3:
        img = np.expand_dims(img, axis=0)
    
    # Si no se especifica capa, usar la última capa convolucional
    if layer_name is None:
        for layer in reversed(model.layers):
            if 'conv' in layer.name:
                layer_name = layer.name
                break
    
    # Obtener capa de salida
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
    
    # Visualizar
    fig, ax = plt.subplots(1, 3, figsize=(15, 5))
    
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
    
    plt.suptitle(f"Visualización de Atención (Grad-CAM) 🔍", fontsize=16, y=0.98)
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'attention_map.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Mapa de atención guardado en {save_path}")
    
    return plt.gcf()

# Función para visualizar métricas de evaluación
def plot_evaluation_metrics(y_true, y_pred, class_names, save_path=None):
    """Visualiza métricas de evaluación por clase.
    
    Args:
        y_true: Etiquetas verdaderas
        y_pred: Etiquetas predichas
        class_names: Diccionario de nombres de clases
        save_path: Ruta donde guardar la visualización
    """
    # Obtener reporte de clasificación como diccionario
    report = classification_report(y_true, y_pred, output_dict=True)
    
    # Extraer métricas por clase
    metrics_df = pd.DataFrame()
    
    for class_idx in sorted(class_names.keys()):
        if str(class_idx) in report:
            class_metrics = report[str(class_idx)]
            metrics_df.loc[class_names.get(class_idx, f"Clase {class_idx}"), 'Precisión'] = class_metrics['precision']
            metrics_df.loc[class_names.get(class_idx, f"Clase {class_idx}"), 'Recall'] = class_metrics['recall']
            metrics_df.loc[class_names.get(class_idx, f"Clase {class_idx}"), 'F1-Score'] = class_metrics['f1-score']
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Graficar métricas
    metrics_df.plot(kind='bar', ax=plt.gca())
    
    # Configurar etiquetas y título
    plt.title('Métricas por Clase 📊', fontsize=16)
    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Valor', fontsize=12)
    plt.ylim(0, 1.1)
    plt.legend(title='Métrica')
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotar etiquetas para mejor visualización
    plt.xticks(rotation=45, ha='right')
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'metrics_by_class.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Métricas por clase guardadas en {save_path}")
    
    return plt.gcf()

# Función para visualizar la distribución de clases
def plot_class_distribution(y, class_names, save_path=None):
    """Visualiza la distribución de clases en el conjunto de datos.
    
    Args:
        y: Etiquetas
        class_names: Diccionario de nombres de clases
        save_path: Ruta donde guardar la visualización
    """
    # Contar ocurrencias de cada clase
    class_counts = np.bincount(y)
    
    # Crear DataFrame para visualización
    df = pd.DataFrame({
        'Clase': [class_names.get(i, f"Clase {i}") for i in range(len(class_counts))],
        'Cantidad': class_counts
    })
    
    # Ordenar por cantidad (descendente)
    df = df.sort_values('Cantidad', ascending=False)
    
    # Crear figura
    plt.figure(figsize=(12, 8))
    
    # Graficar distribución
    ax = sns.barplot(x='Clase', y='Cantidad', data=df)
    
    # Añadir etiquetas con cantidades
    for i, p in enumerate(ax.patches):
        ax.annotate(f"{p.get_height()}", 
                   (p.get_x() + p.get_width() / 2., p.get_height()),
                   ha='center', va='bottom')
    
    # Configurar etiquetas y título
    plt.title('Distribución de Clases 📊', fontsize=16)
    plt.xlabel('Clase', fontsize=12)
    plt.ylabel('Cantidad de Muestras', fontsize=12)
    plt.grid(axis='y', linestyle='--', alpha=0.7)
    
    # Rotar etiquetas para mejor visualización
    plt.xticks(rotation=45, ha='right')
    
    # Ajustar diseño
    plt.tight_layout()
    
    # Guardar figura si se especifica ruta
    if save_path is None:
        save_path = os.path.join(RESULTS_DIR, 'class_distribution.png')
    
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Distribución de clases guardada en {save_path}")
    
    return plt.gcf()

# Función para generar un informe visual completo
def generate_visual_report(model, X_test, y_test, class_names, history=None):
    """Genera un informe visual completo del modelo.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
        class_names: Diccionario de nombres de clases
        history: Historial de entrenamiento (opcional)
    """
    print("Generando informe visual...")
    
    # Crear directorio para el informe
    report_dir = os.path.join(RESULTS_DIR, 'visual_report')
    os.makedirs(report_dir, exist_ok=True)
    
    # Realizar predicciones
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # 1. Visualizar historial de entrenamiento
    if history is not None:
        plot_training_history(history, save_path=os.path.join(report_dir, '01_training_history.png'))
    
    # 2. Visualizar distribución de clases
    plot_class_distribution(y_test, class_names, save_path=os.path.join(report_dir, '02_class_distribution.png'))
    
    # 3. Visualizar matriz de confusión
    plot_confusion_matrix(y_test, y_pred_classes, class_names, 
                         save_path=os.path.join(report_dir, '03_confusion_matrix.png'))
    
    # 4. Visualizar matriz de confusión normalizada
    plot_confusion_matrix(y_test, y_pred_classes, class_names, normalize=True,
                         save_path=os.path.join(report_dir, '04_normalized_confusion_matrix.png'))
    
    # 5. Visualizar métricas por clase
    plot_evaluation_metrics(y_test, y_pred_classes, class_names,
                           save_path=os.path.join(report_dir, '05_metrics_by_class.png'))
    
    # 6. Visualizar ejemplos de predicciones
    plot_predictions(X_test, y_test, y_pred_classes, class_names, n_samples=10,
                    save_path=os.path.join(report_dir, '06_prediction_examples.png'))
    
    # 7. Visualizar errores de clasificación
    plot_classification_errors(X_test, y_test, y_pred_classes, class_names, n_samples=10,
                              save_path=os.path.join(report_dir, '07_classification_errors.png'))
    
    # 8. Visualizar mapas de atención para algunas muestras
    # Seleccionar algunas muestras aleatorias para visualizar atención
    for i in range(3):
        idx = np.random.randint(0, len(X_test))
        img = X_test[idx:idx+1]
        true_class = y_test[idx]
        plot_attention_maps(model, img, true_class,
                           save_path=os.path.join(report_dir, f'08_attention_map_{i+1}.png'))
    
    print(f"Informe visual generado en {report_dir}")

# Función principal para probar las visualizaciones
def test_visualizations():
    """Función para probar las funciones de visualización."""
    print("Probando funciones de visualización...")
    
    # Crear datos de prueba
    np.random.seed(42)
    n_samples = 100
    n_classes = 5
    
    # Crear imágenes aleatorias
    X = np.random.rand(n_samples, 32, 32, 3)
    
    # Crear etiquetas aleatorias
    y_true = np.random.randint(0, n_classes, n_samples)
    y_pred = np.copy(y_true)
    
    # Introducir algunos errores
    error_indices = np.random.choice(n_samples, 20, replace=False)
    for idx in error_indices:
        y_pred[idx] = (y_true[idx] + 1) % n_classes
    
    # Crear nombres de clases
    class_names = {i: f"Señal {i}" for i in range(n_classes)}
    
    # Probar visualización de distribución de clases
    plot_class_distribution(y_true, class_names)
    
    # Probar visualización de matriz de confusión
    plot_confusion_matrix(y_true, y_pred, class_names)
    
    # Probar visualización de errores
    plot_classification_errors(X, y_true, y_pred, class_names)
    
    print("Pruebas completadas")

# Punto de entrada para pruebas
if __name__ == "__main__":
    test_visualizations()