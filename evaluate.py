#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoadSigns Classification - Evaluación

Este módulo contiene funciones para evaluar el rendimiento del modelo
de clasificación de señales viales.

Autor: Tu Nombre
Fecha: 2024
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, precision_score, recall_score, f1_score
import tensorflow as tf
from tensorflow.keras.models import load_model
import json
import time
from pathlib import Path

# Importar módulos propios
from config import MODELS_DIR, RESULTS_DIR
from visualize import plot_confusion_matrix, plot_evaluation_metrics, plot_classification_errors

# Crear directorios si no existen
os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(RESULTS_DIR, exist_ok=True)

# Función para cargar el modelo
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
            raise FileNotFoundError(f"No se encontraron modelos en {MODELS_DIR}")
        
        # Ordenar por fecha de modificación (más reciente primero)
        model_files.sort(key=lambda x: os.path.getmtime(os.path.join(MODELS_DIR, x)), reverse=True)
        model_path = os.path.join(MODELS_DIR, model_files[0])
    
    print(f"Cargando modelo desde {model_path}")
    model = load_model(model_path)
    print(f"Modelo cargado: {model.name}")
    
    return model

# Función para cargar el mapeo de clases
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
        print(f"Advertencia: No se encontró el archivo de mapeo de clases en {filepath}")
        return {}
    
    df = pd.read_csv(filepath)
    class_names = dict(zip(df['ClassId'], df['SignName']))
    print(f"Mapeo de clases cargado desde {filepath}")
    return class_names

# Función para evaluar el modelo
def evaluate_model(model, X_test, y_test, class_names=None, batch_size=32):
    """Evalúa el rendimiento del modelo.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
        class_names: Diccionario de nombres de clases
        batch_size: Tamaño del lote para evaluación
        
    Returns:
        Diccionario con métricas de evaluación
    """
    print("Evaluando modelo...")
    start_time = time.time()
    
    # Evaluar modelo
    test_loss, test_acc = model.evaluate(X_test, tf.keras.utils.to_categorical(y_test), batch_size=batch_size, verbose=1)
    
    # Realizar predicciones
    y_pred_prob = model.predict(X_test, batch_size=batch_size)
    y_pred = np.argmax(y_pred_prob, axis=1)
    
    # Calcular métricas
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average='weighted')
    
    # Calcular matriz de confusión
    cm = confusion_matrix(y_test, y_pred)
    
    # Generar reporte de clasificación
    report = classification_report(y_test, y_pred, output_dict=True)
    
    # Calcular tiempo de inferencia promedio por muestra
    start_inference = time.time()
    model.predict(X_test[:100], batch_size=batch_size)
    end_inference = time.time()
    inference_time = (end_inference - start_inference) / 100  # segundos por muestra
    
    # Tiempo total de evaluación
    evaluation_time = time.time() - start_time
    
    # Crear diccionario de resultados
    results = {
        'accuracy': float(accuracy),
        'precision': float(precision),
        'recall': float(recall),
        'f1_score': float(f1),
        'test_loss': float(test_loss),
        'inference_time_ms': float(inference_time * 1000),  # convertir a ms
        'evaluation_time_s': float(evaluation_time),
        'confusion_matrix': cm.tolist(),
        'classification_report': report
    }
    
    # Imprimir resultados
    print(f"\nResultados de evaluación:")
    print(f"  Precisión: {accuracy:.4f}")
    print(f"  Recall: {recall:.4f}")
    print(f"  F1-Score: {f1:.4f}")
    print(f"  Pérdida: {test_loss:.4f}")
    print(f"  Tiempo de inferencia: {inference_time*1000:.2f} ms por muestra")
    print(f"  Tiempo total de evaluación: {evaluation_time:.2f} segundos")
    
    # Guardar resultados
    results_path = os.path.join(RESULTS_DIR, 'evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(results, f, indent=2)
    print(f"Resultados guardados en {results_path}")
    
    # Generar visualizaciones si se proporcionan nombres de clases
    if class_names:
        # Matriz de confusión
        plot_confusion_matrix(y_test, y_pred, class_names, 
                             save_path=os.path.join(RESULTS_DIR, 'confusion_matrix.png'))
        
        # Matriz de confusión normalizada
        plot_confusion_matrix(y_test, y_pred, class_names, normalize=True,
                             save_path=os.path.join(RESULTS_DIR, 'normalized_confusion_matrix.png'))
        
        # Métricas por clase
        plot_evaluation_metrics(y_test, y_pred, class_names,
                               save_path=os.path.join(RESULTS_DIR, 'metrics_by_class.png'))
        
        # Errores de clasificación
        plot_classification_errors(X_test, y_test, y_pred, class_names, n_samples=10,
                                  save_path=os.path.join(RESULTS_DIR, 'classification_errors.png'))
    
    return results

# Función para evaluar robustez del modelo
def evaluate_robustness(model, X_test, y_test, class_names=None, batch_size=32):
    """Evalúa la robustez del modelo ante diferentes perturbaciones.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
        class_names: Diccionario de nombres de clases
        batch_size: Tamaño del lote para evaluación
        
    Returns:
        Diccionario con métricas de robustez
    """
    print("Evaluando robustez del modelo...")
    
    # Crear directorio para resultados de robustez
    robustness_dir = os.path.join(RESULTS_DIR, 'robustness')
    os.makedirs(robustness_dir, exist_ok=True)
    
    # Definir perturbaciones
    perturbations = {
        'original': lambda x: x,
        'ruido_gaussiano': lambda x: np.clip(x + np.random.normal(0, 0.1, x.shape), 0, 1),
        'brillo': lambda x: np.clip(x + 0.2, 0, 1),
        'oscuridad': lambda x: np.clip(x - 0.2, 0, 1),
        'rotacion': lambda x: tf.keras.preprocessing.image.random_rotation(x, 15, row_axis=0, col_axis=1, channel_axis=2),
        'zoom': lambda x: tf.keras.preprocessing.image.random_zoom(x, (0.8, 0.8), row_axis=0, col_axis=1, channel_axis=2),
        'desenfoque': lambda x: tf.image.resize(tf.image.resize(x, (16, 16)), (32, 32)).numpy()
    }
    
    # Evaluar cada perturbación
    robustness_results = {}
    
    for name, perturbation in perturbations.items():
        print(f"\nEvaluando perturbación: {name}")
        
        # Aplicar perturbación
        X_perturbed = np.copy(X_test)
        for i in range(len(X_perturbed)):
            X_perturbed[i] = perturbation(X_perturbed[i])
        
        # Realizar predicciones
        y_pred_prob = model.predict(X_perturbed, batch_size=batch_size)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Guardar resultados
        robustness_results[name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        print(f"  Precisión: {accuracy:.4f}")
        
        # Guardar ejemplos de imágenes perturbadas
        if class_names:
            # Seleccionar algunas muestras aleatorias
            indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
            
            plt.figure(figsize=(15, 3))
            for i, idx in enumerate(indices):
                plt.subplot(1, 5, i + 1)
                
                # Mostrar imagen perturbada
                plt.imshow(X_perturbed[idx])
                
                # Añadir título con predicción
                true_class = class_names.get(y_test[idx], f"Clase {y_test[idx]}")
                pred_class = class_names.get(y_pred[idx], f"Clase {y_pred[idx]}")
                
                if y_test[idx] == y_pred[idx]:
                    color = 'green'
                else:
                    color = 'red'
                
                plt.title(f"R: {true_class}\nP: {pred_class}", color=color, fontsize=8)
                plt.axis('off')
            
            plt.suptitle(f"Ejemplos con perturbación: {name}", fontsize=14)
            plt.tight_layout()
            
            # Guardar figura
            plt.savefig(os.path.join(robustness_dir, f"{name}_examples.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Visualizar resultados de robustez
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Extraer valores para cada perturbación
        values = [robustness_results[p][metric] for p in perturbations.keys()]
        
        # Crear gráfico de barras
        bars = plt.bar(perturbations.keys(), values)
        
        # Añadir etiquetas con valores
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Configurar etiquetas y título
        plt.title(f'Robustez del Modelo - {metric.capitalize()} 🛡️', fontsize=14)
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotar etiquetas para mejor visualización
        plt.xticks(rotation=45, ha='right')
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(os.path.join(robustness_dir, f"robustness_{metric}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Guardar resultados
    results_path = os.path.join(robustness_dir, 'robustness_results.json')
    with open(results_path, 'w') as f:
        json.dump(robustness_results, f, indent=2)
    print(f"Resultados de robustez guardados en {results_path}")
    
    return robustness_results

# Función para evaluar el modelo en diferentes condiciones de iluminación
def evaluate_lighting_conditions(model, X_test, y_test, class_names=None, batch_size=32):
    """Evalúa el rendimiento del modelo en diferentes condiciones de iluminación.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
        class_names: Diccionario de nombres de clases
        batch_size: Tamaño del lote para evaluación
        
    Returns:
        Diccionario con métricas por condición de iluminación
    """
    print("Evaluando modelo en diferentes condiciones de iluminación...")
    
    # Crear directorio para resultados
    lighting_dir = os.path.join(RESULTS_DIR, 'lighting_conditions')
    os.makedirs(lighting_dir, exist_ok=True)
    
    # Definir condiciones de iluminación
    lighting_conditions = {
        'original': lambda x: x,
        'muy_oscuro': lambda x: np.clip(x * 0.3, 0, 1),
        'oscuro': lambda x: np.clip(x * 0.6, 0, 1),
        'brillante': lambda x: np.clip(x * 1.4, 0, 1),
        'muy_brillante': lambda x: np.clip(x * 1.8, 0, 1),
        'bajo_contraste': lambda x: np.clip(x * 0.7 + 0.15, 0, 1),
        'alto_contraste': lambda x: np.clip((x - 0.5) * 1.5 + 0.5, 0, 1)
    }
    
    # Evaluar cada condición
    lighting_results = {}
    
    for name, condition in lighting_conditions.items():
        print(f"\nEvaluando condición: {name}")
        
        # Aplicar condición
        X_modified = np.copy(X_test)
        for i in range(len(X_modified)):
            X_modified[i] = condition(X_modified[i])
        
        # Realizar predicciones
        y_pred_prob = model.predict(X_modified, batch_size=batch_size)
        y_pred = np.argmax(y_pred_prob, axis=1)
        
        # Calcular métricas
        accuracy = accuracy_score(y_test, y_pred)
        precision = precision_score(y_test, y_pred, average='weighted')
        recall = recall_score(y_test, y_pred, average='weighted')
        f1 = f1_score(y_test, y_pred, average='weighted')
        
        # Guardar resultados
        lighting_results[name] = {
            'accuracy': float(accuracy),
            'precision': float(precision),
            'recall': float(recall),
            'f1_score': float(f1)
        }
        
        print(f"  Precisión: {accuracy:.4f}")
        
        # Guardar ejemplos de imágenes modificadas
        if class_names:
            # Seleccionar algunas muestras aleatorias
            indices = np.random.choice(len(X_test), min(5, len(X_test)), replace=False)
            
            plt.figure(figsize=(15, 3))
            for i, idx in enumerate(indices):
                plt.subplot(1, 5, i + 1)
                
                # Mostrar imagen modificada
                plt.imshow(X_modified[idx])
                
                # Añadir título con predicción
                true_class = class_names.get(y_test[idx], f"Clase {y_test[idx]}")
                pred_class = class_names.get(y_pred[idx], f"Clase {y_pred[idx]}")
                
                if y_test[idx] == y_pred[idx]:
                    color = 'green'
                else:
                    color = 'red'
                
                plt.title(f"R: {true_class}\nP: {pred_class}", color=color, fontsize=8)
                plt.axis('off')
            
            plt.suptitle(f"Ejemplos con iluminación: {name}", fontsize=14)
            plt.tight_layout()
            
            # Guardar figura
            plt.savefig(os.path.join(lighting_dir, f"{name}_examples.png"), dpi=300, bbox_inches='tight')
            plt.close()
    
    # Visualizar resultados
    metrics = ['accuracy', 'precision', 'recall', 'f1_score']
    for metric in metrics:
        plt.figure(figsize=(10, 6))
        
        # Extraer valores para cada condición
        values = [lighting_results[c][metric] for c in lighting_conditions.keys()]
        
        # Crear gráfico de barras
        bars = plt.bar(lighting_conditions.keys(), values)
        
        # Añadir etiquetas con valores
        for bar in bars:
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom', fontsize=8)
        
        # Configurar etiquetas y título
        plt.title(f'Rendimiento por Iluminación - {metric.capitalize()} 💡', fontsize=14)
        plt.ylabel(metric.capitalize())
        plt.ylim(0, 1.1)
        plt.grid(axis='y', linestyle='--', alpha=0.7)
        
        # Rotar etiquetas para mejor visualización
        plt.xticks(rotation=45, ha='right')
        
        # Ajustar diseño
        plt.tight_layout()
        
        # Guardar figura
        plt.savefig(os.path.join(lighting_dir, f"lighting_{metric}.png"), dpi=300, bbox_inches='tight')
        plt.close()
    
    # Guardar resultados
    results_path = os.path.join(lighting_dir, 'lighting_results.json')
    with open(results_path, 'w') as f:
        json.dump(lighting_results, f, indent=2)
    print(f"Resultados de iluminación guardados en {results_path}")
    
    return lighting_results

# Función para generar un informe completo de evaluación
def generate_evaluation_report(model, X_test, y_test, class_names=None, batch_size=32):
    """Genera un informe completo de evaluación del modelo.
    
    Args:
        model: Modelo entrenado
        X_test: Datos de prueba
        y_test: Etiquetas verdaderas
        class_names: Diccionario de nombres de clases
        batch_size: Tamaño del lote para evaluación
        
    Returns:
        Diccionario con todos los resultados de evaluación
    """
    print("Generando informe completo de evaluación...")
    
    # Crear directorio para el informe
    report_dir = os.path.join(RESULTS_DIR, 'evaluation_report')
    os.makedirs(report_dir, exist_ok=True)
    
    # 1. Evaluación básica
    basic_results = evaluate_model(model, X_test, y_test, class_names, batch_size)
    
    # 2. Evaluación de robustez
    robustness_results = evaluate_robustness(model, X_test, y_test, class_names, batch_size)
    
    # 3. Evaluación de condiciones de iluminación
    lighting_results = evaluate_lighting_conditions(model, X_test, y_test, class_names, batch_size)
    
    # Combinar resultados
    all_results = {
        'basic_evaluation': basic_results,
        'robustness_evaluation': robustness_results,
        'lighting_evaluation': lighting_results
    }
    
    # Guardar resultados completos
    results_path = os.path.join(report_dir, 'full_evaluation_results.json')
    with open(results_path, 'w') as f:
        json.dump(all_results, f, indent=2)
    print(f"Informe completo guardado en {report_dir}")
    
    return all_results

# Función principal
def main():
    """Función principal para ejecutar la evaluación."""
    import argparse
    from utils import load_images_from_directory, split_data
    
    # Configurar parser de argumentos
    parser = argparse.ArgumentParser(description='Evaluar modelo de clasificación de señales viales')
    parser.add_argument('--data_dir', type=str, help='Directorio con datos de prueba')
    parser.add_argument('--model_path', type=str, help='Ruta al modelo guardado')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del lote para evaluación')
    parser.add_argument('--report_type', type=str, default='basic', 
                        choices=['basic', 'robustness', 'lighting', 'full'],
                        help='Tipo de informe a generar')
    
    args = parser.parse_args()
    
    # Cargar modelo
    model = load_trained_model(args.model_path)
    
    # Cargar mapeo de clases
    class_names = load_class_mapping()
    
    # Cargar datos de prueba
    if args.data_dir:
        print(f"Cargando datos de prueba desde {args.data_dir}...")
        X, y, loaded_class_names = load_images_from_directory(args.data_dir)
        
        # Si no se cargó el mapeo de clases, usar el de los datos
        if not class_names and loaded_class_names:
            class_names = loaded_class_names
        
        # Dividir datos
        _, _, (X_test, y_test) = split_data(X, y)
    else:
        # Cargar datos de prueba desde archivo
        data_path = os.path.join(RESULTS_DIR, 'test_data.npz')
        if not os.path.isfile(data_path):
            raise FileNotFoundError(f"No se encontraron datos de prueba en {data_path}. Especifique --data_dir")
        
        print(f"Cargando datos de prueba desde {data_path}...")
        data = np.load(data_path)
        X_test, y_test = data['X_test'], data['y_test']
    
    # Generar informe según el tipo especificado
    if args.report_type == 'basic':
        evaluate_model(model, X_test, y_test, class_names, args.batch_size)
    elif args.report_type == 'robustness':
        evaluate_robustness(model, X_test, y_test, class_names, args.batch_size)
    elif args.report_type == 'lighting':
        evaluate_lighting_conditions(model, X_test, y_test, class_names, args.batch_size)
    elif args.report_type == 'full':
        generate_evaluation_report(model, X_test, y_test, class_names, args.batch_size)

# Punto de entrada
if __name__ == "__main__":
    main()