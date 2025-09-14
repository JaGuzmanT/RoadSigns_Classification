#!/usr/bin/env python
# -*- coding: utf-8 -*-

"""
RoadSigns Classification - Modelo de clasificación de señales viales

Este script entrena un modelo de red neuronal convolucional para clasificar
imágenes de señales de tráfico utilizando el dataset GTSRB (German Traffic Sign
Recognition Benchmark) u otros datasets similares.

Autor: Tu Nombre
Fecha: 2024
"""

import os
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import tensorflow as tf
from tensorflow.keras import layers, models, optimizers, callbacks
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, confusion_matrix
import cv2
from tqdm import tqdm
import time

# Configuración de semillas para reproducibilidad
def set_seeds(seed=42):
    """Configura semillas para reproducibilidad."""
    np.random.seed(seed)
    tf.random.set_seed(seed)
    os.environ['PYTHONHASHSEED'] = str(seed)
    os.environ['TF_DETERMINISTIC_OPS'] = '1'

# Función para cargar y preprocesar datos
def load_data(data_dir, img_size=(32, 32), test_size=0.2, val_size=0.1):
    """Carga y preprocesa las imágenes de señales de tráfico.
    
    Args:
        data_dir: Directorio que contiene las imágenes organizadas por clase
        img_size: Tamaño al que se redimensionarán las imágenes
        test_size: Proporción de datos para prueba
        val_size: Proporción de datos para validación
        
    Returns:
        Conjuntos de datos de entrenamiento, validación y prueba
    """
    print("🔍 Cargando y preprocesando datos...")
    
    # Aquí iría el código para cargar las imágenes desde el directorio
    # Este es un ejemplo simplificado. En un caso real, se cargarían las imágenes
    # desde las carpetas correspondientes a cada clase.
    
    # Simulación de carga de datos para este ejemplo
    X = np.random.rand(1000, img_size[0], img_size[1], 3)  # 1000 imágenes RGB aleatorias
    y = np.random.randint(0, 43, size=(1000,))  # 43 clases (como en GTSRB)
    
    # División en conjuntos de entrenamiento, validación y prueba
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size/(1-test_size), random_state=42, stratify=y_train)
    
    print(f"✅ Datos cargados: {X_train.shape[0]} entrenamiento, {X_val.shape[0]} validación, {X_test.shape[0]} prueba")
    
    return (X_train, y_train), (X_val, y_val), (X_test, y_test)

# Función para crear el modelo CNN
def create_model(input_shape=(32, 32, 3), num_classes=43):
    """Crea un modelo CNN para clasificación de señales de tráfico.
    
    Args:
        input_shape: Forma de las imágenes de entrada
        num_classes: Número de clases de señales
        
    Returns:
        Modelo compilado
    """
    print("🧠 Creando arquitectura del modelo...")
    
    model = models.Sequential([
        # Primera capa convolucional
        layers.Conv2D(32, (3, 3), activation='relu', padding='same', input_shape=input_shape),
        layers.BatchNormalization(),
        layers.Conv2D(32, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.2),
        
        # Segunda capa convolucional
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(64, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.3),
        
        # Tercera capa convolucional
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.Conv2D(128, (3, 3), activation='relu', padding='same'),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Dropout(0.4),
        
        # Capa de aplanamiento y densas
        layers.Flatten(),
        layers.Dense(512, activation='relu'),
        layers.BatchNormalization(),
        layers.Dropout(0.5),
        layers.Dense(num_classes, activation='softmax')
    ])
    
    # Compilación del modelo
    model.compile(
        optimizer=optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    print("✅ Modelo creado y compilado")
    model.summary()
    
    return model

# Función para entrenar el modelo
def train_model(model, train_data, val_data, epochs=50, batch_size=32, augment=True):
    """Entrena el modelo con los datos proporcionados.
    
    Args:
        model: Modelo a entrenar
        train_data: Datos de entrenamiento (X_train, y_train)
        val_data: Datos de validación (X_val, y_val)
        epochs: Número de épocas de entrenamiento
        batch_size: Tamaño del lote
        augment: Si se debe usar aumento de datos
        
    Returns:
        Historial de entrenamiento
    """
    X_train, y_train = train_data
    X_val, y_val = val_data
    
    print(f"🚀 Iniciando entrenamiento por {epochs} épocas con batch_size={batch_size}")
    
    # Callbacks
    callbacks_list = [
        callbacks.ModelCheckpoint(
            filepath='Models/model_best.h5',
            monitor='val_accuracy',
            save_best_only=True,
            verbose=1
        ),
        callbacks.EarlyStopping(
            monitor='val_loss',
            patience=10,
            verbose=1
        ),
        callbacks.ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=5,
            min_lr=1e-6,
            verbose=1
        )
    ]
    
    # Aumento de datos si está habilitado
    if augment:
        print("🔄 Aplicando aumento de datos...")
        datagen = ImageDataGenerator(
            rotation_range=15,
            width_shift_range=0.1,
            height_shift_range=0.1,
            shear_range=0.1,
            zoom_range=0.1,
            horizontal_flip=False,  # No voltear horizontalmente las señales
            fill_mode='nearest'
        )
        datagen.fit(X_train)
        
        # Entrenamiento con aumento de datos
        history = model.fit(
            datagen.flow(X_train, y_train, batch_size=batch_size),
            steps_per_epoch=len(X_train) // batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
    else:
        # Entrenamiento sin aumento de datos
        history = model.fit(
            X_train, y_train,
            batch_size=batch_size,
            epochs=epochs,
            validation_data=(X_val, y_val),
            callbacks=callbacks_list,
            verbose=1
        )
    
    print("✅ Entrenamiento completado")
    return history

# Función para evaluar el modelo
def evaluate_model(model, test_data):
    """Evalúa el modelo con los datos de prueba.
    
    Args:
        model: Modelo entrenado
        test_data: Datos de prueba (X_test, y_test)
        
    Returns:
        Métricas de evaluación
    """
    X_test, y_test = test_data
    
    print("📊 Evaluando modelo en datos de prueba...")
    
    # Evaluación del modelo
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    print(f"📈 Precisión en prueba: {test_acc:.4f}")
    print(f"📉 Pérdida en prueba: {test_loss:.4f}")
    
    # Predicciones
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    # Reporte de clasificación
    print("\n📋 Reporte de clasificación:")
    print(classification_report(y_test, y_pred_classes))
    
    # Matriz de confusión
    print("\n🔢 Generando matriz de confusión...")
    cm = confusion_matrix(y_test, y_pred_classes)
    
    # Guardar matriz de confusión como imagen
    plt.figure(figsize=(12, 10))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
    plt.title('Matriz de Confusión')
    plt.xlabel('Predicción')
    plt.ylabel('Valor Real')
    
    # Crear directorio de resultados si no existe
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/confusion_matrix.png')
    plt.close()
    
    print("✅ Evaluación completada y resultados guardados")
    
    return test_acc, test_loss

# Función para visualizar el historial de entrenamiento
def plot_training_history(history):
    """Visualiza el historial de entrenamiento.
    
    Args:
        history: Historial de entrenamiento del modelo
    """
    print("📈 Generando gráficas de entrenamiento...")
    
    # Gráfica de precisión
    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(history.history['accuracy'], label='Entrenamiento')
    plt.plot(history.history['val_accuracy'], label='Validación')
    plt.title('Precisión del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Precisión')
    plt.legend()
    
    # Gráfica de pérdida
    plt.subplot(1, 2, 2)
    plt.plot(history.history['loss'], label='Entrenamiento')
    plt.plot(history.history['val_loss'], label='Validación')
    plt.title('Pérdida del Modelo')
    plt.xlabel('Época')
    plt.ylabel('Pérdida')
    plt.legend()
    
    plt.tight_layout()
    
    # Crear directorio de resultados si no existe
    os.makedirs('results', exist_ok=True)
    plt.savefig('results/training_history.png')
    plt.close()
    
    print("✅ Gráficas generadas y guardadas")

# Función principal
def main():
    """Función principal que ejecuta el flujo completo de entrenamiento."""
    # Configurar argumentos de línea de comandos
    parser = argparse.ArgumentParser(description='Entrenamiento de modelo para clasificación de señales de tráfico')
    parser.add_argument('--data_dir', type=str, default='Data', help='Directorio de datos')
    parser.add_argument('--epochs', type=int, default=50, help='Número de épocas')
    parser.add_argument('--batch_size', type=int, default=32, help='Tamaño del lote')
    parser.add_argument('--img_size', type=int, default=32, help='Tamaño de imagen')
    parser.add_argument('--no_augment', action='store_true', help='Desactivar aumento de datos')
    args = parser.parse_args()
    
    # Crear directorios necesarios
    os.makedirs('Models', exist_ok=True)
    os.makedirs('results', exist_ok=True)
    
    # Configurar semillas para reproducibilidad
    set_seeds()
    
    # Cargar datos
    train_data, val_data, test_data = load_data(
        data_dir=args.data_dir,
        img_size=(args.img_size, args.img_size)
    )
    
    # Crear modelo
    model = create_model(input_shape=(args.img_size, args.img_size, 3))
    
    # Entrenar modelo
    history = train_model(
        model=model,
        train_data=train_data,
        val_data=val_data,
        epochs=args.epochs,
        batch_size=args.batch_size,
        augment=not args.no_augment
    )
    
    # Cargar el mejor modelo guardado
    print("\n🔄 Cargando el mejor modelo guardado...")
    best_model = tf.keras.models.load_model('Models/model_best.h5')
    
    # Evaluar modelo
    test_acc, test_loss = evaluate_model(best_model, test_data)
    
    # Visualizar historial de entrenamiento
    plot_training_history(history)
    
    print(f"\n🎉 ¡Proceso completado! Precisión final: {test_acc:.4f}")
    print(f"📊 Resultados guardados en el directorio 'results/'")
    print(f"💾 Modelo guardado como 'Models/model_best.h5'")

# Punto de entrada
if __name__ == "__main__":
    start_time = time.time()
    print("\n🚀 Iniciando RoadSigns Classification - Entrenamiento de modelo")
    print("=" * 70)
    
    main()
    
    elapsed_time = time.time() - start_time
    hours, remainder = divmod(elapsed_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"\n⏱️ Tiempo total de ejecución: {int(hours):02}:{int(minutes):02}:{int(seconds):02}")
    print("=" * 70)