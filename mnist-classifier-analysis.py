# Importación de las bibliotecas necesarias
import matplotlib.pyplot as plt
import os
import numpy as np
from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import SGDClassifier
from sklearn.decomposition import PCA
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

import time  # Importa la biblioteca time para medir el tiempo de ejecución

# Carga del Conjunto de Datos MNIST
print("Cargando el conjunto de datos MNIST...")
# Carga el conjunto de datos MNIST
mnist = fetch_openml('mnist_784', parser='auto')
X, y = mnist["data"], mnist["target"]  # Divide los datos y las etiquetas

# División en conjunto de entrenamiento y prueba
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=10000, random_state=42)  # Divide los datos en conjuntos de entrenamiento y prueba

# Entrenamiento sin PCA (Random Forest)
print("\nEntrenando el clasificador de Random Forest sin PCA...")
start_time = time.time()  # Inicia el temporizador
# Crea el clasificador de Bosques Aleatorios
rf_classifier = RandomForestClassifier(n_estimators=100, random_state=42)
rf_classifier.fit(X_train, y_train)  # Entrena el clasificador
# Calcula el tiempo de entrenamiento
training_time_without_pca_rf = time.time() - start_time

# Evaluar el rendimiento
# Realiza predicciones en el conjunto de prueba
y_pred_rf = rf_classifier.predict(X_test)
accuracy_without_pca_rf = accuracy_score(
    y_test, y_pred_rf)  # Calcula la precisión
precision_without_pca_rf = precision_score(
    y_test, y_pred_rf, average='weighted')  # Calcula la precisión promedio
# Calcula la recuperación promedio
recall_without_pca_rf = recall_score(y_test, y_pred_rf, average='weighted')
# Calcula la puntuación F1 promedio
f1_without_pca_rf = f1_score(y_test, y_pred_rf, average='weighted')

# Aplicar PCA
print("\nAplicando PCA...")
# Crea una instancia de PCA con un umbral de varianza explicada del 95%
pca = PCA(n_components=0.95)
# Aplica PCA al conjunto de entrenamiento
X_train_reduced = pca.fit_transform(X_train)
X_test_reduced = pca.transform(X_test)  # Aplica PCA al conjunto de prueba
# Obtiene el número de dimensiones después de aplicar PCA
n_dimensions_after_pca = X_train_reduced.shape[1]
print("Número de dimensiones después de aplicar PCA:", n_dimensions_after_pca)

# Entrenamiento con PCA (Random Forest)
print("\nEntrenando el clasificador de Random Forest con PCA...")
start_time = time.time()  # Inicia el temporizador
# Crea el clasificador de Bosques Aleatorios con PCA
rf_classifier_pca = RandomForestClassifier(n_estimators=100, random_state=42)
# Entrena el clasificador con los datos reducidos por PCA
rf_classifier_pca.fit(X_train_reduced, y_train)
# Calcula el tiempo de entrenamiento con PCA
training_time_with_pca_rf = time.time() - start_time

# Evaluar el rendimiento
# Realiza predicciones en el conjunto de prueba con datos reducidos por PCA
y_pred_rf_pca = rf_classifier_pca.predict(X_test_reduced)
accuracy_with_pca_rf = accuracy_score(
    y_test, y_pred_rf_pca)  # Calcula la precisión con PCA
precision_with_pca_rf = precision_score(
    y_test, y_pred_rf_pca, average='weighted')  # Calcula la precisión promedio con PCA
# Calcula la recuperación promedio con PCA
recall_with_pca_rf = recall_score(y_test, y_pred_rf_pca, average='weighted')
# Calcula la puntuación F1 promedio con PCA
f1_with_pca_rf = f1_score(y_test, y_pred_rf_pca, average='weighted')

# Experimentación con SGDClassifier
print("\nEntrenando el clasificador SGDClassifier sin PCA...")
start_time = time.time()  # Inicia el temporizador
sgd_classifier = SGDClassifier(random_state=42)  # Crea el clasificador SGD
sgd_classifier.fit(X_train, y_train)  # Entrena el clasificador
# Calcula el tiempo de entrenamiento sin PCA
training_time_without_pca_sgd = time.time() - start_time

# Evaluar el rendimiento
# Realiza predicciones en el conjunto de prueba
y_pred_sgd = sgd_classifier.predict(X_test)
accuracy_without_pca_sgd = accuracy_score(
    y_test, y_pred_sgd)  # Calcula la precisión
precision_without_pca_sgd = precision_score(
    y_test, y_pred_sgd, average='weighted')  # Calcula la precisión promedio
recall_without_pca_sgd = recall_score(
    y_test, y_pred_sgd, average='weighted')  # Calcula la recuperación promedio
# Calcula la puntuación F1 promedio
f1_without_pca_sgd = f1_score(y_test, y_pred_sgd, average='weighted')

# Entrenamiento con PCA (SGDClassifier)
print("\nEntrenando el clasificador SGDClassifier con PCA...")
start_time = time.time()  # Inicia el temporizador
# Crea el clasificador SGD con PCA
sgd_classifier_pca = SGDClassifier(random_state=42)
# Entrena el clasificador con los datos reducidos por PCA
sgd_classifier_pca.fit(X_train_reduced, y_train)
# Calcula el tiempo de entrenamiento con PCA
training_time_with_pca_sgd = time.time() - start_time

# Evaluar el rendimiento
y_pred_sgd_pca = sgd_classifier_pca.predict(X_test_reduced)
accuracy_with_pca_sgd = accuracy_score(y_test, y_pred_sgd_pca)
precision_with_pca_sgd = precision_score(
    y_test, y_pred_sgd_pca, average='weighted')
recall_with_pca_sgd = recall_score(y_test, y_pred_sgd_pca, average='weighted')
f1_with_pca_sgd = f1_score(y_test, y_pred_sgd_pca, average='weighted')

# Tiempo de entrenamiento y métricas de rendimiento para Random Forest sin PCA
tiempo_entrenamiento_rf_sin_pca = training_time_without_pca_rf
precision_rf_sin_pca = precision_without_pca_rf
recall_rf_sin_pca = recall_without_pca_rf
f1_rf_sin_pca = f1_without_pca_rf


# Directorio donde se guardarán las imágenes
results_dir = "results"
if not os.path.exists(results_dir):
    os.makedirs(results_dir)

# Definir los nombres de las métricas y los valores correspondientes para los resultados sin PCA
metrics_without_pca = ['Tiempo de entrenamiento RF', 'Precisión RF', 'Recall RF', 'F1-score RF',
                       'Tiempo de entrenamiento SGD', 'Precisión SGD', 'Recall SGD', 'F1-score SGD']
values_without_pca = [training_time_without_pca_rf, precision_without_pca_rf, recall_without_pca_rf, f1_without_pca_rf,
                      training_time_without_pca_sgd, precision_without_pca_sgd, recall_without_pca_sgd, f1_without_pca_sgd]

# Definir los nombres de las métricas y los valores correspondientes para los resultados con PCA
metrics_with_pca = ['Tiempo de entrenamiento RF con PCA', 'Precisión RF con PCA', 'Recall RF con PCA', 'F1-score RF con PCA',
                    'Tiempo de entrenamiento SGD con PCA', 'Precisión SGD con PCA', 'Recall SGD con PCA', 'F1-score SGD con PCA']
values_with_pca = [training_time_with_pca_rf, precision_with_pca_rf, recall_with_pca_rf, f1_with_pca_rf,
                   training_time_with_pca_sgd, precision_with_pca_sgd, recall_with_pca_sgd, f1_with_pca_sgd]

# Crear un gráfico de barras para los resultados sin PCA
plt.figure(figsize=(10, 6))
plt.bar(metrics_without_pca, values_without_pca, color='blue')
plt.xlabel('Métricas')
plt.ylabel('Valor')
plt.title('Resultados sin PCA')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Guardar la imagen en la carpeta "results"
plt.savefig(os.path.join(results_dir, "resultados_sin_pca.png"))
plt.close()

# Crear un gráfico de barras para los resultados con PCA
plt.figure(figsize=(10, 6))
plt.bar(metrics_with_pca, values_with_pca, color='green')
plt.xlabel('Métricas')
plt.ylabel('Valor')
plt.title('Resultados con PCA')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
# Guardar la imagen en la carpeta "results"
plt.savefig(os.path.join(results_dir, "resultados_con_pca.png"))
plt.close()
