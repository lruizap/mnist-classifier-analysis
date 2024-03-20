# mnist-classifier-analysis

## Introducción del proyecto

Este proyecto realiza un análisis y experimentación con clasificadores en Python utilizando el conjunto de datos MNIST, que consiste en imágenes de dígitos escritos a mano. Se exploran dos clasificadores: Random Forest y SGDClassifier, tanto con como sin reducción de dimensionalidad utilizando PCA (Análisis de Componentes Principales).

### 1. Conjunto de Datos MNIST

El **conjunto de datos MNIST** es un recurso fundamental en el campo del aprendizaje automático y la visión por computadora. Contiene **60,000 imágenes de entrenamiento** y **10,000 imágenes de prueba**, todas ellas representando dígitos escritos a mano del 0 al 9. Cada imagen tiene una resolución de **28x28 píxeles**. El MNIST se utiliza ampliamente como punto de partida para tareas de clasificación de imágenes debido a su simplicidad y relevancia .

![MNIST Dataset](https://upload.wikimedia.org/wikipedia/commons/2/27/MnistExamples.png)

### 2. Random Forest

El **Random Forest** es un algoritmo de aprendizaje supervisado que combina múltiples árboles de decisión para obtener una predicción más robusta y precisa. Cada árbol en el bosque vota por la clase de destino, y la clase con más votos se selecciona como la predicción final. Los Random Forests son especialmente útiles para manejar características no lineales y evitar el sobreajuste. Su versatilidad los hace adecuados para problemas de clasificación y regresión .

![Random Forest](https://upload.wikimedia.org/wikipedia/commons/7/76/Random_forest_diagram_complete.png)

### 3. Análisis de Componentes Principales (PCA)

El **PCA** es una técnica de reducción de dimensionalidad que busca transformar las características originales en un nuevo conjunto de características llamadas **componentes principales**. Estos componentes capturan la mayor parte de la varianza en los datos. Al mantener el 95% de la varianza explicada, PCA reduce la complejidad del modelo y mejora la eficiencia computacional sin sacrificar significativamente el rendimiento. Es especialmente útil cuando se trabaja con datos de alta dimensionalidad .

![PCA](https://upload.wikimedia.org/wikipedia/commons/thumb/f/f5/GaussianScatterPCA.svg/800px-GaussianScatterPCA.svg.png)

### 4. Clasificador de Descenso de Gradiente Estocástico (SGDClassifier)

El **SGDClassifier** es un clasificador lineal que utiliza el descenso de gradiente estocástico para optimizar la función de pérdida. A diferencia de los métodos tradicionales que calculan gradientes en todo el conjunto de datos, el SGD actualiza los pesos del modelo de manera incremental. Esto lo hace especialmente útil para grandes conjuntos de datos. Se utiliza ampliamente en tareas de clasificación binaria y multiclase .

![SGDClassifier](https://upload.wikimedia.org/wikipedia/commons/thumb/e/e5/Gradient_descent.jpg/800px-Gradient_descent.jpg)

En resumen, tu proyecto implica explorar y evaluar diferentes algoritmos de clasificación, desde Random Forest hasta SGDClassifier, aplicando técnicas como PCA para mejorar la eficiencia y el rendimiento.

## Descripción del Código

El código está escrito en Python y realiza las siguientes acciones:

1. Carga el conjunto de datos MNIST, que consta de imágenes de dígitos escritos a mano y sus etiquetas correspondientes.
2. Divide el conjunto de datos en conjuntos de entrenamiento y prueba.
3. Entrena un clasificador Random Forest sin aplicar PCA y evalúa su rendimiento.
4. Aplica PCA para reducir la dimensionalidad de los datos manteniendo el 95% de la varianza explicada.
5. Entrena un clasificador Random Forest con los datos reducidos por PCA y evalúa su rendimiento.
6. Experimenta con el clasificador SGDClassifier sin PCA y evalúa su rendimiento.
7. Entrena el clasificador SGDClassifier con los datos reducidos por PCA y evalúa su rendimiento.

El código calcula métricas de evaluación como precisión, recall y puntuación F1 para cada clasificador tanto con como sin PCA, y también registra el tiempo de entrenamiento para comparar la eficiencia de los métodos.

## Código más a fondo

Este código es un ejemplo de cómo aplicar el algoritmo de Bosques Aleatorios (Random Forest) y el algoritmo SGDClassifier para clasificar el conjunto de datos MNIST, que contiene imágenes de dígitos escritos a mano. Aquí está el análisis paso a paso:

1. **Instalación de bibliotecas**: Se incluyen algunos comandos de instalación de paquetes usando pip, pero estos son comentarios (líneas que comienzan con '#') y no se ejecutarán si este código se ejecuta como un script normal. Estos comentarios parecen estar destinados a ser ejecutados en un entorno Jupyter Notebook, donde los signos '%' se usan para ejecutar comandos de shell o mágicos.

2. **Importación de bibliotecas necesarias**: Se importan las bibliotecas necesarias para cargar el conjunto de datos, entrenar modelos, aplicar PCA (Análisis de Componentes Principales) y evaluar el rendimiento de los modelos.

3. **Carga del conjunto de datos MNIST**: Utiliza la función `fetch_openml` de Scikit-learn para car☺gar el conjunto de datos MNIST, que consiste en imágenes de 28x28 píxeles de dígitos escritos a mano y sus respectivas etiquetas.

4. **División en conjunto de entrenamiento y prueba**: Divide el conjunto de datos en conjuntos de entrenamiento y prueba usando la función `train_test_split` de Scikit-learn. Se utiliza un tamaño de prueba de 10000 muestras y una semilla aleatoria para garantizar la reproducibilidad.

5. **Entrenamiento de modelos sin PCA (Random Forest y SGDClassifier)**: Se entrenan dos modelos de clasificación, uno usando el algoritmo de Bosques Aleatorios (`RandomForestClassifier`) y otro usando el algoritmo SGDClassifier (`SGDClassifier`). Ambos modelos se entrenan con los datos sin procesar (sin aplicar PCA).

6. **Evaluación del rendimiento sin PCA**: Se evalúa el rendimiento de los modelos entrenados sin PCA utilizando métricas como precisión, recuperación (recall) y puntuación F1.

7. **Aplicación de PCA**: Se aplica PCA para reducir la dimensionalidad de los datos. El PCA se ajusta al conjunto de entrenamiento y luego transforma tanto el conjunto de entrenamiento como el conjunto de prueba. Se establece un umbral de varianza explicada del 95%.

8. **Entrenamiento de modelos con PCA (Random Forest y SGDClassifier)**: Se entrenan los mismos modelos que antes, pero esta vez se utilizan los datos transformados por PCA.

9. **Evaluación del rendimiento con PCA**: Se evalúa el rendimiento de los modelos entrenados con PCA utilizando las mismas métricas que antes.

10. **Imprimir resultados**: Se imprimen los tiempos de entrenamiento y las métricas de rendimiento para los modelos entrenados con y sin PCA.

En resumen, este código carga el conjunto de datos MNIST, entrena modelos de clasificación usando Bosques Aleatorios y SGDClassifier tanto con datos sin procesar como con datos transformados por PCA, y evalúa el rendimiento de los modelos en términos de precisión, recuperación y puntuación F1.

## Ejecución del Código en un Entorno Virtual

Para ejecutar el código en un entorno virtual con `env`, sigue estos pasos:

1. Clona este repositorio en tu máquina local:

   ```
   git clone https://github.com/lruizap/mnist-classifier-analysis.git
   ```

2. Navega hasta el directorio del repositorio clonado:

   ```
   cd nombre_del_repositorio
   ```

3. Crea un entorno virtual con `env`:

   ```
   python -m venv env
   ```

4. Activa el entorno virtual:

   - En Windows:

     ```
     .\env\Scripts\activate
     ```

   - En macOS y Linux:

     ```
     source env/bin/activate
     ```

5. Instala las dependencias necesarias utilizando pip:

   ```
   pip install -r requirements.txt
   ```

6. Ejecuta el script Python:

   ```
   python main.py
   ```

7. Observa la salida en la consola, que mostrará el rendimiento de los clasificadores y los tiempos de entrenamiento.

¡Y eso es todo! Ahora puedes analizar los resultados y experimentar con diferentes configuraciones de clasificadores y técnicas de reducción de dimensionalidad dentro de un entorno virtual limpio y aislado.

## Webgrafía

1. **TensorFlow Datasets**:

   - **Descripción**: El **conjunto de datos MNIST** contiene imágenes de dígitos escritos a mano y se utiliza ampliamente en tareas de clasificación de imágenes.
   - **Fuente**: [MNIST en TensorFlow Datasets](https://www.tensorflow.org/datasets/catalog/mnist).

2. **Hugging Face Datasets**:

   - **Descripción**: El MNIST consta de **70,000 imágenes en blanco y negro de 28x28 píxeles** de dígitos escritos a mano. Hay 60,000 imágenes en el conjunto de entrenamiento y 10,000 en el conjunto de validación, con una clase por dígito (un total de 10 clases).
   - **Fuente**: [MNIST en Hugging Face Datasets](https://huggingface.co/datasets/mnist).

3. **Torchvision**:

   - **Descripción**: El MNIST es un conjunto de datos de imágenes de dígitos escritos a mano.
   - **Fuente**: [Documentación de Torchvision](https://pytorch.org/vision/main/generated/torchvision.datasets.MNIST.html).

4. **Machine Learning Datasets**:

   - **Descripción**: El conjunto de entrenamiento del MNIST está compuesto por 30,000 patrones de SD-3 y 30,000 patrones de SD-1. El conjunto de prueba consta de 5,000 patrones de SD-3 y 5,000 patrones de SD-1.
   - **Fuente**: [MNIST en Machine Learning Datasets](https://datasets.activeloop.ai/docs/ml/datasets/mnist/).

5. **Papers With Code**:
   - **Descripción**: El MNIST es una gran colección de dígitos escritos a mano con un conjunto de entrenamiento de 60,000 ejemplos y un conjunto de prueba de 10,000 ejemplos.
   - **Fuente**: [MNIST en Papers With Code](https://paperswithcode.com/dataset/mnist).
