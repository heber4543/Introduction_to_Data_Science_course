# Práctica 3. Predicción de la deserción estudiantil

Clasificador que predice, al cierre del primer semestre, la situación que tendrá un estudiante al final de la duración normal de su carrera (graduado, desertor o todavía inscrito). Se comparan cinco clasificadores optimizados con validación cruzada (árbol de decisión, Naive Bayes, kNN, SVM y regresión logística), y el mejor genera las predicciones del conjunto de prueba.

Práctica de la materia Introducción a la Ciencia de Datos, Posgrado en Ciencias de la Computación, CICESE.

## Dataset

Datos reales de 4424 estudiantes de licenciatura del Instituto Politécnico de Portalegre (Portugal) que ingresaron entre los ciclos 2008/09 y 2018/19. Cada estudiante tiene 36 atributos demográficos, socioeconómicos, macroeconómicos y de desempeño académico en el primer y segundo semestre, y la clase `Target` (Graduate, Dropout o Enrolled).

- `datos/desercion-entrenamiento.csv` tiene 3539 estudiantes con sus 36 atributos y la clase.
- `datos/desercion-prueba.csv` tiene 885 estudiantes sin la clase y solo con los atributos conocidos al cierre del primer semestre.

Fuente. Realinho, V., Vieira Martins, M., Machado, J. y Baptista, L. (2021). *Predict Students' Dropout and Academic Success* [Dataset]. UCI Machine Learning Repository. https://archive.ics.uci.edu/dataset/697 (licencia CC BY 4.0).

## Contenido del repositorio

| Archivo | Descripción |
|---|---|
| `practica3.ipynb` | Notebook con el análisis completo |
| `predicciones.csv` | Predicciones del clasificador final para los 885 estudiantes de prueba (columnas `id` y `Target`) |
| `datos/` | Conjuntos de entrenamiento y prueba |

## Requisitos

Python 3.13 con las siguientes librerías (versiones con las que se ejecutó).

| Librería | Versión |
|---|---|
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| scikit-learn | 1.6.1 |

## Uso

1. Clonar el repositorio y entrar a su carpeta.
2. Abrir `practica3.ipynb` desde esa carpeta (Jupyter, VS Code o Colab), para que las rutas relativas a `datos/` funcionen.
3. Ejecutar todo desde el inicio (*Restart & Run All*). El notebook genera `predicciones.csv` en la misma carpeta.

En Colab, si no se clona el repositorio, basta con comentar las líneas de carga con ruta relativa y descomentar las que leen los datos desde la URL.

La búsqueda de hiperparámetros del SVM es la más lenta (alrededor de 11 minutos en Colab); la ejecución completa toma unos 20 minutos.

## Resultado

El mejor clasificador fue el SVM con kernel RBF (C = 30, gamma = 0.003, pesos balanceados por clase), con F1 macro de validación cruzada de 0.688 ± 0.006.
