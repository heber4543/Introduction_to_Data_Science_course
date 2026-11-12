# Práctica 4. Meta aprendizaje

Introducción a la Ciencia de Datos. Heber Abraham Zapata Robles.

El notebook `practica4.ipynb` mide cuánto ganan los ensambles (bagging, boosting, votación y stacking) sobre los cinco clasificadores base de la Práctica 3 en la deserción estudiantil, y después busca el mejor modelo para Bank Marketing.

## Conjuntos de datos

**Deserción estudiantil.** Archivo `desercion-entrenamiento.csv` del repositorio del curso, con 3,539 estudiantes y la variable objetivo `Target` en tres clases (Graduate, Dropout y Enrolled). Se usan los 29 atributos disponibles al cierre del primer semestre, sin `Nacionality` ni los atributos del segundo semestre, como en la Práctica 3. La evaluación usa 5 pliegues estratificados con `random_state=42`. El notebook lo lee directamente del repositorio del curso en GitHub; la línea con la ruta relativa `datos/desercion-entrenamiento.csv` está comentada en la misma celda.

**Bank Marketing.** Archivo `bank-additional-full.csv` del repositorio UCI (dataset 222), con 41,188 clientes contactados por una campaña de telemercadeo bancario. La variable objetivo `y` indica si el cliente contrató un depósito a plazo y se codifica como 1 (yes) y 0 (no). Se elimina la columna `duration`, así que quedan 19 atributos (10 categóricos y 9 numéricos). La evaluación usa `StratifiedKFold(n_splits=5, shuffle=True, random_state=2026)`. Si el archivo no está en la carpeta `datos/`, el notebook lo descarga de UCI.

## Requisitos

El notebook se ejecutó en Google Colab con Python 3.13 y estas versiones.

| Librería | Versión |
|---|---|
| numpy | 2.4.6 |
| pandas | 2.2.3 |
| matplotlib | 3.10.0 |
| seaborn | 0.13.2 |
| scikit-learn | 1.6.1 |
| lightgbm | 4.6.0 |
| catboost | 1.2.10 |
| joblib | 1.6.0 |

Para instalarlas fuera de Colab.

```bash
pip install numpy==2.4.6 pandas==2.2.3 matplotlib==3.10.0 seaborn==0.13.2 scikit-learn==1.6.1 lightgbm==4.6.0 catboost==1.2.10 joblib==1.6.0
```

En Colab basta con correr el notebook, porque la primera celda instala catboost; las demás librerías ya vienen instaladas.

## Uso

Abrir `practica4.ipynb` en Colab o Jupyter y ejecutar **Restart & Run All**. Las semillas están fijas, así que los resultados se repiten. La corrida completa tarda bastante, sobre todo por el stacking con el SVM calibrado y las búsquedas de hiperparámetros.

Al final, el notebook reentrena el mejor modelo de Bank Marketing (LightGBM optimizado con umbral ajustado) con los 41,188 clientes y lo guarda como `Heber_Zapata-lightgbm_optimizado_umbral_ajustado.joblib`. El archivo contiene el pipeline completo (preprocesamiento, modelo y umbral), así que recibe los datos tal como vienen en el CSV, sin `duration` ni `y`.

```python
import joblib
import pandas as pd

modelo = joblib.load("Heber_Zapata-lightgbm_optimizado_umbral_ajustado.joblib")
X = pd.read_csv("datos/bank-additional-full.csv", sep=";").drop(columns=["duration", "y"])
predicciones = modelo.predict(X)  # 1 = yes, 0 = no
```

## Archivos

| Archivo | Contenido |
|---|---|
| `practica4.ipynb` | Notebook de la práctica |
| `Heber_Zapata-lightgbm_optimizado_umbral_ajustado.joblib` | Mejor modelo de Bank Marketing |
| `README.md` | Este archivo |
