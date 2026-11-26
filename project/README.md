# Predicción de parámetros del modelo RLWM con una red neuronal artificial

Código del proyecto final de la materia Introducción a la Ciencia de Datos (CICESE).

Se simularon 713,169,765 agentes del modelo de aprendizaje por refuerzo y memoria de trabajo (RLWM) en la versión de 90 ensayos de la tarea. De cada agente se calcularon 59 métricas conductuales y con ellas se entrenó un perceptrón multicapa para predecir los parámetros α, ϕ, ρ, ε y K. Después se evaluó si los parámetros predichos reproducen la conducta igual que los verdaderos (soluciones múltiples), en agentes simulados y en 37 participantes.

Autor. Heber Abraham Zapata Robles, CICESE.

## Datos

Los datos no se incluyen en este repositorio porque son bastante pesados (solo las métricas de entrenamiento ocupan unos 90 GB en memoria). Se pueden solicitar al autor y se hacen llegar.

Los datos de los participantes están seudonimizados y se recolectaron bajo el protocolo SIPCEINV/122/2025.

Contacto. hzapata@cicese.edu.mx

## Estructura

| Archivo | Qué hace |
|---|---|
| `rlwm_sim/model.py` | Ecuaciones del modelo RLWM. La simulación y la verosimilitud usan el mismo código |
| `rlwm_sim/environment.py` | Diseño de la tarea de la fase 3 (bloques, estímulos y respuestas correctas) |
| `rlwm_sim/random_env.py` | Genera la secuencia de estímulos propia de cada agente, lo simula y calcula sus métricas |
| `rlwm_sim/grid.py` | Rejilla de parámetros y conversión del número de agente a sus parámetros |
| `rlwm_sim/features.py` | Definición y cálculo de las 59 métricas conductuales |
| `rlwm_sim/runio.py` | Configuraciones, rutas, manifiesto y huella del código de cada corrida |
| `rlwm_sim/prepare_run.py` | Paso 1. Crea la corrida y su manifiesto |
| `rlwm_sim/run_shard_random.py` | Paso 2. Simula un fragmento de la rejilla y guarda sus métricas |
| `rlwm_sim/train_ann.py` | Paso 3. Selección de características con group lasso y entrenamiento de la red |
| `rlwm_sim/prep_soluciones.py` | Paso 4a. Exporta la red a NumPy y prepara los datos de los participantes |
| `rlwm_sim/eval_soluciones.py` | Paso 4b. Evaluación de soluciones múltiples en participantes y en agentes simulados |
| `rlwm_sim/muestra_metricas.py` | Paso 5. Métricas de 200,000 agentes para el análisis de redundancia |
| `configs/` | Configuración de cada paso (rejilla, semillas, red y evaluación) |
| `notebooks/graficas_rlwm_ann.ipynb` | Figuras y tablas del reporte (Google Colab) |

## Requisitos

Python 3.12 con NumPy, Numba, pandas, pyarrow y PyTorch. Las figuras se hicieron en Google Colab.

## Orden de ejecución

Todo se ejecutó en el clúster Ixachi del CICESE, en este orden y desde la carpeta del repositorio. La carpeta de resultados se define en `configs/fase3_rejilla001_secaleatoria.json`.

```bash
# 1. Preparar la corrida
python -m rlwm_sim.prepare_run configs/fase3_rejilla001_secaleatoria.json

# 2. Simular los agentes y calcular sus métricas, un fragmento a la vez (0 a 199)
python -m rlwm_sim.run_shard_random configs/fase3_rejilla001_secaleatoria.json --shard 0

# 3. Selección de características y entrenamiento de la red
python -m rlwm_sim.train_ann configs/ann_v2.json

# 4. Soluciones múltiples (participantes y agentes simulados)
python -m rlwm_sim.prep_soluciones configs/eval_soluciones.json
python -m rlwm_sim.eval_soluciones configs/eval_soluciones.json

# 5. Muestra de métricas para el análisis de redundancia
python -m rlwm_sim.muestra_metricas configs/eval_soluciones.json
```

La parte de participantes del paso 4 necesita sus datos, que no están en el repositorio. Para correr solo la parte de agentes simulados se agrega `--parte simulados`.

Las semillas están en `configs/` (plan de la tarea 20261125, simulación 20261126, división de los datos 2026, evaluación 20261127 y 20261128, y muestra de métricas 20261129), por lo que cada paso se puede reproducir.

## Figuras

El notebook lee los resultados de los pasos 3 a 5 desde Google Drive. Para usarlo, copia esos archivos a tu Drive y cambia la variable `CARPETA` de la primera celda por tu carpeta.

## Decisiones del modelo

- Política. w = ρ·min(1, K/TC) y π = (1 − ε)(w·π_WM + (1 − w)·π_RL) + ε/3, con β = 17 fijo y sin sesgo de aprendizaje.
- Rejilla. α, ϕ y ρ de 0.01 a 0.99, ε de 0.01 a 0.49, K de 2 a 4, con pasos de 0.01 y 5 réplicas por combinación.
- Cada agente recibe su propia secuencia de estímulos, generada con las reglas de la fase 3, y su azar depende solo de la semilla, del agente y del ensayo.
- División 60/20/20 por combinación de parámetros, para que las 5 réplicas de una combinación queden en el mismo conjunto.
