# TP Final — Herramientas para Grandes Volúmenes de Datos

Proyecto de MLOps end-to-end en Databricks: predicción de reproducción de
contenido (`playback`) a partir de datos de clickstream tipo Netflix.

## Problema

A partir de eventos crudos de navegación (`All_Clickstream.csv`), se
construyen **sesiones de usuario** y se entrena un modelo de clasificación
binaria que predice si una sesión terminará en reproducción de contenido,
usando solo el comportamiento previo dentro de esa misma sesión.

## Stack

- **PySpark** (Databricks) — ingesta, construcción de sesiones, feature engineering
- **MLflow Tracking + Model Registry** — experimentación y versionado del modelo
- **Optuna** — búsqueda de hiperparámetros
- **SHAP** — interpretabilidad del modelo
- **Databricks Model Serving** — endpoint de inferencia
- **Evidently** — monitoreo de data drift y performance del modelo

## Estructura del repo

| Archivo | Descripción |
|---|---|
| `01_feature_engineering_pyspark.py` | Ingesta del CSV, construcción de sesiones, features y target |
| `02_modelado_mlflow.py` | Entrenamiento, experimentos con Optuna, registro del modelo, SHAP |
| `03_invoke_endpoint.py` | Ejemplo de invocación del endpoint de Model Serving |
| `04_evidently_monitoring.py` | Reporte de data drift y performance del modelo |
| `05_informe_final.py` | Informe completo: dataset, experimentos, métricas, justificación, conclusiones |
| `resultados_evidently.html` | Reporte de Evidently exportado (generado por `04`) |

## Cómo reproducirlo

Ver la sección 6 ("Instrucciones para reproducir el proyecto") dentro de
`05_informe_final.py` para el paso a paso completo.

Resumen rápido:
1. Subir `All_Clickstream.csv` a un Volume de Databricks.
2. Importar y correr los notebooks en orden (`01` → `02` → crear endpoint desde la UI → `03` → `04`).
3. Ver conclusiones y justificación del modelo en `05_informe_final.py`.

## Dataset

Clickstream de navegación (~12.100 eventos, 9 perfiles de usuario),
transformado en ~975 sesiones de usuario mediante ingeniería de features
en PySpark. Ver la sección 1 de `05_informe_final.py` para el detalle
completo y la justificación de su adecuación al curso.
