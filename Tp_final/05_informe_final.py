# Databricks notebook source
# MAGIC %md
# MAGIC # TP Final - Big Data & MLOps
# MAGIC ## Informe Final del Proyecto
# MAGIC
# MAGIC Este notebook consolida toda la documentación pedida por la consigna del
# MAGIC TP. Los notebooks de código (`01` a `04`) contienen la implementación;
# MAGIC este es el resumen narrativo para la evaluación y la presentación.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Descripción del dataset y su adecuación al curso
# MAGIC
# MAGIC **Dataset:** Clickstream de navegación tipo Netflix (`All_Clickstream.csv`),
# MAGIC 12.104 eventos de click de 9 perfiles de usuario, con columnas de perfil,
# MAGIC fuente de tráfico, tipo de acción de navegación (`Navigation Level`), URL
# MAGIC y timestamp.
# MAGIC
# MAGIC **Adecuación al curso:** si bien el volumen crudo (12k filas) es
# MAGIC modesto, el proyecto está diseñado con un pipeline pensado para escalar:
# MAGIC todo el procesamiento (parseo de timestamps, detección de sesiones por
# MAGIC usuario mediante funciones de ventana, agregación de features) se
# MAGIC implementa en **PySpark sobre Databricks**, con una arquitectura que
# MAGIC funcionaría igual sobre un dataset de clickstream de millones de
# MAGIC eventos sin cambiar una línea de lógica (solo escalando el clúster).
# MAGIC Se trata el dataset como un caso de "small big data" con pipeline
# MAGIC escalable.
# MAGIC
# MAGIC **Problema definido:** a partir de un dataset de eventos crudo sin
# MAGIC target, se construyó un problema de **clasificación binaria**:
# MAGIC predecir si una sesión de navegación terminará en reproducción de
# MAGIC contenido (`had_playback`), usando solamente el comportamiento previo
# MAGIC del usuario dentro de esa misma sesión (evitando data leakage: las
# MAGIC features excluyen explícitamente los eventos posteriores al primer
# MAGIC playback).
# MAGIC
# MAGIC Resultado de la ingeniería de features (ver `01_feature_engineering_pyspark`):
# MAGIC **975 sesiones** utilizables, con un balance de clases razonable
# MAGIC (~57% con playback / ~43% sin playback).

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Experimentos realizados (hiperparámetros, features probadas)
# MAGIC
# MAGIC Todos los experimentos quedan registrados en MLflow Tracking, bajo el
# MAGIC experimento `/Shared/tp_bigdata_clickstream` (ver `02_modelado_mlflow`):
# MAGIC
# MAGIC | Run | Descripción |
# MAGIC |---|---|
# MAGIC | `baseline_rf` | RandomForestClassifier con hiperparámetros por defecto de sklearn |
# MAGIC | `optuna_trial_0` a `optuna_trial_7` | 8 corridas de búsqueda de hiperparámetros con Optuna (nested runs), variando `n_estimators` (50-300), `max_depth` (3-15), `min_samples_split` (2-10), `min_samples_leaf` (1-8) |
# MAGIC | `final_best_model` | Reentrenamiento con los mejores hiperparámetros encontrados por Optuna |
# MAGIC | `evidently_monitoring_report` | Run adicional donde se registra el reporte de Evidently como artefacto |
# MAGIC
# MAGIC Esto supera de sobra el mínimo de 4 corridas pedido por la consigna.
# MAGIC
# MAGIC **Features utilizadas:** `n_clicks`, `n_distinct_actions`, `n_distinct_urls`,
# MAGIC `n_search`, `n_movieDetails`, `n_browseTitles`, `n_browseGallery`,
# MAGIC `n_postPlay`, `n_profilesGate`, `session_duration_sec`, `hour_of_day`,
# MAGIC `day_of_week`, y el one-hot encoding de `source`.
# MAGIC
# MAGIC **Split:** temporal (no aleatorio) — 80% de las sesiones más antiguas
# MAGIC para entrenamiento, 20% más recientes para test. Esta decisión se tomó
# MAGIC para simular un escenario realista de producción y para que el análisis
# MAGIC de drift de la sección 5 tenga sentido (comparamos pasado vs. futuro,
# MAGIC no una partición aleatoria arbitraria).
# MAGIC
# MAGIC *(Completar acá con los valores exactos de `study.best_params` que te
# MAGIC imprimió el notebook `02_modelado_mlflow` al correrlo)*

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Métricas comparativas
# MAGIC
# MAGIC
# MAGIC | Run | Accuracy | Precision | Recall | F1 | ROC-AUC |
# MAGIC |---|---|---|---|---|---|
# MAGIC | baseline_rf | 0.6615384615384615 | 0.7033898305084746 | 0.7280701754385965 | 0.7155172413793104 | 0.7497292614251678 |
# MAGIC | final_best_model (Optuna) | 0.7589743589743589 | 0.845360824742268 | 0.7192982456140351 | 0.7772511848341233 | 0.8398310591293048* |
# MAGIC
# MAGIC Estos mismos números del modelo final coinciden (o deberían ser muy
# MAGIC similares) con los que reporta Evidently para el dataset "Current" en
# MAGIC la sección 5 de este informe.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Razones técnicas y prácticas para la elección del modelo final
# MAGIC
# MAGIC Se eligió el modelo **`final_best_model`** (RandomForest con
# MAGIC hiperparámetros optimizados por Optuna) por sobre el baseline, dado que:
# MAGIC
# MAGIC - **ROC-AUC** fue la métrica priorizada para la selección, porque el
# MAGIC   target está razonablemente balanceado (~57/43) y ROC-AUC resume bien
# MAGIC   la capacidad de discriminación del modelo en ambas clases sin
# MAGIC   depender de un umbral de corte fijo.
# MAGIC - RandomForest es interpretable vía SHAP (importante para entender
# MAGIC   qué comportamientos predicen mejor la conversión a reproducción,
# MAGIC   algo valioso desde el punto de vista de negocio) y es
# MAGIC   suficientemente robusto para un dataset de este tamaño sin
# MAGIC   necesidad de arquitecturas más complejas (ej. redes neuronales),
# MAGIC   que estarían sobredimensionadas para ~780 sesiones de entrenamiento.
# MAGIC - La búsqueda de Optuna permitió explorar el espacio de
# MAGIC   hiperparámetros de forma más sistemática que una búsqueda manual,
# MAGIC   documentando cada intento como run de MLflow para trazabilidad
# MAGIC   completa.
# MAGIC - Vemos como el modelo con los hiperparametros optmizados por Optuna mejora todas menos unas de las metricas de medicion del modelo, por ende, el modelo final es el que tiene los hiperparámetros optimizados.
# MAGIC

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Monitoreo de métricas y drift de los datos (Evidently)
# MAGIC
# MAGIC Se comparó el set de **referencia** (sesiones de entrenamiento, más
# MAGIC antiguas) contra el set **actual** (sesiones de test, más recientes),
# MAGIC simulando un escenario real de monitoreo post-despliegue
# MAGIC (ver `04_evidently_monitoring`, reporte completo en
# MAGIC `resultados_evidently.html`).
# MAGIC
# MAGIC **Data Drift:**
# MAGIC - 7 de 18 columnas mostraron drift estadístico (38.9%).
# MAGIC - Como el umbral de Evidently para declarar "Dataset Drift" es 50%,
# MAGIC   el veredicto global fue: **Dataset Drift NOT detected**.
# MAGIC - Conclusión: el comportamiento general de navegación se mantuvo
# MAGIC   razonablemente estable entre el período de entrenamiento y el de
# MAGIC   test, aunque hay features puntuales que cambiaron.
# MAGIC
# MAGIC **Performance del modelo (Reference vs. Current):**
# MAGIC
# MAGIC | Métrica | Reference (train) | Current (test) | Diferencia |
# MAGIC |---|---|---|---|
# MAGIC | Accuracy | 0.879 | 0.759 | -0.120 |
# MAGIC | Precision | 0.926 | 0.845 | -0.081 |
# MAGIC | Recall | 0.854 | 0.719 | -0.135 |
# MAGIC | F1 | 0.888 | 0.777 | -0.111 |
# MAGIC | ROC-AUC | 0.944 | 0.840 | -0.104 |
# MAGIC
# MAGIC **Conclusión importante:** no se detectó drift de datos fuerte, pero
# MAGIC sí una **degradación de performance de ~10-13 puntos** en todas las
# MAGIC métricas entre entrenamiento y test. Esto es un caso de **concept
# MAGIC drift** (la relación entre features y target cambia con el tiempo,
# MAGIC aunque la distribución de las features en sí no cambie tanto) más que
# MAGIC de data drift puro. En un escenario de producción real, esto sugiere:
# MAGIC
# MAGIC 1. Reentrenar el modelo periódicamente con una ventana de datos más
# MAGIC    reciente, en vez de asumir que un modelo entrenado una sola vez
# MAGIC    sigue siendo válido indefinidamente.
# MAGIC 2. Instrumentar una alerta automática si la performance en producción
# MAGIC    cae por debajo de un umbral (ej. ROC-AUC < 0.80).
# MAGIC 3. Investigar si hay variables adicionales (no capturadas en este
# MAGIC    dataset) que expliquen el cambio de comportamiento entre usuarios
# MAGIC    "viejos" y "nuevos".

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Instrucciones para reproducir el proyecto
# MAGIC
# MAGIC 1. **Prerrequisitos:** cuenta de Databricks (probado en Free
# MAGIC    Edition/Community), acceso a un volumen de Unity Catalog.
# MAGIC 2. **Subir el dataset:** cargar `All_Clickstream.csv` a
# MAGIC    `/Volumes/workspace/default/tp_bigdata/All_Clickstream.csv`
# MAGIC    (Catalog → Data Ingestion → Upload files to a volume).
# MAGIC 3. **Importar y correr en orden** (Workspace → Import → seleccionar
# MAGIC    cada `.py`, Databricks los reconoce automáticamente como notebooks):
# MAGIC    1. `01_feature_engineering_pyspark.py` → genera la tabla
# MAGIC       `clickstream_session_features`.
# MAGIC    2. `02_modelado_mlflow.py` → entrena, registra experimentos en
# MAGIC       MLflow, registra el modelo en el Model Registry, corre SHAP, y
# MAGIC       guarda `clickstream_reference_data` / `clickstream_current_data`.
# MAGIC    3. Crear el endpoint de Model Serving desde la UI de Databricks
# MAGIC       (Models → `clickstream_playback_predictor` → Serve this model),
# MAGIC       apuntando a la última versión registrada.
# MAGIC    4. `03_invoke_endpoint.py` → prueba el endpoint con ejemplos reales.
# MAGIC    5. `04_evidently_monitoring.py` → genera el reporte de drift y
# MAGIC       performance, y lo guarda como `resultados_evidently.html` dentro
# MAGIC       de esta misma carpeta.
# MAGIC 4. **Dependencias:** cada notebook instala lo que necesita vía `%pip`
# MAGIC    (`optuna`, `shap`, `evidently`) — no requiere instalación manual
# MAGIC    fuera de Databricks.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Conclusiones generales
# MAGIC
# MAGIC - Se construyó un pipeline integro (ingesta → PySpark →
# MAGIC   MLflow → Model Registry → Serving → Evidently) sobre un problema de
# MAGIC   negocio real: predecir si una sesión de navegación termina en
# MAGIC   reproducción de contenido.
# MAGIC - El modelo final (RandomForest + Optuna) logró un ROC-AUC de 0.84 en
# MAGIC   test, con buena capacidad de discriminación pese al tamaño acotado
# MAGIC   del dataset.
# MAGIC - El monitoreo con Evidently no detectó drift de datos fuerte, pero sí
# MAGIC   una degradación de performance relevante entre entrenamiento y test,
# MAGIC   lo que valida la importancia de tener observabilidad en producción
# MAGIC   más allá de solo mirar la distribución de las features de entrada.
