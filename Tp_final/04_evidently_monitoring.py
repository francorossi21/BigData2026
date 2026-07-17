# Databricks notebook source
# MAGIC %md
# MAGIC # TP Final - Big Data & MLOps
# MAGIC ## Fase 4: Monitoreo con Evidently (Data Drift + Performance)
# MAGIC
# MAGIC Compara las sesiones de **entrenamiento** (`clickstream_reference_data`,
# MAGIC las más viejas en el tiempo) contra las sesiones de **test**
# MAGIC (`clickstream_current_data`, las más recientes), simulando un escenario
# MAGIC real de monitoreo en producción.

# COMMAND ----------

# MAGIC %pip install --upgrade pydantic
# MAGIC %pip uninstall -y evidently
# MAGIC %pip install evidently
# MAGIC %restart_python

# COMMAND ----------

import numpy as np
# Restaurar el alias eliminado para compatibilidad con versiones nuevas de numpy
np.float_ = np.float64

import pandas as pd
from evidently import Report
from evidently.presets import DataDriftPreset, ClassificationPreset
from evidently.future.datasets import Dataset, DataDefinition
from evidently import MulticlassClassification

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.1 Leer las tablas de referencia y actual

# COMMAND ----------

data_reference = spark.table("clickstream_reference_data").toPandas()
data_current = spark.table("clickstream_current_data").toPandas()

print(f"Referencia (train): {data_reference.shape[0]} filas")
print(f"Actual (test): {data_current.shape[0]} filas")
display(data_reference.head())

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.2 Definir el esquema para Evidently
# MAGIC Le indicamos cuál es la columna target, cuál la predicción, y cuál la
# MAGIC probabilidad predicha (ya vienen calculadas desde la Fase 2).

# COMMAND ----------

data_definition = DataDefinition(
    classification=[MulticlassClassification(
        target="target",
        prediction_labels="prediction",
        prediction_probas="proba",
    )]
)

reference_dataset = Dataset.from_pandas(data_reference, data_definition=data_definition)
current_dataset = Dataset.from_pandas(data_current, data_definition=data_definition)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.3 Generar el reporte (Data Drift + Performance del modelo)

# COMMAND ----------

report = Report(metrics=[DataDriftPreset(), ClassificationPreset()])

snapshot = report.run(
    current_data=current_dataset,
    reference_data=reference_dataset,
)

snapshot.save_html("/tmp/resultados_evidently.html")

import os
file_size = os.path.getsize("/tmp/resultados_evidently.html")
print(f"Reporte guardado. Tamaño del archivo: {file_size} bytes")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4 Ver el reporte
# MAGIC Databricks puede mostrar HTML directo dentro del notebook.

# COMMAND ----------

with open("/tmp/resultados_evidently.html", "r") as f:
    html_content = f.read()

print(f"Largo del HTML leído: {len(html_content)} caracteres")
displayHTML(html_content)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.4.b Guardar el HTML dentro de la carpeta del TP en Workspace
# MAGIC `/tmp` es una carpeta temporal del clúster: se borra y no queda en tu
# MAGIC repo. Acá lo guardamos directo en la misma carpeta donde está este
# MAGIC notebook (ej: `TP_Final`), para que quede como archivo real que podés
# MAGIC ver, descargar, y versionar en Git junto con el resto del proyecto.

# COMMAND ----------

notebook_path = (
    dbutils.notebook.entry_point.getDbutils().notebook().getContext().notebookPath().get()
)
folder_path = "/".join(notebook_path.split("/")[:-1])
output_path = f"/Workspace{folder_path}/resultados_evidently.html"

with open(output_path, "w") as f:
    f.write(html_content)

print(f"HTML guardado en: {output_path}")
print("Refrescá la carpeta en el explorador de Workspace (F5 o volver a entrar) para verlo listado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.5 Guardar el reporte como artefacto en MLflow
# MAGIC Así queda versionado junto con el resto del experimento, y es fácil de
# MAGIC encontrar/mostrar en la presentación final.

# COMMAND ----------

import mlflow

with mlflow.start_run(run_name="evidently_monitoring_report"):
    mlflow.log_artifact("/tmp/resultados_evidently.html")
    print("Reporte de Evidently registrado como artefacto en MLflow.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 4.6 Conclusiones (completar en el informe final)
# MAGIC
# MAGIC Revisar el reporte de arriba y completar:
# MAGIC
# MAGIC 1. **Data Drift**: ¿cuántas columnas driftearon? ¿Cuál es el `share
# MAGIC    drifted`? ¿Se detectó `Dataset Drift`? ¿Qué features cambiaron más
# MAGIC    (ej: subió `n_search`, bajó `n_postPlay`, etc.) y qué explicación de
# MAGIC    negocio le darías a ese cambio?
# MAGIC 2. **Performance del modelo**: comparar accuracy/precision/recall/F1/
# MAGIC    ROC-AUC entre `reference` y `current`. ¿El modelo mantiene su
# MAGIC    performance en datos más recientes o se degrada?
# MAGIC 3. **Acción recomendada**: si hubiera drift significativo o caída de
# MAGIC    performance, ¿qué harían? (ej: reentrenar el modelo periódicamente
# MAGIC    con una ventana de datos más reciente, agregar alertas automáticas
# MAGIC    de drift, etc.)
