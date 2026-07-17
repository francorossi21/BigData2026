# Databricks notebook source
# MAGIC %md
# MAGIC # TP Final - Big Data & MLOps
# MAGIC ## Fase 3: Probar el endpoint de Model Serving
# MAGIC
# MAGIC Adaptado de `testinvokemodel.ipynb`: en vez de usar un token guardado en
# MAGIC un archivo y el dataset de cáncer de mama, usamos el token de la sesión
# MAGIC actual de Databricks (más simple y seguro) y un ejemplo real de nuestro
# MAGIC dataset de clickstream.
# MAGIC
# MAGIC **Antes de correr esto:** confirmá en la pestaña "Serving" que tu endpoint
# MAGIC `clickstream-playback-endpoint` esté en estado **Ready** (verde), no
# MAGIC "Creating" ni "Pending".

# COMMAND ----------

# MAGIC %pip install requests pandas
# MAGIC %restart_python

# COMMAND ----------

import requests
import pandas as pd
import json

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Obtener el token y la URL del workspace automáticamente
# MAGIC No hace falta crear un archivo de token a mano: Databricks nos da el
# MAGIC token de la sesión actual del notebook.

# COMMAND ----------

ctx = dbutils.notebook.entry_point.getDbutils().notebook().getContext()
token = ctx.apiToken().get()
workspace_url = ctx.apiUrl().get()

# TODO: confirmar que este es el nombre exacto que le pusiste al endpoint
ENDPOINT_NAME = "clickstream-playback-endpoint"

url = f"{workspace_url}/serving-endpoints/{ENDPOINT_NAME}/invocations"
print("URL del endpoint:", url)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Función para invocar el endpoint
# MAGIC Mismo patrón que `testinvokemodel.ipynb` (función `score_model`), solo
# MAGIC que la URL y el token ahora se arman automáticamente arriba.

# COMMAND ----------

def score_model(dataset: pd.DataFrame):
    headers = {
        "Authorization": f"Bearer {token}",
        "Content-Type": "application/json",
    }
    ds_dict = {"dataframe_split": dataset.to_dict(orient="split")}
    data_json = json.dumps(ds_dict, allow_nan=True)
    response = requests.post(url=url, headers=headers, data=data_json)
    if response.status_code != 200:
        raise Exception(f"Request failed with status {response.status_code}, {response.text}")
    return response.json()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.3 Traer un par de sesiones reales de ejemplo
# MAGIC Usamos la tabla `clickstream_current_data` que guardamos en la Fase 2
# MAGIC (son las sesiones de test, con sus features reales).

# COMMAND ----------

df_current = spark.table("clickstream_current_data").toPandas()

# Quitamos las columnas que NO son features de entrada (son el target y las
# predicciones que ya calculamos nosotros mismos en la Fase 2, no van al modelo)
cols_to_drop = [c for c in ["target", "prediction", "proba"] if c in df_current.columns]
sample = df_current.drop(columns=cols_to_drop).head(3)

print("Ejemplo de sesiones que le vamos a mandar al endpoint:")
display(sample)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.4 Invocar el endpoint

# COMMAND ----------

result = score_model(sample)
print(json.dumps(result, indent=2))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.5 Comparar predicción del endpoint vs. la real
# MAGIC Esto es lo que va en el informe como "ejemplo de uso del endpoint".

# COMMAND ----------

real_values = df_current.loc[sample.index, "target"].tolist()

comparison = pd.DataFrame({
    "prediccion_endpoint": result.get("predictions", result),
    "valor_real": real_values,
})
display(comparison)
