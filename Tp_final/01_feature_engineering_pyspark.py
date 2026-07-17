# Databricks notebook source
# MAGIC %md
# MAGIC # TP Final - Big Data & MLOps
# MAGIC ## Fase 1: Ingesta y Feature Engineering con PySpark
# MAGIC
# MAGIC **Dataset:** Clickstream tipo Netflix (`All_Clickstream.csv`)
# MAGIC **Objetivo del problema:** predecir si una sesión de navegación terminará en
# MAGIC reproducción de contenido (`playback`) a partir del comportamiento previo
# MAGIC del usuario dentro de esa misma sesión.
# MAGIC
# MAGIC **Por qué esta definición de target:**
# MAGIC - Es un problema de negocio real (¿este patrón de navegación va a convertir en reproducción?).
# MAGIC - Se puede resolver con clasificación binaria, en línea con el notebook de referencia (RandomForest).
# MAGIC - Evita data leakage: las features solo usan eventos ANTERIORES al primer `playback` de la sesión.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1 Carga del dataset
# MAGIC Subir `All_Clickstream.csv` a un Volume o a DBFS y ajustar el `path` de abajo.

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql.window import Window

path = "/Volumes/workspace/default/tp_bigdata/All_Clickstream.csv"

df = (
    spark.read
    .option("header", True)
    .option("inferSchema", True)
    .csv(path)
)

print(f"Filas: {df.count()} | Columnas: {len(df.columns)}")
df.printSchema()
display(df.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.1.b Renombrar columnas
# MAGIC Delta (el formato de tabla de Databricks) no permite espacios en los
# MAGIC nombres de columna. El CSV original trae columnas como `"Profile Name"`,
# MAGIC así que las renombramos a snake_case/underscore antes de seguir.

# COMMAND ----------

rename_map = {
    "Profile Name": "Profile_Name",
    "Navigation Level": "Navigation_Level",
    "Referrer Url": "Referrer_Url",
    "Webpage Url": "Webpage_Url",
    "Click Utc Ts": "Click_Utc_Ts",
}
for old_name, new_name in rename_map.items():
    if old_name in df.columns:
        df = df.withColumnRenamed(old_name, new_name)

df.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.2 Limpieza básica
# MAGIC - `Referrer Url` está vacía en el 100% de los registros -> se descarta.
# MAGIC - Se parsea el timestamp y se descartan filas sin timestamp válido.

# COMMAND ----------

df = df.drop("Referrer_Url")

df = df.withColumn("Click_Ts", F.to_timestamp(F.col("Click_Utc_Ts")))
n_before = df.count()
df = df.filter(F.col("Click_Ts").isNotNull())
n_after = df.count()
print(f"Filas descartadas por timestamp inválido: {n_before - n_after}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.3 Construcción de sesiones
# MAGIC Se define una nueva sesión por usuario cuando pasan más de 30 minutos
# MAGIC entre dos clicks consecutivos. Este umbral es un parámetro configurable
# MAGIC (documentar en el informe la justificación / probar sensibilidad si da tiempo).

# COMMAND ----------

SESSION_GAP_MINUTES = 30

window_user = Window.partitionBy("Profile_Name").orderBy("Click_Ts")

df = df.withColumn("prev_ts", F.lag("Click_Ts").over(window_user))
df = df.withColumn(
    "gap_minutes",
    (F.col("Click_Ts").cast("long") - F.col("prev_ts").cast("long")) / 60.0
)
df = df.withColumn(
    "new_session",
    F.when(F.col("prev_ts").isNull(), F.lit(1))
     .when(F.col("gap_minutes") > SESSION_GAP_MINUTES, F.lit(1))
     .otherwise(F.lit(0))
)
df = df.withColumn("session_idx", F.sum("new_session").over(window_user))
df = df.withColumn(
    "session_id",
    F.concat_ws("_", F.col("Profile_Name"), F.col("session_idx"))
)

n_sessions = df.select("session_id").distinct().count()
print(f"Sesiones detectadas: {n_sessions}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.4 Definición del target y prevención de data leakage
# MAGIC Para cada sesión se busca el timestamp del primer `playback`.
# MAGIC El **target** (`had_playback`) es a nivel sesión.
# MAGIC Las **features** se calculan usando SOLO los eventos anteriores a ese
# MAGIC primer playback (si no hubo playback, se usan todos los eventos de la sesión).

# COMMAND ----------

first_playback = (
    df.filter(F.col("Navigation_Level") == "playback")
      .groupBy("session_id")
      .agg(F.min("Click_Ts").alias("first_playback_ts"))
)

df = df.join(first_playback, on="session_id", how="left")
df = df.withColumn("had_playback", F.col("first_playback_ts").isNotNull())

pre_playback = df.filter(
    F.col("first_playback_ts").isNull() | (F.col("Click_Ts") < F.col("first_playback_ts"))
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.5 Agregación de features por sesión

# COMMAND ----------

session_features = (
    pre_playback.groupBy("session_id", "Profile_Name")
    .agg(
        F.count("*").alias("n_clicks"),
        F.countDistinct("Navigation_Level").alias("n_distinct_actions"),
        F.countDistinct("Webpage_Url").alias("n_distinct_urls"),
        F.sum(F.when(F.col("Navigation_Level") == "search", 1).otherwise(0)).alias("n_search"),
        F.sum(F.when(F.col("Navigation_Level") == "movieDetails", 1).otherwise(0)).alias("n_movieDetails"),
        F.sum(F.when(F.col("Navigation_Level") == "browseTitles", 1).otherwise(0)).alias("n_browseTitles"),
        F.sum(F.when(F.col("Navigation_Level") == "browseTitlesGallery", 1).otherwise(0)).alias("n_browseGallery"),
        F.sum(F.when(F.col("Navigation_Level") == "postPlay", 1).otherwise(0)).alias("n_postPlay"),
        F.sum(F.when(F.col("Navigation_Level") == "profilesGate", 1).otherwise(0)).alias("n_profilesGate"),
        F.min("Click_Ts").alias("session_start"),
        F.max("Click_Ts").alias("session_end"),
        F.first("Source").alias("source"),
    )
)

session_features = session_features.withColumn(
    "session_duration_sec",
    F.col("session_end").cast("long") - F.col("session_start").cast("long")
)
session_features = session_features.withColumn("hour_of_day", F.hour("session_start"))
session_features = session_features.withColumn("day_of_week", F.dayofweek("session_start"))

# Filtrar sesiones sin ningún evento pre-playback (no aportan información predictiva)
session_features = session_features.filter(F.col("n_clicks") > 0)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.6 Unión con el target final

# COMMAND ----------

target = df.select("session_id", "had_playback").dropDuplicates(["session_id"])

final_df = session_features.join(target, on="session_id", how="left")
final_df = final_df.withColumn("had_playback", F.col("had_playback").cast("int"))

print("Distribución del target:")
final_df.groupBy("had_playback").count().show()

display(final_df)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 1.7 Guardar como tabla Delta
# MAGIC Esta tabla es el punto de partida para la Fase 2 (modelado + MLflow).

# COMMAND ----------

final_df.write.mode("overwrite").saveAsTable("clickstream_session_features")

print("Tabla 'clickstream_session_features' guardada correctamente.")
print(f"Total de sesiones utilizables: {final_df.count()}")
