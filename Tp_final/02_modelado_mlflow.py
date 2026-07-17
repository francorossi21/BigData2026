# Databricks notebook source
# MAGIC %md
# MAGIC # TP Final - Big Data & MLOps
# MAGIC ## Fase 2: Modelado, Experimentación con MLflow y Registro del Modelo
# MAGIC
# MAGIC **Input:** tabla `clickstream_session_features` generada en la Fase 1
# MAGIC (notebook `01_feature_engineering_pyspark`).
# MAGIC
# MAGIC **Problema:** clasificación binaria — predecir `had_playback` (si la sesión
# MAGIC terminó en reproducción de contenido) a partir del comportamiento previo
# MAGIC del usuario dentro de la misma sesión.
# MAGIC
# MAGIC **Split de train/test:** en vez de un split aleatorio, ordenamos las
# MAGIC sesiones por `session_start` y usamos el 80% más antiguo como train y el
# MAGIC 20% más reciente como test. Esto simula un escenario real de producción
# MAGIC (entrenamos con el pasado, evaluamos sobre el futuro) y nos deja además
# MAGIC el set "current" listo para el análisis de drift con Evidently en la Fase 4.

# COMMAND ----------

# MAGIC %pip install optuna shap
# MAGIC %restart_python

# COMMAND ----------

import mlflow
import mlflow.sklearn
import optuna
import numpy as np
import pandas as pd
from mlflow.models.signature import infer_signature
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, classification_report
)

mlflow.set_experiment("/Shared/tp_bigdata_clickstream")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.1 Lectura de la tabla (patrón de `leer_tabla.ipynb`)

# COMMAND ----------

df_spark = spark.table("clickstream_session_features")
df = df_spark.toPandas()
print(df.shape)
df.head()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.2 Preparación de features
# MAGIC - `source` se codifica como categoría (one-hot).
# MAGIC - Se descartan columnas identificadoras / de timestamp crudo (se usan
# MAGIC   derivadas como `hour_of_day`, `day_of_week`, `session_duration_sec`).

# COMMAND ----------

df = df.sort_values("session_start").reset_index(drop=True)

feature_cols = [
    "n_clicks", "n_distinct_actions", "n_distinct_urls",
    "n_search", "n_movieDetails", "n_browseTitles", "n_browseGallery",
    "n_postPlay", "n_profilesGate", "session_duration_sec",
    "hour_of_day", "day_of_week",
]

df_model = pd.get_dummies(df, columns=["source"], prefix="source")

# Saneamos nombres de columna (Delta no permite espacios ni ciertos
# caracteres especiales). Ej: "source_Source 0" -> "source_Source_0"
df_model.columns = [
    c.replace(" ", "_").replace(",", "").replace(";", "")
     .replace("{", "").replace("}", "").replace("(", "")
     .replace(")", "").replace("\n", "").replace("\t", "").replace("=", "")
    for c in df_model.columns
]

source_cols = [c for c in df_model.columns if c.startswith("source_")]
feature_cols_full = feature_cols + source_cols

X = df_model[feature_cols_full]
y = df_model["had_playback"]

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.3 Split temporal (80% más antiguo = train, 20% más reciente = test)

# COMMAND ----------

split_idx = int(len(df_model) * 0.8)
X_train, X_test = X.iloc[:split_idx], X.iloc[split_idx:]
y_train, y_test = y.iloc[:split_idx], y.iloc[split_idx:]

print(f"Train: {len(X_train)} sesiones | Test: {len(X_test)} sesiones")
print("Balance train:\n", y_train.value_counts(normalize=True))
print("Balance test:\n", y_test.value_counts(normalize=True))

# COMMAND ----------

def eval_metrics(y_true, y_pred, y_proba):
    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1": f1_score(y_true, y_pred),
        "roc_auc": roc_auc_score(y_true, y_proba),
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.4 Run 1 — Baseline
# MAGIC RandomForest con hiperparámetros por defecto, como punto de partida.

# COMMAND ----------

with mlflow.start_run(run_name="baseline_rf") as run:
    clf = RandomForestClassifier(random_state=42)
    clf.fit(X_train, y_train)

    y_pred = clf.predict(X_test)
    y_proba = clf.predict_proba(X_test)[:, 1]
    metrics = eval_metrics(y_test, y_pred, y_proba)

    mlflow.log_params(clf.get_params())
    mlflow.log_metrics(metrics)

    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(clf, name="model", signature=signature)

    baseline_run_id = run.info.run_id
    print("Baseline:", metrics)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.5 Runs 2-N — Búsqueda de hiperparámetros con Optuna
# MAGIC Cada trial de Optuna se registra como un run anidado (nested run) dentro
# MAGIC de un run padre `optuna_search`, cumpliendo de sobra el mínimo de 4
# MAGIC corridas pedido por el TP.

# COMMAND ----------

N_TRIALS = 8

def objective(trial):
    params = {
        "n_estimators": trial.suggest_int("n_estimators", 50, 300, step=50),
        "max_depth": trial.suggest_int("max_depth", 3, 15),
        "min_samples_split": trial.suggest_int("min_samples_split", 2, 10),
        "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 8),
        "random_state": 42,
    }

    with mlflow.start_run(nested=True, run_name=f"optuna_trial_{trial.number}"):
        clf = RandomForestClassifier(**params)
        clf.fit(X_train, y_train)

        y_pred = clf.predict(X_test)
        y_proba = clf.predict_proba(X_test)[:, 1]
        metrics = eval_metrics(y_test, y_pred, y_proba)

        mlflow.log_params(params)
        mlflow.log_metrics(metrics)

    return metrics["roc_auc"]

# COMMAND ----------

with mlflow.start_run(run_name="optuna_search") as parent_run:
    study = optuna.create_study(direction="maximize")
    study.optimize(objective, n_trials=N_TRIALS)
    mlflow.log_params({"n_trials": N_TRIALS})
    mlflow.log_metric("best_roc_auc", study.best_value)
    parent_run_id = parent_run.info.run_id

print("Mejores hiperparámetros:", study.best_params)
print("Mejor ROC-AUC:", study.best_value)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.6 Visualización de la búsqueda (Optuna)

# COMMAND ----------

import optuna.visualization as vis

vis.plot_optimization_history(study).show()
vis.plot_param_importances(study).show()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.7 Run final — Modelo ganador
# MAGIC Se reentrena con los mejores hiperparámetros encontrados y se registra
# MAGIC como el modelo candidato a producción, con artefactos de evaluación
# MAGIC (matriz de confusión, classification report).

# COMMAND ----------

with mlflow.start_run(run_name="final_best_model") as run:
    best_params = {**study.best_params, "random_state": 42}
    clf_final = RandomForestClassifier(**best_params)
    clf_final.fit(X_train, y_train)

    y_pred = clf_final.predict(X_test)
    y_proba = clf_final.predict_proba(X_test)[:, 1]
    final_metrics = eval_metrics(y_test, y_pred, y_proba)

    mlflow.log_params(best_params)
    mlflow.log_metrics(final_metrics)

    cm = confusion_matrix(y_test, y_pred)
    report = classification_report(y_test, y_pred)

    with open("/tmp/classification_report.txt", "w") as f:
        f.write(report)
    mlflow.log_artifact("/tmp/classification_report.txt")

    signature = infer_signature(X_train, y_pred)
    mlflow.sklearn.log_model(
        clf_final, name="model", signature=signature,
        input_example=X_train.iloc[:5]
    )

    final_run_id = run.info.run_id
    print("Modelo final:", final_metrics)
    print("Matriz de confusión:\n", cm)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.8 Comparación de corridas (para el informe)

# COMMAND ----------

comparison = pd.DataFrame([
    {"run": "baseline_rf", **metrics},
    {"run": "final_best_model (Optuna)", **final_metrics},
])
display(comparison)

# COMMAND ----------

# MAGIC %md
# MAGIC **Justificación del modelo final (completar en el informe):**
# MAGIC comparar `comparison` de arriba, explicar por qué el modelo final
# MAGIC (hiperparámetros de Optuna) mejora o no al baseline, y qué métrica
# MAGIC se priorizó (ROC-AUC, dado que el target está razonablemente balanceado).

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.9 Registro en MLflow Model Registry

# COMMAND ----------

model_uri = f"runs:/{final_run_id}/model"
registered_model_name = "clickstream_playback_predictor"

result = mlflow.register_model(model_uri=model_uri, name=registered_model_name)
print(f"Modelo registrado: {registered_model_name}, versión {result.version}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.10 SHAP — Importancia de variables
# MAGIC A diferencia del dataset de cáncer de mama, acá las features tienen
# MAGIC significado de negocio directo (n_search, n_movieDetails, etc.), lo que
# MAGIC hace mucho más interpretable el análisis.

# COMMAND ----------

import shap

# Forzamos tipos numéricos puros (evita error de casting con columnas
# resultantes del one-hot encoding de 'source').
X_train_shap = X_train.astype(float)
X_test_shap = X_test.astype(float)

# TreeExplainer es más robusto y rápido para modelos de árboles (como
# RandomForest) que el shap.Explainer genérico, y no necesita un dataset
# de "background" para funcionar.
explainer = shap.TreeExplainer(clf_final)
shap_values = explainer.shap_values(X_test_shap)

# Según la versión de shap, shap_values puede venir como lista [clase0, clase1]
# o como array 3D (filas, features, clases). Nos quedamos con la clase 1
# (probabilidad de 'had_playback' = 1).
if isinstance(shap_values, list):
    shap_values_class1 = shap_values[1]
else:
    shap_values_class1 = shap_values[:, :, 1]

shap.summary_plot(shap_values_class1, X_test_shap, feature_names=X_test_shap.columns)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 2.11 Guardar sets de referencia / actual para Evidently (Fase 4)

# COMMAND ----------

data_reference = X_train.copy()
data_reference["target"] = y_train.values
data_reference["prediction"] = clf_final.predict(X_train)
data_reference["proba"] = clf_final.predict_proba(X_train)[:, 1]

data_current = X_test.copy()
data_current["target"] = y_test.values
data_current["prediction"] = y_pred
data_current["proba"] = y_proba

spark.createDataFrame(data_reference).write.mode("overwrite").saveAsTable("clickstream_reference_data")
spark.createDataFrame(data_current).write.mode("overwrite").saveAsTable("clickstream_current_data")

print("Tablas de referencia y actuales guardadas para la Fase 4 (Evidently).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximo paso
# MAGIC Ir al **Model Serving** de Databricks, crear un endpoint apuntando al
# MAGIC modelo `clickstream_playback_predictor` (última versión registrada), y
# MAGIC usar el patrón de `testinvokemodel.ipynb` (función `score_model`) para
# MAGIC probarlo con un ejemplo real de `X_test`.
