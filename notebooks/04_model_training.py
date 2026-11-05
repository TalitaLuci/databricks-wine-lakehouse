# Databricks notebook source
# MAGIC %md
# MAGIC # 04 · Modelagem — Classificação de vinhos "bons" com Spark MLlib
# MAGIC
# MAGIC Este notebook treina e compara modelos de classificação para prever se um vinho tem `quality >= 7`,
# MAGIC usando a base de modelagem da camada Gold.
# MAGIC
# MAGIC **Origem:** `workspace.wine_gold.wine_ml_dataset` e `workspace.wine_gold.ml_feature_catalog`
# MAGIC **Destinos (schema `wine_gold`):**
# MAGIC - `model_metrics`: métricas de cada modelo por execução (histórico)
# MAGIC - `model_predictions`: previsões no teste do modelo selecionado
# MAGIC - `model_feature_importance`: importância dos atributos em cada modelo
# MAGIC
# MAGIC **Desenho do experimento**
# MAGIC - 2 algoritmos: `LogisticRegression` (referência interpretável) e `RandomForestClassifier`.
# MAGIC - 2 conjuntos de atributos: `base` (medições originais + tipo de vinho) e `base_plus_engineered` (inclui os atributos derivados).
# MAGIC - Total: 4 candidatos, cada um com busca de hiperparâmetros por validação cruzada **somente no treino**.
# MAGIC - O teste é consultado apenas para o candidato escolhido pela validação cruzada.
# MAGIC - **Desbalanceamento:** pesos de classe calculados no treino (`class_weight`). A métrica principal é a **AUC-PR da classe positiva**,
# MAGIC   e não a acurácia, que seria alta mesmo para um modelo que nunca prevê vinho bom.
# MAGIC - **Folds estratificados e determinísticos:** a coluna `fold` é criada por hash do `wine_id`, mantendo a proporção de cada tipo e classe.
# MAGIC - **Padronização sem vazamento:** o `StandardScaler` faz parte do `Pipeline` e é ajustado apenas com os dados de treino de cada ajuste.
# MAGIC - **Seleção pelo desempenho na validação cruzada, nunca pelo teste.** O teste só mede, ao final, quão bem o modelo generaliza.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração

# COMMAND ----------

import json
from datetime import datetime, timezone

from pyspark.ml import Pipeline
from pyspark.ml.classification import LogisticRegression, RandomForestClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.feature import StandardScaler, VectorAssembler
from pyspark.ml.functions import vector_to_array
from pyspark.ml.tuning import ParamGridBuilder
from pyspark.sql import Window
from pyspark.sql import functions as F

CATALOG = "workspace"
GOLD_SCHEMA = "wine_gold"

ML_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.wine_ml_dataset"
FEATURE_CATALOG_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.ml_feature_catalog"
METRICS_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.model_metrics"
PREDICTIONS_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.model_predictions"
IMPORTANCE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.model_feature_importance"

SEED = 42
NUM_FOLDS = 3
ENABLE_MLFLOW = True

MODEL_TYPES = ["logistic_regression", "random_forest"]

# Colunas que a base de modelagem precisa ter (contrato com o notebook 03).
REQUIRED_COLUMNS = {"wine_id", "wine_type", "split", "label", "quality"}

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
print(f"Execução: {RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Teste do Spark MLlib no ambiente
# MAGIC
# MAGIC Antes de qualquer treino real, treina modelos mínimos com dados sintéticos para confirmar que os componentes usados aqui
# MAGIC funcionam na computação do workspace. Se esta seção falhar, anote a mensagem: ela vira uma limitação documentada
# MAGIC no README e define o plano alternativo.

# COMMAND ----------

try:
    rows = [(0.0, 1.0, 0.0), (1.0, 0.0, 1.0), (0.5, 0.5, 0.0), (0.9, 0.1, 1.0)] * 10
    smoke_df = spark.createDataFrame(rows, "a double, b double, label double")

    smoke_pipeline = Pipeline(
        stages=[
            VectorAssembler(inputCols=["a", "b"], outputCol="features_raw"),
            StandardScaler(inputCol="features_raw", outputCol="features"),
            LogisticRegression(featuresCol="features", labelCol="label"),
        ]
    )
    smoke_scored = smoke_pipeline.fit(smoke_df).transform(smoke_df)

    smoke_rf = Pipeline(
        stages=[
            VectorAssembler(inputCols=["a", "b"], outputCol="features"),
            RandomForestClassifier(featuresCol="features", labelCol="label", numTrees=5, seed=SEED),
        ]
    )
    smoke_rf_scored = smoke_rf.fit(smoke_df).transform(smoke_df)

    auc = BinaryClassificationEvaluator(metricName="areaUnderPR").evaluate(smoke_scored)
    smoke_scored.select(vector_to_array("probability")[1].alias("p")).count()
    smoke_rf_scored.count()
    print(f"OK: VectorAssembler, StandardScaler, LogisticRegression e RandomForest funcionam (AUC-PR de teste: {auc:.2f}).")
except Exception as error:
    raise RuntimeError(
        "O Spark MLlib falhou neste ambiente. Registre o erro no README e use o plano alternativo "
        f"(por exemplo, scikit-learn sobre a tabela Gold). Detalhe: {type(error).__name__}: {error}"
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verificar pré-requisitos

# COMMAND ----------

spark.sql(f"USE CATALOG `{CATALOG}`")

assert spark.catalog.tableExists(ML_TABLE), f"Tabela {ML_TABLE} não encontrada. Execute o notebook 03."
assert spark.catalog.tableExists(FEATURE_CATALOG_TABLE), (
    f"Tabela {FEATURE_CATALOG_TABLE} não encontrada. Execute o notebook 03."
)

ml = spark.table(ML_TABLE)

missing_columns = REQUIRED_COLUMNS - set(ml.columns)
assert not missing_columns, f"Colunas ausentes em {ML_TABLE}: {sorted(missing_columns)}"

split_values = {row["split"] for row in ml.select("split").distinct().collect()}
assert split_values == {"train", "test"}, f"Valores inesperados em split: {split_values}"

label_values = {row["label"] for row in ml.select("label").distinct().collect()}
assert label_values == {0.0, 1.0}, f"Valores inesperados em label: {label_values}"

print(f"Base de modelagem validada: {ml.count()} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Conjuntos de atributos
# MAGIC
# MAGIC Os nomes vêm do catálogo de atributos criado no notebook 03, e não de listas repetidas no código.

# COMMAND ----------

feature_catalog = spark.table(FEATURE_CATALOG_TABLE)

catalog_required = {"feature_name", "feature_group", "feature_role"}
catalog_missing = catalog_required - set(feature_catalog.columns)
assert not catalog_missing, (
    f"Colunas ausentes no catálogo de atributos: {sorted(catalog_missing)}. "
    "Execute a versão atualizada do notebook 03."
)

model_feature_catalog = feature_catalog.filter(F.col("feature_role") == "feature")
features_by_group = {}
for row in model_feature_catalog.select("feature_name", "feature_group").collect():
    features_by_group.setdefault(row["feature_group"], []).append(row["feature_name"])

base_features = sorted(
    features_by_group.get("base_physicochemical", []) + features_by_group.get("base_wine_type", [])
)
engineered_features = sorted(features_by_group.get("engineered", []))

FEATURE_SETS = {
    "base": base_features,
    "base_plus_engineered": base_features + engineered_features,
}

for set_name, columns in FEATURE_SETS.items():
    missing = set(columns) - set(ml.columns)
    assert not missing, f"Atributos do catálogo ausentes na base ({set_name}): {sorted(missing)}"
    print(f"{set_name}: {len(columns)} atributos")

assert engineered_features, "Nenhum atributo derivado encontrado no catálogo."

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Treino, teste, folds e pesos de classe
# MAGIC
# MAGIC - `fold`: número de 0 a `NUM_FOLDS - 1`, distribuído dentro de cada grupo (`wine_type` × `label`) por hash do `wine_id`.
# MAGIC - `class_weight`: calculado novamente em cada partição de ajuste, sem usar os rótulos do fold de validação.

# COMMAND ----------

train_raw = ml.filter("split = 'train'")
test = ml.filter("split = 'test'")

stratum = Window.partitionBy("wine_type", "label")
fold_order = stratum.orderBy(
    F.sha2(F.concat_ws("|", F.col("wine_id"), F.lit(f"cv-{SEED}")), 256)
)

train = train_raw.withColumn(
    "fold", ((F.row_number().over(fold_order) - 1) % NUM_FOLDS).cast("int")
)

n_train = train.count()
n_positive = train.filter("label = 1.0").count()
n_negative = n_train - n_positive
n_test = test.count()
n_test_positive = test.filter("label = 1.0").count()

assert n_train > 0 and n_test > 0, "Os conjuntos de treino e teste não podem estar vazios."
assert n_positive > 0 and n_negative > 0, "O treino precisa conter as duas classes."
assert n_test_positive > 0 and n_test_positive < n_test, "O teste precisa conter as duas classes."
assert n_positive >= NUM_FOLDS and n_negative >= NUM_FOLDS, (
    "Há poucos exemplos de uma classe para distribuir entre todos os folds."
)

fold_class_counts = {
    (row["fold"], row["label"]): row["count"]
    for row in train.groupBy("fold", "label").count().collect()
}
for fold in range(NUM_FOLDS):
    assert fold_class_counts.get((fold, 0.0), 0) > 0, f"Fold {fold} sem exemplos negativos."
    assert fold_class_counts.get((fold, 1.0), 0) > 0, f"Fold {fold} sem exemplos positivos."

print(f"Treino: {n_train} registros, {n_positive} positivos ({n_positive / n_train:.1%})")
print(f"Teste:  {n_test} registros, {n_test_positive} positivos ({n_test_positive / n_test:.1%})")
print("Folds validados: cada fold contém exemplos das duas classes.")

display(
    train.groupBy("fold", "label").count().orderBy("fold", "label")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Funções do experimento
# MAGIC
# MAGIC - `build_pipeline` e `build_grid`: montam o pipeline e a grade de hiperparâmetros de cada algoritmo.
# MAGIC - `tune`: validação cruzada manual e determinística com os folds definidos por `fold`; os pesos de classe são
# MAGIC   recalculados dentro de cada partição de ajuste.
# MAGIC - `compute_metrics`: métricas da classe positiva a partir das previsões.

# COMMAND ----------

def build_pipeline(model_type: str, feature_columns: list) -> Pipeline:
    if model_type == "logistic_regression":
        # A regressão logística precisa de atributos padronizados; o scaler é ajustado só com o treino.
        return Pipeline(
            stages=[
                VectorAssembler(inputCols=feature_columns, outputCol="features_raw"),
                StandardScaler(
                    inputCol="features_raw", outputCol="features", withMean=True, withStd=True
                ),
                LogisticRegression(
                    featuresCol="features",
                    labelCol="label",
                    weightCol="class_weight",
                    maxIter=200,
                    standardization=False,
                ),
            ]
        )

    # Árvores não dependem da escala dos atributos.
    return Pipeline(
        stages=[
            VectorAssembler(inputCols=feature_columns, outputCol="features"),
            RandomForestClassifier(
                featuresCol="features",
                labelCol="label",
                weightCol="class_weight",
                seed=SEED,
            ),
        ]
    )


def build_grid(model_type: str, pipeline: Pipeline) -> list:
    classifier = pipeline.getStages()[-1]

    if model_type == "logistic_regression":
        return (
            ParamGridBuilder()
            .addGrid(classifier.regParam, [0.0, 0.01, 0.1])
            .addGrid(classifier.elasticNetParam, [0.0, 0.5])
            .build()
        )

    return (
        ParamGridBuilder()
        .addGrid(classifier.numTrees, [100])
        .addGrid(classifier.maxDepth, [6, 10])
        .addGrid(classifier.minInstancesPerNode, [1, 5])
        .build()
    )


PR_EVALUATOR = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderPR"
)
ROC_EVALUATOR = BinaryClassificationEvaluator(
    labelCol="label", rawPredictionCol="rawPrediction", metricName="areaUnderROC"
)


def add_class_weights(df):
    # Calcula pesos apenas com os dados usados no ajuste atual.
    counts = {
        row["label"]: row["count"]
        for row in df.groupBy("label").count().collect()
    }
    n_pos = counts.get(1.0, 0)
    n_neg = counts.get(0.0, 0)
    n_total = n_pos + n_neg
    assert n_pos > 0 and n_neg > 0, (
        "A partição usada no ajuste precisa conter as duas classes."
    )
    weight_pos = n_total / (2.0 * n_pos)
    weight_neg = n_total / (2.0 * n_neg)
    return df.withColumn(
        "class_weight",
        F.when(F.col("label") == 1.0, F.lit(weight_pos))
        .otherwise(F.lit(weight_neg)),
    )


def manual_cross_validation(pipeline, grid, train_df):
    # Folds determinísticos definidos acima; os pesos são recalculados em cada fit.
    scores = []
    for params in grid:
        fold_scores = []
        for fold in range(NUM_FOLDS):
            fit_df = train_df.filter(F.col("fold") != fold)
            valid_df = train_df.filter(F.col("fold") == fold)

            fit_weighted = add_class_weights(fit_df)
            model = pipeline.fit(fit_weighted, params)
            fold_scores.append(PR_EVALUATOR.evaluate(model.transform(valid_df)))

        scores.append(sum(fold_scores) / len(fold_scores))

    best_index = max(range(len(grid)), key=lambda i: scores[i])
    final_train = add_class_weights(train_df)
    best_model = pipeline.fit(final_train, grid[best_index])
    return best_model, float(scores[best_index])


def tune(pipeline, grid, train_df):
    # A validação manual permite recalcular os pesos sem usar os rótulos do fold validado.
    best_model, score = manual_cross_validation(pipeline, grid, train_df)
    return best_model, score, "manual_stratified"


TUNED_PARAMS = {"regParam", "elasticNetParam", "numTrees", "maxDepth", "minInstancesPerNode"}


def describe_params(model) -> dict:
    # Hiperparâmetros escolhidos, lidos do classificador final do pipeline.
    stage = model.stages[-1]
    return {
        param.name: value
        for param, value in stage.extractParamMap().items()
        if param.name in TUNED_PARAMS
    }


def compute_metrics(predictions) -> dict:
    # Métricas da classe positiva (vinho bom) com limiar padrão de 0,5.
    counts = predictions.select(
        F.sum(((F.col("prediction") == 1.0) & (F.col("label") == 1.0)).cast("int")).alias("tp"),
        F.sum(((F.col("prediction") == 1.0) & (F.col("label") == 0.0)).cast("int")).alias("fp"),
        F.sum(((F.col("prediction") == 0.0) & (F.col("label") == 1.0)).cast("int")).alias("fn"),
        F.sum(((F.col("prediction") == 0.0) & (F.col("label") == 0.0)).cast("int")).alias("tn"),
    ).first()

    tp, fp, fn, tn = (int(counts[k]) for k in ("tp", "fp", "fn", "tn"))
    total = tp + fp + fn + tn

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    specificity = tn / (tn + fp) if (tn + fp) else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

    return {
        "auc_pr": float(PR_EVALUATOR.evaluate(predictions)),
        "auc_roc": float(ROC_EVALUATOR.evaluate(predictions)),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "balanced_accuracy": (recall + specificity) / 2,
        "accuracy": (tp + tn) / total,
        "positive_rate": (tp + fn) / total,
    }

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Treinar e avaliar os candidatos
# MAGIC
# MAGIC Para cada combinação de conjunto de atributos e algoritmo: busca de hiperparâmetros por validação cruzada no treino,
# MAGIC avaliação do melhor modelo no treino (para medir sobreajuste) e no teste.
# MAGIC Esta é a etapa mais demorada do notebook.

# COMMAND ----------

runs = []

for feature_set_name, feature_columns in FEATURE_SETS.items():
    for model_type in MODEL_TYPES:
        print(f"Treinando: {model_type} | {feature_set_name} ({len(feature_columns)} atributos)")

        pipeline = build_pipeline(model_type, feature_columns)
        grid = build_grid(model_type, pipeline)
        best_model, cv_auc_pr, tuner = tune(pipeline, grid, train)

        # Métrica de treino é diagnóstica; a seleção continua baseada apenas na validação cruzada.
        train_auc_pr = float(
            PR_EVALUATOR.evaluate(best_model.transform(add_class_weights(train)))
        )

        runs.append(
            {
                "model_name": model_type,
                "feature_set": feature_set_name,
                "feature_columns": feature_columns,
                "model": best_model,
                "predictions": None,
                "best_params": describe_params(best_model),
                "tuner": tuner,
                "cv_auc_pr": cv_auc_pr,
                "train_auc_pr": train_auc_pr,
                "metrics": None,
            }
        )

        print(f"  CV AUC-PR = {cv_auc_pr:.3f} | treino AUC-PR = {train_auc_pr:.3f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Seleção do modelo
# MAGIC
# MAGIC O modelo selecionado é o de **maior AUC-PR média na validação cruzada**.
# MAGIC O conjunto de teste é avaliado somente uma vez, após a seleção, para preservar sua função de avaliação final.

# COMMAND ----------

selected = max(runs, key=lambda run: run["cv_auc_pr"])

print(
    f"Modelo selecionado: {selected['model_name']} | {selected['feature_set']} "
    f"(CV AUC-PR = {selected['cv_auc_pr']:.3f})"
)

selected_test_predictions = selected["model"].transform(test)
selected["predictions"] = selected_test_predictions
selected["metrics"] = compute_metrics(selected_test_predictions)

print(
    f"Teste final — AUC-PR = {selected['metrics']['auc_pr']:.3f} | "
    f"AUC-ROC = {selected['metrics']['auc_roc']:.3f} | "
    f"recall = {selected['metrics']['recall']:.3f} | "
    f"precisão = {selected['metrics']['precision']:.3f}"
)

# COMMAND ----------

results_rows = [
    (
        RUN_ID,
        run["model_name"],
        run["feature_set"],
        len(run["feature_columns"]),
        run["tuner"],
        run["cv_auc_pr"],
        run["train_auc_pr"],
        run["metrics"]["auc_pr"] if run is selected else None,
        run["metrics"]["auc_roc"] if run is selected else None,
        run["metrics"]["precision"] if run is selected else None,
        run["metrics"]["recall"] if run is selected else None,
        run["metrics"]["f1"] if run is selected else None,
        run["metrics"]["balanced_accuracy"] if run is selected else None,
        run["metrics"]["accuracy"] if run is selected else None,
        run["metrics"]["positive_rate"] if run is selected else None,
        json.dumps(run["best_params"], sort_keys=True),
        run is selected,
    )
    for run in runs
]

results = spark.createDataFrame(
    results_rows,
    """
    run_id string, model_name string, feature_set string, n_features int, tuner string,
    cv_auc_pr double, train_auc_pr double, test_auc_pr double, test_auc_roc double,
    test_precision double, test_recall double, test_f1 double,
    test_balanced_accuracy double, test_accuracy double, test_positive_rate double,
    best_params string, selected boolean
    """,
).withColumn("run_timestamp", F.current_timestamp())

display(
    results.select(
        "model_name", "feature_set", "cv_auc_pr", "train_auc_pr", "test_auc_pr", "test_auc_roc",
        "test_precision", "test_recall", "test_f1", "test_balanced_accuracy", "best_params", "selected",
    ).orderBy(F.col("cv_auc_pr").desc())
)

# COMMAND ----------

# MAGIC %md
# MAGIC **Como ler a tabela**
# MAGIC - `test_positive_rate` é a proporção de vinhos bons no teste; serve como referência para interpretar a AUC-PR.
# MAGIC - As colunas `test_*` são preenchidas somente para o candidato selecionado. Os demais não foram avaliados no teste.
# MAGIC - Uma diferença grande entre `train_auc_pr` e `cv_auc_pr` pode indicar sobreajuste.
# MAGIC - O teste tem poucos vinhos bons; suas métricas têm incerteza e não devem ser tratadas como estimativas definitivas.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Gravar métricas, previsões e importância dos atributos

# COMMAND ----------

(
    results.write.format("delta")
    .mode("append")
    .option("mergeSchema", "true")
    .saveAsTable(METRICS_TABLE)
)
spark.sql(
    f"COMMENT ON TABLE {METRICS_TABLE} IS "
    "'Gold: metricas de cada candidato por execucao (validacao cruzada, treino e teste), com indicacao do modelo selecionado'"
)
print(f"Métricas gravadas: {METRICS_TABLE}")

# COMMAND ----------

predictions_out = (
    selected["predictions"]
    .select(
        "wine_id",
        "wine_type",
        "quality",
        "label",
        "prediction",
        vector_to_array("probability")[1].alias("prob_good"),
    )
    .withColumn("model_name", F.lit(selected["model_name"]))
    .withColumn("feature_set", F.lit(selected["feature_set"]))
    .withColumn("run_id", F.lit(RUN_ID))
)

(
    predictions_out.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(PREDICTIONS_TABLE)
)
spark.sql(
    f"COMMENT ON TABLE {PREDICTIONS_TABLE} IS "
    "'Gold: previsoes no conjunto de teste do modelo selecionado, com probabilidade de vinho bom'"
)
print(f"Previsões gravadas: {PREDICTIONS_TABLE}")

# COMMAND ----------

importance_rows = []

for run in runs:
    stage = run["model"].stages[-1]
    if run["model_name"] == "random_forest":
        importance_type = "rf_feature_importance"
        values = stage.featureImportances.toArray()
    else:
        # Coeficientes sobre atributos padronizados: comparáveis entre si pela magnitude.
        importance_type = "lr_standardized_coefficient"
        values = stage.coefficients.toArray()

    for feature, value in zip(run["feature_columns"], values):
        importance_rows.append(
            (RUN_ID, run["model_name"], run["feature_set"], feature, importance_type, float(value))
        )

importance = spark.createDataFrame(
    importance_rows,
    "run_id string, model_name string, feature_set string, feature string, importance_type string, importance double",
)

(
    importance.write.format("delta")
    .mode("overwrite")
    .option("overwriteSchema", "true")
    .saveAsTable(IMPORTANCE_TABLE)
)
spark.sql(
    f"COMMENT ON TABLE {IMPORTANCE_TABLE} IS "
    "'Gold: importancia dos atributos (floresta aleatoria) e coeficientes padronizados (regressao logistica) da ultima execucao'"
)
print(f"Importância gravada: {IMPORTANCE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Modelo selecionado: matriz de confusão e atributos mais influentes

# COMMAND ----------

display(
    selected["predictions"]
    .groupBy("label", "prediction")
    .count()
    .withColumnRenamed("label", "real")
    .withColumnRenamed("prediction", "previsto")
    .orderBy("real", "previsto")
)

# COMMAND ----------

display(
    spark.table(IMPORTANCE_TABLE)
    .filter(
        (F.col("model_name") == selected["model_name"])
        & (F.col("feature_set") == selected["feature_set"])
    )
    .withColumn("abs_importance", F.abs("importance"))
    .orderBy(F.col("abs_importance").desc())
    .select("feature", "importance_type", "importance")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Desempenho por tipo de vinho
# MAGIC
# MAGIC Um modelo único para tinto e branco pode ter qualidade diferente em cada tipo. Esta visão mostra a diferença.

# COMMAND ----------

by_type = (
    spark.table(PREDICTIONS_TABLE)
    .groupBy("wine_type")
    .agg(
        F.count("*").alias("records"),
        F.sum(F.col("label").cast("int")).alias("good_wines"),
        F.sum(((F.col("prediction") == 1.0) & (F.col("label") == 1.0)).cast("int")).alias("true_positives"),
        F.sum(((F.col("prediction") == 1.0) & (F.col("label") == 0.0)).cast("int")).alias("false_positives"),
    )
    .withColumn("recall", F.round(F.col("true_positives") / F.col("good_wines"), 3))
    .withColumn(
        "precision",
        F.round(F.col("true_positives") / (F.col("true_positives") + F.col("false_positives")), 3),
    )
    .orderBy("wine_type")
)

display(by_type)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Registro no MLflow (opcional)
# MAGIC
# MAGIC Registra parâmetros e métricas de cada candidato, se o MLflow estiver disponível no workspace.
# MAGIC A falha nesta etapa não interrompe o notebook. Os modelos em si não são salvos aqui.

# COMMAND ----------

def log_runs_to_mlflow(all_runs) -> None:
    try:
        import mlflow

        user = spark.sql("SELECT current_user()").first()[0]
        mlflow.set_experiment(f"/Users/{user}/wine_quality_lakehouse")

        for run in all_runs:
            with mlflow.start_run(run_name=f"{run['model_name']}__{run['feature_set']}"):
                mlflow.set_tag("run_id", RUN_ID)
                mlflow.set_tag("selected", str(run is selected))
                mlflow.log_params(
                    {
                        "model_name": run["model_name"],
                        "feature_set": run["feature_set"],
                        "n_features": len(run["feature_columns"]),
                        "num_folds": NUM_FOLDS,
                        "seed": SEED,
                        **run["best_params"],
                    }
                )
                mlflow.log_metrics(
                    {
                        "cv_auc_pr": run["cv_auc_pr"],
                        "train_auc_pr": run["train_auc_pr"],
                        **(
                            {
                                "test_auc_pr": run["metrics"]["auc_pr"],
                                "test_auc_roc": run["metrics"]["auc_roc"],
                                "test_precision": run["metrics"]["precision"],
                                "test_recall": run["metrics"]["recall"],
                                "test_f1": run["metrics"]["f1"],
                            }
                            if run is selected else {}
                        ),
                    }
                )
        print("Execuções registradas no MLflow.")
    except Exception as error:
        print(f"Aviso: registro no MLflow não realizado ({type(error).__name__}: {error})")


if ENABLE_MLFLOW:
    log_runs_to_mlflow(runs)
else:
    print("MLflow desativado (ENABLE_MLFLOW = False).")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Limitações a documentar no README
# MAGIC
# MAGIC - **Poucos vinhos bons no teste.** As estimativas de métricas têm incerteza considerável, principalmente no vinho tinto.
# MAGIC - **Limiar fixo de 0,5.** Em um uso real, o limiar deveria ser escolhido conforme o custo de errar em cada direção, usando validação, e não o teste.
# MAGIC - **A nota é subjetiva.** `quality` vem de avaliações sensoriais, o que impõe um limite à previsibilidade a partir de medições físico-químicas.
# MAGIC - **Associação, não causalidade.** A importância dos atributos descreve o que o modelo usa, não o que causa qualidade.
# MAGIC - **Um único particionamento treino/teste.** Repetir o experimento com outras sementes mostraria a estabilidade dos resultados.

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximo passo
# MAGIC
# MAGIC No notebook `05_insights_and_consumption.py`: consultas SQL sobre as tabelas Gold (analíticas e de modelo), gráficos para o README
# MAGIC e um resumo das decisões de arquitetura e dos resultados.
