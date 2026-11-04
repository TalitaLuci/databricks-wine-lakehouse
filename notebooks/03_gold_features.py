# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 03 · Camada Gold — Atributos para modelagem e tabelas analíticas
# MAGIC
# MAGIC Este notebook lê a tabela Silver e produz dois tipos de saída:
# MAGIC 1. a base de modelagem, com alvo binário, atributos e divisão treino/teste;
# MAGIC 2. tabelas analíticas prontas para consulta e apoio à decisão.
# MAGIC
# MAGIC **Origem:** `workspace.wine_silver.wine_quality_clean`
# MAGIC **Destinos (schema `wine_gold`):**
# MAGIC - `wine_ml_dataset`: base de modelagem (atributos, alvo e coluna `split`)
# MAGIC - `ml_feature_catalog`: dicionário dos atributos, com grupo e descrição
# MAGIC - `quality_summary_by_type`: resumo de qualidade por tipo de vinho
# MAGIC - `quality_by_alcohol_band`: proporção de vinhos bons por faixa de álcool
# MAGIC - `feature_profile_by_class`: perfil médio das medições por tipo e classe
# MAGIC
# MAGIC **Decisões da Gold**
# MAGIC - **Alvo binário:** `label = 1` quando `quality >= 7` (vinho "bom"). As classes são desbalanceadas, o que será
# MAGIC   tratado na modelagem (métricas e pesos de classe).
# MAGIC - **Divisão treino/teste determinística e estratificada** por tipo de vinho e classe (80% / 20%). A ordem dentro de cada
# MAGIC   grupo vem de um hash estável, então a divisão não muda entre execuções.
# MAGIC - **Prevenção de vazamento:** além de remover duplicatas exatas na Silver, este notebook verifica se há medições
# MAGIC   idênticas associadas a notas diferentes. Se houver, interrompe a geração da Gold para evitar que observações
# MAGIC   equivalentes sejam distribuídas entre treino e teste.
# MAGIC - A padronização (`StandardScaler`) **não** é feita aqui: ela deve ser ajustada só com o treino, dentro do pipeline do MLlib.
# MAGIC - `occurrences`, `source_file` e `ingestion_timestamp` ficam **fora** da base de modelagem.
# MAGIC - **Escrita com sobrescrita:** a Gold é derivada da Silver e a divisão é determinística, então reconstruí-la é idempotente.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração

# COMMAND ----------

from pyspark.sql import functions as F
from pyspark.sql import Window

CATALOG = "workspace"
SILVER_SCHEMA = "wine_silver"
GOLD_SCHEMA = "wine_gold"

SILVER_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.wine_quality_clean"
ML_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.wine_ml_dataset"
FEATURE_CATALOG_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.ml_feature_catalog"
SUMMARY_TYPE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.quality_summary_by_type"
ALCOHOL_BAND_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.quality_by_alcohol_band"
PROFILE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.feature_profile_by_class"

BASE_FEATURES = [
    "fixed_acidity", "volatile_acidity", "citric_acid", "residual_sugar",
    "chlorides", "free_sulfur_dioxide", "total_sulfur_dioxide", "density",
    "ph", "sulphates", "alcohol",
]
TARGET = "quality"

GOOD_QUALITY_THRESHOLD = 7
TRAIN_FRACTION = 0.8
SPLIT_SEED = 42

# Valores esperados para a versão original do dataset UCI usada neste projeto.
EXPECTED_SILVER_ROWS = {"red": 1359, "white": 3961}
EXPECTED_POSITIVES = {"red": 184, "white": 825}
EXPECTED_SPLIT = {"train": 4256, "test": 1064}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verificar pré-requisitos

# COMMAND ----------

spark.sql(f"USE CATALOG `{CATALOG}`")

schemas = {row[0] for row in spark.sql(f"SHOW SCHEMAS IN `{CATALOG}`").collect()}
assert GOLD_SCHEMA in schemas, f"Schema '{GOLD_SCHEMA}' não encontrado. Execute o notebook 00."
assert spark.catalog.tableExists(SILVER_TABLE), (
    f"Tabela {SILVER_TABLE} não encontrada. Execute o notebook 02."
)

silver = spark.table(SILVER_TABLE)

silver_counts = {
    row["wine_type"]: row["n"]
    for row in silver.groupBy("wine_type").agg(F.count("*").alias("n")).collect()
}
assert silver_counts == EXPECTED_SILVER_ROWS, (
    f"Contagens inesperadas na Silver: {silver_counts}; esperado {EXPECTED_SILVER_ROWS}."
)

print(f"Silver validada: {silver.count()} registros {silver_counts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Alvo e atributos
# MAGIC
# MAGIC - `label`: 1.0 se `quality >= 7`, senão 0.0.
# MAGIC - `is_white`: indicador numérico do tipo de vinho (1.0 branco, 0.0 tinto).
# MAGIC - `free_so2_ratio`: fração do dióxido de enxofre livre sobre o total.
# MAGIC - `total_acidity`: soma das acidezes fixa, volátil e cítrica.
# MAGIC
# MAGIC Os dois últimos são atributos derivados propostos como hipótese; seu valor será comparado no notebook de modelagem.
# MAGIC Antes do cálculo, verificamos denominadores iguais a zero para não gerar razões indefinidas.

# COMMAND ----------

zero_so2_total = silver.filter(F.col("total_sulfur_dioxide") == 0).count()
assert zero_so2_total == 0, (
    f"{zero_so2_total} registros têm total_sulfur_dioxide igual a zero. "
    "Não é possível calcular free_so2_ratio nesses casos; investigue a Silver antes de continuar."
)

# Também mantemos a proteção no próprio cálculo. Se a origem mudar futuramente e surgir zero,
# o resultado será nulo e a validação da seção 8 impedirá a gravação de uma base incompleta.
featured = (
    silver.withColumn("label", (F.col(TARGET) >= GOOD_QUALITY_THRESHOLD).cast("double"))
    .withColumn("is_white", (F.col("wine_type") == "white").cast("double"))
    .withColumn(
        "free_so2_ratio",
        F.when(
            F.col("total_sulfur_dioxide") > 0,
            F.col("free_sulfur_dioxide") / F.col("total_sulfur_dioxide"),
        ).otherwise(F.lit(None).cast("double")),
    )
    .withColumn(
        "total_acidity",
        F.col("fixed_acidity") + F.col("volatile_acidity") + F.col("citric_acid"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Verificar conflitos de rótulo e dividir treino/teste
# MAGIC
# MAGIC Antes da divisão, verificamos se o mesmo conjunto de medições físico-químicas aparece associado a notas
# MAGIC diferentes dentro do mesmo tipo de vinho. Se houver conflitos, o notebook para: dividir por `wine_id`
# MAGIC poderia colocar medições idênticas em conjuntos diferentes e causar vazamento entre treino e teste.

# COMMAND ----------

label_conflicts = (
    silver.groupBy("wine_type", *BASE_FEATURES)
    .agg(F.countDistinct(TARGET).alias("distinct_labels"))
    .filter(F.col("distinct_labels") > 1)
    .count()
)

assert label_conflicts == 0, (
    f"Foram encontrados {label_conflicts} grupos de medições idênticas com notas diferentes. "
    "A divisão por wine_id pode causar vazamento entre treino e teste. "
    "Investigue os conflitos antes de gerar a Gold."
)

print("Verificação de conflitos de rótulo: nenhum grupo conflitante encontrado.")

# COMMAND ----------

# MAGIC %md
# MAGIC A divisão é estratificada por `wine_type` × `label`. Dentro de cada estrato, os registros são ordenados
# MAGIC por um hash de `wine_id` com semente fixa. Os primeiros 80% vão para treino e o restante para teste.

# COMMAND ----------

stratum = Window.partitionBy("wine_type", "label")
stratum_ordered = stratum.orderBy(
    F.sha2(F.concat_ws("|", F.col("wine_id"), F.lit(str(SPLIT_SEED))), 256)
)

split_df = (
    featured.withColumn("stratum_rank", F.row_number().over(stratum_ordered))
    .withColumn("stratum_size", F.count(F.lit(1)).over(stratum))
    .withColumn(
        "split",
        F.when(
            F.col("stratum_rank") <= F.round(F.col("stratum_size") * F.lit(TRAIN_FRACTION)),
            F.lit("train"),
        ).otherwise(F.lit("test")),
    )
)

ml_dataset = split_df.select(
    "wine_id", "wine_type", "split", "label", TARGET,
    *BASE_FEATURES, "is_white", "free_so2_ratio", "total_acidity",
)

ml_dataset.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Gravar a base de modelagem
# MAGIC
# MAGIC Após a escrita, restrições do Delta garantem que `label` e `split` só aceitem valores válidos.

# COMMAND ----------

def write_gold(df, table: str, comment: str) -> None:
    (df.write.format("delta").mode("overwrite")
        .option("overwriteSchema", "true").saveAsTable(table))
    spark.sql(f"COMMENT ON TABLE {table} IS '{comment}'")
    print(f"Gravada: {table}")


def add_check_constraint(table: str, name: str, condition: str) -> None:
    try:
        spark.sql(f"ALTER TABLE {table} ADD CONSTRAINT {name} CHECK ({condition})")
        print(f"Restrição criada: {name}")
    except Exception as error:
        if "already exists" in str(error).lower():
            print(f"Restrição já existe: {name}")
        else:
            raise


write_gold(
    ml_dataset,
    ML_TABLE,
    "Gold: base de modelagem do Wine Quality com alvo binario (quality >= 7), atributos e divisao treino/teste estratificada",
)

add_check_constraint(ML_TABLE, "ck_label_binary", "label IN (0.0, 1.0)")
add_check_constraint(ML_TABLE, "ck_split_valid", "split IN ('train', 'test')")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Catálogo de atributos
# MAGIC
# MAGIC O catálogo diferencia atributos de entrada, alvo e metadados descritivos. `wine_type` permanece na tabela
# MAGIC para análises, mas não deve ser enviado ao VectorAssembler: sua representação numérica para o modelo é `is_white`.

# COMMAND ----------

feature_rows = [
    ("fixed_acidity", "base_physicochemical", "feature", "Acidez fixa, em ácido tartárico (g/dm³)"),
    ("volatile_acidity", "base_physicochemical", "feature", "Acidez volátil, em ácido acético (g/dm³)"),
    ("citric_acid", "base_physicochemical", "feature", "Ácido cítrico (g/dm³)"),
    ("residual_sugar", "base_physicochemical", "feature", "Açúcar residual (g/dm³)"),
    ("chlorides", "base_physicochemical", "feature", "Cloretos, em cloreto de sódio (g/dm³)"),
    ("free_sulfur_dioxide", "base_physicochemical", "feature", "Dióxido de enxofre livre (mg/dm³)"),
    ("total_sulfur_dioxide", "base_physicochemical", "feature", "Dióxido de enxofre total (mg/dm³)"),
    ("density", "base_physicochemical", "feature", "Densidade (g/cm³)"),
    ("ph", "base_physicochemical", "feature", "pH"),
    ("sulphates", "base_physicochemical", "feature", "Sulfatos, em sulfato de potássio (g/dm³)"),
    ("alcohol", "base_physicochemical", "feature", "Teor alcoólico (% vol)"),
    ("is_white", "base_wine_type", "feature", "Indicador do tipo de vinho: 1.0 branco, 0.0 tinto"),
    ("free_so2_ratio", "engineered", "feature", "Fração de SO₂ livre sobre o SO₂ total"),
    ("total_acidity", "engineered", "feature", "Soma de acidez fixa, volátil e ácido cítrico"),
    ("quality", "target", "target", "Nota original de qualidade (alvo multiclasses de origem)"),
    ("label", "target", "target", "Alvo binário: 1.0 se quality >= 7; caso contrário, 0.0"),
    ("wine_id", "metadata", "metadata", "Identificador determinístico do registro"),
    ("wine_type", "metadata", "metadata", "Tipo de vinho: red ou white; não usar diretamente no VectorAssembler"),
    ("split", "metadata", "metadata", "Partição determinística: train ou test"),
]

feature_catalog = spark.createDataFrame(
    feature_rows,
    "feature_name string, feature_group string, feature_role string, description string",
)

write_gold(
    feature_catalog,
    FEATURE_CATALOG_TABLE,
    "Gold: dicionario de atributos, alvos e metadados da base de modelagem",
)

display(spark.table(FEATURE_CATALOG_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Tabelas analíticas

# COMMAND ----------

ml = spark.table(ML_TABLE)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.1 Resumo de qualidade por tipo de vinho

# COMMAND ----------

summary_by_type = ml.groupBy("wine_type").agg(
    F.count("*").alias("records"),
    F.round(F.avg(TARGET), 2).alias("avg_quality"),
    F.sum(F.col("label").cast("int")).alias("good_wines"),
    F.round(F.avg("label") * 100, 1).alias("pct_good"),
    F.round(F.avg("alcohol"), 2).alias("avg_alcohol"),
    F.round(F.avg("volatile_acidity"), 3).alias("avg_volatile_acidity"),
    F.round(F.avg("residual_sugar"), 2).alias("avg_residual_sugar"),
)

write_gold(
    summary_by_type, SUMMARY_TYPE_TABLE,
    "Gold: resumo de qualidade por tipo de vinho (registros, nota media, percentual de vinhos bons)",
)
display(spark.table(SUMMARY_TYPE_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.2 Proporção de vinhos bons por faixa de álcool

# COMMAND ----------

alcohol = F.col("alcohol")
alcohol_band = (
    F.when(alcohol < 9, F.lit("< 9"))
    .when(alcohol < 10, F.lit("9 a 10"))
    .when(alcohol < 11, F.lit("10 a 11"))
    .when(alcohol < 12, F.lit("11 a 12"))
    .when(alcohol < 13, F.lit("12 a 13"))
    .otherwise(F.lit(">= 13"))
)
alcohol_band_order = (
    F.when(alcohol < 9, 1).when(alcohol < 10, 2).when(alcohol < 11, 3)
    .when(alcohol < 12, 4).when(alcohol < 13, 5).otherwise(6)
)

quality_by_alcohol = (
    ml.withColumn("alcohol_band", alcohol_band)
    .withColumn("band_order", alcohol_band_order)
    .groupBy("wine_type", "band_order", "alcohol_band")
    .agg(
        F.count("*").alias("records"),
        F.round(F.avg(TARGET), 2).alias("avg_quality"),
        F.sum(F.col("label").cast("int")).alias("good_wines"),
        F.round(F.avg("label") * 100, 1).alias("pct_good"),
    )
    .orderBy("wine_type", "band_order")
)

write_gold(
    quality_by_alcohol, ALCOHOL_BAND_TABLE,
    "Gold: proporcao de vinhos bons (quality >= 7) por faixa de teor alcoolico e tipo de vinho",
)
display(spark.table(ALCOHOL_BAND_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ### 7.3 Perfil médio das medições por tipo e classe
# MAGIC
# MAGIC Esta é uma visão descritiva; diferenças entre médias não demonstram causalidade.

# COMMAND ----------

feature_profile = (
    ml.withColumn(
        "class_label",
        F.when(F.col("label") == 1.0, F.lit("good (>= 7)")).otherwise(F.lit("not_good (< 7)")),
    )
    .groupBy("wine_type", "class_label")
    .agg(F.count("*").alias("records"), *[F.round(F.avg(c), 3).alias(c) for c in BASE_FEATURES])
    .orderBy("wine_type", "class_label")
)

write_gold(
    feature_profile, PROFILE_TABLE,
    "Gold: media das medicoes fisico-quimicas por tipo de vinho e classe (bom ou nao bom)",
)
display(spark.table(PROFILE_TABLE))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Validar a Gold

# COMMAND ----------

ml = spark.table(ML_TABLE)
total = ml.count()

assert total == sum(EXPECTED_SILVER_ROWS.values()), f"Total inesperado na Gold: {total}."
assert ml.select("wine_id").distinct().count() == total, "wine_id duplicado na Gold."

model_columns = ["label", "is_white", "free_so2_ratio", "total_acidity"] + BASE_FEATURES
null_counts = ml.select(*[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in model_columns]).first()
assert all(null_counts[c] == 0 for c in model_columns), (
    f"Há nulos nas colunas de modelagem: { {c: null_counts[c] for c in model_columns if null_counts[c] != 0} }"
)

for forbidden in ("occurrences", "source_file", "ingestion_timestamp"):
    assert forbidden not in ml.columns, f"Coluna '{forbidden}' não deveria estar na base de modelagem."

positives = {
    row["wine_type"]: row["n"]
    for row in ml.filter("label = 1.0").groupBy("wine_type").agg(F.count("*").alias("n")).collect()
}
assert positives == EXPECTED_POSITIVES, f"Vinhos bons inesperados: {positives}."

split_counts = {
    row["split"]: row["n"]
    for row in ml.groupBy("split").agg(F.count("*").alias("n")).collect()
}
assert split_counts == EXPECTED_SPLIT, f"Divisão inesperada: {split_counts}."

strata = ml.groupBy("wine_type", "label").agg(
    F.count("*").alias("n"),
    F.sum((F.col("split") == "train").cast("int")).alias("n_train"),
).collect()
for row in strata:
    expected_train = TRAIN_FRACTION * row["n"]
    assert abs(row["n_train"] - expected_train) <= 1, (
        f"Grupo ({row['wine_type']}, {row['label']}): {row['n_train']} no treino de {row['n']}."
    )

print(f"OK — {total} registros; treino {split_counts['train']}, teste {split_counts['test']}")
print(f"OK — vinhos bons: {positives}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8.1 Impressão digital da divisão
# MAGIC
# MAGIC O hash resume quais registros estão em treino e teste. O separador evita ambiguidades na concatenação
# MAGIC dos campos antes do cálculo da impressão digital.

# COMMAND ----------

fingerprint = ml.agg(
    F.md5(
        F.concat_ws(
            "",
            F.sort_array(
                F.collect_list(F.concat_ws("|", F.col("wine_id"), F.col("split")))
            ),
        )
    ).alias("fp")
).first()["fp"]

print(f"Impressão digital da divisão: {fingerprint}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 8.2 Balanço de classes por conjunto

# COMMAND ----------

display(
    ml.groupBy("split", "wine_type")
    .agg(
        F.count("*").alias("records"),
        F.sum(F.col("label").cast("int")).alias("good_wines"),
        F.round(F.avg("label") * 100, 1).alias("pct_good"),
    )
    .orderBy("split", "wine_type")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Histórico Delta das tabelas Gold

# COMMAND ----------

display(
    spark.sql(f"DESCRIBE HISTORY {ML_TABLE}").select(
        "version", "timestamp", "operation", "operationMetrics"
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximo passo
# MAGIC
# MAGIC No notebook `04_model_training.py`:
# MAGIC 1. testar o Spark MLlib no ambiente com um modelo mínimo;
# MAGIC 2. carregar `wine_ml_dataset` e separar os conjuntos pela coluna `split`;
# MAGIC 3. montar um `Pipeline` com `VectorAssembler` e `StandardScaler`, ajustando a padronização apenas com o treino;
# MAGIC 4. treinar modelos de referência e comparar métricas adequadas ao desbalanceamento;
# MAGIC 5. ajustar hiperparâmetros apenas no treino;
# MAGIC 6. comparar atributos base com os atributos derivados;
# MAGIC 7. avaliar no teste e registrar no MLflow se disponível.