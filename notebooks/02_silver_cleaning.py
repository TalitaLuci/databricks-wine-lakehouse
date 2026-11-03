# Databricks notebook source
# MAGIC %md
# MAGIC # 02 · Camada Silver — Tipagem, qualidade e deduplicação
# MAGIC
# MAGIC Este notebook lê a tabela Bronze, converte os tipos, aplica regras de qualidade,
# MAGIC remove duplicatas exatas e grava a tabela Silver.
# MAGIC
# MAGIC **Origem:** `workspace.wine_bronze.wine_quality_raw`
# MAGIC **Destinos (schema `wine_silver`):**
# MAGIC - `wine_quality_clean`: dados tipados, validados e sem duplicatas exatas
# MAGIC - `wine_quality_rejected`: registros que violaram alguma regra de qualidade (quarentena)
# MAGIC - `dq_results`: histórico dos resultados das verificações de qualidade, por execução
# MAGIC
# MAGIC **Decisões da Silver**
# MAGIC - Colunas físico-químicas viram `double` e `quality` vira `int`. Valores que não convertem viram nulos
# MAGIC   (`try_cast`) e são capturados pelas regras, em vez de derrubar o pipeline.
# MAGIC - Registros que violam regras vão para a quarentena, não são descartados em silêncio.
# MAGIC - **Duplicatas exatas são removidas** (mesmo tipo de vinho, mesmas 11 medições e mesma nota). O número de
# MAGIC   ocorrências original é preservado na coluna `occurrences`. Motivo: linhas idênticas podem cair no treino e no
# MAGIC   teste ao mesmo tempo e inflar as métricas do modelo.
# MAGIC - **Outliers não são removidos.** Medições extremas são valores plausíveis neste tipo de dado; apenas
# MAGIC   impossibilidades físicas (pH fora de 0-14, SO₂ livre maior que o total etc.) são rejeitadas.
# MAGIC - Escrita com `MERGE` por chave (`wine_id`), para reexecução idempotente.
# MAGIC   O `ingestion_timestamp` não é usado sozinho como critério de atualização, pois pode mudar
# MAGIC   quando a Bronze é reprocessada sem que o conteúdo do registro tenha mudado.
# MAGIC - Restrições `CHECK` e `NOT NULL` do Delta reforçam as regras de qualidade na tabela.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração

# COMMAND ----------

from datetime import datetime, timezone
from pyspark.sql import functions as F
from pyspark.sql import Window

CATALOG = "workspace"
BRONZE_SCHEMA = "wine_bronze"
SILVER_SCHEMA = "wine_silver"

BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.wine_quality_raw"
SILVER_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.wine_quality_clean"
REJECTED_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.wine_quality_rejected"
DQ_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.dq_results"

FEATURES = [
    "fixed_acidity",
    "volatile_acidity",
    "citric_acid",
    "residual_sugar",
    "chlorides",
    "free_sulfur_dioxide",
    "total_sulfur_dioxide",
    "density",
    "ph",
    "sulphates",
    "alcohol",
]
TARGET = "quality"

# Uma duplicata exata repete todas estas colunas.
KEY_COLUMNS = ["wine_type"] + FEATURES + [TARGET]

# Valores esperados para o dataset original (conferidos com pandas antes da escrita deste notebook).
EXPECTED_BRONZE_ROWS = 6497
EXPECTED_SILVER_ROWS = {"red": 1359, "white": 3961}

RUN_ID = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%SZ")
print(f"Execução: {RUN_ID}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verificar pré-requisitos
# MAGIC
# MAGIC O notebook 01 deve ter sido executado antes. Aqui verificamos se a tabela Bronze existe
# MAGIC e se tem o número esperado de registros.

# COMMAND ----------

spark.sql(f"USE CATALOG `{CATALOG}`")

schemas = {row[0] for row in spark.sql(f"SHOW SCHEMAS IN `{CATALOG}`").collect()}
assert SILVER_SCHEMA in schemas, (
    f"Schema '{SILVER_SCHEMA}' não encontrado. Execute o notebook 00."
)
assert spark.catalog.tableExists(BRONZE_TABLE), (
    f"Tabela {BRONZE_TABLE} não encontrada. Execute o notebook 01."
)

bronze = spark.table(BRONZE_TABLE)
bronze_rows = bronze.count()
assert bronze_rows == EXPECTED_BRONZE_ROWS, (
    f"Bronze com {bronze_rows} registros; esperado {EXPECTED_BRONZE_ROWS}."
)

print(f"Bronze validada: {bronze_rows} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Converter tipos
# MAGIC
# MAGIC `try_cast` devolve nulo quando o valor não pode ser convertido, em vez de gerar erro.
# MAGIC Assim, problemas de conversão aparecem como falhas de qualidade e não interrompem o notebook.

# COMMAND ----------

typed = bronze.select(
    "wine_type",
    *[F.expr(f"try_cast(trim({c}) AS DOUBLE)").alias(c) for c in FEATURES],
    F.expr(f"try_cast(trim({TARGET}) AS INT)").alias(TARGET),
    "source_file",
    "ingestion_timestamp",
)

typed.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.1 Relatório de conversão
# MAGIC
# MAGIC Compara os nulos antes e depois da conversão. Qualquer diferença indica um valor que não pôde ser
# MAGIC convertido (incluindo texto vazio).

# COMMAND ----------

value_columns = FEATURES + [TARGET]

raw_nulls = bronze.select(
    *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in value_columns]
).first()
typed_nulls = typed.select(
    *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in value_columns]
).first()

conversion_report = spark.createDataFrame(
    [
        (c, int(raw_nulls[c]), int(typed_nulls[c]), int(typed_nulls[c] - raw_nulls[c]))
        for c in value_columns
    ],
    "column string, nulls_before long, nulls_after long, conversion_failures long",
)

display(conversion_report)

parse_failures = conversion_report.agg(F.sum("conversion_failures")).first()[0]
print(f"Total de falhas de conversão: {parse_failures}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Regras de qualidade
# MAGIC
# MAGIC Cada regra é uma expressão SQL que deve ser verdadeira para um registro válido.
# MAGIC Valores nulos reprovam a regra. Os registros reprovados vão para a quarentena.

# COMMAND ----------

RULES = {
    "all_fields_present": {
        "description": "Todas as colunas numéricas presentes e convertidas",
        "expr": " AND ".join(f"{c} IS NOT NULL" for c in value_columns),
    },
    "non_negative_measures": {
        "description": "Medições físico-químicas não negativas",
        "expr": " AND ".join(f"{c} >= 0" for c in FEATURES),
    },
    "ph_in_range": {
        "description": "pH entre 0 e 14",
        "expr": "ph BETWEEN 0 AND 14",
    },
    "density_positive": {
        "description": "Densidade maior que zero",
        "expr": "density > 0",
    },
    "alcohol_in_range": {
        "description": "Teor alcoólico entre 0 e 100 (% vol)",
        "expr": "alcohol BETWEEN 0 AND 100",
    },
    "quality_in_range": {
        "description": "Nota de qualidade entre 0 e 10",
        "expr": "quality BETWEEN 0 AND 10",
    },
    "free_so2_not_above_total": {
        "description": "SO₂ livre não pode exceder o SO₂ total",
        "expr": "free_sulfur_dioxide <= total_sulfur_dioxide",
    },
    "wine_type_valid": {
        "description": "Tipo de vinho é red ou white",
        "expr": "wine_type IN ('red', 'white')",
    },
}


def rule_failed(expr: str):
    # Um registro falha a regra se a expressão for falsa ou nula.
    return ~F.coalesce(F.expr(expr), F.lit(False))


checked = typed.withColumn(
    "failed_rules",
    # concat_ws ignora nulos: o resultado é vazio quando nenhuma regra falha.
    F.concat_ws(
        ",",
        *[F.when(rule_failed(r["expr"]), F.lit(name)) for name, r in RULES.items()],
    ),
).withColumn("is_valid", F.col("failed_rules") == "")

# COMMAND ----------

violations = checked.select(
    *[
        F.sum(F.when(rule_failed(r["expr"]), 1).otherwise(0)).alias(name)
        for name, r in RULES.items()
    ]
).first()

rule_report = spark.createDataFrame(
    [(name, r["description"], int(violations[name])) for name, r in RULES.items()],
    "rule string, description string, violations long",
)

display(rule_report)

# COMMAND ----------

valid = checked.filter("is_valid").drop("failed_rules", "is_valid")
rejected = checked.filter("NOT is_valid")

valid_rows = valid.count()
rejected_rows = rejected.count()

print(f"Registros válidos: {valid_rows}")
print(f"Registros rejeitados (quarentena): {rejected_rows}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Duplicatas exatas
# MAGIC
# MAGIC Primeiro medimos, depois removemos. A coluna `occurrences` guarda quantas vezes cada registro apareceu
# MAGIC na origem, preservando a informação mesmo após a remoção.

# COMMAND ----------

window_key = Window.partitionBy(*KEY_COLUMNS)
window_ordered = window_key.orderBy("ingestion_timestamp")

deduped = (
    valid.withColumn("occurrences", F.count(F.lit(1)).over(window_key))
    .withColumn("row_in_group", F.row_number().over(window_ordered))
    .filter("row_in_group = 1")
    .drop("row_in_group")
)

deduped_rows = deduped.count()
duplicates_removed = valid_rows - deduped_rows

print(f"Duplicatas exatas removidas: {duplicates_removed}")

# COMMAND ----------

display(
    valid.groupBy("wine_type").count().withColumnRenamed("count", "rows_before")
    .join(
        deduped.groupBy("wine_type").count().withColumnRenamed("count", "rows_after"),
        "wine_type",
    )
    .withColumn("duplicates_removed", F.col("rows_before") - F.col("rows_after"))
)

# COMMAND ----------

display(
    deduped.groupBy("occurrences").count().orderBy("occurrences")
    .withColumnRenamed("count", "distinct_records")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Conflito de rótulos
# MAGIC
# MAGIC Depois de remover as duplicatas exatas, ainda pode haver registros com as mesmas 11 medições mas notas
# MAGIC diferentes. Isso seria ambíguo para o modelo. Esperamos zero.

# COMMAND ----------

label_conflicts = (
    deduped.groupBy("wine_type", *FEATURES).count().filter("count > 1").count()
)
print(f"Grupos com medições idênticas e notas diferentes: {label_conflicts}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Montar a tabela Silver
# MAGIC
# MAGIC `wine_id` é um identificador determinístico (hash SHA-256 da chave natural do registro). Ele permite
# MAGIC usar `MERGE` e garante que o mesmo vinho receba sempre o mesmo identificador.

# COMMAND ----------

silver = deduped.withColumn(
    "wine_id",
    F.sha2(F.concat_ws("|", *[F.col(c).cast("string") for c in KEY_COLUMNS]), 256),
).select(
    "wine_id",
    "wine_type",
    *FEATURES,
    TARGET,
    "occurrences",
    "source_file",
    "ingestion_timestamp",
)

silver.printSchema()
display(silver.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Gravar quarentena e histórico de qualidade
# MAGIC
# MAGIC A quarentena é reescrita a cada execução (reflete o estado atual). O histórico de qualidade é
# MAGIC acrescentado, porque é um registro por execução.

# COMMAND ----------

rejected.write.format("delta").mode("overwrite").option(
    "overwriteSchema", "true"
).saveAsTable(REJECTED_TABLE)

dq_rows = (
    [("conversion_failures", "Valores que não puderam ser convertidos", int(parse_failures), bronze_rows)]
    + [(name, r["description"], int(violations[name]), bronze_rows) for name, r in RULES.items()]
    + [
        ("rejected_rows", "Registros enviados para a quarentena", rejected_rows, bronze_rows),
        ("exact_duplicates_removed", "Duplicatas exatas removidas", duplicates_removed, valid_rows),
        ("label_conflicts", "Grupos com mesmas medições e notas diferentes", label_conflicts, deduped_rows),
    ]
)

(
    spark.createDataFrame(
        dq_rows,
        "check_name string, description string, affected_rows long, total_rows long",
    )
    .withColumn("run_id", F.lit(RUN_ID))
    .withColumn("run_timestamp", F.current_timestamp())
    .select("run_id", "run_timestamp", "check_name", "description", "affected_rows", "total_rows")
    .write.format("delta")
    .mode("append")
    .saveAsTable(DQ_TABLE)
)

print(f"Quarentena: {REJECTED_TABLE}")
print(f"Histórico de qualidade: {DQ_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Gravar a tabela Silver com MERGE
# MAGIC
# MAGIC Na primeira execução, a tabela é criada. Nas seguintes, o `MERGE` compara os registros pela chave `wine_id`:
# MAGIC insere os novos, atualiza apenas os que mudaram e remove os que deixaram de existir na origem.
# MAGIC Se nada mudou, nenhum registro é reescrito.

# COMMAND ----------

silver.createOrReplaceTempView("silver_stage")

if not spark.catalog.tableExists(SILVER_TABLE):
    silver.write.format("delta").mode("overwrite").saveAsTable(SILVER_TABLE)
    print(f"Tabela criada: {SILVER_TABLE}")
else:
    spark.sql(
        f"""
        MERGE INTO {SILVER_TABLE} AS t
        USING silver_stage AS s
        ON t.wine_id = s.wine_id
        WHEN MATCHED AND (
            NOT (t.occurrences <=> s.occurrences)
        ) THEN UPDATE SET *
        WHEN NOT MATCHED THEN INSERT *
        WHEN NOT MATCHED BY SOURCE THEN DELETE
        """
    )
    print(f"MERGE concluído: {SILVER_TABLE}")

spark.sql(
    f"COMMENT ON TABLE {SILVER_TABLE} IS "
    "'Silver: Wine Quality tipado, validado e sem duplicatas exatas; occurrences guarda a contagem original'"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 9. Restrições do Delta
# MAGIC
# MAGIC Restrições `CHECK` e `NOT NULL` fazem o próprio Delta recusar gravações inválidas no futuro.
# MAGIC A célula pode ser reexecutada: restrições que já existem são ignoradas.

# COMMAND ----------

CHECK_CONSTRAINTS = {
    "ck_non_negative_measures": " AND ".join(f"{c} >= 0" for c in FEATURES),
    "ck_ph_range": "ph BETWEEN 0 AND 14",
    "ck_density_positive": "density > 0",
    "ck_alcohol_range": "alcohol BETWEEN 0 AND 100",
    "ck_quality_range": "quality BETWEEN 0 AND 10",
    "ck_so2_consistency": "free_sulfur_dioxide <= total_sulfur_dioxide",
    "ck_wine_type": "wine_type IN ('red', 'white')",
}

for name, condition in CHECK_CONSTRAINTS.items():
    try:
        spark.sql(f"ALTER TABLE {SILVER_TABLE} ADD CONSTRAINT {name} CHECK ({condition})")
        print(f"Restrição criada: {name}")
    except Exception as error:
        if "already exists" in str(error).lower():
            print(f"Restrição já existe: {name}")
        else:
            raise

NOT_NULL_COLUMNS = ["wine_id", "wine_type"] + FEATURES + [TARGET]

for column in NOT_NULL_COLUMNS:
    spark.sql(f"ALTER TABLE {SILVER_TABLE} ALTER COLUMN `{column}` SET NOT NULL")

display(
    spark.sql(f"SHOW TBLPROPERTIES {SILVER_TABLE}").filter(
        F.col("key").startswith("delta.constraints")
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 10. Validar a Silver

# COMMAND ----------

silver_table = spark.table(SILVER_TABLE)

counts = {
    row["wine_type"]: row["n"]
    for row in silver_table.groupBy("wine_type").agg(F.count("*").alias("n")).collect()
}

for wine_type, expected in EXPECTED_SILVER_ROWS.items():
    actual = counts.get(wine_type, 0)
    assert actual == expected, f"{wine_type}: esperado {expected}, encontrado {actual}."
    print(f"OK — {wine_type}: {actual} registros")

total = silver_table.count()
assert total == sum(EXPECTED_SILVER_ROWS.values()), f"Total inesperado na Silver: {total}."

distinct_ids = silver_table.select("wine_id").distinct().count()
assert distinct_ids == total, "wine_id duplicado na Silver."

null_values = silver_table.select(
    *[F.sum(F.col(c).isNull().cast("int")).alias(c) for c in value_columns]
).first()
assert all(null_values[c] == 0 for c in value_columns), "Há nulos nas colunas numéricas."

# Cada registro da Silver representa 'occurrences' linhas válidas da Bronze.
represented = silver_table.agg(F.sum("occurrences")).first()[0]
assert represented == valid_rows, (
    f"Soma de occurrences ({represented}) diferente dos registros válidos ({valid_rows})."
)

print(f"Total na Silver: {total} registros, todos com wine_id único e sem nulos")
print(f"Rastreabilidade: {represented} linhas de origem representadas por {total} registros")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 11. Perfil dos dados
# MAGIC
# MAGIC Resumo estatístico para orientar a engenharia de atributos da Gold. Nenhum outlier foi removido.

# COMMAND ----------

display(silver_table.select(*FEATURES).summary("min", "25%", "50%", "75%", "max"))

# COMMAND ----------

display(
    silver_table.groupBy("wine_type")
    .pivot(TARGET)
    .count()
    .na.fill(0)
    .orderBy("wine_type")
)

# COMMAND ----------

# Proporção de vinhos com nota >= 7, candidata a classe positiva na Gold (alvo binário).
display(
    silver_table.groupBy("wine_type").agg(
        F.count("*").alias("records"),
        F.sum((F.col(TARGET) >= 7).cast("int")).alias("quality_ge_7"),
        F.round(F.avg((F.col(TARGET) >= 7).cast("int")) * 100, 1).alias("pct_quality_ge_7"),
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 12. Testar reexecução e consultar histórico Delta
# MAGIC
# MAGIC Para testar a idempotência, execute novamente as seções 7, 8 e 10. As contagens da Silver devem permanecer
# MAGIC em 1.359 (tinto) e 3.961 (branco). No histórico, a operação `MERGE` mostra em `operationMetrics`
# MAGIC quantos registros foram inseridos, atualizados e removidos. Se os dados e a contagem de ocorrências
# MAGIC não mudaram, a reexecução não deve atualizar registros apenas porque `ingestion_timestamp` mudou na Bronze.

# COMMAND ----------

display(
    spark.sql(f"DESCRIBE HISTORY {SILVER_TABLE}").select(
        "version", "timestamp", "operation", "operationMetrics"
    )
)

# COMMAND ----------

display(spark.table(DQ_TABLE).orderBy(F.col("run_timestamp").desc(), "check_name"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximo passo
# MAGIC
# MAGIC No notebook `03_gold_features.py`, vamos criar o alvo binário (`quality >= 7`), separar treino e teste
# MAGIC sem vazamento, montar a tabela de atributos para o Spark MLlib e gerar tabelas analíticas
# MAGIC para consumo. A coluna `occurrences` não deve entrar como atributo do modelo.
