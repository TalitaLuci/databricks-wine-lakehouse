# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 01 · Camada Bronze — Ingestão do Wine Quality
# MAGIC
# MAGIC Este notebook lê os CSVs já disponibilizados no Volume do Unity Catalog e os grava
# MAGIC em uma tabela Delta na camada Bronze.
# MAGIC
# MAGIC A criação dos schemas, a configuração do Volume e a verificação inicial dos arquivos
# MAGIC são responsabilidades do notebook `00_setup_catalog_schemas.py`.
# MAGIC
# MAGIC **Origem:** `/Volumes/workspace/wine_quality/raw_files`
# MAGIC **Destino:** `workspace.wine_bronze.wine_quality_raw`
# MAGIC
# MAGIC **Princípios da Bronze**
# MAGIC - Preservar os valores originais como `string`.
# MAGIC - Não remover duplicatas, tratar outliers ou aplicar regras de negócio.
# MAGIC - Padronizar os nomes das colunas em `snake_case`.
# MAGIC - Adicionar metadados de rastreabilidade.
# MAGIC - Permitir reexecução sem duplicar registros.

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração

# COMMAND ----------

import re
from pyspark.sql import functions as F

CATALOG = "workspace"
SOURCE_SCHEMA = "wine_quality"
SOURCE_VOLUME = "raw_files"
BRONZE_SCHEMA = "wine_bronze"

RAW_PATH = f"/Volumes/{CATALOG}/{SOURCE_SCHEMA}/{SOURCE_VOLUME}"
BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.wine_quality_raw"

SOURCES = {
    "winequality-red.csv": "red",
    "winequality-white.csv": "white",
}

EXPECTED_ROWS = {
    "winequality-red.csv": 1599,
    "winequality-white.csv": 4898,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verificar pré-requisitos
# MAGIC
# MAGIC O notebook 00 deve ter sido executado antes deste. Aqui apenas verificamos
# MAGIC se os schemas e os arquivos necessários estão disponíveis.

# COMMAND ----------

schemas = {
    row[0] for row in spark.sql(f"SHOW SCHEMAS IN `{CATALOG}`").collect()
}

assert SOURCE_SCHEMA in schemas, (
    f"Schema de origem '{SOURCE_SCHEMA}' não encontrado. Execute o notebook 00."
)
assert BRONZE_SCHEMA in schemas, (
    f"Schema Bronze '{BRONZE_SCHEMA}' não encontrado. Execute o notebook 00."
)

spark.sql(f"USE CATALOG `{CATALOG}`")

source_files = {item.name.rstrip("/") for item in dbutils.fs.ls(RAW_PATH)}
missing_files = set(SOURCES) - source_files
assert not missing_files, (
    f"Arquivos ausentes em {RAW_PATH}: {sorted(missing_files)}"
)

print(f"Origem validada: {RAW_PATH}")
print(f"Destino: {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Funções de leitura

# COMMAND ----------

def to_snake_case(name: str) -> str:
    # Converte nomes como 'fixed acidity' e 'pH' para snake_case.
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()


def read_raw(file_name: str, wine_type: str):
    # Lê o CSV preservando os valores como texto.
    path = f"{RAW_PATH}/{file_name}"

    df = (
        spark.read.format("csv")
        .option("header", True)
        .option("sep", ";")
        .option("quote", '"')
        .option("inferSchema", False)
        .load(path)
    )

    # Padroniza somente os nomes das colunas, sem alterar os valores.
    df = df.toDF(*[to_snake_case(column) for column in df.columns])

    # Adiciona metadados para rastrear a origem dos registros.
    return (
        df.withColumn("wine_type", F.lit(wine_type))
        .withColumn("source_file", F.lit(file_name))
        .withColumn("source_path", F.lit(path))
        .withColumn("ingestion_timestamp", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Gravar os dados na tabela Delta Bronze
# MAGIC
# MAGIC Na primeira execução, a tabela é criada. Nas seguintes, `replaceWhere` substitui
# MAGIC somente as linhas do arquivo que está sendo reprocessado, identificado por `source_file`.
# MAGIC Assim, reexecutar a ingestão não deve acrescentar uma segunda cópia dos registros.

# COMMAND ----------

for file_name, wine_type in SOURCES.items():
    df = read_raw(file_name, wine_type)

    writer = df.write.format("delta").mode("overwrite")

    if spark.catalog.tableExists(BRONZE_TABLE):
        writer = writer.option(
            "replaceWhere",
            f"source_file = '{file_name}'"
        )

    writer.saveAsTable(BRONZE_TABLE)
    print(f"Ingerido: {file_name} | tipo: {wine_type}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Adicionar descrição à tabela

# COMMAND ----------

spark.sql(
    f"COMMENT ON TABLE {BRONZE_TABLE} IS "
    "'Bronze: Wine Quality, valores originais como string com metadados de rastreabilidade'"
)
print(f"Descrição aplicada: {BRONZE_TABLE}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validar a ingestão

# COMMAND ----------

bronze = spark.table(BRONZE_TABLE)

counts = {
    row["source_file"]: row["n"]
    for row in (
        bronze.groupBy("source_file")
        .agg(F.count("*").alias("n"))
        .collect()
    )
}

for file_name, expected_count in EXPECTED_ROWS.items():
    actual_count = counts.get(file_name, 0)
    assert actual_count == expected_count, (
        f"{file_name}: esperado {expected_count} registros, "
        f"mas foram encontrados {actual_count}."
    )
    print(f"OK — {file_name}: {actual_count} registros")

total_rows = bronze.count()
expected_total = sum(EXPECTED_ROWS.values())

assert total_rows == expected_total, (
    f"Total esperado: {expected_total}; total encontrado: {total_rows}."
)

print(f"Total na Bronze: {total_rows} registros")
bronze.printSchema()

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Visualizar uma amostra

# COMMAND ----------

display(bronze.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Testar reexecução e consultar histórico Delta
# MAGIC
# MAGIC Para testar a idempotência, execute novamente a seção 4 e depois a seção 6.
# MAGIC As contagens devem continuar em 1.599 registros de vinho tinto, 4.898 de vinho branco
# MAGIC e 6.497 no total. O histórico Delta registra as operações realizadas na tabela.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {BRONZE_TABLE}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximo passo
# MAGIC
# MAGIC No notebook `02_silver_cleaning.py`, vamos converter as colunas físico-químicas
# MAGIC para tipos numéricos, verificar valores nulos, inválidos e duplicados e documentar
# MAGIC as regras de qualidade aplicadas.