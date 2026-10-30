# Databricks notebook source
# /// script
# [tool.databricks.environment]
# environment_version = "6"
# ///
# MAGIC %md
# MAGIC # 00 · Setup — Catálogo, schemas e arquivos de origem
# MAGIC
# MAGIC Este notebook prepara os schemas da arquitetura medalhão e verifica os CSVs
# MAGIC já enviados ao Volume `workspace.wine_quality.raw_files`.
# MAGIC A ingestão dos arquivos para Delta é responsabilidade do notebook 01.
# MAGIC
# MAGIC ```
# MAGIC workspace
# MAGIC ├── wine_quality/raw_files  <- CSVs originais já enviados
# MAGIC ├── wine_bronze             <- tabelas Delta brutas
# MAGIC ├── wine_silver
# MAGIC └── wine_gold
# MAGIC ```
# MAGIC
# MAGIC Este notebook não faz upload, cópia nem ingestão dos CSVs.

# COMMAND ----------

CATALOG = "workspace"
SOURCE_SCHEMA = "wine_quality"
SOURCE_VOLUME = "raw_files"

LAYER_SCHEMAS = {
    "wine_bronze": "Camada Bronze: dados brutos em Delta com metadados de ingestão",
    "wine_silver": "Camada Silver: dados tipados, limpos e validados",
    "wine_gold": "Camada Gold: features para modelagem e tabelas analíticas",
}

RAW_PATH = f"/Volumes/{CATALOG}/{SOURCE_SCHEMA}/{SOURCE_VOLUME}"
EXPECTED_FILES = {"winequality-red.csv", "winequality-white.csv"}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Verificar o catálogo

# COMMAND ----------

available_catalogs = [row[0] for row in spark.sql("SHOW CATALOGS").collect()]
print("Catálogos disponíveis:", available_catalogs)

assert CATALOG in available_catalogs, (
    f"Catálogo '{CATALOG}' não encontrado. Disponíveis: {available_catalogs}"
)

spark.sql(f"USE CATALOG `{CATALOG}`")
print(f"Catálogo em uso: {CATALOG}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Criar os schemas das camadas
# MAGIC
# MAGIC O schema `wine_quality` e o Volume `raw_files` já existem e guardam os CSVs.

# COMMAND ----------

for schema_name, comment in LAYER_SCHEMAS.items():
    spark.sql(
        f"CREATE SCHEMA IF NOT EXISTS `{CATALOG}`.`{schema_name}` "
        f"COMMENT '{comment}'"
    )
    print(f"OK: {CATALOG}.{schema_name}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Verificar o Volume e os arquivos CSV
# MAGIC
# MAGIC Esta etapa verifica os arquivos já enviados.

# COMMAND ----------

available_schemas = {
    row[0] for row in spark.sql(f"SHOW SCHEMAS IN `{CATALOG}`").collect()
}
assert SOURCE_SCHEMA in available_schemas, (
    f"Schema de origem '{SOURCE_SCHEMA}' não encontrado em '{CATALOG}'."
)

volume_rows = spark.sql(
    f"SHOW VOLUMES IN `{CATALOG}`.`{SOURCE_SCHEMA}`"
).collect()
available_volumes = {row["volume_name"] for row in volume_rows}

assert SOURCE_VOLUME in available_volumes, (
    f"Volume '{SOURCE_VOLUME}' não encontrado em {CATALOG}.{SOURCE_SCHEMA}."
)

print(f"Volume de origem: {RAW_PATH}")

# COMMAND ----------

files = {item.name.rstrip("/") for item in dbutils.fs.ls(RAW_PATH)}
print("Arquivos encontrados:", sorted(files))

missing_files = EXPECTED_FILES - files
assert not missing_files, (
    f"Arquivos ausentes em {RAW_PATH}: {sorted(missing_files)}"
)
print("OK: os dois CSVs estão disponíveis para ingestão.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Resumo da estrutura

# COMMAND ----------

display(spark.sql(f"SHOW SCHEMAS IN `{CATALOG}`"))

# COMMAND ----------

display(spark.sql(f"SHOW VOLUMES IN `{CATALOG}`.`{SOURCE_SCHEMA}`"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximo passo
# MAGIC
# MAGIC Execute `01_bronze_ingestion.py` para ler os dois CSVs do Volume e gravar
# MAGIC a tabela Delta `workspace.wine_bronze.wine_quality_raw`.