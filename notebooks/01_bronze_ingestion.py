# Databricks notebook source
# MAGIC %md
# MAGIC # 01 · Camada Bronze — Ingestão do Wine Quality
# MAGIC
# MAGIC **Objetivo:** levar os dados brutos (vinho tinto e branco, UCI) para uma tabela Delta governada pelo Unity Catalog,
# MAGIC preservando o conteúdo original e adicionando apenas metadados de rastreabilidade.
# MAGIC
# MAGIC **Princípios da Bronze neste projeto**
# MAGIC - Nenhuma regra de negócio: sem remover duplicatas, sem tratar outliers, sem converter tipos.
# MAGIC - Todas as colunas de dados ficam como `string` (a tipagem é responsabilidade da Silver).
# MAGIC - Única alteração nos dados: nomes de colunas em `snake_case` (o Delta não aceita espaços nos nomes por padrão).
# MAGIC - Metadados adicionados: `wine_type`, `source_file`, `source_path`, `ingestion_timestamp`.
# MAGIC - Ingestão **idempotente**: rodar o notebook mais de uma vez não duplica dados.
# MAGIC
# MAGIC **Fonte:** P. Cortez, A. Cerdeira, F. Almeida, T. Matos e J. Reis. *Modeling wine preferences by data mining from
# MAGIC physicochemical properties.* Decision Support Systems, 47(4):547-553, 2009. https://doi.org/10.1016/j.dss.2009.05.016

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração
# MAGIC
# MAGIC Ajuste `CATALOG` se o seu workspace usar outro catálogo (no Free Edition o padrão costuma ser `workspace`).

# COMMAND ----------

import re
from pyspark.sql import functions as F

CATALOG = "workspace"
SCHEMA_BRONZE = "wine_bronze"
SCHEMA_SILVER = "wine_silver"
SCHEMA_GOLD = "wine_gold"
VOLUME = "raw_files"

RAW_PATH = f"/Volumes/{CATALOG}/{SCHEMA_BRONZE}/{VOLUME}"
BRONZE_TABLE = f"{CATALOG}.{SCHEMA_BRONZE}.wine_quality_raw"

# arquivo de origem -> tipo de vinho
SOURCES = {
    "winequality-red.csv": "red",
    "winequality-white.csv": "white",
}

# contagens esperadas, conforme winequality.names
EXPECTED_ROWS = {
    "winequality-red.csv": 1599,
    "winequality-white.csv": 4898,
}

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Setup do catálogo (idempotente)
# MAGIC
# MAGIC Cria um schema por camada da arquitetura medalhão e o Volume que recebe os arquivos brutos.

# COMMAND ----------

for schema in (SCHEMA_BRONZE, SCHEMA_SILVER, SCHEMA_GOLD):
    spark.sql(f"CREATE SCHEMA IF NOT EXISTS {CATALOG}.{schema}")

spark.sql(f"CREATE VOLUME IF NOT EXISTS {CATALOG}.{SCHEMA_BRONZE}.{VOLUME}")

print(f"Volume pronto em: {RAW_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Upload dos arquivos brutos
# MAGIC
# MAGIC Faça o upload manual **uma vez**:
# MAGIC `Catalog` → `workspace` → `wine_bronze` → `raw_files` → **Upload to this volume**,
# MAGIC enviando `winequality-red.csv` e `winequality-white.csv`.
# MAGIC
# MAGIC A célula abaixo confere se os arquivos estão lá antes de continuar.

# COMMAND ----------

present = {f.name for f in dbutils.fs.ls(RAW_PATH)}
missing = set(SOURCES) - present
assert not missing, f"Arquivos não encontrados no Volume {RAW_PATH}: {sorted(missing)}"

display(dbutils.fs.ls(RAW_PATH))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Funções de leitura
# MAGIC
# MAGIC Os CSVs da UCI usam `;` como separador e cabeçalho entre aspas. Lemos tudo como `string`
# MAGIC (`inferSchema=False`), mantendo o dado bruto.

# COMMAND ----------

def to_snake_case(name: str) -> str:
    """'fixed acidity' -> 'fixed_acidity'; 'pH' -> 'ph'."""
    return re.sub(r"[^0-9a-zA-Z]+", "_", name.strip()).strip("_").lower()


def read_raw(file_name: str, wine_type: str):
    path = f"{RAW_PATH}/{file_name}"
    df = (
        spark.read.format("csv")
        .option("header", True)
        .option("sep", ";")
        .option("quote", '"')
        .option("inferSchema", False)
        .load(path)
    )
    df = df.toDF(*[to_snake_case(c) for c in df.columns])
    return (
        df.withColumn("wine_type", F.lit(wine_type))
        .withColumn("source_file", F.lit(file_name))
        .withColumn("source_path", F.lit(path))
        .withColumn("ingestion_timestamp", F.current_timestamp())
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Ingestão idempotente na tabela Bronze
# MAGIC
# MAGIC Para cada arquivo, usamos `replaceWhere` com `source_file`: se o arquivo já foi carregado, **somente as linhas dele
# MAGIC são substituídas**; se for a primeira carga, as linhas são inseridas. Assim, reexecutar o notebook não gera duplicatas.

# COMMAND ----------

for file_name, wine_type in SOURCES.items():
    df = read_raw(file_name, wine_type)
    writer = df.write.format("delta").mode("overwrite")

    if spark.catalog.tableExists(BRONZE_TABLE):
        writer = writer.option("replaceWhere", f"source_file = '{file_name}'")

    writer.saveAsTable(BRONZE_TABLE)
    print(f"Ingerido: {file_name} ({wine_type})")

spark.sql(
    f"COMMENT ON TABLE {BRONZE_TABLE} IS "
    "'Bronze: Wine Quality (Cortez et al., 2009), vinho tinto e branco, dados brutos como string + metadados de ingestao'"
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Validações da Bronze
# MAGIC
# MAGIC Conferimos a contagem por arquivo contra o esperado e verificamos o schema.

# COMMAND ----------

bronze = spark.table(BRONZE_TABLE)

counts = {
    r["source_file"]: r["n"]
    for r in bronze.groupBy("source_file").agg(F.count("*").alias("n")).collect()
}

for file_name, expected in EXPECTED_ROWS.items():
    got = counts.get(file_name, 0)
    assert got == expected, f"{file_name}: esperado {expected}, obtido {got}"
    print(f"OK  {file_name}: {got} linhas")

print(f"Total na Bronze: {bronze.count()} linhas")
bronze.printSchema()

# COMMAND ----------

display(bronze.limit(10))

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Evidência de idempotência e histórico Delta
# MAGIC
# MAGIC **Teste:** execute novamente a seção 5 e depois a seção 6. As contagens devem continuar iguais
# MAGIC (1599 + 4898 = 6497), e o histórico abaixo mostrará novas versões da tabela.

# COMMAND ----------

display(spark.sql(f"DESCRIBE HISTORY {BRONZE_TABLE}"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Próximo passo
# MAGIC
# MAGIC `02_silver_cleaning`: conversão de tipos, checagens de qualidade, tratamento de duplicatas
# MAGIC (240 no tinto e 937 no branco) e padronização da variável `quality`.
