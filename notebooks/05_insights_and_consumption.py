# Databricks notebook source
# MAGIC %md
# MAGIC # 05 · Insights e consumo — Da camada Gold ao apoio à decisão
# MAGIC
# MAGIC Este notebook mostra como as tabelas Gold são consumidas: auditoria do fluxo, consultas SQL de apoio à decisão,
# MAGIC gráficos para o README e um resumo dos resultados gerado a partir dos próprios dados.
# MAGIC
# MAGIC **Origem:** tabelas das camadas Bronze, Silver e Gold (incluindo as tabelas do modelo)
# MAGIC **Destino:** Volume `workspace.wine_gold.reports`, com os gráficos (`figures/`), `results_summary.md` e `model_comparison.csv`
# MAGIC
# MAGIC **O que este notebook não faz:** não treina nem altera nenhum modelo e não grava nas tabelas das camadas.
# MAGIC Todos os números do resumo vêm das tabelas, e nenhum é escrito à mão.
# MAGIC
# MAGIC **Seções**
# MAGIC 1. Configuração e pré-requisitos
# MAGIC 2. Auditoria do fluxo (reconciliação entre camadas e qualidade dos dados)
# MAGIC 3. Resultados do modelo (última execução)
# MAGIC 4. Gráficos
# MAGIC 5. Consultas SQL de apoio à decisão
# MAGIC 6. Histórico Delta das camadas
# MAGIC 7. Resumo dos resultados e arquivos para o repositório

# COMMAND ----------

# MAGIC %md
# MAGIC ## 1. Configuração

# COMMAND ----------

import os
from datetime import datetime, timezone

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from pyspark.sql import functions as F

CATALOG = "workspace"
BRONZE_SCHEMA = "wine_bronze"
SILVER_SCHEMA = "wine_silver"
GOLD_SCHEMA = "wine_gold"

BRONZE_TABLE = f"{CATALOG}.{BRONZE_SCHEMA}.wine_quality_raw"
SILVER_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.wine_quality_clean"
REJECTED_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.wine_quality_rejected"
DQ_TABLE = f"{CATALOG}.{SILVER_SCHEMA}.dq_results"
ML_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.wine_ml_dataset"
SUMMARY_TYPE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.quality_summary_by_type"
ALCOHOL_BAND_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.quality_by_alcohol_band"
METRICS_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.model_metrics"
PREDICTIONS_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.model_predictions"
IMPORTANCE_TABLE = f"{CATALOG}.{GOLD_SCHEMA}.model_feature_importance"

REPORTS_VOLUME = "reports"
REPORTS_PATH = f"/Volumes/{CATALOG}/{GOLD_SCHEMA}/{REPORTS_VOLUME}"
FIGURES_PATH = f"{REPORTS_PATH}/figures"

RED_COLOR = "#8B1E3F"
WHITE_COLOR = "#C9A227"
NEUTRAL_COLOR = "#4C6A92"
LIGHT_COLOR = "#A9B8CC"

# COMMAND ----------

# MAGIC %md
# MAGIC ## 2. Verificar pré-requisitos
# MAGIC
# MAGIC Os notebooks 00 a 04 devem ter sido executados. Também criamos o Volume que recebe os arquivos gerados aqui.

# COMMAND ----------

spark.sql(f"USE CATALOG `{CATALOG}`")

required_tables = [
    BRONZE_TABLE, SILVER_TABLE, REJECTED_TABLE, DQ_TABLE, ML_TABLE,
    SUMMARY_TYPE_TABLE, ALCOHOL_BAND_TABLE, METRICS_TABLE, PREDICTIONS_TABLE, IMPORTANCE_TABLE,
]
missing_tables = [t for t in required_tables if not spark.catalog.tableExists(t)]
assert not missing_tables, (
    f"Tabelas ausentes: {missing_tables}. Execute os notebooks 00 a 04 antes deste."
)

spark.sql(
    f"CREATE VOLUME IF NOT EXISTS `{CATALOG}`.`{GOLD_SCHEMA}`.`{REPORTS_VOLUME}` "
    "COMMENT 'Graficos e resumos gerados a partir das tabelas Gold, para uso no repositorio'"
)
os.makedirs(FIGURES_PATH, exist_ok=True)

print(f"Todas as {len(required_tables)} tabelas encontradas.")
print(f"Arquivos serão gravados em: {REPORTS_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 3. Auditoria do fluxo
# MAGIC
# MAGIC ### 3.1 Reconciliação entre camadas
# MAGIC
# MAGIC Quantos registros existem em cada camada e se a conta fecha: cada registro da Bronze deve estar na Silver
# MAGIC (como uma linha, podendo representar várias duplicatas exatas, registradas em `occurrences`) ou na quarentena.

# COMMAND ----------

bronze_rows = spark.table(BRONZE_TABLE).count()
silver_rows = spark.table(SILVER_TABLE).count()
rejected_rows = spark.table(REJECTED_TABLE).count()
ml_rows = spark.table(ML_TABLE).count()

represented_rows = spark.table(SILVER_TABLE).agg(F.sum("occurrences")).first()[0] or 0
duplicates_removed = represented_rows - silver_rows

layer_counts = spark.createDataFrame(
    [
        ("1. Bronze", BRONZE_TABLE, bronze_rows),
        ("2. Silver", SILVER_TABLE, silver_rows),
        ("2. Quarentena", REJECTED_TABLE, rejected_rows),
        ("3. Gold (base de modelagem)", ML_TABLE, ml_rows),
    ],
    "layer string, table_name string, records long",
)
display(layer_counts)

reconciled = bronze_rows == represented_rows + rejected_rows
print(f"Bronze: {bronze_rows} | Silver: {silver_rows} | Quarentena: {rejected_rows} | Gold: {ml_rows}")
print(f"Duplicatas exatas removidas: {duplicates_removed}")
print(
    "Reconciliação: " + ("OK (Bronze = Silver representada + quarentena)" if reconciled
                         else f"ATENÇÃO: Bronze ({bronze_rows}) difere de Silver representada ({represented_rows}) + quarentena ({rejected_rows})")
)
print("Silver e Gold com o mesmo número de registros: " + ("sim" if silver_rows == ml_rows else "NÃO"))
if not reconciled:
    print(
        "ATENÇÃO: a reconciliação entre Bronze, Silver representada e quarentena não fechou. "
        "Investigue antes de publicar os resultados."
    )
if silver_rows != ml_rows:
    print(
        "ATENÇÃO: Silver e Gold têm contagens diferentes. Verifique filtros e divisão treino/teste "
        "antes de interpretar a base de modelagem."
    )

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.2 Qualidade dos dados (última execução da Silver)

# COMMAND ----------

dq = spark.table(DQ_TABLE)
latest_dq_run = dq.agg(F.max("run_id")).first()[0]

display(
    dq.filter(F.col("run_id") == latest_dq_run)
    .select("check_name", "description", "affected_rows", "total_rows")
    .orderBy(F.col("affected_rows").desc(), "check_name")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 4. Resultados do modelo
# MAGIC
# MAGIC Usamos a execução mais recente do notebook 04. Como o conjunto de teste é avaliado **somente para o candidato
# MAGIC selecionado pela validação cruzada**, as colunas `test_*` dos demais candidatos ficam vazias, e isso é intencional.

# COMMAND ----------

metrics_all = spark.table(METRICS_TABLE)
assert metrics_all.limit(1).count() > 0, "A tabela de métricas está vazia. Execute o notebook 04."
latest_run_id = (
    metrics_all.orderBy(F.col("run_timestamp").desc(), F.col("run_id").desc())
    .select("run_id").first()["run_id"]
)
latest = metrics_all.filter(F.col("run_id") == latest_run_id)

display(
    latest.select(
        "model_name", "feature_set", "n_features", "cv_auc_pr", "train_auc_pr",
        "test_auc_pr", "test_auc_roc", "test_precision", "test_recall", "test_f1",
        "best_params", "selected",
    ).orderBy(F.col("cv_auc_pr").desc())
)

selected_rows = latest.filter(F.col("selected") == True).limit(2).collect()
assert len(selected_rows) == 1, (
    f"A execução {latest_run_id} deve ter exatamente um modelo selecionado; "
    f"foram encontrados {len(selected_rows)}."
)
selected_row = selected_rows[0]

required_test_metrics = [
    "test_auc_pr", "test_auc_roc", "test_precision", "test_recall",
    "test_f1", "test_positive_rate",
]
missing_test_metrics = [
    name for name in required_test_metrics if selected_row[name] is None
]
assert not missing_test_metrics, (
    f"Métricas de teste ausentes para o modelo selecionado: {missing_test_metrics}. "
    "Verifique a execução do notebook 04."
)

# O notebook 04 mantém histórico de métricas, mas sobrescreve previsões e importâncias
# a cada execução. Validamos o run_id para impedir a mistura de execuções diferentes.
latest_predictions = (
    spark.table(PREDICTIONS_TABLE)
    .filter(F.col("run_id") == latest_run_id)
)
assert latest_predictions.limit(1).count() == 1, (
    f"As previsões não correspondem à execução mais recente ({latest_run_id}). "
    "Execute novamente o notebook 04 antes deste notebook."
)

prediction_metadata = latest_predictions.select(
    "model_name", "feature_set"
).distinct().collect()
assert len(prediction_metadata) == 1, "A tabela de previsões contém mais de um modelo/feature set."
assert (
    prediction_metadata[0]["model_name"] == selected_row["model_name"]
    and prediction_metadata[0]["feature_set"] == selected_row["feature_set"]
), "As previsões não correspondem ao modelo selecionado na tabela de métricas."

prediction_count = latest_predictions.count()
assert prediction_count > 0, "A tabela de previsões está vazia."
prediction_nulls = latest_predictions.filter(
    F.col("label").isNull()
    | F.col("prediction").isNull()
    | F.col("prob_good").isNull()
).count()
assert prediction_nulls == 0, f"Há {prediction_nulls} previsões com valores nulos."
assert latest_predictions.filter(
    (F.col("prob_good") < 0) | (F.col("prob_good") > 1)
).count() == 0, "Há probabilidades fora do intervalo [0, 1]."

latest_predictions.createOrReplaceTempView("latest_model_predictions")

print(f"Execução do modelo: {latest_run_id}")
print(f"Modelo selecionado: {selected_row['model_name']} | {selected_row['feature_set']}")
print(f"Previsões validadas: {prediction_count} registros.")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 5. Gráficos
# MAGIC
# MAGIC Os gráficos são salvos como PNG no Volume `reports`, prontos para a pasta `docs/` do repositório.

# COMMAND ----------

def save_figure(fig, name: str) -> None:
    path = f"{FIGURES_PATH}/{name}.png"
    fig.savefig(path, dpi=150, bbox_inches="tight")
    print(f"Gráfico salvo: {path}")
    plt.show()
    plt.close(fig)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.1 Distribuição das notas por tipo de vinho
# MAGIC
# MAGIC Em percentual de cada tipo, porque o branco tem muito mais registros que o tinto.

# COMMAND ----------

quality_counts = (
    spark.table(ML_TABLE).groupBy("wine_type", "quality").count().toPandas()
)
quality_counts["pct"] = (
    quality_counts["count"] / quality_counts.groupby("wine_type")["count"].transform("sum") * 100
)

qualities = sorted(quality_counts["quality"].unique())
x = np.arange(len(qualities))
width = 0.38

fig, ax = plt.subplots(figsize=(8, 4.5))
for offset, wine_type, color, label in [(-0.5, "red", RED_COLOR, "Tinto"), (0.5, "white", WHITE_COLOR, "Branco")]:
    sub = quality_counts[quality_counts["wine_type"] == wine_type].set_index("quality").reindex(qualities)
    bars = ax.bar(x + offset * width, sub["pct"].fillna(0), width, label=label, color=color)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)

ax.set_xticks(x)
ax.set_xticklabels([str(q) for q in qualities])
ax.set_xlabel("Nota de qualidade")
ax.set_ylabel("% dos vinhos do tipo")
ax.set_title("Distribuição das notas por tipo de vinho")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)

save_figure(fig, "01_quality_distribution")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Vinhos bons por faixa de álcool
# MAGIC
# MAGIC Vinho "bom" é o de `quality >= 7`. A relação é descritiva e não prova que o álcool cause a qualidade.

# COMMAND ----------

alcohol_pdf = spark.table(ALCOHOL_BAND_TABLE).toPandas()
bands = (
    alcohol_pdf.drop_duplicates("band_order").sort_values("band_order")["alcohol_band"].tolist()
)
x = np.arange(len(bands))

fig, ax = plt.subplots(figsize=(8, 4.5))
for offset, wine_type, color, label in [(-0.5, "red", RED_COLOR, "Tinto"), (0.5, "white", WHITE_COLOR, "Branco")]:
    sub = alcohol_pdf[alcohol_pdf["wine_type"] == wine_type].set_index("alcohol_band").reindex(bands)
    bars = ax.bar(x + offset * width, sub["pct_good"].fillna(0), width, label=label, color=color)
    ax.bar_label(bars, fmt="%.0f%%", fontsize=8, padding=2)

ax.set_xticks(x)
ax.set_xticklabels(bands)
ax.set_xlabel("Teor alcoólico (% vol)")
ax.set_ylabel("% de vinhos bons (nota ≥ 7)")
ax.set_title("Proporção de vinhos bons por faixa de álcool")
ax.legend(frameon=False)
ax.spines[["top", "right"]].set_visible(False)

save_figure(fig, "02_good_wines_by_alcohol")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.3 Comparação dos candidatos
# MAGIC
# MAGIC AUC-PR na validação cruzada (critério de seleção) e no treino (diagnóstico de sobreajuste). A linha tracejada é a proporção de
# MAGIC vinhos bons no teste, a referência de um modelo sem informação. O losango é a AUC-PR no teste, medida apenas para o modelo selecionado.

# COMMAND ----------

candidates = latest.toPandas().sort_values("cv_auc_pr", ascending=False).reset_index(drop=True)
labels = [
    f"{row.model_name.replace('_', ' ')}\n{row.feature_set.replace('_', ' ')}"
    for row in candidates.itertuples()
]
x = np.arange(len(candidates))

fig, ax = plt.subplots(figsize=(9, 4.8))
ax.bar(x - 0.2, candidates["cv_auc_pr"], 0.4, label="Validação cruzada", color=NEUTRAL_COLOR)
ax.bar(x + 0.2, candidates["train_auc_pr"], 0.4, label="Treino", color=LIGHT_COLOR)

selected_index = candidates.index[candidates["selected"]].tolist()[0]
ax.scatter(
    [selected_index], [selected_row["test_auc_pr"]],
    marker="D", s=70, color="black", zorder=3, label="Teste (modelo selecionado)",
)
ax.axhline(
    selected_row["test_positive_rate"], linestyle="--", linewidth=1, color="gray",
    label="Referência sem informação",
)

ax.set_xticks(x)
ax.set_xticklabels(labels, fontsize=8)
ax.set_ylabel("AUC-PR")
ax.set_title("Comparação dos candidatos (AUC-PR)")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

save_figure(fig, "03_model_comparison")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.4 Atributos mais influentes no modelo selecionado
# MAGIC
# MAGIC Para florestas aleatórias, é a importância do atributo; para regressão logística, o coeficiente sobre atributos padronizados
# MAGIC (o sinal indica a direção). Descreve o que o modelo usa, e não o que causa qualidade.

# COMMAND ----------

importance_pdf = (
    spark.table(IMPORTANCE_TABLE)
    .filter(
        (F.col("run_id") == latest_run_id)
        & (F.col("model_name") == selected_row["model_name"])
        & (F.col("feature_set") == selected_row["feature_set"])
    )
    .toPandas()
)
assert not importance_pdf.empty, (
    f"Sem importâncias para o modelo selecionado na execução {latest_run_id}. "
    "Execute novamente o notebook 04."
)

importance_type = importance_pdf["importance_type"].iloc[0]
top = (
    importance_pdf.assign(abs_importance=importance_pdf["importance"].abs())
    .sort_values("abs_importance", ascending=False)
    .head(12)
    .sort_values("abs_importance")
)

fig, ax = plt.subplots(figsize=(8, 5))
colors = [NEUTRAL_COLOR if v >= 0 else RED_COLOR for v in top["importance"]]
ax.barh(top["feature"], top["importance"], color=colors)
ax.set_xlabel(
    "Importância do atributo" if importance_type == "rf_feature_importance"
    else "Coeficiente (atributos padronizados)"
)
ax.set_title(f"Atributos mais influentes: {selected_row['model_name'].replace('_', ' ')}")
ax.spines[["top", "right"]].set_visible(False)

save_figure(fig, "04_feature_importance")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.5 Curva precisão × recall do modelo selecionado (teste)
# MAGIC
# MAGIC O ponto marca o desempenho com o limiar padrão de 0,5. Esta curva descreve o modelo, mas **não deve ser usada para escolher
# MAGIC um limiar**: isso exigiria validação, e o teste só serve para a avaliação final.

# COMMAND ----------

pred_pdf = (
    latest_predictions.select("label", "prob_good").toPandas()
    .sort_values("prob_good", ascending=False)
    .reset_index(drop=True)
)

assert not pred_pdf.empty, "Não há previsões para construir a curva precisão-recall."
total_positives = int(pred_pdf["label"].sum())
assert total_positives > 0, "O conjunto de teste não contém exemplos positivos; não é possível calcular recall."
true_positives = pred_pdf["label"].cumsum()
precision_curve = true_positives / np.arange(1, len(pred_pdf) + 1)
recall_curve = true_positives / total_positives

fig, ax = plt.subplots(figsize=(6.5, 5))
ax.step(recall_curve, precision_curve, where="post", color=NEUTRAL_COLOR, label="Modelo selecionado")
ax.axhline(
    selected_row["test_positive_rate"], linestyle="--", linewidth=1, color="gray",
    label="Referência sem informação",
)
ax.scatter(
    [selected_row["test_recall"]], [selected_row["test_precision"]],
    color="black", zorder=3, label="Limiar 0,5",
)
ax.set_xlabel("Recall (vinhos bons encontrados)")
ax.set_ylabel("Precisão (acerto entre os indicados)")
ax.set_xlim(0, 1.02)
ax.set_ylim(0, 1.02)
ax.set_title("Curva precisão × recall no teste")
ax.legend(frameon=False, fontsize=8)
ax.spines[["top", "right"]].set_visible(False)

save_figure(fig, "05_precision_recall_curve")

# COMMAND ----------

# MAGIC %md
# MAGIC ## 6. Consultas SQL de apoio à decisão
# MAGIC
# MAGIC Consultas sobre as tabelas Gold, que podem ser coladas no SQL Editor do Databricks ou ligadas a um painel.
# MAGIC Os resultados dependem do modelo selecionado e são **descritivos**: o teste não deve ser usado para escolher limiares.

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.1 O que o modelo faz com as probabilidades: taxa real por faixa
# MAGIC
# MAGIC Se o modelo ordena bem os vinhos, a proporção de vinhos realmente bons deve crescer a cada faixa de probabilidade.
# MAGIC Como o treino usou pesos de classe, as probabilidades **não são calibradas** (a probabilidade média não é a taxa real).

# COMMAND ----------

probability_bands = spark.sql(
    f"""
    SELECT
        CASE
            WHEN prob_good < 0.2 THEN '0.0 a 0.2'
            WHEN prob_good < 0.4 THEN '0.2 a 0.4'
            WHEN prob_good < 0.6 THEN '0.4 a 0.6'
            WHEN prob_good < 0.8 THEN '0.6 a 0.8'
            ELSE '0.8 a 1.0'
        END AS probability_band,
        COUNT(*) AS wines,
        SUM(CAST(label AS INT)) AS actually_good,
        ROUND(AVG(label) * 100, 1) AS pct_actually_good,
        ROUND(AVG(prob_good), 3) AS avg_predicted_probability
    FROM latest_model_predictions
    GROUP BY 1
    ORDER BY 1
    """
)
display(probability_bands)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.2 Quantos vinhos revisar e quantos vinhos bons encontrar
# MAGIC
# MAGIC Cenário de decisão: se apenas os vinhos com probabilidade acima de um limiar fossem separados para uma avaliação mais cuidadosa,
# MAGIC quanto do conjunto seria revisado e que parte dos vinhos bons seria encontrada? Um limiar mais alto revisa menos vinhos e acerta mais
# MAGIC entre os indicados, mas deixa escapar mais vinhos bons.

# COMMAND ----------

review_scenarios = spark.sql(
    f"""
    WITH thresholds AS (
        SELECT explode(array(0.3, 0.4, 0.5, 0.6, 0.7, 0.8)) AS threshold
    ),
    base AS (
        SELECT CAST(COUNT(*) AS DOUBLE) AS n,
               CAST(SUM(CAST(label AS INT)) AS DOUBLE) AS n_good
        FROM latest_model_predictions
    ),
    flagged AS (
        SELECT t.threshold,
               SUM(CASE WHEN p.prob_good >= t.threshold THEN 1 ELSE 0 END) AS wines_flagged,
               SUM(CASE WHEN p.prob_good >= t.threshold AND p.label = 1.0 THEN 1 ELSE 0 END) AS good_found
        FROM latest_model_predictions p
        CROSS JOIN thresholds t
        GROUP BY t.threshold
    )
    SELECT
        CAST(f.threshold AS DOUBLE) AS threshold,
        f.wines_flagged,
        ROUND(100.0 * f.wines_flagged / b.n, 1) AS pct_of_wines_reviewed,
        f.good_found,
        ROUND(100.0 * f.good_found / b.n_good, 1) AS pct_of_good_wines_found,
        ROUND(100.0 * f.good_found / NULLIF(f.wines_flagged, 0), 1) AS pct_correct_among_flagged
    FROM flagged f
    CROSS JOIN base b
    ORDER BY f.threshold
    """
)
display(review_scenarios)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.3 Análise de erros: onde o modelo mais se engana
# MAGIC
# MAGIC Os vinhos que o modelo indicou como bons com mais confiança e não eram (falsos positivos), e os vinhos bons que ele
# MAGIC considerou menos prováveis (falsos negativos). Servem para investigar o que o modelo não captura nas medições.

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT p.wine_id, p.wine_type, d.quality, ROUND(p.prob_good, 3) AS prob_good,
               d.alcohol, d.volatile_acidity, d.sulphates, d.density
        FROM latest_model_predictions p
        JOIN {ML_TABLE} d ON p.wine_id = d.wine_id
        WHERE p.label = 0.0 AND p.prediction = 1.0
        ORDER BY p.prob_good DESC
        LIMIT 10
        """
    )
)

# COMMAND ----------

display(
    spark.sql(
        f"""
        SELECT p.wine_id, p.wine_type, d.quality, ROUND(p.prob_good, 3) AS prob_good,
               d.alcohol, d.volatile_acidity, d.sulphates, d.density
        FROM latest_model_predictions p
        JOIN {ML_TABLE} d ON p.wine_id = d.wine_id
        WHERE p.label = 1.0 AND p.prediction = 0.0
        ORDER BY p.prob_good ASC
        LIMIT 10
        """
    )
)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 6.4 Desempenho por tipo de vinho

# COMMAND ----------

by_type = spark.sql(
    f"""
    SELECT
        wine_type,
        COUNT(*) AS wines,
        SUM(CAST(label AS INT)) AS good_wines,
        SUM(CASE WHEN prediction = 1.0 AND label = 1.0 THEN 1 ELSE 0 END) AS true_positives,
        SUM(CASE WHEN prediction = 1.0 AND label = 0.0 THEN 1 ELSE 0 END) AS false_positives,
        ROUND(SUM(CASE WHEN prediction = 1.0 AND label = 1.0 THEN 1 ELSE 0 END)
              / NULLIF(SUM(CAST(label AS INT)), 0), 3) AS recall,
        ROUND(SUM(CASE WHEN prediction = 1.0 AND label = 1.0 THEN 1 ELSE 0 END)
              / NULLIF(SUM(CASE WHEN prediction = 1.0 THEN 1 ELSE 0 END), 0), 3) AS precision
    FROM latest_model_predictions
    GROUP BY wine_type
    ORDER BY wine_type
    """
)
display(by_type)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 7. Histórico Delta das camadas
# MAGIC
# MAGIC Número de versões e última operação de cada tabela principal. O Delta preserva as versões anteriores, que podem ser consultadas
# MAGIC com `VERSION AS OF` (viagem no tempo).

# COMMAND ----------

history_tables = [BRONZE_TABLE, SILVER_TABLE, ML_TABLE, METRICS_TABLE, PREDICTIONS_TABLE]
history_rows = []

for table in history_tables:
    history = spark.sql(f"DESCRIBE HISTORY {table}")
    latest_version = history.agg(F.max("version")).first()[0]
    last_operation = (
        history.orderBy(F.col("version").desc()).select("operation").first()
        if latest_version is not None
        else None
    )
    history_rows.append(
        (table, int((latest_version or 0) + 1), last_operation["operation"] if last_operation else None)
    )

display(
    spark.createDataFrame(history_rows, "table_name string, versions long, last_operation string")
)

# COMMAND ----------

# MAGIC %md
# MAGIC ## 8. Resumo dos resultados e arquivos para o repositório
# MAGIC
# MAGIC Gera `results_summary.md` e `model_comparison.csv` a partir das tabelas, sem nenhum número digitado à mão.

# COMMAND ----------

type_summary = spark.table(SUMMARY_TYPE_TABLE).toPandas().set_index("wine_type")


def pct(value) -> str:
    return f"{value * 100:.1f}%".replace(".", ",")


def num(value, digits: int = 3) -> str:
    if value is None or pd.isna(value):
        return "não disponível"
    return f"{float(value):.{digits}f}".replace(".", ",")


selected_label = f"{selected_row['model_name'].replace('_', ' ')} ({selected_row['feature_set'].replace('_', ' ')})"

lines = [
    "# Resumo dos resultados",
    "",
    f"Gerado em {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M UTC')} a partir das tabelas do projeto "
    f"(execução do modelo: `{latest_run_id}`).",
    "",
    "## Fluxo de dados",
    "",
    f"- Bronze: {bronze_rows} registros (vinho tinto e branco).",
    f"- Silver: {silver_rows} registros após a remoção de {duplicates_removed} duplicatas exatas; {rejected_rows} registros na quarentena.",
    f"- Gold (base de modelagem): {ml_rows} registros.",
    "",
    "## Vinhos bons (nota >= 7)",
    "",
]

for wine_type, label in [("red", "Tinto"), ("white", "Branco")]:
    if wine_type in type_summary.index:
        row = type_summary.loc[wine_type]
        lines.append(
            f"- {label}: {int(row['records'])} vinhos, {int(row['good_wines'])} bons ({str(row['pct_good']).replace('.', ',')}%), "
            f"nota média {str(row['avg_quality']).replace('.', ',')}."
        )

lines += [
    "",
    "## Modelo selecionado",
    "",
    f"Selecionado pela AUC-PR média na validação cruzada: **{selected_label}**.",
    "",
    f"- AUC-PR na validação cruzada: {num(selected_row['cv_auc_pr'])}",
    f"- AUC-PR no treino: {num(selected_row['train_auc_pr'])}",
    f"- AUC-PR no teste: {num(selected_row['test_auc_pr'])} (referência sem informação: {num(selected_row['test_positive_rate'])})",
    f"- Previsões avaliadas no teste: {prediction_count}",
    f"- AUC-ROC no teste: {num(selected_row['test_auc_roc'])}",
    f"- Limiar 0,5 no teste: precisão {num(selected_row['test_precision'])}, recall {num(selected_row['test_recall'])}, F1 {num(selected_row['test_f1'])}",
    "",
    "O conjunto de teste foi avaliado apenas para o candidato selecionado. As métricas de teste têm incerteza considerável,",
    "porque o teste contém poucos vinhos bons.",
    "",
    "## Candidatos comparados (validação cruzada)",
    "",
    "| Modelo | Atributos | AUC-PR (CV) | AUC-PR (treino) |",
    "|---|---|---|---|",
]

for row in candidates.itertuples():
    lines.append(
        f"| {row.model_name.replace('_', ' ')} | {row.feature_set.replace('_', ' ')} | {num(row.cv_auc_pr)} | {num(row.train_auc_pr)} |"
    )

summary_path = f"{REPORTS_PATH}/results_summary.md"
with open(summary_path, "w", encoding="utf-8") as file:
    file.write("\n".join(lines) + "\n")

comparison_path = f"{REPORTS_PATH}/model_comparison.csv"
candidates.drop(columns=["run_timestamp"], errors="ignore").to_csv(comparison_path, index=False)

print(f"Resumo salvo: {summary_path}")
print(f"Tabela salva: {comparison_path}")
print()
print("\n".join(lines))

# COMMAND ----------

# MAGIC %md
# MAGIC ### Arquivos gerados

# COMMAND ----------

generated = []
for folder, _, files in os.walk(REPORTS_PATH):
    for name in sorted(files):
        path = os.path.join(folder, name)
        generated.append((path.replace(REPORTS_PATH + "/", ""), os.path.getsize(path)))

display(spark.createDataFrame(sorted(generated), "file string, size_bytes long"))

# COMMAND ----------

# MAGIC %md
# MAGIC ## Como levar os arquivos para o GitHub
# MAGIC
# MAGIC 1. No Databricks, abra `Catalog` → `workspace` → `wine_gold` → `reports` e baixe os arquivos (os PNGs de `figures/`, `results_summary.md` e `model_comparison.csv`).
# MAGIC 2. Coloque os PNGs em `docs/figures/` do repositório e referencie-os no README.
# MAGIC 3. Use `results_summary.md` como base para a seção de resultados do README, revisando o texto antes de publicar.
# MAGIC 4. Exporte os notebooks do Databricks como `.py` (formato fonte) para a pasta `notebooks/`.
# MAGIC 5. Não versione os CSVs originais; eles permanecem no Volume. Revise os resultados e imagens antes de publicar.
# MAGIC
# MAGIC ## Próximo passo
# MAGIC
# MAGIC `README.md` do projeto: visão geral, diagrama da arquitetura medalhão, decisões de projeto, resultados, limitações e como reproduzir.
