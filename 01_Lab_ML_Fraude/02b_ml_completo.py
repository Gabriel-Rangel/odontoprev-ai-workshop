# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 01 — Deteccao de Fraude em Reembolsos Odontologicos
# MAGIC ## Gabarito (solucao de referencia)
# MAGIC
# MAGIC **Objetivo:** construir um modelo que priorize a fila da auditoria de reembolsos,
# MAGIC usando **PySpark, scikit-learn, MLflow e Unity Catalog**.
# MAGIC
# MAGIC ### Contexto de negocio
# MAGIC
# MAGIC A Odontoprev paga reembolso quando o beneficiario usa a **livre escolha**: ele
# MAGIC procura um dentista fora da rede, paga, e pede o valor de volta apresentando nota
# MAGIC fiscal. A operadora nao viu o atendimento — so ve o documento.
# MAGIC
# MAGIC Hoje a regra e simples: **pedido acima de R$ 1.500 vai para analise manual.**
# MAGIC O resultado e o pior dos dois mundos:
# MAGIC
# MAGIC - a auditoria analisa uma montanha de pedidos legitimos e caros (implante, protese);
# MAGIC - e as fraudes bem feitas passam batido, porque foram desenhadas para ficar **logo
# MAGIC   abaixo** do teto.
# MAGIC
# MAGIC **Sua missao:** substituir a regra por um modelo que ordene os pedidos por risco,
# MAGIC e definir o ponto de corte que **maximiza o beneficio financeiro** da operacao.
# MAGIC
# MAGIC ### O que vamos construir
# MAGIC
# MAGIC | Etapa | O que faz | Tecnica |
# MAGIC |-------|-----------|---------|
# MAGIC | **Diagnostico** | Medir a regra atual e achar vazamento de alvo | PySpark + analise critica |
# MAGIC | **Feature Engineering** | Transformar linhas isoladas em contexto relacional | Window Functions |
# MAGIC | **Treinamento** | Classificador com classes desbalanceadas | scikit-learn + MLflow |
# MAGIC | **Decisao** | Threshold pelo custo do negocio, nao por 0,5 | curva Precision-Recall |
# MAGIC | **Producao** | Registro no UC + fila de auditoria priorizada | UC Model Registry + pyfunc |
# MAGIC
# MAGIC ### Como usar este notebook
# MAGIC
# MAGIC Sao **9 tarefas**, marcadas como `TO-DO`. Cada uma tem a explicacao do *porque*
# MAGIC antes do codigo — leia a explicacao, ela e o conteudo do lab.
# MAGIC
# MAGIC > **Dupla de notebooks:** `02a_ml_to_do` (codigo das tarefas comentado, para voce
# MAGIC > escrever) e `02b_ml_completo` (mesmo texto, codigo ativo). Sao identicos no
# MAGIC > conteudo — use o gabarito se travar.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dependencias
# MAGIC
# MAGIC Instala as bibliotecas e reinicia o Python. Rode estas duas celulas primeiro.

# COMMAND ----------

# MAGIC %pip install -q mlflow>=2.16 scikit-learn pandas

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuracao
# MAGIC
# MAGIC O schema e derivado do seu usuario — cada participante trabalha isolado dentro do
# MAGIC catalogo compartilhado `workshop_databricks`.

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()
CATALOGO = "workshop_databricks"
SCHEMA = nome
TETO_AUDITORIA = 1500.00

spark.sql(f"USE CATALOG {CATALOGO}")
spark.sql(f"USE SCHEMA {SCHEMA}")
print(f"Ambiente: {CATALOGO}.{SCHEMA}")

# COMMAND ----------

from pyspark.sql import Window
from pyspark.sql.functions import *

# `from pyspark.sql.functions import *` sobrescreve os builtins max/min/sum/round
# do Python. Guardamos os originais para usar em codigo Python puro.
from builtins import max as pymax, min as pymin, round as pyround, sum as pysum

import mlflow
import numpy as np
import pandas as pd
from mlflow import MlflowClient
from mlflow.models import infer_signature
from sklearn.ensemble import HistGradientBoostingClassifier, RandomForestClassifier
from sklearn.metrics import (average_precision_score, precision_recall_curve,
                             precision_score, recall_score, roc_auc_score)

mlflow.set_registry_uri("databricks-uc")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Etapa 1: Conhecer os dados
# MAGIC
# MAGIC Cinco tabelas. Repare que **quatro delas nao sao o pedido de reembolso** — e e
# MAGIC justamente nelas que esta o contexto que denuncia a fraude.
# MAGIC
# MAGIC | Tabela | Papel |
# MAGIC |--------|-------|
# MAGIC | `silver_pedidos_reembolso` | A tabela alvo — um pedido por linha |
# MAGIC | `silver_beneficiarios` | UF, plano, segmento, `data_adesao` |
# MAGIC | `silver_prestadores` | Quem e credenciado e quem esta **fora da rede** |
# MAGIC | `silver_procedimentos_rede` | O que a rede credenciada executou (cruzamento) |
# MAGIC | `silver_planos` | `percentual_reembolso` de cada produto |

# COMMAND ----------

df_pedidos = spark.table("silver_pedidos_reembolso")
df_beneficiarios = spark.table("silver_beneficiarios")
df_prestadores = spark.table("silver_prestadores")
df_rede = spark.table("silver_procedimentos_rede")
df_planos = spark.table("silver_planos")

df_pedidos.printSchema()

# COMMAND ----------

df_pedidos.limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Qual o tamanho do problema?

# COMMAND ----------

df_pedidos.groupBy("fraude_confirmada").agg(
    count("*").alias("pedidos"),
    round(avg("valor_solicitado"), 2).alias("ticket_medio"),
    round(sum("valor_solicitado"), 2).alias("valor_total"),
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### O baseline: a regra do teto de R$ 1.500
# MAGIC
# MAGIC Antes de treinar qualquer modelo, meça o que a operadora **ja faz hoje**.
# MAGIC Sem isso voce nao tem como provar que o modelo vale a pena — e essa e a primeira
# MAGIC pergunta que a area de negocio vai fazer.

# COMMAND ----------

regra = df_pedidos.withColumn(
    "sinalizado_pela_regra", when(col("valor_solicitado") > TETO_AUDITORIA, 1).otherwise(0)
)

regra.groupBy("sinalizado_pela_regra", "fraude_confirmada").count().orderBy(
    "sinalizado_pela_regra", "fraude_confirmada"
).display()

# COMMAND ----------

m = regra.select(
    sum(when((col("sinalizado_pela_regra") == 1) & (col("fraude_confirmada") == 1), 1).otherwise(0)).alias("vp"),
    sum(when((col("sinalizado_pela_regra") == 1) & (col("fraude_confirmada") == 0), 1).otherwise(0)).alias("fp"),
    sum(when((col("sinalizado_pela_regra") == 0) & (col("fraude_confirmada") == 1), 1).otherwise(0)).alias("fn"),
).collect()[0]

print(f"Precision da regra: {m.vp / pymax(m.vp + m.fp, 1):.3f}")
print(f"Recall da regra:    {m.vp / pymax(m.vp + m.fn, 1):.3f}")
print(f"\nA regra manda {m.vp + m.fp:,} pedidos para auditoria e acerta {m.vp:,}.")
print(f"Deixa passar {m.fn:,} fraudes.")

# COMMAND ----------

# MAGIC %md
# MAGIC > **Pare e discuta:** a regra tem precision de ~2%. De cada 100 pedidos que a
# MAGIC > auditoria analisa, ~98 sao legitimos. E ainda assim ela deixa passar ~95% das
# MAGIC > fraudes. Por que? Porque o valor alto e **normal** em odontologia: implante e
# MAGIC > protese custam milhares de reais legitimamente.
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC ## Etapa 2: Vazamento de alvo (target leakage)
# MAGIC
# MAGIC ### **TO-DO 1** — Encontrar as colunas que vazam a resposta
# MAGIC
# MAGIC Olhe o schema de `silver_pedidos_reembolso`. **Quatro colunas nao podem entrar
# MAGIC no modelo**, mesmo sendo muito preditivas.
# MAGIC
# MAGIC A pergunta que revela cada uma: *"essa informacao existe no momento em que o
# MAGIC pedido entra na fila, ou so depois da auditoria decidir?"*
# MAGIC
# MAGIC Liste as quatro colunas em `COLUNAS_VAZAMENTO`.
# MAGIC
# MAGIC > **Dica:** uma delas contem o motivo da glosa escrito em texto. Se o modelo ler
# MAGIC > "Duplicidade identificada", ele nao esta prevendo nada — esta lendo a resposta.
# MAGIC
# MAGIC **Por que isso e a falha nº 1 em projetos de fraude:** um modelo com vazamento
# MAGIC mostra metricas excelentes na validacao e desempenho proximo de zero em producao,
# MAGIC porque em producao essas colunas estao vazias no momento do score.

# COMMAND ----------

COLUNAS_VAZAMENTO = ["status", "valor_reembolsado", "motivo_glosa", "data_analise"]

# COMMAND ----------

# MAGIC %md
# MAGIC Verificacao: veja o quanto `status` "prediz" o alvo. A correlacao alta e
# MAGIC justamente o sintoma do vazamento — nao um bom sinal.

# COMMAND ----------

demo = df_pedidos.select(
    when(col("status") == "negado", 1).otherwise(0).alias("foi_negado"),
    "fraude_confirmada",
).toPandas()

print(f"Correlacao de 'foi_negado' com o alvo: "
      f"{demo.foi_negado.corr(demo.fraude_confirmada):.3f}")
print("\nAlta — e por isso mesmo ela esta PROIBIDA: no momento do score,")
print("nenhum pedido tem status ainda.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Etapa 3: Feature Engineering
# MAGIC
# MAGIC Esta e a etapa que decide o resultado do lab. Um pedido de reembolso, **olhado
# MAGIC isoladamente**, quase nunca parece fraudulento — R$ 1.400 numa restauracao pode
# MAGIC ser exagero ou pode ser um caso complexo.
# MAGIC
# MAGIC O que denuncia a fraude e o **padrao entre linhas**:
# MAGIC
# MAGIC - a mesma clinica fora da rede atendendo dezenas de beneficiarios diferentes;
# MAGIC - cinco pedidos do mesmo par (beneficiario, prestador) em uma semana;
# MAGIC - a mesma nota fiscal aparecendo duas vezes;
# MAGIC - um canal cobrado em dente que ja foi extraido.
# MAGIC
# MAGIC ### 3.1 Enriquecer o pedido
# MAGIC
# MAGIC Joins com beneficiario, prestador e plano, mais o **valor de referencia** de cada
# MAGIC procedimento (a mediana cobrada na rede credenciada). O `valor_ref` e a base do
# MAGIC `razao_valor_ref` na proxima secao.

# COMMAND ----------

df = (
    df_pedidos
    .join(
        df_beneficiarios.select("beneficiario_id", "uf", "plano_id", "segmento", "data_adesao"),
        "beneficiario_id", "left",
    )
    .join(
        df_prestadores.select("prestador_id", "credenciado", "especialidade",
                             col("uf").alias("uf_prestador")),
        "prestador_id", "left",
    )
    .join(df_planos.select("plano_id", "percentual_reembolso"), "plano_id", "left")
)

# Valor de referencia de cada procedimento = mediana cobrada na rede credenciada
df_ref = (
    df_rede.groupBy("codigo_tuss")
    .agg(round(percentile_approx("valor_procedimento", 0.5), 2).alias("valor_ref"))
)
df = df.join(df_ref, "codigo_tuss", "left")

print(f"Pedidos enriquecidos: {df.count():,}")
df.select("pedido_id", "valor_solicitado", "valor_ref", "credenciado", "uf", "uf_prestador").limit(5).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### 3.2 — **TO-DO 2**: Features do proprio pedido
# MAGIC
# MAGIC Crie 7 colunas:
# MAGIC
# MAGIC | Coluna | Regra |
# MAGIC |--------|-------|
# MAGIC | `razao_valor_ref` | `valor_solicitado / valor_ref` |
# MAGIC | `dias_desde_adesao` | dias entre `data_adesao` e `data_procedimento` |
# MAGIC | `carencia_recente` | 1 se `dias_desde_adesao <= 120` |
# MAGIC | `fora_da_rede` | 1 se o prestador **nao** e credenciado |
# MAGIC | `uf_divergente` | 1 se a UF do beneficiario difere da do prestador |
# MAGIC | `prox_teto` | 1 se o valor esta entre 85% do teto e o teto |
# MAGIC | `dias_proc_solic` | dias entre procedimento e solicitacao |
# MAGIC
# MAGIC **Por que `razao_valor_ref` e a feature mais importante do modelo final?** Porque
# MAGIC normaliza o valor pelo procedimento. R$ 1.400 e absurdo numa limpeza e barato num
# MAGIC implante — o valor sozinho nao diz nada, a **razao** diz tudo.
# MAGIC
# MAGIC E `prox_teto` codifica a inteligencia do fraudador: quem conhece o teto de
# MAGIC R$ 1.500 pede R$ 1.450.
# MAGIC
# MAGIC > **Dica:** `datediff(fim, inicio)`, `when(cond, 1).otherwise(0)`, `~col("x")`

# COMMAND ----------

df = (
    df
    .withColumn("razao_valor_ref", round(col("valor_solicitado") / col("valor_ref"), 4))
    .withColumn("dias_desde_adesao", datediff(col("data_procedimento"), col("data_adesao")))
    .withColumn("carencia_recente", when(col("dias_desde_adesao") <= 120, 1).otherwise(0))
    .withColumn("fora_da_rede", when(~col("credenciado"), 1).otherwise(0))
    .withColumn("uf_divergente", when(col("uf") != col("uf_prestador"), 1).otherwise(0))
    .withColumn(
        "prox_teto",
        when(
            (col("valor_solicitado") > TETO_AUDITORIA * 0.85)
            & (col("valor_solicitado") <= TETO_AUDITORIA), 1,
        ).otherwise(0),
    )
    .withColumn("dias_proc_solic", datediff(col("data_solicitacao"), col("data_procedimento")))
)

df.select("valor_solicitado", "valor_ref", "razao_valor_ref", "prox_teto",
          "dias_desde_adesao", "fora_da_rede").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### 3.3 — **TO-DO 3**: Features de concentracao no prestador
# MAGIC
# MAGIC **A feature que pega a fraude mais cara do dataset.**
# MAGIC
# MAGIC No esquema de *reembolso assistido*, uma clinica de fachada recruta beneficiarios,
# MAGIC emite notas infladas e fica com parte do reembolso. Nenhum pedido individual
# MAGIC parece estranho. O que denuncia e a **concentracao**: um prestador fora da rede
# MAGIC com 40 beneficiarios distintos e valores sempre colados no teto.
# MAGIC
# MAGIC Use `Window.partitionBy("prestador_id")` e crie:
# MAGIC
# MAGIC | Coluna | O que mede |
# MAGIC |--------|-----------|
# MAGIC | `prest_n_pedidos` | total de pedidos do prestador |
# MAGIC | `prest_n_benef` | beneficiarios **distintos** do prestador |
# MAGIC | `prest_valor_medio` | ticket medio do prestador |
# MAGIC | `prest_pct_prox_teto` | % dos pedidos dele colados no teto |
# MAGIC
# MAGIC > **Dica:** para contagem distinta em Window use
# MAGIC > `size(collect_set("beneficiario_id").over(w))` — `countDistinct` nao funciona
# MAGIC > em Window no Spark.

# COMMAND ----------

w_prest = Window.partitionBy("prestador_id")

df = (
    df
    .withColumn("prest_n_pedidos", count("pedido_id").over(w_prest))
    .withColumn("prest_n_benef", size(collect_set("beneficiario_id").over(w_prest)))
    .withColumn("prest_valor_medio", round(avg("valor_solicitado").over(w_prest), 2))
    .withColumn("prest_pct_prox_teto", round(avg("prox_teto").over(w_prest), 4))
)

# COMMAND ----------

# MAGIC %md
# MAGIC Os prestadores mais concentrados — as clinicas de fachada devem aparecer no topo.
# MAGIC Repare na combinacao: `fora_da_rede = 1`, muitos beneficiarios distintos e
# MAGIC `prest_pct_prox_teto` alto.

# COMMAND ----------

(df.select("prestador_id", "fora_da_rede", "prest_n_pedidos", "prest_n_benef",
           "prest_valor_medio", "prest_pct_prox_teto")
   .distinct()
   .orderBy(col("prest_n_benef").desc())
   .limit(15)
   .display())

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### 3.4 — **TO-DO 4**: Comportamento do beneficiario e rajada de pedidos
# MAGIC
# MAGIC Crie as features do beneficiario (`Window.partitionBy("beneficiario_id")`):
# MAGIC `benef_n_pedidos`, `benef_valor_total`, `benef_n_prestadores`.
# MAGIC
# MAGIC E a mais interessante: **`pedidos_par_30d`** — quantos pedidos aquele par
# MAGIC (beneficiario, prestador) fez nos ultimos 30 dias. Isso captura
# MAGIC **fracionamento**: um tratamento de R$ 3.000 quebrado em 5 pedidos de R$ 600
# MAGIC para escapar do teto.
# MAGIC
# MAGIC Atencao: precisa ser uma janela de **30 dias**, nao de 30 linhas. Para isso,
# MAGIC ordene pela data convertida em segundos e use `rangeBetween`:
# MAGIC
# MAGIC ```python
# MAGIC w_par_30d = (
# MAGIC     Window.partitionBy("beneficiario_id", "prestador_id")
# MAGIC     .orderBy(col("data_procedimento").cast("timestamp").cast("long"))
# MAGIC     .rangeBetween(-30 * 86400, 0)
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC > **Por que `rangeBetween` e nao `rowsBetween`?** `rowsBetween(-4, 0)` conta as
# MAGIC > 5 linhas anteriores independente da data — se o beneficiario tem pedidos
# MAGIC > espalhados em 2 anos, elas nao formam rajada nenhuma.

# COMMAND ----------

w_benef = Window.partitionBy("beneficiario_id")

df = (
    df
    .withColumn("benef_n_pedidos", count("pedido_id").over(w_benef))
    .withColumn("benef_valor_total", round(sum("valor_solicitado").over(w_benef), 2))
    .withColumn("benef_n_prestadores", size(collect_set("prestador_id").over(w_benef)))
)

# Rajada: janela deslizante de 30 dias no par (beneficiario, prestador)
w_par_30d = (
    Window.partitionBy("beneficiario_id", "prestador_id")
    .orderBy(col("data_procedimento").cast("timestamp").cast("long"))
    .rangeBetween(-30 * 86400, 0)
)

df = df.withColumn("pedidos_par_30d", count("pedido_id").over(w_par_30d))

df.select("beneficiario_id", "prestador_id", "data_procedimento",
          "pedidos_par_30d", "benef_n_pedidos").orderBy(
    col("pedidos_par_30d").desc()).limit(10).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ### 3.5 — **TO-DO 5**: Sinais de cruzamento entre tabelas
# MAGIC
# MAGIC Tres fraudes que **so aparecem cruzando tabelas**:
# MAGIC
# MAGIC **a) `hash_repetido`** — a mesma nota fiscal apresentada mais de uma vez.
# MAGIC Use `Window.partitionBy("hash_documento")` e marque 1 quando a contagem > 1.
# MAGIC
# MAGIC **b) `existe_na_rede`** — o procedimento **tambem** foi pago pela rede
# MAGIC credenciada. A operadora pagaria duas vezes pelo mesmo dente. Faca um `left join`
# MAGIC de `df_rede` por (`beneficiario_id`, `codigo_tuss`, `dente`) e marque 1 quando
# MAGIC casar. Cuidado: `dente` tem null — use `coalesce(col("dente"), lit("NA"))`.
# MAGIC
# MAGIC **c) `dente_ja_extraido`** — restauracao ou canal em dente com **exodontia
# MAGIC anterior** (TUSS `81000600`). Impossibilidade clinica: dente extraido nao volta.
# MAGIC Agregue as extracoes por (`beneficiario_id`, `dente`) pegando `min(data)`, junte,
# MAGIC e marque 1 quando `data_procedimento > data_extracao`.
# MAGIC
# MAGIC > Repare que (c) e uma regra que um **auditor humano** conhece. Boa parte do
# MAGIC > feature engineering em fraude e traduzir conhecimento de dominio em coluna.

# COMMAND ----------

# a) Documento reciclado
w_hash = Window.partitionBy("hash_documento")
df = df.withColumn(
    "hash_repetido", when(count("pedido_id").over(w_hash) > 1, 1).otherwise(0)
)

# b) Duplicidade rede x reembolso — mesmo beneficiario, TUSS e dente
df_rede_key = (
    df_rede.select(
        "beneficiario_id", "codigo_tuss",
        coalesce(col("dente"), lit("NA")).alias("dente_k"),
    ).distinct().withColumn("existe_na_rede", lit(1))
)

df = (
    df.withColumn("dente_k", coalesce(col("dente"), lit("NA")))
      .join(df_rede_key, ["beneficiario_id", "codigo_tuss", "dente_k"], "left")
      .withColumn("existe_na_rede", coalesce(col("existe_na_rede"), lit(0)))
)

# c) Dente ja extraido — exodontia (81000600) anterior no mesmo dente
df_extracoes = (
    df_rede.filter(col("codigo_tuss") == "81000600")
    .groupBy("beneficiario_id", "dente")
    .agg(min("data_realizacao").alias("data_extracao"))
)

df = (
    df.join(df_extracoes, ["beneficiario_id", "dente"], "left")
      .withColumn(
          "dente_ja_extraido",
          when(
              col("data_extracao").isNotNull()
              & (col("data_procedimento") > col("data_extracao")), 1,
          ).otherwise(0),
      )
)

df.select(
    round(avg("hash_repetido") * 100, 2).alias("pct_hash_repetido"),
    round(avg("existe_na_rede") * 100, 2).alias("pct_existe_na_rede"),
    round(avg("dente_ja_extraido") * 100, 2).alias("pct_dente_extraido"),
).display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### 3.6 Gravar a tabela de features (Gold)
# MAGIC
# MAGIC 20 features no total. Materializar em uma tabela Gold garante que treino e
# MAGIC inferencia usem **exatamente** a mesma definicao de feature — a causa mais comum
# MAGIC de *training/serving skew*.

# COMMAND ----------

FEATURES = [
    "valor_solicitado", "razao_valor_ref", "dias_desde_adesao", "carencia_recente",
    "fora_da_rede", "uf_divergente", "prox_teto", "dias_proc_solic",
    "prest_n_pedidos", "prest_n_benef", "prest_valor_medio", "prest_pct_prox_teto",
    "benef_n_pedidos", "benef_valor_total", "benef_n_prestadores", "pedidos_par_30d",
    "hash_repetido", "existe_na_rede", "dente_ja_extraido", "percentual_reembolso",
]

df_features = df.select(
    "pedido_id", "beneficiario_id", "prestador_id", "codigo_tuss",
    "data_solicitacao", *FEATURES, "fraude_confirmada",
).dropDuplicates(["pedido_id"])

(df_features.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOGO}.{SCHEMA}.gold_features_fraude"))

print(f"gold_features_fraude: {df_features.count():,} linhas | {len(FEATURES)} features")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Etapa 4 — **TO-DO 6**: Split temporal (e por que nao aleatorio)
# MAGIC
# MAGIC Faca o split usando o **percentil 80 de `data_solicitacao`** como corte:
# MAGIC treino = pedidos mais antigos, teste = os 20% mais recentes.
# MAGIC
# MAGIC **Por que nao `train_test_split` aleatorio?** Porque em fraude ele vaza:
# MAGIC os pedidos de um mesmo esquema (mesma clinica de fachada, mesma quadrilha) cairiam
# MAGIC metade no treino e metade no teste. O modelo memorizaria `prestador_id = 412` em
# MAGIC vez de aprender o **padrao**, e sua metrica ficaria otimista demais.
# MAGIC
# MAGIC Em producao o modelo enfrenta esquemas que nunca viu. O teste tem que imitar isso.
# MAGIC
# MAGIC > **Dica:** `pdf["data_solicitacao"].quantile(0.80)`

# COMMAND ----------

pdf = spark.table(f"{CATALOGO}.{SCHEMA}.gold_features_fraude").toPandas()
pdf["data_solicitacao"] = pd.to_datetime(pdf["data_solicitacao"])

data_corte = pdf["data_solicitacao"].quantile(0.80)

treino = pdf[pdf.data_solicitacao <= data_corte]
teste = pdf[pdf.data_solicitacao > data_corte]

X_train = treino[FEATURES].fillna(-1)
y_train = treino["fraude_confirmada"]
X_test = teste[FEATURES].fillna(-1)
y_test = teste["fraude_confirmada"]

print(f"Data de corte: {data_corte.date()}")
print(f"Treino: {len(treino):,} pedidos ({y_train.mean()*100:.2f}% fraude)")
print(f"Teste:  {len(teste):,} pedidos ({y_test.mean()*100:.2f}% fraude)")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Etapa 5 — **TO-DO 7**: Treinar o modelo com a metrica correta
# MAGIC
# MAGIC Treine um `HistGradientBoostingClassifier` e logue no MLflow.
# MAGIC
# MAGIC **A parte conceitual mais importante do lab: qual metrica usar?**
# MAGIC
# MAGIC | Metrica | Por que serve ou nao |
# MAGIC |---------|---------------------|
# MAGIC | **Acuracia** | Inutil. Com ~5% de fraude, "nunca e fraude" da 95% de acuracia. |
# MAGIC | **ROC AUC** | Otimista com classes raras — o eixo de falso positivo e diluido pela massa de negativos. |
# MAGIC | **PR AUC** | **A correta.** Olha so a classe rara: das que eu sinalizei, quantas eram fraude? |
# MAGIC
# MAGIC Compare sempre o PR AUC com a **prevalencia** (`y_test.mean()`): ela e o PR AUC
# MAGIC de um modelo aleatorio. A razao entre os dois e o **lift** real.
# MAGIC
# MAGIC ### Um detalhe que decide o Lab 02: a *signature*
# MAGIC
# MAGIC Vamos logar o modelo com `signature=infer_signature(...)` e um `input_example`.
# MAGIC A signature e o **contrato de entrada**: quais colunas, em que ordem, de que tipo.
# MAGIC
# MAGIC | Sem signature | Com signature |
# MAGIC |---|---|
# MAGIC | O Model Serving nao valida o payload | Valida nome, ordem e tipo de cada feature |
# MAGIC | Erro 400 opaco, dificil de depurar | Mensagem dizendo qual coluna faltou |
# MAGIC | Ordem trocada passa e da **score errado, silenciosamente** | Ordem trocada e recusada |
# MAGIC
# MAGIC Esse ultimo caso e o perigoso: 20 features numericas, se o app mandar na ordem
# MAGIC errada, o modelo responde com confianca um numero sem sentido.
# MAGIC
# MAGIC > Por isso usamos `autolog(log_models=False)` e logamos o modelo na mao: o autolog
# MAGIC > sozinho registra o modelo **sem** signature.
# MAGIC
# MAGIC ### E uma pegadinha que arruina o projeto inteiro: `predict` vs `predict_proba`
# MAGIC
# MAGIC Um classificador sklearn tem dois metodos de saida:
# MAGIC
# MAGIC | Metodo | Devolve | Serve para priorizar fila? |
# MAGIC |---|---|---|
# MAGIC | `predict()` | a **classe**: 0 ou 1 | ❌ nao — nao da para ordenar |
# MAGIC | `predict_proba()` | a **probabilidade**: 0.0 a 1.0 | ✅ sim |
# MAGIC
# MAGIC Quando voce loga com `mlflow.sklearn.log_model`, o MLflow embala o modelo chamando
# MAGIC **`predict()`**. Resultado: o endpoint e a inferencia em batch devolvem `0.0` ou
# MAGIC `1.0` — e **todo o exercicio de threshold perde sentido**, porque nao existe
# MAGIC gradiente para cortar.
# MAGIC
# MAGIC O sintoma e traicoeiro: nada quebra. As metricas de treino ficam otimas (elas usam
# MAGIC `predict_proba` direto), e so a saida servida vem binaria.
# MAGIC
# MAGIC Por isso embalamos o modelo num **wrapper pyfunc** que expoe explicitamente a
# MAGIC probabilidade da classe positiva (`[:, 1]` = P(fraude)).

# COMMAND ----------

from datetime import datetime as dt


class ScoreDeFraude(mlflow.pyfunc.PythonModel):
    """Wrapper que faz o modelo servir PROBABILIDADE, nao classe.

    `mlflow.sklearn.log_model` embala o modelo chamando `predict()`, que devolve
    0 ou 1. Aqui expomos `predict_proba(...)[:, 1]` — a probabilidade de fraude —
    que e o que a fila de auditoria e o app precisam para ordenar por risco.
    """

    def __init__(self, modelo):
        self.modelo = modelo

    def predict(self, context, model_input):
        return self.modelo.predict_proba(model_input)[:, 1]


# O path `/Users/{usuario}/` ja garante a unicidade: cada participante escreve
# o experimento na sua propria pasta pessoal.
EXPERIMENT_PATH = f"/Users/{usuario}/odontoprev_workshop_fraude"

mlflow.set_experiment(EXPERIMENT_PATH)
run_name = f"fraude_hgb_{nome}_{dt.now().strftime('%Y%m%d_%H%M%S')}"

print(f"Experimento: {EXPERIMENT_PATH}")
print(f"Run:         {run_name}")

# COMMAND ----------

mlflow.sklearn.autolog(silent=True, log_models=False)

with mlflow.start_run(run_name=run_name) as run:
    modelo = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_depth=8, random_state=42
    )
    modelo.fit(X_train, y_train)

    scores = modelo.predict_proba(X_test)[:, 1]

    roc = roc_auc_score(y_test, scores)
    pr_auc = average_precision_score(y_test, scores)

    mlflow.log_metric("test_roc_auc", roc)
    mlflow.log_metric("test_pr_auc", pr_auc)
    mlflow.log_metric("prevalencia_teste", float(y_test.mean()))

    # Signature = CONTRATO DE ENTRADA do modelo (nomes, ordem e tipos das 20 features).
    # Sem ela o Model Serving do Lab 02 nao consegue validar o payload.
    signature = infer_signature(X_train, modelo.predict_proba(X_train)[:, 1])
    mlflow.pyfunc.log_model(
        name="model",
        python_model=ScoreDeFraude(modelo),
        signature=signature,
        input_example=X_train.head(3),
    )

    run_id = run.info.run_id

print(f"ROC AUC:     {roc:.4f}")
print(f"PR AUC:      {pr_auc:.4f}")
print(f"Baseline PR: {y_test.mean():.4f}  (um modelo aleatorio)")
print(f"Lift:        {pr_auc / y_test.mean():.1f}x")

# COMMAND ----------

# MAGIC %md
# MAGIC ### O experimento que prova o valor do feature engineering
# MAGIC
# MAGIC Treine o **mesmo algoritmo** usando so as features do proprio pedido
# MAGIC (sem as de prestador/beneficiario/cruzamento) e compare o PR AUC.
# MAGIC
# MAGIC Essa diferenca **e** o valor do trabalho da Etapa 3 — e o numero que voce
# MAGIC leva para a area de negocio.

# COMMAND ----------

FEATURES_PEDIDO = [
    "valor_solicitado", "razao_valor_ref", "dias_proc_solic",
    "fora_da_rede", "uf_divergente", "prox_teto", "percentual_reembolso",
]

with mlflow.start_run(run_name=f"{run_name}_so_pedido"):
    modelo_simples = HistGradientBoostingClassifier(
        max_iter=250, learning_rate=0.08, max_depth=8, random_state=42
    )
    modelo_simples.fit(treino[FEATURES_PEDIDO].fillna(-1), y_train)
    scores_simples = modelo_simples.predict_proba(teste[FEATURES_PEDIDO].fillna(-1))[:, 1]
    pr_simples = average_precision_score(y_test, scores_simples)
    mlflow.log_metric("test_pr_auc", pr_simples)

print(f"{'='*62}")
print(f"  PR AUC so com features do pedido:  {pr_simples:.4f}")
print(f"  PR AUC com features relacionais:   {pr_auc:.4f}")
print(f"  Ganho: {(pr_auc/pr_simples - 1)*100:+.1f}%")
print(f"{'='*62}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Etapa 6 — **TO-DO 8**: Escolher o threshold pelo dinheiro
# MAGIC
# MAGIC O modelo devolve uma probabilidade. A auditoria precisa de uma **decisao**.
# MAGIC
# MAGIC O corte **nao e 0,5** — 0,5 nao tem nenhum significado de negocio. O corte certo
# MAGIC vem do custo dos dois erros:
# MAGIC
# MAGIC - **Falso negativo:** a operadora paga a fraude → perde `valor_solicitado`;
# MAGIC - **Falso positivo:** a auditoria analisa um pedido honesto → custa
# MAGIC   `CUSTO_ANALISE` (R$ 40) e gera atrito com um beneficiario legitimo.
# MAGIC
# MAGIC Varra thresholds de 0,05 a 0,95 e para cada um calcule:
# MAGIC
# MAGIC | Coluna | Como calcular |
# MAGIC |--------|--------------|
# MAGIC | `sinalizados` | VP + FP |
# MAGIC | `precision` / `recall` | `precision_score`, `recall_score` |
# MAGIC | `fraude_evitada_R$` | soma de `valor_solicitado` dos VP |
# MAGIC | `custo_analise_R$` | (VP + FP) × R$ 40 |
# MAGIC | `beneficio_liquido_R$` | `fraude_evitada − custo_analise` |
# MAGIC
# MAGIC Escolha o threshold de **maior beneficio liquido**.
# MAGIC
# MAGIC > **Discussao:** o otimo puramente financeiro pode sinalizar muita coisa. Se a
# MAGIC > auditoria so tem capacidade para 300 analises/dia, a **capacidade** e a
# MAGIC > restricao real — e o threshold sai dela, nao da curva.
# MAGIC
# MAGIC > **Atencao ao `round`:** aqui os valores sao floats do Python, nao colunas Spark.
# MAGIC > Use `pyround` (o builtin que guardamos no inicio), senao o `round` do PySpark
# MAGIC > tenta interpretar o numero como coluna e falha.

# COMMAND ----------

CUSTO_ANALISE = 40.00  # custo medio de uma analise manual

resultados = []
for thr in np.arange(0.05, 0.96, 0.05):
    pred = (scores >= thr).astype(int)
    vp = int(((pred == 1) & (y_test == 1)).sum())
    fp = int(((pred == 1) & (y_test == 0)).sum())
    fn = int(((pred == 0) & (y_test == 1)).sum())

    # Valor da fraude evitada (o que deixamos de pagar)
    evitado = float(teste.loc[(pred == 1) & (y_test == 1), "valor_solicitado"].sum())
    # Valor da fraude que passou
    perdido = float(teste.loc[(pred == 0) & (y_test == 1), "valor_solicitado"].sum())
    custo_op = (vp + fp) * CUSTO_ANALISE

    resultados.append({
        "threshold": pyround(thr, 2),
        "sinalizados": vp + fp,
        "precision": pyround(precision_score(y_test, pred, zero_division=0), 3),
        "recall": pyround(recall_score(y_test, pred, zero_division=0), 3),
        "fraude_evitada_R$": pyround(evitado, 2),
        "fraude_perdida_R$": pyround(perdido, 2),
        "custo_analise_R$": pyround(custo_op, 2),
        "beneficio_liquido_R$": pyround(evitado - custo_op, 2),
    })

df_thr = pd.DataFrame(resultados)
melhor = df_thr.loc[df_thr["beneficio_liquido_R$"].idxmax()]

display(spark.createDataFrame(df_thr))

# COMMAND ----------

THRESHOLD = float(melhor.threshold)

print(f"{'='*62}")
print(f"  THRESHOLD OTIMO (por beneficio liquido): {THRESHOLD:.2f}")
print(f"{'='*62}")
print(f"  Pedidos sinalizados:  {int(melhor.sinalizados):,}")
print(f"  Precision:            {melhor.precision:.3f}")
print(f"  Recall:               {melhor.recall:.3f}")
print(f"  Fraude evitada:       R$ {melhor['fraude_evitada_R$']:,.2f}")
print(f"  Custo de analise:     R$ {melhor['custo_analise_R$']:,.2f}")
print(f"  Beneficio liquido:    R$ {melhor['beneficio_liquido_R$']:,.2f}")
print(f"{'='*62}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Etapa 7: Governanca no Unity Catalog
# MAGIC
# MAGIC Registre o modelo e promova como `champion`. Grave o **threshold como tag** da
# MAGIC versao: ele e parte da decisao de negocio, nao do codigo — quem consumir o modelo
# MAGIC precisa saber qual corte foi homologado.

# COMMAND ----------

MODEL_NAME = f"{CATALOGO}.{SCHEMA}.modelo_fraude_reembolso"

client = MlflowClient()
resultado_v1 = mlflow.register_model(f"runs:/{run_id}/model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "champion", resultado_v1.version)

# Guardar o threshold junto do modelo — ele faz parte da decisao, nao do codigo
client.set_model_version_tag(MODEL_NAME, resultado_v1.version, "threshold", str(THRESHOLD))
client.set_model_version_tag(MODEL_NAME, resultado_v1.version, "pr_auc", f"{pr_auc:.4f}")

print(f"Modelo {MODEL_NAME} v{resultado_v1.version} registrado como CHAMPION")
print(f"Threshold gravado como tag: {THRESHOLD}")

# COMMAND ----------

# MAGIC %md
# MAGIC ## Etapa 8: Challenger (RandomForest com `class_weight="balanced"`)
# MAGIC
# MAGIC Treine um segundo modelo, registre como `challenger` e compare pelo **PR AUC**.
# MAGIC Se for melhor, mova o alias `champion` para ele.
# MAGIC
# MAGIC `class_weight="balanced"` faz o modelo penalizar mais o erro na classe rara —
# MAGIC uma alternativa a reamostragem (SMOTE/undersampling).

# COMMAND ----------

with mlflow.start_run(run_name=f"fraude_rf_challenger_{nome}") as run_v2:
    modelo_v2 = RandomForestClassifier(
        n_estimators=300, max_depth=14, min_samples_leaf=5,
        class_weight="balanced", n_jobs=-1, random_state=42,
    )
    modelo_v2.fit(X_train, y_train)

    scores_v2 = modelo_v2.predict_proba(X_test)[:, 1]
    roc_v2 = roc_auc_score(y_test, scores_v2)
    pr_auc_v2 = average_precision_score(y_test, scores_v2)

    mlflow.log_metric("test_roc_auc", roc_v2)
    mlflow.log_metric("test_pr_auc", pr_auc_v2)

    # Mesmo contrato de entrada do champion — o challenger tem que ser servivel tambem
    mlflow.pyfunc.log_model(
        name="model",
        python_model=ScoreDeFraude(modelo_v2),
        signature=infer_signature(X_train, modelo_v2.predict_proba(X_train)[:, 1]),
        input_example=X_train.head(3),
    )

    run_id_v2 = run_v2.info.run_id

resultado_v2 = mlflow.register_model(f"runs:/{run_id_v2}/model", MODEL_NAME)
client.set_registered_model_alias(MODEL_NAME, "challenger", resultado_v2.version)

print(f"{'='*62}")
print(f"  CHAMPION (v{resultado_v1.version})  vs  CHALLENGER (v{resultado_v2.version})")
print(f"{'='*62}")
print(f"  {'Metrica':<12} {'Champion':>12} {'Challenger':>12} {'Melhor':>12}")
print(f"  {'-'*50}")
print(f"  {'ROC AUC':<12} {roc:>12.4f} {roc_v2:>12.4f} "
      f"{('Challenger' if roc_v2 > roc else 'Champion'):>12}")
print(f"  {'PR AUC':<12} {pr_auc:>12.4f} {pr_auc_v2:>12.4f} "
      f"{('Challenger' if pr_auc_v2 > pr_auc else 'Champion'):>12}")
print(f"{'='*62}")

if pr_auc_v2 > pr_auc:
    client.set_registered_model_alias(MODEL_NAME, "champion", resultado_v2.version)
    print(f"\nChallenger v{resultado_v2.version} PROMOVIDO a Champion (PR AUC maior).")
else:
    print(f"\nChampion v{resultado_v1.version} mantido.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## 🎁 BONUS (nao vamos rodar) — Hyperparameter tuning distribuido com Optuna
# MAGIC
# MAGIC Nas Etapas 5 e 8 escolhemos os hiperparametros **na mao**: `max_iter=250`,
# MAGIC `learning_rate=0.08`, `max_depth=8`. Funcionou, mas foi um chute informado — nao
# MAGIC exploramos o espaco de busca.
# MAGIC
# MAGIC Em um projeto real esse ajuste e sistematico, e o **Optuna** e a ferramenta padrao:
# MAGIC ele usa **TPE** (Tree-structured Parzen Estimator) para aprender das tentativas
# MAGIC anteriores e concentrar a busca nas regioes promissoras — em vez de varrer um grid
# MAGIC inteiro as cegas.
# MAGIC
# MAGIC | Abordagem | Como escolhe o proximo teste | Custo |
# MAGIC |---|---|---|
# MAGIC | Grid Search | todas as combinacoes | explode com o nº de parametros |
# MAGIC | Random Search | sorteio | ignora o que ja aprendeu |
# MAGIC | **Optuna (TPE)** | **modela o que deu certo e refina** | poucas dezenas de trials |
# MAGIC
# MAGIC ### O que torna isso interessante no Databricks
# MAGIC
# MAGIC O `MlflowSparkStudy` distribui os **trials pelos executores do cluster** — cada worker
# MAGIC treina um modelo com hiperparametros diferentes, em paralelo. E o `MlflowStorage`
# MAGIC usa o proprio MLflow Tracking como storage compartilhado do estudo, entao:
# MAGIC
# MAGIC - os workers coordenam a busca (nao repetem combinacoes);
# MAGIC - **cada trial vira um run** no seu experimento, com metricas e parametros;
# MAGIC - a comparacao fica visivel na UI do MLflow, sem codigo extra.
# MAGIC
# MAGIC > ⚠️ **Deixamos comentado de proposito.** Rodar 20+ trials levaria mais tempo do que
# MAGIC > temos no workshop, e o objetivo aqui e o **fluxo completo** (feature engineering →
# MAGIC > threshold → serving → app), nao arrancar o ultimo ponto de PR AUC.
# MAGIC >
# MAGIC > Fica como referencia para quando voces levarem isso para o ambiente de voces.
# MAGIC >
# MAGIC > 📄 Doc: https://docs.databricks.com/aws/en/machine-learning/automl-hyperparam-tuning/optuna

# COMMAND ----------

# ============================================================================
# BONUS — 100% comentado. Nao faz parte do lab.
#
# Requer:  %pip install -q optuna 'mlflow>=3.0'
# ============================================================================
#
# import optuna
# from mlflow.optuna.storage import MlflowStorage
# from mlflow.pyspark.optuna.study import MlflowSparkStudy
#
#
# def objetivo(trial):
#     """Um trial = um conjunto de hiperparametros. Devolve o que queremos MAXIMIZAR.
#
#     Note a metrica: PR AUC, a mesma do Lab. Otimizar acuracia aqui levaria o
#     Optuna a encontrar um modelo que nunca acusa fraude (e acerta 94,5%).
#     """
#     params = {
#         "max_iter":       trial.suggest_int("max_iter", 100, 500, step=50),
#         "learning_rate":  trial.suggest_float("learning_rate", 0.01, 0.3, log=True),
#         "max_depth":      trial.suggest_int("max_depth", 3, 12),
#         "min_samples_leaf": trial.suggest_int("min_samples_leaf", 5, 60),
#         "l2_regularization": trial.suggest_float("l2_regularization", 1e-3, 10.0, log=True),
#     }
#
#     modelo_trial = HistGradientBoostingClassifier(random_state=42, **params)
#     modelo_trial.fit(X_train, y_train)
#
#     scores_trial = modelo_trial.predict_proba(X_test)[:, 1]
#     return average_precision_score(y_test, scores_trial)   # PR AUC
#
#
# # Storage compartilhado: o MLflow Tracking coordena os trials entre os workers
# # e cada trial aparece como um run no experimento.
# storage = MlflowStorage(experiment_id=mlflow.get_experiment_by_name(EXPERIMENT_PATH).experiment_id)
#
# estudo = MlflowSparkStudy(
#     study_name=f"optuna_fraude_{nome}",
#     storage=storage,
#     direction="maximize",        # PR AUC: maior e melhor
# )
#
# # n_trials = quantas combinacoes testar | n_jobs = quantas em PARALELO no cluster
# estudo.optimize(objetivo, n_trials=40, n_jobs=8)
#
# print(f"Melhor PR AUC: {estudo.best_value:.4f}")
# print(f"Melhores parametros: {estudo.best_params}")
#
# # A partir daqui o fluxo e o MESMO do lab: treinar com best_params, logar com
# # signature + wrapper de probabilidade, registrar no UC e comparar com o champion
# # atual antes de promover. O tuning muda COMO se acha o modelo, nao o que se faz
# # com ele depois.
#
# # modelo_tunado = HistGradientBoostingClassifier(random_state=42, **estudo.best_params)
# # modelo_tunado.fit(X_train, y_train)

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Etapa 9 — **TO-DO 9**: Inferencia em batch — a fila da auditoria
# MAGIC
# MAGIC Carregue o modelo pelo alias `champion` e escore a base inteira, gravando a
# MAGIC fila priorizada em `gold_fila_auditoria`.
# MAGIC
# MAGIC Crie a coluna `prioridade` para a auditoria trabalhar:
# MAGIC
# MAGIC | Faixa | Prioridade |
# MAGIC |-------|-----------|
# MAGIC | score >= 0,80 | CRITICA |
# MAGIC | score >= THRESHOLD | ALTA |
# MAGIC | score >= THRESHOLD/2 | MEDIA |
# MAGIC | resto | BAIXA |
# MAGIC
# MAGIC > **Dica:** carregue **sempre pelo alias**, nunca pela versao fixa — trocar o
# MAGIC > modelo em producao passa a ser mover um alias, sem mexer no codigo.
# MAGIC > ```python
# MAGIC > modelo_champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
# MAGIC > ```
# MAGIC
# MAGIC ### Duas formas de escorar em batch — e quando usar cada uma
# MAGIC
# MAGIC | | `load_model` + pandas | `spark_udf` |
# MAGIC |---|---|---|
# MAGIC | Onde executa | driver | **distribuido nos workers** |
# MAGIC | Limite | o que couber na memoria do driver | praticamente ilimitado |
# MAGIC | Bom para | ate alguns milhoes de linhas | **big data** |
# MAGIC
# MAGIC **Neste lab usamos `load_model`**, porque a base tem ~40 mil linhas e cabe
# MAGIC folgado no driver — nao ha ganho em distribuir.
# MAGIC
# MAGIC > 📌 **Em producao, com volume real, o caminho e o Spark UDF.** Ele aplica o
# MAGIC > modelo direto sobre o DataFrame distribuido, sem trazer dados para o driver:
# MAGIC >
# MAGIC > ```python
# MAGIC > predict_udf = mlflow.pyfunc.spark_udf(
# MAGIC >     spark, f"models:/{MODEL_NAME}@champion",
# MAGIC >     result_type="double", env_manager="local")
# MAGIC >
# MAGIC > df_scored = df.withColumn("score_fraude", predict_udf(*[col(c) for c in FEATURES]))
# MAGIC > ```
# MAGIC >
# MAGIC > O resultado e **identico** ao do `load_model` — muda so onde a conta acontece.

# COMMAND ----------

# Carrega SEMPRE pelo alias — promover outro modelo nao exige mexer neste codigo
modelo_champion = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")

# `fillna(-1)` = o mesmo tratamento usado no treino (prestador novo, sem historico)
pdf_score = (
    spark.table(f"{CATALOGO}.{SCHEMA}.gold_features_fraude")
    .select("pedido_id", *FEATURES)
    .toPandas()
    .fillna(-1)
)

pdf_score["score_fraude"] = [
    float(s) for s in modelo_champion.predict(pdf_score[FEATURES])
]

df_scored = (
    spark.table(f"{CATALOGO}.{SCHEMA}.gold_features_fraude")
    .na.fill(-1)
    .join(
        spark.createDataFrame(pdf_score[["pedido_id", "score_fraude"]]),
        "pedido_id", "inner",
    )
    .withColumn(
        "prioridade",
        when(col("score_fraude") >= 0.80, "CRITICA")
        .when(col("score_fraude") >= THRESHOLD, "ALTA")
        .when(col("score_fraude") >= THRESHOLD * 0.5, "MEDIA")
        .otherwise("BAIXA"),
    )
    .withColumn("valor_em_risco", round(col("score_fraude") * col("valor_solicitado"), 2))
)

(df_scored.write.mode("overwrite").option("overwriteSchema", "true")
    .saveAsTable(f"{CATALOGO}.{SCHEMA}.gold_fila_auditoria"))

print(f"gold_fila_auditoria: {df_scored.count():,} pedidos escorados")

# COMMAND ----------

# MAGIC %md
# MAGIC ### A fila que a auditoria realmente recebe
# MAGIC
# MAGIC Compare o `pct_fraude_real` de cada faixa com os ~5,5% da base inteira: e isso
# MAGIC que significa "priorizar".

# COMMAND ----------

(spark.table(f"{CATALOGO}.{SCHEMA}.gold_fila_auditoria")
    .groupBy("prioridade")
    .agg(
        count("*").alias("pedidos"),
        round(sum("valor_solicitado"), 2).alias("valor_solicitado"),
        round(avg("fraude_confirmada") * 100, 2).alias("pct_fraude_real"),
    )
    .orderBy(col("valor_solicitado").desc())
    .display())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Top 20 casos para investigar hoje
# MAGIC
# MAGIC Repare que as colunas de feature funcionam como **justificativa** para o auditor:
# MAGIC `prest_n_benef` alto, `hash_repetido = 1` ou `dente_ja_extraido = 1` explicam
# MAGIC por que aquele pedido subiu na fila.

# COMMAND ----------

(spark.table(f"{CATALOGO}.{SCHEMA}.gold_fila_auditoria")
    .filter(col("prioridade").isin("CRITICA", "ALTA"))
    .select("pedido_id", "beneficiario_id", "prestador_id", "codigo_tuss",
            "valor_solicitado", "razao_valor_ref", "prest_n_benef",
            "pedidos_par_30d", "hash_repetido", "existe_na_rede",
            "dente_ja_extraido", "score_fraude", "prioridade")
    .orderBy(col("score_fraude").desc())
    .limit(20)
    .display())

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Etapa 10: Onde o modelo acerta e onde erra (recall por persona)
# MAGIC
# MAGIC A tabela `_gabarito_personas_fraude` diz qual **tipo** de fraude cada pedido era.
# MAGIC Em producao voce nao tem esse rotulo — mas aqui ele mostra algo importante: o
# MAGIC modelo nao trata todas as fraudes igual.

# COMMAND ----------

df_gabarito = spark.table(f"{CATALOGO}.{SCHEMA}._gabarito_personas_fraude")

(spark.table(f"{CATALOGO}.{SCHEMA}.gold_fila_auditoria")
    .filter(col("fraude_confirmada") == 1)
    .join(df_gabarito, "pedido_id", "left")
    .groupBy("persona_fraude")
    .agg(
        count("*").alias("fraudes"),
        round(avg("score_fraude"), 3).alias("score_medio"),
        round(avg(when(col("score_fraude") >= THRESHOLD, 1.0).otherwise(0.0)) * 100, 1)
            .alias("pct_detectada"),
        round(sum("valor_solicitado"), 2).alias("valor_total"),
    )
    .orderBy(col("pct_detectada").desc())
    .display())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Leitura dos resultados
# MAGIC
# MAGIC As personas com assinatura **deterministica** (`documento_reciclado`,
# MAGIC `implausibilidade_clinica`, `fracionamento`) sao praticamente todas capturadas —
# MAGIC existe uma feature que as denuncia diretamente.
# MAGIC
# MAGIC `duplicidade_rede` e `adesao_oportunista` sao as mais dificeis: dependem de
# MAGIC contexto que se confunde com comportamento legitimo (um beneficiario novo
# MAGIC realmente pode precisar de tratamento caro logo apos entrar no plano).
# MAGIC
# MAGIC > **Conversa de negocio:** esse quadro e o que diz onde investir. Se
# MAGIC > `duplicidade_rede` vale milhoes, vale a pena criar features especificas
# MAGIC > (janela de tolerancia de datas, match por valor) em vez de trocar o algoritmo.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Resumo
# MAGIC
# MAGIC | Conceito | O que voce praticou |
# MAGIC |----------|--------------------|
# MAGIC | **Baseline primeiro** | Medir a regra atual antes de propor ML |
# MAGIC | **Target leakage** | Identificar colunas preenchidas apos a decisao |
# MAGIC | **Feature relacional** | Window Functions: concentracao, rajada, cruzamento |
# MAGIC | **Conhecimento de dominio** | `dente_ja_extraido`, `prox_teto` — regras de auditor viram coluna |
# MAGIC | **Split temporal** | Evitar vazamento entre esquemas de fraude |
# MAGIC | **Metrica correta** | PR AUC vs acuracia/ROC em classe rara |
# MAGIC | **Ablacao** | Provar em numeros o ganho das features relacionais |
# MAGIC | **Threshold economico** | Corte por beneficio liquido, nao 0,5 |
# MAGIC | **Governanca** | UC Model Registry, aliases, threshold como tag |
# MAGIC | **Producao** | Fila priorizada por risco (pyfunc; Spark UDF p/ big data) |
# MAGIC
# MAGIC ### Aliases no Unity Catalog
# MAGIC
# MAGIC | Alias | Significado | Como carregar |
# MAGIC |-------|------------|---------------|
# MAGIC | `champion` | Modelo em producao | `models:/{nome}@champion` |
# MAGIC | `challenger` | Candidato a producao | `models:/{nome}@challenger` |
# MAGIC
# MAGIC ---
# MAGIC
# MAGIC **Proximo:** Lab 02 — GenAI aplicado ao mesmo caso de uso.
