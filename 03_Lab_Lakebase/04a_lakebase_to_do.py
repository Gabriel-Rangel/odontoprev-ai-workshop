# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 03 — Lakebase: features online para inferencia em tempo real
# MAGIC ## Exercicios (To-Do)
# MAGIC
# MAGIC **Objetivo:** resolver o problema que o Lab 02 deixou aberto — de onde o app tira
# MAGIC as features **no momento** em que o pedido de reembolso chega.
# MAGIC
# MAGIC ### O problema, em uma frase
# MAGIC
# MAGIC No Lab 02 nos "trapaceamos": para montar o payload do endpoint, lemos as 20
# MAGIC features de uma tabela que **ja estava calculada**. Um app de verdade nao tem isso.
# MAGIC Quando o pedido entra, o app conhece **o pedido** — e nada mais.
# MAGIC
# MAGIC Olhe as 20 features do modelo separadas por origem:
# MAGIC
# MAGIC | Origem | Qtd | Exemplos | Da para calcular na hora? |
# MAGIC |---|---|---|---|
# MAGIC | **Do proprio pedido** | 13 | `valor_solicitado`, `prox_teto`, `dente_ja_extraido` | ✅ sim |
# MAGIC | **Agregado do prestador** | 4 | `prest_n_benef`, `prest_pct_prox_teto` | ❌ nao |
# MAGIC | **Agregado do beneficiario** | 3 | `benef_valor_total`, `benef_n_prestadores` | ❌ nao |
# MAGIC
# MAGIC Aquelas **7 features sao memoria de 2 anos de historico**. `prest_n_benef` significa
# MAGIC "quantos beneficiarios distintos ja pediram reembolso neste prestador" — uma
# MAGIC agregacao sobre a base inteira. Rodar isso num job Spark leva **minutos**; o app
# MAGIC tem **milissegundos**.
# MAGIC
# MAGIC ### A solucao: separar *calcular* de *servir*
# MAGIC
# MAGIC ```
# MAGIC   BATCH (1x/dia, Spark)              ONLINE (a cada pedido, ms)
# MAGIC   ─────────────────────              ──────────────────────────
# MAGIC   agrega 2 anos de historico   ──►   le por CHAVE PRIMARIA
# MAGIC   gold_features_prestador            SELECT ... WHERE prestador_id = 412
# MAGIC   gold_features_beneficiario         (Lakebase / Postgres)
# MAGIC ```
# MAGIC
# MAGIC O **Lakebase** e o Postgres gerenciado do Databricks. Usado assim — guardando
# MAGIC features pre-agregadas para leitura por chave — ele faz o papel de
# MAGIC **online feature store**.
# MAGIC
# MAGIC > **Importante:** o modelo **nao muda em nada**. Ele continua recebendo as mesmas
# MAGIC > 20 colunas. O que muda e **de onde 7 delas vem** na hora do score.
# MAGIC
# MAGIC ### O que vamos fazer
# MAGIC
# MAGIC | Passo | O que faz |
# MAGIC |-------|-----------|
# MAGIC | 1 | Construir as 2 tabelas de agregado (com PK e CDF) |
# MAGIC | 2 | Criar a **instancia Lakebase** (pela UI) |
# MAGIC | 3 | Criar as 2 **synced tables** Delta ➜ Postgres (pela UI) |
# MAGIC | 4 | Ler por chave primaria e comparar latencia com o SQL Warehouse |
# MAGIC
# MAGIC ## 🔧 O que personalizar neste lab
# MAGIC
# MAGIC | Valor | Onde | Sugestao |
# MAGIC |---|---|---|
# MAGIC | **Nome da instancia Lakebase** | Passo 2 (UI) + codigo | `lakebase-<seu-nome>` |
# MAGIC | **Nome das synced tables** | Passo 3 (UI) | `online_features_prestador` / `..._beneficiario` |
# MAGIC
# MAGIC Procure por **`👉 ALTERE`** nas celulas.
# MAGIC
# MAGIC > **Pre-requisito:** Lab 01 concluido (precisamos de `gold_features_fraude`).
# MAGIC > O Lakebase esta em **Public Preview**.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dependencias

# COMMAND ----------

# MAGIC %pip install -q psycopg2-binary 'databricks-sdk>=0.59'

# COMMAND ----------

# MAGIC %md
# MAGIC > O `databricks-sdk` do runtime serverless e antigo e **nao** tem a API
# MAGIC > `w.database` (Lakebase e recente). Por isso fixamos `>=0.59`.

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuracao

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()
CATALOGO = "workshop_databricks"
SCHEMA = nome

# 👉 ALTERE se usar outro nome. Instancias nao aceitam "_" — usamos "-".
INSTANCIA = f"lakebase-{nome.replace('_', '-')}"

spark.sql(f"USE CATALOG {CATALOGO}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Schema:    {CATALOGO}.{SCHEMA}")
print(f"Instancia: {INSTANCIA}")

# COMMAND ----------

import json
import time
import uuid

from pyspark.sql.functions import *

from builtins import abs as pyabs, max as pymax, round as pyround

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 1 — Construir as tabelas de agregado
# MAGIC
# MAGIC A `gold_features_fraude` do Lab 01 e chaveada por `pedido_id` — **nao serve** para
# MAGIC lookup por entidade. Precisamos de duas tabelas novas, uma linha por entidade:
# MAGIC
# MAGIC | Tabela | Chave | Linhas | Colunas de feature |
# MAGIC |---|---|---|---|
# MAGIC | `gold_features_prestador` | `prestador_id` | ~700 | 4 |
# MAGIC | `gold_features_beneficiario` | `beneficiario_id` | ~7.600 | 3 |
# MAGIC
# MAGIC ### Tres requisitos que a synced table impoe
# MAGIC
# MAGIC | Requisito | Por que |
# MAGIC |---|---|
# MAGIC | **Chave primaria** declarada | e por ela que o Postgres indexa e busca |
# MAGIC | Coluna da PK **NOT NULL** | linhas com chave nula sao descartadas no sync |
# MAGIC | **Change Data Feed** habilitado | necessario para os modos `TRIGGERED`/`CONTINUOUS` |
# MAGIC
# MAGIC > Vamos usar `SNAPSHOT` (que **nao** exige CDF), mas habilitamos o CDF de qualquer
# MAGIC > forma: assim voce pode trocar de modo depois sem refazer as tabelas.
# MAGIC
# MAGIC ### **TO-DO 1** — Criar as duas tabelas com PK, NOT NULL e CDF
# MAGIC
# MAGIC Os agregados ja estao calculados em `gold_features_fraude` (repetidos em cada
# MAGIC pedido da mesma entidade) — basta pegar um valor por entidade com `MAX(...)` e
# MAGIC agrupar pela chave.
# MAGIC
# MAGIC > **Dica:** `ALTER TABLE ... ALTER COLUMN x SET NOT NULL`,
# MAGIC > `ADD CONSTRAINT pk_x PRIMARY KEY (x)`,
# MAGIC > `SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')`

# COMMAND ----------

# ============================================================
# TO-DO: escreva seu codigo aqui.
#
# A solucao de referencia esta comentada abaixo — tente primeiro,
# depois compare (ou veja 04b_lakebase_completo).
# ============================================================
# # --- Agregado por PRESTADOR (4 features) ---
# spark.sql(f"""
#     CREATE OR REPLACE TABLE {CATALOGO}.{SCHEMA}.gold_features_prestador AS
#     SELECT
#       prestador_id,
#       MAX(prest_n_pedidos)     AS prest_n_pedidos,
#       MAX(prest_n_benef)       AS prest_n_benef,
#       MAX(prest_valor_medio)   AS prest_valor_medio,
#       MAX(prest_pct_prox_teto) AS prest_pct_prox_teto
#     FROM {CATALOGO}.{SCHEMA}.gold_features_fraude
#     GROUP BY prestador_id
# """)
#
# spark.sql(f"ALTER TABLE {CATALOGO}.{SCHEMA}.gold_features_prestador "
#           f"ALTER COLUMN prestador_id SET NOT NULL")
# spark.sql(f"ALTER TABLE {CATALOGO}.{SCHEMA}.gold_features_prestador "
#           f"ADD CONSTRAINT pk_prestador PRIMARY KEY (prestador_id)")
# spark.sql(f"ALTER TABLE {CATALOGO}.{SCHEMA}.gold_features_prestador "
#           f"SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
#
# # --- Agregado por BENEFICIARIO (3 features) ---
# spark.sql(f"""
#     CREATE OR REPLACE TABLE {CATALOGO}.{SCHEMA}.gold_features_beneficiario AS
#     SELECT
#       beneficiario_id,
#       MAX(benef_n_pedidos)     AS benef_n_pedidos,
#       MAX(benef_valor_total)   AS benef_valor_total,
#       MAX(benef_n_prestadores) AS benef_n_prestadores
#     FROM {CATALOGO}.{SCHEMA}.gold_features_fraude
#     GROUP BY beneficiario_id
# """)
#
# spark.sql(f"ALTER TABLE {CATALOGO}.{SCHEMA}.gold_features_beneficiario "
#           f"ALTER COLUMN beneficiario_id SET NOT NULL")
# spark.sql(f"ALTER TABLE {CATALOGO}.{SCHEMA}.gold_features_beneficiario "
#           f"ADD CONSTRAINT pk_benef PRIMARY KEY (beneficiario_id)")
# spark.sql(f"ALTER TABLE {CATALOGO}.{SCHEMA}.gold_features_beneficiario "
#           f"SET TBLPROPERTIES ('delta.enableChangeDataFeed' = 'true')")
#
# n_prest = spark.table(f"{CATALOGO}.{SCHEMA}.gold_features_prestador").count()
# n_benef = spark.table(f"{CATALOGO}.{SCHEMA}.gold_features_beneficiario").count()
# print(f"gold_features_prestador:    {n_prest:>6,} linhas (PK prestador_id)")
# print(f"gold_features_beneficiario: {n_benef:>6,} linhas (PK beneficiario_id)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quem sao os prestadores mais concentrados?
# MAGIC
# MAGIC Vale olhar o conteudo do que vamos servir: sao exatamente esses numeros que fazem
# MAGIC o modelo suspeitar de uma clinica de fachada.

# COMMAND ----------

# ============================================================
# TO-DO: escreva seu codigo aqui.
#
# A solucao de referencia esta comentada abaixo — tente primeiro,
# depois compare (ou veja 04b_lakebase_completo).
# ============================================================
# (spark.table(f"{CATALOGO}.{SCHEMA}.gold_features_prestador")
#     .orderBy(col("prest_n_benef").desc())
#     .limit(10)
#     .display())

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 2 (UI) — Criar a instancia Lakebase
# MAGIC
# MAGIC > ⏱️ Leva alguns minutos. Dispare agora e siga lendo.
# MAGIC
# MAGIC 1. Menu lateral > **Compute** > aba **Database instances** (ou busque "Lakebase").
# MAGIC 2. **Create database instance**.
# MAGIC 3. **Name** = `lakebase-<seu-nome>` &nbsp; 👉 **ALTERE**
# MAGIC    (o valor impresso na celula de configuracao; use `-`, nao `_`)
# MAGIC 4. **Capacity** = `CU_1` (o menor; suficiente para o lab)
# MAGIC 5. **Create** e aguarde o status **Available**.
# MAGIC
# MAGIC O que voce acabou de criar e um **Postgres gerenciado** (PG 16), com DNS proprio de
# MAGIC leitura/escrita. Ele nao substitui o lakehouse — ele **serve** o lakehouse para
# MAGIC quem precisa de resposta em milissegundos.
# MAGIC
# MAGIC > 📄 Doc: https://docs.databricks.com/aws/en/oltp/
# MAGIC
# MAGIC ### Opcao B (CLI) — se preferir criar por codigo
# MAGIC
# MAGIC ```bash
# MAGIC databricks database create-database-instance lakebase-<seu-nome> \
# MAGIC   --capacity CU_1 -p gabriel-dev
# MAGIC ```
# MAGIC
# MAGIC > 💰 **Atencao ao custo:** a instancia cobra **enquanto existir**, mesmo sem uso.
# MAGIC > O `99_Cleanup` remove ela no fim do workshop.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confirmar que a instancia esta disponivel

# COMMAND ----------

# ============================================================
# TO-DO: escreva seu codigo aqui.
#
# A solucao de referencia esta comentada abaixo — tente primeiro,
# depois compare (ou veja 04b_lakebase_completo).
# ============================================================
# from databricks.sdk import WorkspaceClient
#
# w = WorkspaceClient()
#
# for tentativa in range(40):
#     inst = w.database.get_database_instance(name=INSTANCIA)
#     estado = str(inst.state)
#     if "AVAILABLE" in estado.upper():
#         print(f"Instancia AVAILABLE")
#         print(f"  host (read-write): {inst.read_write_dns}")
#         print(f"  PG version:        {inst.pg_version}")
#         break
#     print(f"  [{tentativa * 15:4d}s] estado = {estado}")
#     time.sleep(15)
#
# HOST_PG = inst.read_write_dns

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 3 (UI) — Criar as synced tables (Delta ➜ Postgres)
# MAGIC
# MAGIC A **synced table** e o mecanismo gerenciado que replica uma tabela do Unity Catalog
# MAGIC para o Postgres do Lakebase. Sem job, sem codigo de ETL.
# MAGIC
# MAGIC ### Tabela 1 — prestador
# MAGIC
# MAGIC 1. **Catalog Explorer** > abra `workshop_databricks.<seu_nome>.gold_features_prestador`.
# MAGIC 2. Botao **Create** > **Synced table**.
# MAGIC 3. Configure:
# MAGIC    - **Database instance** = `lakebase-<seu-nome>` &nbsp; 👉 **ALTERE**
# MAGIC    - **Primary key** = `prestador_id`
# MAGIC    - **Sync mode** = **Snapshot**
# MAGIC    - **Destination (UC)** = `workshop_databricks.<seu_nome>.online_features_prestador`
# MAGIC      → no Postgres aparece como **`<seu_nome>.online_features_prestador`**
# MAGIC        (o schema do Postgres espelha o schema do UC)
# MAGIC 4. **Create** e acompanhe ate **Online**.
# MAGIC
# MAGIC ### Tabela 2 — beneficiario
# MAGIC
# MAGIC Repita para `gold_features_beneficiario`, com **Primary key** = `beneficiario_id`
# MAGIC e destino `online_features_beneficiario`.
# MAGIC
# MAGIC ### Escolha do *Sync mode*
# MAGIC
# MAGIC | Modo | Como funciona | Quando usar |
# MAGIC |---|---|---|
# MAGIC | **Snapshot** | recarga completa a cada sync | **o lab** — simples, dispensa CDF |
# MAGIC | **Triggered** | so o delta, sob demanda/agendado | producao com cadencia definida |
# MAGIC | **Continuous** | pipeline sempre ligada | quando minutos de atraso importam |
# MAGIC
# MAGIC Para features agregadas de 2 anos, **Snapshot diario e o suficiente**: `prest_n_benef`
# MAGIC nao muda de forma relevante em uma hora. Já para uma feature como "pedidos deste
# MAGIC beneficiario na ultima hora", ai voce precisaria de `Continuous`.
# MAGIC
# MAGIC > 📄 Doc: https://docs.databricks.com/aws/en/oltp/sync-data/sync-table
# MAGIC
# MAGIC ### Opcao B (SDK) — criar por codigo
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk.service.database import (
# MAGIC     SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy)
# MAGIC
# MAGIC w.database.create_synced_database_table(
# MAGIC     SyncedDatabaseTable(
# MAGIC         name=f"{CATALOGO}.{SCHEMA}.online_features_prestador",
# MAGIC         database_instance_name=INSTANCIA,
# MAGIC         logical_database_name="databricks_postgres",
# MAGIC         spec=SyncedTableSpec(
# MAGIC             source_table_full_name=f"{CATALOGO}.{SCHEMA}.gold_features_prestador",
# MAGIC             primary_key_columns=["prestador_id"],
# MAGIC             scheduling_policy=SyncedTableSchedulingPolicy.SNAPSHOT,
# MAGIC             create_database_objects_if_missing=True,
# MAGIC         ),
# MAGIC     )
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC > Equivalente no CLI: `databricks database create-synced-database-table --json @resources/synced_prestador.json`

# COMMAND ----------

# MAGIC %md
# MAGIC ### Acompanhar o sync até ficar `ONLINE`
# MAGIC
# MAGIC A celula abaixo **cria as synced tables se ainda nao existirem** (util se voce
# MAGIC preferiu nao usar a UI) e depois espera as duas ficarem `ONLINE`.

# COMMAND ----------

# ============================================================
# TO-DO: escreva seu codigo aqui.
#
# A solucao de referencia esta comentada abaixo — tente primeiro,
# depois compare (ou veja 04b_lakebase_completo).
# ============================================================
# from databricks.sdk.errors import AlreadyExists, NotFound
# from databricks.sdk.service.database import (
#     SyncedDatabaseTable, SyncedTableSpec, SyncedTableSchedulingPolicy,
# )
#
# ONLINE_PREST = f"{CATALOGO}.{SCHEMA}.online_features_prestador"
# ONLINE_BENEF = f"{CATALOGO}.{SCHEMA}.online_features_beneficiario"
#
# # (destino_online, tabela_fonte, coluna de chave primaria)
# SYNCS = [
#     (ONLINE_PREST, f"{CATALOGO}.{SCHEMA}.gold_features_prestador", "prestador_id"),
#     (ONLINE_BENEF, f"{CATALOGO}.{SCHEMA}.gold_features_beneficiario", "beneficiario_id"),
# ]
#
# for destino, fonte, chave in SYNCS:
#     try:
#         w.database.get_synced_database_table(name=destino)
#         print(f"{destino.split('.')[-1]:<32} ja existe (criada na UI)")
#         continue
#     except NotFound:
#         pass
#
#     try:
#         w.database.create_synced_database_table(
#             SyncedDatabaseTable(
#                 name=destino,
#                 database_instance_name=INSTANCIA,
#                 logical_database_name="databricks_postgres",
#                 spec=SyncedTableSpec(
#                     source_table_full_name=fonte,
#                     primary_key_columns=[chave],
#                     scheduling_policy=SyncedTableSchedulingPolicy.SNAPSHOT,
#                     create_database_objects_if_missing=True,
#                 ),
#             )
#         )
#         print(f"{destino.split('.')[-1]:<32} criada por codigo")
#     except AlreadyExists:
#         # Apagar a synced table no UC nao remove a tabela fisica no Postgres.
#         # Se voce ja rodou este lab antes, sobra o "orfao" — derrube-o e recrie.
#         print(f"{destino.split('.')[-1]:<32} orfa no Postgres — veja a celula seguinte")
#         raise
#
# for tabela in (ONLINE_PREST, ONLINE_BENEF):
#     for tentativa in range(40):
#         t = w.database.get_synced_database_table(name=tabela)
#         estado = str(t.data_synchronization_status.detailed_state)
#         if "ONLINE" in estado:
#             print(f"{tabela.split('.')[-1]:<32} ONLINE")
#             break
#         if "FAILED" in estado:
#             print(f"{tabela.split('.')[-1]:<32} FALHOU: {estado}")
#             break
#         time.sleep(15)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔧 Se der `AlreadyExists` na celula acima
# MAGIC
# MAGIC Apagar uma synced table no Unity Catalog **nao remove** a tabela fisica que ela
# MAGIC criou no Postgres. Se voce ja rodou este lab antes, sobra um "orfao" com o mesmo
# MAGIC nome e a criacao falha.
# MAGIC
# MAGIC Rode a celula abaixo **apenas nesse caso** — ela derruba as tabelas orfas no
# MAGIC Postgres para voce recriar as synced tables.
# MAGIC
# MAGIC > É a mesma pegadinha que aparece em qualquer replicacao gerenciada: o destino tem
# MAGIC > ciclo de vida proprio.

# COMMAND ----------

# Descomente SOMENTE se a celula anterior falhou com AlreadyExists:
#
# import psycopg2
# cred_tmp = w.database.generate_database_credential(
#     request_id=str(uuid.uuid4()), instance_names=[INSTANCIA])
# c_tmp = psycopg2.connect(
#     host=w.database.get_database_instance(name=INSTANCIA).read_write_dns,
#     port=5432, dbname="databricks_postgres", user=usuario,
#     password=cred_tmp.token, sslmode="require")
# c_tmp.autocommit = True
# with c_tmp.cursor() as cur_tmp:
#     for t in ("online_features_prestador", "online_features_beneficiario"):
#         cur_tmp.execute(f'DROP TABLE IF EXISTS "{SCHEMA}".{t} CASCADE')
#         print(f"dropada no Postgres: {SCHEMA}.{t}")
# c_tmp.close()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 4 — Ler as features por chave primaria
# MAGIC
# MAGIC Agora a parte que importa: conectar no Postgres e buscar as features de **um**
# MAGIC prestador — exatamente o que o app do Lab 04 vai fazer a cada pedido.
# MAGIC
# MAGIC ### Como o app se autentica no Lakebase
# MAGIC
# MAGIC Nao existe senha fixa. Voce gera uma **credencial temporaria** (OAuth) e usa como
# MAGIC senha do Postgres:
# MAGIC
# MAGIC | Campo | Valor |
# MAGIC |---|---|
# MAGIC | `host` | o `read_write_dns` da instancia |
# MAGIC | `user` | seu usuario Databricks (ou o *service principal* do app) |
# MAGIC | `password` | token de `generate_database_credential` |
# MAGIC | `sslmode` | `require` |
# MAGIC
# MAGIC ### **TO-DO 2** — Conectar e ler por chave
# MAGIC
# MAGIC > **Dica:**
# MAGIC > ```python
# MAGIC > cred = w.database.generate_database_credential(
# MAGIC >     request_id=str(uuid.uuid4()), instance_names=[INSTANCIA])
# MAGIC > psycopg2.connect(host=HOST_PG, port=5432, dbname="databricks_postgres",
# MAGIC >                  user=usuario, password=cred.token, sslmode="require")
# MAGIC > ```

# COMMAND ----------

# ============================================================
# TO-DO: escreva seu codigo aqui.
#
# A solucao de referencia esta comentada abaixo — tente primeiro,
# depois compare (ou veja 04b_lakebase_completo).
# ============================================================
# import psycopg2
#
# cred = w.database.generate_database_credential(
#     request_id=str(uuid.uuid4()), instance_names=[INSTANCIA]
# )
#
# conn = psycopg2.connect(
#     host=HOST_PG, port=5432, dbname="databricks_postgres",
#     user=usuario, password=cred.token, sslmode="require", connect_timeout=15,
# )
# cur = conn.cursor()
#
# # O schema no Postgres espelha o schema do UC
# cur.execute("""
#     SELECT table_schema, table_name
#     FROM information_schema.tables
#     WHERE table_name LIKE 'online_features%'
#     ORDER BY table_name
# """)
# print("Tabelas visiveis no Postgres:")
# for esquema, tabela in cur.fetchall():
#     print(f"  {esquema}.{tabela}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### O lookup que o app faz a cada pedido
# MAGIC
# MAGIC Duas queries por chave primaria. E isso que substitui um job Spark de minutos.

# COMMAND ----------

# ============================================================
# TO-DO: escreva seu codigo aqui.
#
# A solucao de referencia esta comentada abaixo — tente primeiro,
# depois compare (ou veja 04b_lakebase_completo).
# ============================================================
# def buscar_features_online(prestador_id: int, beneficiario_id: int) -> dict:
#     """Busca os 7 agregados no Lakebase. E o coracao do Lab 04."""
#     feats = {}
#     cur.execute(
#         f'SELECT prest_n_pedidos, prest_n_benef, prest_valor_medio, prest_pct_prox_teto '
#         f'FROM "{SCHEMA}".online_features_prestador WHERE prestador_id = %s',
#         (prestador_id,),
#     )
#     linha = cur.fetchone()
#     if linha:
#         feats.update(zip(
#             ["prest_n_pedidos", "prest_n_benef", "prest_valor_medio", "prest_pct_prox_teto"],
#             linha,
#         ))
#
#     cur.execute(
#         f'SELECT benef_n_pedidos, benef_valor_total, benef_n_prestadores '
#         f'FROM "{SCHEMA}".online_features_beneficiario WHERE beneficiario_id = %s',
#         (beneficiario_id,),
#     )
#     linha = cur.fetchone()
#     if linha:
#         feats.update(zip(
#             ["benef_n_pedidos", "benef_valor_total", "benef_n_prestadores"], linha
#         ))
#     return feats
#
#
# # Pega um par (prestador, beneficiario) real para testar
# amostra = (
#     spark.table(f"{CATALOGO}.{SCHEMA}.gold_features_fraude")
#     .select("prestador_id", "beneficiario_id")
#     .limit(1).collect()[0]
# )
# pid, bid = int(amostra.prestador_id), int(amostra.beneficiario_id)
#
# inicio = time.perf_counter()
# features_online = buscar_features_online(pid, bid)
# ms_lookup = (time.perf_counter() - inicio) * 1000
#
# print(f"Lookup de prestador={pid}, beneficiario={bid} em {ms_lookup:.1f} ms\n")
# for k, v in features_online.items():
#     print(f"  {k:<24} {v}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### **TO-DO 3** — Comparar com o SQL Warehouse
# MAGIC
# MAGIC Faca a **mesma** busca por chave, mas pelo Spark/warehouse (lendo a tabela Delta),
# MAGIC e compare os tempos. Os dois numeros lado a lado sao o argumento inteiro do OLTP.
# MAGIC
# MAGIC > Nao e que o warehouse seja "ruim" — ele e otimizado para **varrer** e agregar
# MAGIC > milhoes de linhas. So nao e feito para responder "me da a linha 412" em 5 ms.

# COMMAND ----------

# ============================================================
# TO-DO: escreva seu codigo aqui.
#
# A solucao de referencia esta comentada abaixo — tente primeiro,
# depois compare (ou veja 04b_lakebase_completo).
# ============================================================
# inicio = time.perf_counter()
# _ = (
#     spark.table(f"{CATALOGO}.{SCHEMA}.gold_features_prestador")
#     .filter(col("prestador_id") == pid)
#     .collect()
# )
# ms_spark = (time.perf_counter() - inicio) * 1000
#
# print(f"{'Caminho':<34} {'Latencia':>12}")
# print(f"{'-'*48}")
# print(f"{'Lakebase (Postgres, por PK)':<34} {ms_lookup:>9.1f} ms")
# print(f"{'Delta via Spark (mesma busca)':<34} {ms_spark:>9.1f} ms")
# print(f"{'-'*48}")
# print(f"{'Diferenca':<34} {ms_spark / pymax(ms_lookup, 0.001):>9.1f}x")
# print()
# print("Num app que precisa responder a cada pedido, essa diferenca decide")
# print("se a decisao acontece ANTES ou DEPOIS do dinheiro sair.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Conferir que os valores batem com o batch
# MAGIC
# MAGIC O Postgres e uma **replica** — os numeros tem que ser identicos aos do Delta.
# MAGIC Se divergirem, o sync esta atrasado (ou parado).

# COMMAND ----------

# ============================================================
# TO-DO: escreva seu codigo aqui.
#
# A solucao de referencia esta comentada abaixo — tente primeiro,
# depois compare (ou veja 04b_lakebase_completo).
# ============================================================
# delta = (
#     spark.table(f"{CATALOGO}.{SCHEMA}.gold_features_prestador")
#     .filter(col("prestador_id") == pid)
#     .toPandas().iloc[0]
# )
#
# print(f"{'Feature':<24} {'Lakebase':>12} {'Delta':>12} {'OK':>5}")
# print("-" * 56)
# for c in ["prest_n_pedidos", "prest_n_benef", "prest_valor_medio", "prest_pct_prox_teto"]:
#     v_pg, v_delta = features_online.get(c), delta[c]
#     ok = "sim" if pyabs(float(v_pg) - float(v_delta)) < 1e-6 else "NAO"
#     print(f"{c:<24} {float(v_pg):>12.4f} {float(v_delta):>12.4f} {ok:>5}")

# COMMAND ----------

# ============================================================
# TO-DO: escreva seu codigo aqui.
#
# A solucao de referencia esta comentada abaixo — tente primeiro,
# depois compare (ou veja 04b_lakebase_completo).
# ============================================================
# cur.close()
# conn.close()
# print("Conexao encerrada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Apendice — e o Feature Store "de verdade"?
# MAGIC
# MAGIC O que fizemos aqui e um **lookup manual**: o app busca as features e monta o vetor.
# MAGIC O Databricks tambem oferece um caminho mais automatizado, com o
# MAGIC `FeatureEngineeringClient`:
# MAGIC
# MAGIC ```python
# MAGIC from databricks.feature_engineering import FeatureEngineeringClient
# MAGIC fe = FeatureEngineeringClient()
# MAGIC fe.create_online_store(name="fraude_online", capacity="CU_1")
# MAGIC fe.publish_table(online_store=..., source_table_name=..., publish_mode="TRIGGERED")
# MAGIC ```
# MAGIC
# MAGIC Nesse modelo, um modelo treinado com **feature spec** faz o lookup **sozinho** no
# MAGIC serving: o app manda so `{"prestador_id": 412, "valor_solicitado": 1450}` e o
# MAGIC endpoint completa o resto.
# MAGIC
# MAGIC | | Lookup manual (este lab) | Feature Store automatico |
# MAGIC |---|---|---|
# MAGIC | Quem busca as features | o app | o endpoint |
# MAGIC | Muda o treino do modelo? | **nao** | **sim** — exige `log_model(feature_spec=...)` |
# MAGIC | Voce ve o que acontece | tudo explicito | abstraido |
# MAGIC
# MAGIC **Por que escolhemos o manual no workshop:** ele nao exige redesenhar o modelo do
# MAGIC Lab 01, e deixa visivel *exatamente* o que acontece entre o pedido chegar e o score
# MAGIC sair. Entendido isso, adotar a versao automatica depois e simples.
# MAGIC
# MAGIC ---
# MAGIC ## Resumo
# MAGIC
# MAGIC | Conceito | O que voce praticou |
# MAGIC |----------|--------------------|
# MAGIC | **Separar calcular de servir** | Spark agrega 1x/dia; Postgres serve por chave |
# MAGIC | **Online feature store** | features pre-agregadas lidas por PK em ms |
# MAGIC | **Synced table** | replicacao gerenciada Delta ➜ Postgres, sem ETL na mao |
# MAGIC | **Requisitos do sync** | PK declarada, NOT NULL, CDF p/ Triggered/Continuous |
# MAGIC | **Escolha de sync mode** | Snapshot vs Triggered vs Continuous, pela volatilidade |
# MAGIC | **Credencial temporaria** | OAuth como senha do Postgres |
# MAGIC | **Consistencia** | replica tem que bater com o Delta |
# MAGIC
# MAGIC > **Proximo:** Lab 04 — juntar tudo num **app** que recebe um pedido novo, busca
# MAGIC > estas 7 features aqui, chama o endpoint do Lab 02 e mostra o score na hora.
