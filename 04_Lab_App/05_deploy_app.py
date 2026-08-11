# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 04 — Databricks App: simulador de inferencia em tempo real
# MAGIC ## Deploy guiado
# MAGIC
# MAGIC **Objetivo:** publicar um app que recebe um **pedido de reembolso novo** e devolve
# MAGIC o score de fraude na hora — juntando o que voce construiu nos tres labs anteriores.
# MAGIC
# MAGIC ```
# MAGIC   [pedido novo no formulario]
# MAGIC        |
# MAGIC        |-- 13 features calculadas do proprio pedido        (features.py)
# MAGIC        |-- 7 agregados buscados no Lakebase por chave      (Lab 03)
# MAGIC        |
# MAGIC        v
# MAGIC   vetor de 20 features  ->  POST /serving-endpoints/.../invocations   (Lab 02)
# MAGIC        |
# MAGIC        v
# MAGIC   [score + prioridade + "por que" + latencia em ms]
# MAGIC ```
# MAGIC
# MAGIC ### O que o app demonstra em sala
# MAGIC
# MAGIC | Bloco | O que mostra |
# MAGIC |---|---|
# MAGIC | Formulario + **presets** | 6 cenarios prontos (rotineiro, colado no teto, clinica de fachada, fracionamento, dente extraido, adesao oportunista) |
# MAGIC | Gauge de risco | score + faixa (CRITICA/ALTA/MEDIA/BAIXA) com o threshold **homologado** |
# MAGIC | Latencia separada | **lookup** (Lakebase) vs **inferencia** (Model Serving) |
# MAGIC | "Por que subiu na fila" | as 20 features com a origem de cada uma — o caso auditavel |
# MAGIC
# MAGIC ## 🔧 O que personalizar neste lab
# MAGIC
# MAGIC | Valor | Onde | Sugestao |
# MAGIC |---|---|---|
# MAGIC | **Nome do app** | Passo 2 (UI) | `fraude-sim-<seu-nome>` |
# MAGIC | `LAKEBASE_INSTANCE` | `app.yaml` | `lakebase-<seu-nome>` (Lab 03) |
# MAGIC | `PG_SCHEMA` / `SCHEMA_UC` | `app.yaml` | `<seu_usuario>` |
# MAGIC | `THRESHOLD` | `app.yaml` | o valor homologado no seu Lab 01 |
# MAGIC
# MAGIC Procure por **`👉 ALTERE`** no `app.yaml`.
# MAGIC
# MAGIC > **Pre-requisitos:** Lab 02 (endpoint `Ready`) e Lab 03 (synced tables `Online`).
# MAGIC > Sem o Lab 03 da para rodar com `USE_LAKEBASE=false` (fallback via SQL Warehouse).
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuracao — descubra os SEUS valores

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()
nome_hifen = nome.replace("_", "-")

print("Preencha o app.yaml com estes valores:\n")
print(f"  LAKEBASE_INSTANCE = lakebase-{nome_hifen}")
print(f"  PG_SCHEMA         = {nome}")
print(f"  SCHEMA_UC         = {nome}")
print(f"\nNome sugerido do app: fraude-sim-{nome_hifen}   (max 30 caracteres!)")
print(f"Endpoint do Lab 02:   fraude-reembolso-{nome_hifen}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Threshold homologado — copie para o `app.yaml`
# MAGIC
# MAGIC O corte de negocio nao fica hardcoded no app: ele foi gravado como **tag da
# MAGIC versao do modelo** no Lab 01. Leia daqui e cole no `app.yaml`.

# COMMAND ----------

# MAGIC %pip install -q 'mlflow>=2.16'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()

import mlflow
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
MODEL_NAME = f"workshop_databricks.{nome}.modelo_fraude_reembolso"

mv = MlflowClient().get_model_version_by_alias(MODEL_NAME, "champion")
print(f"Champion: v{mv.version}")
print(f"THRESHOLD = {mv.tags.get('threshold', '0.30')}   <- cole no app.yaml")
print(f"PR AUC    = {mv.tags.get('pr_auc', 'n/d')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 1 — Editar o `app.yaml`
# MAGIC
# MAGIC Abra [`app.yaml`](./app.yaml) nesta mesma pasta e troque os `<seu_usuario>` /
# MAGIC `<seu-usuario>` pelos valores impressos acima. Confira tambem o `THRESHOLD`.
# MAGIC
# MAGIC Repare em duas formas diferentes de configurar:
# MAGIC
# MAGIC | Forma | Quando usar |
# MAGIC |---|---|
# MAGIC | `valueFrom: serving-endpoint` | o valor vem do **resource** que voce adiciona na UI |
# MAGIC | `value: "lakebase-..."` | valor literal, que voce mesmo escreve |
# MAGIC
# MAGIC ---
# MAGIC ## Passo 2 (UI) — Criar e publicar o app
# MAGIC
# MAGIC 1. Menu lateral > **Compute** > aba **Apps** > **Create app** > **Custom**.
# MAGIC 2. **Name** = `fraude-sim-<seu-nome>` &nbsp; 👉 **ALTERE** > **Create**
# MAGIC    (⚠️ nome de app tem **maximo 30 caracteres** e nao aceita `_`)
# MAGIC    (isso provisiona o app **e** um *service principal* proprio para ele)
# MAGIC 3. **Configure** > **+ Add resource** — adicione **tres**:
# MAGIC    - **Serving endpoint** = `fraude-reembolso-<seu-nome>` (**Can query**)
# MAGIC      → e o que alimenta `valueFrom: serving-endpoint`
# MAGIC    - **SQL warehouse** = o seu warehouse serverless (**Can use**)
# MAGIC      → alimenta `valueFrom: sql-warehouse`
# MAGIC    - **Database** = `lakebase-<seu-nome>` (**Can connect and create**)
# MAGIC 4. **User authorization (Preview):** deixe **vazio** — o app roda como service
# MAGIC    principal, nao em nome do usuario.
# MAGIC 5. **Compute** > **Instance size** = `Medium`.
# MAGIC 6. **Deploy**, apontando para **esta pasta** no Workspace.
# MAGIC 7. Abra a **URL** que aparece no topo da pagina do app.
# MAGIC
# MAGIC > 📄 Doc: https://docs.databricks.com/aws/en/dev-tools/databricks-apps/

# COMMAND ----------

# MAGIC %md
# MAGIC ### Opcao B (CLI) — deploy pelo terminal
# MAGIC
# MAGIC Mesma coisa, por linha de comando. Comentado — use se preferir.
# MAGIC
# MAGIC ```bash
# MAGIC APP=fraude-sim-<seu-nome>
# MAGIC WSPATH=/Workspace/Users/<voce>@databricks.com/odontoprev-ai-workshop/04_Lab_App
# MAGIC
# MAGIC databricks apps create $APP -p gabriel-dev
# MAGIC databricks apps update $APP --json @../resources/app-resources.json -p gabriel-dev
# MAGIC databricks workspace import-dir 04_Lab_App "$WSPATH" --overwrite -p gabriel-dev
# MAGIC databricks apps deploy $APP --source-code-path "$WSPATH" -p gabriel-dev
# MAGIC databricks apps get $APP -p gabriel-dev      # pega a URL e o service principal
# MAGIC ```

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 3 — Permissoes do service principal (obrigatorio)
# MAGIC
# MAGIC Esta e a pegadinha mais comum de Databricks Apps com Lakebase: o **resource
# MAGIC Database da a conexao, mas nao da privilegio nas tabelas**. Sem os GRANTs abaixo
# MAGIC o app conecta e depois falha com `permission denied for table`.
# MAGIC
# MAGIC 1. Pegue o `service_principal_client_id` do app:
# MAGIC    **Compute > Apps > seu app > aba Overview** (ou `databricks apps get <app>`).
# MAGIC 2. No **SQL editor da instancia Lakebase** (Compute > Database instances > sua
# MAGIC    instancia > **New query**), rode, trocando `<sp_client_id>` e `<seu_usuario>`:
# MAGIC
# MAGIC ```sql
# MAGIC GRANT USAGE ON SCHEMA "<seu_usuario>" TO "<sp_client_id>";
# MAGIC GRANT SELECT ON "<seu_usuario>".online_features_prestador   TO "<sp_client_id>";
# MAGIC GRANT SELECT ON "<seu_usuario>".online_features_beneficiario TO "<sp_client_id>";
# MAGIC
# MAGIC -- Cada re-sync RECRIA as tabelas. Sem isto, o acesso se perde no proximo sync:
# MAGIC ALTER DEFAULT PRIVILEGES IN SCHEMA "<seu_usuario>"
# MAGIC   GRANT SELECT ON TABLES TO "<sp_client_id>";
# MAGIC ```
# MAGIC
# MAGIC 3. No Unity Catalog (aqui no notebook), libere as tabelas Delta que o app le
# MAGIC    pelo SQL Warehouse (listas de apoio e fallback):
# MAGIC
# MAGIC > `<sp_client_id>` e o valor de **PGUSER** no app. *Role membership*
# MAGIC > (`GRANT <voce> TO <sp>`) **nao** e permitido no Lakebase — use os GRANTs acima.

# COMMAND ----------

# 👉 ALTERE o SP_CLIENT_ID e rode para liberar as tabelas do UC
SP_CLIENT_ID = "<sp_client_id>"

if SP_CLIENT_ID != "<sp_client_id>":
    usuario = spark.sql("SELECT current_user()").collect()[0][0]
    nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()
    CATALOGO, SCHEMA = "workshop_databricks", nome

    spark.sql(f"GRANT USE CATALOG ON CATALOG {CATALOGO} TO `{SP_CLIENT_ID}`")
    spark.sql(f"GRANT USE SCHEMA ON SCHEMA {CATALOGO}.{SCHEMA} TO `{SP_CLIENT_ID}`")
    for t in ("gold_features_prestador", "gold_features_beneficiario"):
        spark.sql(f"GRANT SELECT ON TABLE {CATALOGO}.{SCHEMA}.{t} TO `{SP_CLIENT_ID}`")
        print(f"  SELECT concedido em {t}")
else:
    print("Preencha SP_CLIENT_ID com o service principal do app e rode de novo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 4 — Roteiro de demonstracao
# MAGIC
# MAGIC Com o app aberto, rode os presets nesta ordem. A progressao conta a historia:
# MAGIC
# MAGIC | # | Preset | O que esperar | O ponto |
# MAGIC |---|---|---|---|
# MAGIC | 1 | **Pedido rotineiro** | score BAIXO | o modelo nao grita por qualquer coisa |
# MAGIC | 2 | **Valor colado no teto** | score sobe | R$ 1.450 numa restauracao de R$ 190 — a regra do teto **nao** pegaria |
# MAGIC | 3 | **Clinica de fachada** | score ALTO | use um `prestador_id` da lista lateral: o sinal vem do **Lakebase**, nao do pedido |
# MAGIC | 4 | **Fracionamento** | score ALTO | 5 pedidos do mesmo par em 30 dias |
# MAGIC | 5 | **Dente ja extraido** | score ALTO | impossibilidade clinica |
# MAGIC | 6 | **Adesao oportunista** | intermediario | implante caro 45 dias apos aderir — caso genuinamente ambiguo |
# MAGIC
# MAGIC ### As duas perguntas que fecham o workshop
# MAGIC
# MAGIC **1. Troque o `prestador_id` do preset "Clinica de fachada" por um prestador comum.**
# MAGIC O pedido e identico — mesmo valor, mesmo procedimento — e o score **cai**. Por que?
# MAGIC Porque 4 das 20 features nao descrevem o pedido, descrevem **o prestador**. Sem o
# MAGIC store online, o app nao teria como saber disso em 50 ms.
# MAGIC
# MAGIC **2. Olhe a latencia do lookup vs a da inferencia.**
# MAGIC Normalmente o **lookup** e a parte mais rapida, e a inferencia domina. Isso e
# MAGIC contraintuitivo e importante: o gargalo nao e "buscar dados", e o modelo. Em
# MAGIC producao e ai que se mexe (`scale_to_zero=False`, replica quente).
# MAGIC
# MAGIC ---
# MAGIC ## Rodar localmente (dev)
# MAGIC
# MAGIC ```bash
# MAGIC cd 04_Lab_App
# MAGIC export DATABRICKS_CONFIG_PROFILE=gabriel-dev
# MAGIC export SERVING_ENDPOINT_NAME=fraude-reembolso-<seu-nome>
# MAGIC export LAKEBASE_INSTANCE=lakebase-<seu-nome>
# MAGIC export PG_SCHEMA=<seu_usuario>
# MAGIC export SCHEMA_UC=<seu_usuario>
# MAGIC export DATABRICKS_WAREHOUSE_ID=<id_do_warehouse>
# MAGIC pip install -r requirements.txt
# MAGIC streamlit run app.py
# MAGIC ```
# MAGIC
# MAGIC ---
# MAGIC ## Resumo do workshop
# MAGIC
# MAGIC | Lab | O que entregou |
# MAGIC |---|---|
# MAGIC | **01 — ML** | modelo governado no UC, threshold economico, fila em batch |
# MAGIC | **02 — Serving** | o mesmo modelo como endpoint REST, latencia medida |
# MAGIC | **03 — Lakebase** | features agregadas servidas por chave em ms |
# MAGIC | **04 — App** | os tres juntos decidindo **antes** do dinheiro sair |
# MAGIC
# MAGIC > 🧹 **Nao esqueca o `99_Cleanup`.** O serving endpoint, o app e a instancia
# MAGIC > Lakebase **cobram enquanto existirem**.
