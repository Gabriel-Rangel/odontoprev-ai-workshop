# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Hands-On Databricks — Odontoprev
# MAGIC ## Limpeza do Ambiente
# MAGIC
# MAGIC Remove **apenas os SEUS recursos**. O catalogo `workshop_databricks` e
# MAGIC compartilhado pela turma e **nao** e apagado.
# MAGIC
# MAGIC > ⚠️ **Rode isto no fim do workshop.** Tres recursos deste workshop
# MAGIC > **cobram enquanto existirem**, mesmo sem uso:
# MAGIC >
# MAGIC > | Recurso | Lab | Custo em idle | Removido aqui? |
# MAGIC > |---|---|---|---|
# MAGIC > | **Serving endpoint** | 02 | zero se `scale to zero` estiver ligado | ✅ |
# MAGIC > | **Instancia Lakebase** | 03 | **sim** — cobra parada | ✅ |
# MAGIC > | **Databricks App** | 04 | **sim** — cobra parado | ✅ |
# MAGIC > | **Knowledge Assistant** | 05 | **sim** — mantem endpoint provisionado | ❌ **so pela UI** |
# MAGIC > | Ferramentas + tabelas do Lab 06 | 06 | nao | ✅ (saem no `DROP SCHEMA`) |
# MAGIC > | App/endpoint exportado do Playground | 06 | **sim**, se exportou | ❌ **so pela UI** |
# MAGIC
# MAGIC > ⚠️ **O Knowledge Assistant do Lab 05 nao sai por este notebook.** O Agent Bricks
# MAGIC > nao expoe API publica de listagem (`GET /api/2.0/custom-llms` devolve
# MAGIC > `ENDPOINT_NOT_FOUND`), entao nao ha como encontra-lo por codigo. Apague a mao:
# MAGIC > **Menu lateral > Agents > seu agente > kebab (⋮) > Delete**.
# MAGIC >
# MAGIC > O Passo 7 no fim deste notebook confere se ficou algum endpoint `ka-*` no ar.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %pip install -q 'databricks-sdk>=0.59'

# COMMAND ----------

dbutils.library.restartPython()

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()
nome_hifen = nome.replace("_", "-")

CATALOGO = "workshop_databricks"
SCHEMA = nome
ENDPOINT = f"fraude-reembolso-{nome_hifen}"
INSTANCIA = f"lakebase-{nome_hifen}"
APP = f"fraude-sim-{nome_hifen}"

print("Serao removidos:")
print(f"  App:            {APP}")
print(f"  Endpoint:       {ENDPOINT}")
print(f"  Lakebase:       {INSTANCIA}")
print(f"  Schema (UC):    {CATALOGO}.{SCHEMA}")

# COMMAND ----------

from databricks.sdk import WorkspaceClient
from databricks.sdk.errors import NotFound, ResourceDoesNotExist

w = WorkspaceClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 1 — Databricks App (Lab 04)

# COMMAND ----------

try:
    w.apps.delete(name=APP)
    print(f"App removido: {APP}")
except (NotFound, ResourceDoesNotExist):
    print(f"App nao encontrado (ok): {APP}")
except Exception as e:  # noqa: BLE001
    print(f"Falha ao remover o app: {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 2 — Serving endpoint (Lab 02)

# COMMAND ----------

try:
    w.serving_endpoints.delete(name=ENDPOINT)
    print(f"Endpoint removido: {ENDPOINT}")
except (NotFound, ResourceDoesNotExist):
    print(f"Endpoint nao encontrado (ok): {ENDPOINT}")
except Exception as e:  # noqa: BLE001
    print(f"Falha ao remover o endpoint: {type(e).__name__}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 3 — Synced tables e instancia Lakebase (Lab 03)
# MAGIC
# MAGIC As synced tables saem primeiro; a instancia depois.

# COMMAND ----------

for tabela in ("online_features_prestador", "online_features_beneficiario"):
    nome_completo = f"{CATALOGO}.{SCHEMA}.{tabela}"
    try:
        w.database.delete_synced_database_table(name=nome_completo)
        print(f"Synced table removida: {tabela}")
    except (NotFound, ResourceDoesNotExist):
        print(f"Synced table nao encontrada (ok): {tabela}")
    except Exception as e:  # noqa: BLE001
        print(f"Falha em {tabela}: {type(e).__name__}: {e}")

# COMMAND ----------

# Remover a instancia derruba tambem as tabelas fisicas do Postgres —
# nao precisa limpar os "orfaos" antes.
try:
    # `purge=True` remove tambem os dados. Nao use `force`: a API de instancia
    # nao aceita esse parametro e devolve erro.
    w.database.delete_database_instance(name=INSTANCIA, purge=True)
    print(f"Instancia Lakebase removida: {INSTANCIA}")
except (NotFound, ResourceDoesNotExist):
    print(f"Instancia nao encontrada (ok): {INSTANCIA}")
except Exception as e:  # noqa: BLE001
    print(f"Falha ao remover a instancia: {type(e).__name__}: {e}")
    print(">>> Confira manualmente em Compute > Database instances — ela COBRA parada.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 4 — Modelos registrados no Unity Catalog
# MAGIC
# MAGIC Modelos precisam sair antes do `DROP SCHEMA`.

# COMMAND ----------

from mlflow import MlflowClient

client = MlflowClient()

try:
    modelos = spark.sql(f"SHOW MODELS IN {CATALOGO}.{SCHEMA}").collect()
except Exception as e:  # noqa: BLE001
    modelos = []
    print(f"Nenhum modelo encontrado ({e})")

for m in modelos:
    nome_completo = f"{CATALOGO}.{SCHEMA}.{m['name']}"
    try:
        client.delete_registered_model(nome_completo)
        print(f"Modelo removido: {nome_completo}")
    except Exception as e:  # noqa: BLE001
        print(f"Falha em {nome_completo}: {e}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 5 — Schema (tabelas + volumes)

# COMMAND ----------

spark.sql(f"DROP SCHEMA IF EXISTS {CATALOGO}.{SCHEMA} CASCADE")
print(f"Schema removido: {CATALOGO}.{SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 6 — Experimento MLflow (opcional)
# MAGIC
# MAGIC Fica no seu Workspace e nao consome storage do catalogo. Descomente se quiser.

# COMMAND ----------

# EXPERIMENT_PATH = f"/Users/{usuario}/odontoprev_workshop_fraude"
# exp = client.get_experiment_by_name(EXPERIMENT_PATH)
# if exp:
#     client.delete_experiment(exp.experiment_id)
#     print(f"Experimento removido: {EXPERIMENT_PATH}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 7 — Knowledge Assistant (Lab 05): conferir e apagar pela UI
# MAGIC
# MAGIC Nao da para apagar por codigo, mas da para **detectar**: todo KA mantem um serving
# MAGIC endpoint com o prefixo `ka-`.

# COMMAND ----------

endpoints_ka = [e for e in w.serving_endpoints.list() if e.name and e.name.startswith("ka-")]

if not endpoints_ka:
    print("Nenhum endpoint de Knowledge Assistant no ar. Nada a fazer.")
else:
    print(f"ATENCAO: {len(endpoints_ka)} endpoint(s) de Knowledge Assistant AINDA NO AR:\n")
    for e in endpoints_ka:
        print(f"  {e.name}")
    print("\n>>> Estes endpoints COBRAM enquanto existirem.")
    print(">>> Apague o agente na UI: Agents > seu agente > kebab (⋮) > Delete")
    print(">>> (apagar o endpoint direto em Serving NAO remove o agente)")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificacao final

# COMMAND ----------

print("=" * 60)
print("  LIMPEZA CONCLUIDA")
print("=" * 60)
print(f"  Removidos: app, endpoint, Lakebase, modelos e schema")
print(f"  Mantido:   catalogo {CATALOGO} (compartilhado)")
print(f"  Manual:    Knowledge Assistant (Lab 05) — apague em Agents")
print(f"             App exportado do Playground (Lab 06), se voce exportou")
print("=" * 60)
print()
print("Confirme na UI que nao sobrou nada cobrando:")
print("  - Compute > Apps")
print("  - Serving")
print("  - Compute > Database instances")
print("  - Agents            <- o KA do Lab 05")
