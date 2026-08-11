# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 02 — Model Serving: do batch para o tempo real
# MAGIC ## Gabarito (solucao de referencia)
# MAGIC
# MAGIC **Objetivo:** publicar o modelo do Lab 01 como um **endpoint REST online** e
# MAGIC provar que ele responde em milissegundos — o que o batch nao consegue fazer.
# MAGIC
# MAGIC ### Por que isso importa neste caso de uso
# MAGIC
# MAGIC O Lab 01 terminou com `gold_fila_auditoria`: uma fila **recalculada de madrugada**.
# MAGIC Isso serve para o auditor comecar o dia priorizando. Mas a fraude de reembolso se
# MAGIC decide **no instante em que o pedido entra**:
# MAGIC
# MAGIC | | Batch (Lab 01) | Online (este lab) |
# MAGIC |---|---|---|
# MAGIC | Quando roda | 1x por dia | a cada pedido |
# MAGIC | Latencia | horas | milissegundos |
# MAGIC | Serve para | dar trabalho ao auditor | **barrar o pagamento antes de sair** |
# MAGIC
# MAGIC Se o score chega no dia seguinte, o dinheiro do reembolso ja foi transferido.
# MAGIC
# MAGIC ### O que vamos fazer
# MAGIC
# MAGIC | Passo | O que faz |
# MAGIC |-------|-----------|
# MAGIC | 1 | Resolver o alias `@champion` para o numero de versao |
# MAGIC | 2 | Criar o **serving endpoint** (pela UI) |
# MAGIC | 3 | Testar pela propria UI (aba **Query endpoint**) |
# MAGIC | 4 | Invocar por codigo, medir latencia e validar contra o batch |
# MAGIC
# MAGIC ## 🔧 O que personalizar neste lab
# MAGIC
# MAGIC | Valor | Onde | Sugestao |
# MAGIC |---|---|---|
# MAGIC | **Nome do endpoint** | Passo 2 (UI) + celulas de codigo | `fraude-reembolso-<seu_nome>` |
# MAGIC
# MAGIC Procure por **`👉 ALTERE`** nas celulas. O resto pode rodar como esta.
# MAGIC
# MAGIC > **Pre-requisito:** ter concluido o **Lab 01** (o modelo
# MAGIC > `modelo_fraude_reembolso` precisa estar registrado com o alias `@champion`).
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Dependencias

# COMMAND ----------

# MAGIC %pip install -q mlflow>=2.16 scikit-learn pandas 'databricks-sdk>=0.59'

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

MODEL_NAME = f"{CATALOGO}.{SCHEMA}.modelo_fraude_reembolso"

# 👉 ALTERE se quiser outro nome de endpoint.
# Endpoints nao aceitam "_" — usamos "-" no nome.
ENDPOINT = f"fraude-reembolso-{nome.replace('_', '-')}"

spark.sql(f"USE CATALOG {CATALOGO}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Modelo:   {MODEL_NAME}")
print(f"Endpoint: {ENDPOINT}")

# COMMAND ----------

import json
import time

# Guardamos builtins do Python: em notebooks Databricks e comum que
# `from pyspark.sql.functions import *` sombreie abs/max/round/sum.
from builtins import abs as pyabs, max as pymax, round as pyround

import mlflow
import pandas as pd
import requests
from mlflow import MlflowClient

mlflow.set_registry_uri("databricks-uc")
client = MlflowClient()

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 1 — Resolver o alias `@champion` para o numero da versao
# MAGIC
# MAGIC No Lab 01 usamos o alias `@champion` para carregar o modelo — e isso e a boa
# MAGIC pratica de governanca: trocar o modelo em producao passa a ser **mover um alias**.
# MAGIC
# MAGIC **Mas o Model Serving nao aceita alias.** O endpoint precisa de um **numero de
# MAGIC versao** explicito. No SDK, `ServedEntityInput` tem `entity_name` e
# MAGIC `entity_version` — **nao existe `entity_alias`**.
# MAGIC
# MAGIC Isso e uma decisao de design, nao uma limitacao: um endpoint que seguisse o alias
# MAGIC automaticamente trocaria o modelo em producao **sem ninguem aprovar o deploy**.
# MAGIC
# MAGIC > **Consequencia pratica:** ao promover um novo champion, o endpoint **nao** muda
# MAGIC > sozinho. Atualizar o endpoint e um passo separado e deliberado.
# MAGIC
# MAGIC ### **TO-DO 1** — Descubra a versao do champion e leia o threshold homologado
# MAGIC
# MAGIC Use `client.get_model_version_by_alias(...)`. Aproveite para ler a **tag
# MAGIC `threshold`** que gravamos no Lab 01 — o corte de negocio viaja junto do modelo.

# COMMAND ----------

versao_champion = client.get_model_version_by_alias(MODEL_NAME, "champion")
VERSAO = versao_champion.version
THRESHOLD = float(versao_champion.tags.get("threshold", 0.30))

print(f"Champion:  v{VERSAO}")
print(f"Threshold: {THRESHOLD}  (lido da tag da versao, nao hardcoded)")
print(f"PR AUC:    {versao_champion.tags.get('pr_auc', 'n/d')}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Confirme que o modelo tem *signature*
# MAGIC
# MAGIC A signature e o **contrato de entrada** do endpoint: nomes, ordem e tipos das 20
# MAGIC features. Sem ela o serving nao valida o payload — e uma ordem trocada de colunas
# MAGIC produziria um score errado **silenciosamente**.

# COMMAND ----------

info = mlflow.models.get_model_info(f"models:/{MODEL_NAME}/{VERSAO}")

if info.signature is None:
    print("ATENCAO: modelo sem signature — volte ao Lab 01 e logue com infer_signature().")
else:
    entradas = info.signature.inputs.input_names()
    print(f"Signature OK — {len(entradas)} features esperadas, nesta ordem:")
    for i, c in enumerate(entradas, 1):
        print(f"  {i:2d}. {c}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 2 (UI) — Criar o serving endpoint
# MAGIC
# MAGIC > ⏱️ **Faca isto AGORA e siga lendo.** O provisionamento leva **10 a 25 minutos**
# MAGIC > (sobe um container com o ambiente do modelo). Dispare a criacao e continue no
# MAGIC > Passo 3 enquanto o Databricks trabalha.
# MAGIC
# MAGIC 1. Menu lateral > **Serving** > **Create serving endpoint**.
# MAGIC 2. **Serving endpoint name** = `fraude-reembolso-<seu_nome>` &nbsp; 👉 **ALTERE**
# MAGIC    (o valor impresso na celula de configuracao acima)
# MAGIC 3. Em **Entity details**:
# MAGIC    - **Entity** > **Unity Catalog model**
# MAGIC    - **Model** = `workshop_databricks.<seu_nome>.modelo_fraude_reembolso` &nbsp; 👉 **ALTERE**
# MAGIC    - **Version** = a versao do **Passo 1** (ex.: `2`)
# MAGIC 4. **Compute scale-out** = **Small**
# MAGIC 5. Marque **Scale to zero** ✅
# MAGIC 6. **Create**.
# MAGIC 7. Acompanhe o estado até **Ready** (fica em *Not Ready / Updating* enquanto provisiona).
# MAGIC
# MAGIC ### Sobre o `Scale to zero`
# MAGIC
# MAGIC | | Scale to zero LIGADO | DESLIGADO |
# MAGIC |---|---|---|
# MAGIC | Custo em idle | **zero** | cobra 24/7 |
# MAGIC | Primeira chamada apos idle | **cold start de 40s a 5+ min** | ~1 s sempre |
# MAGIC | Bom para | workshop, dev, batch esporadico | **producao antifraude** |
# MAGIC
# MAGIC Ligamos no lab para nao queimar credito. Em producao, um cold start na hora
# MAGIC de aprovar um reembolso e inaceitavel — ali se paga por ter replica quente.
# MAGIC
# MAGIC > ⚠️ **Quanto dura o cold start, na pratica?** Medimos: **de 40 segundos a
# MAGIC > mais de 5 minutos**. Nao e "alguns milissegundos a mais" — o container sobe
# MAGIC > do zero, instala o ambiente do modelo e carrega os artefatos.
# MAGIC >
# MAGIC > Isso tem consequencia direta no **Lab 04**: se o app for demonstrado depois de
# MAGIC > um intervalo, o primeiro clique pode estourar o timeout. Duas saidas:
# MAGIC >
# MAGIC > 1. **Aqueca o endpoint** com uma chamada alguns minutos antes de apresentar; ou
# MAGIC > 2. **desmarque `Scale to zero`** (o que faremos antes do Lab 04) — passa a
# MAGIC >    custar por hora, mas responde sempre em ~1s.
# MAGIC >
# MAGIC > Essa e exatamente a decisao de arquitetura que um projeto real enfrenta.
# MAGIC
# MAGIC > 📄 Doc: https://docs.databricks.com/aws/en/machine-learning/model-serving/create-manage-serving-endpoints

# COMMAND ----------

# MAGIC %md
# MAGIC ### Opcao B (codigo) — criar o mesmo endpoint pelo SDK
# MAGIC
# MAGIC A UI e o caminho recomendado no workshop porque mostra cada campo. Em producao,
# MAGIC porem, isso vira **codigo versionado** (CI/CD). O bloco abaixo esta **comentado**
# MAGIC de proposito: use apenas se preferir criar por codigo em vez da UI.
# MAGIC
# MAGIC ```python
# MAGIC from databricks.sdk import WorkspaceClient
# MAGIC from databricks.sdk.service.serving import EndpointCoreConfigInput, ServedEntityInput
# MAGIC
# MAGIC w = WorkspaceClient()
# MAGIC w.serving_endpoints.create_and_wait(
# MAGIC     name=ENDPOINT,
# MAGIC     config=EndpointCoreConfigInput(
# MAGIC         served_entities=[
# MAGIC             ServedEntityInput(
# MAGIC                 entity_name=MODEL_NAME,
# MAGIC                 entity_version=VERSAO,        # numero, nunca alias
# MAGIC                 workload_size="Small",
# MAGIC                 scale_to_zero_enabled=True,
# MAGIC             )
# MAGIC         ]
# MAGIC     ),
# MAGIC     timeout=datetime.timedelta(minutes=40),
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC > Equivalente no CLI:
# MAGIC > ```bash
# MAGIC > databricks serving-endpoints create --json '{
# MAGIC >   "name": "fraude-reembolso-<seu_nome>",
# MAGIC >   "config": {"served_entities": [{
# MAGIC >     "entity_name": "workshop_databricks.<seu_nome>.modelo_fraude_reembolso",
# MAGIC >     "entity_version": "2", "workload_size": "Small", "scale_to_zero_enabled": true
# MAGIC >   }]}}' -p gabriel-dev
# MAGIC > ```

# COMMAND ----------

# MAGIC %md
# MAGIC ### Acompanhar a criacao daqui
# MAGIC
# MAGIC Rode a celula abaixo para esperar o endpoint ficar pronto. Ela consulta o estado
# MAGIC a cada 30 s — e o que a UI mostra na coluna **State**.

# COMMAND ----------

from databricks.sdk import WorkspaceClient

w = WorkspaceClient()

for tentativa in range(60):
    ep = w.serving_endpoints.get(ENDPOINT)
    pronto = str(ep.state.ready) if ep.state else "?"
    update = str(ep.state.config_update) if ep.state else "?"
    if "READY" in pronto.upper():
        print(f"\nEndpoint PRONTO ({tentativa * 30}s)")
        break
    if "FAILED" in update.upper():
        print(f"\nFALHA no update: {update}")
        print("Veja o motivo na UI: Serving > seu endpoint > aba **Events**.")
        print("Se a mensagem for 'did not become available in time', e timeout de")
        print("infraestrutura — basta tentar de novo (proxima celula).")
        break
    print(f"  [{tentativa * 30:4d}s] ready={pronto} | update={update}")
    time.sleep(30)

# COMMAND ----------

# MAGIC %md
# MAGIC ### 🔧 Se o update falhar com `DEPLOYMENT_ABORTED`
# MAGIC
# MAGIC Acontece: o Databricks desiste de provisionar o container depois de um tempo e
# MAGIC devolve *"served entity creation aborted because the resource did not become
# MAGIC available in time"*. **Nao e erro do seu modelo** — e capacidade de infraestrutura.
# MAGIC
# MAGIC Repare em dois detalhes importantes:
# MAGIC
# MAGIC - o endpoint continua **`READY`**, servindo a versao **anterior**. Um deploy que
# MAGIC   falha nao derruba o que ja estava no ar — e exatamente o comportamento que voce
# MAGIC   quer em producao;
# MAGIC - por isso `ready` e `config_update` sao **estados separados**.
# MAGIC
# MAGIC A correcao e simplesmente tentar de novo:
# MAGIC
# MAGIC ```python
# MAGIC w.serving_endpoints.update_config(
# MAGIC     name=ENDPOINT,
# MAGIC     served_entities=[ServedEntityInput(
# MAGIC         entity_name=MODEL_NAME, entity_version=VERSAO,
# MAGIC         workload_size="Small", scale_to_zero_enabled=True)],
# MAGIC )
# MAGIC ```
# MAGIC
# MAGIC > Na UI: **Serving > seu endpoint > Edit endpoint > Update**.

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 3 (UI) — Testar pela propria interface
# MAGIC
# MAGIC Antes de escrever codigo, teste na UI. E o caminho mais rapido para validar o
# MAGIC contrato de entrada.
# MAGIC
# MAGIC 1. Menu lateral > **Serving** > clique no seu endpoint.
# MAGIC 2. Botao **Query endpoint** (canto superior direito).
# MAGIC 3. Cole o JSON abaixo (a proxima celula gera um exemplo real da sua base) e **Send**.
# MAGIC 4. A resposta e `{"predictions": [<probabilidade de fraude>]}`.
# MAGIC
# MAGIC > Se der **erro 400**, quase sempre e o contrato: falta uma feature, o nome esta
# MAGIC > diferente, ou o tipo nao bate. A mensagem indica qual coluna causou.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Gerar um JSON de exemplo a partir de um pedido real
# MAGIC
# MAGIC Pegamos um pedido que o batch do Lab 01 marcou como **CRITICA** — assim podemos
# MAGIC comparar o score online com o score batch no Passo 4.

# COMMAND ----------

FEATURES = list(info.signature.inputs.input_names())

exemplo = (
    spark.table(f"{CATALOGO}.{SCHEMA}.gold_fila_auditoria")
    .filter("prioridade = 'CRITICA'")
    .select("pedido_id", "score_fraude", *FEATURES)
    .limit(1)
    .toPandas()
)

pedido_id = int(exemplo.pedido_id.iloc[0])
score_batch = float(exemplo.score_fraude.iloc[0])
payload = {"dataframe_records": exemplo[FEATURES].to_dict(orient="records")}

print(f"pedido_id  = {pedido_id}")
print(f"score batch = {score_batch:.6f}")
print("\nCole este JSON na aba Query endpoint:\n")
print(json.dumps(payload, indent=2, default=float))

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 4 — Invocar por codigo e medir a latencia
# MAGIC
# MAGIC ### **TO-DO 2** — Chame o endpoint e cronometre
# MAGIC
# MAGIC O contrato REST e:
# MAGIC
# MAGIC ```
# MAGIC POST {host}/serving-endpoints/{ENDPOINT}/invocations
# MAGIC body: {"dataframe_records": [ {feature: valor, ...} ]}
# MAGIC ```
# MAGIC
# MAGIC Meça **duas** chamadas seguidas e compare:
# MAGIC
# MAGIC - a **1ª** pode pegar o container frio (cold start) — segundos;
# MAGIC - a **2ª** pega o container quente — dezenas de milissegundos.
# MAGIC
# MAGIC E essa diferenca que o `Scale to zero` introduz.
# MAGIC
# MAGIC > **Dica:** autentique com o SDK — dentro do notebook o token sai de graca:
# MAGIC > ```python
# MAGIC > from databricks.sdk.core import Config
# MAGIC > cfg = Config()
# MAGIC > headers = cfg.authenticate()
# MAGIC > ```

# COMMAND ----------

from databricks.sdk.core import Config

cfg = Config()
host = cfg.host if cfg.host.startswith("http") else f"https://{cfg.host}"
url = f"{host.rstrip('/')}/serving-endpoints/{ENDPOINT}/invocations"


def scorar(payload: dict) -> tuple[float, float]:
    """Devolve (score, latencia_ms)."""
    headers = cfg.authenticate()
    headers["Content-Type"] = "application/json"
    inicio = time.perf_counter()
    resp = requests.post(url, headers=headers, json=payload, timeout=120)
    latencia_ms = (time.perf_counter() - inicio) * 1000
    resp.raise_for_status()
    pred = resp.json()["predictions"]
    return float(pred[0]), latencia_ms


score_1, ms_1 = scorar(payload)   # pode ser cold start
score_2, ms_2 = scorar(payload)   # container quente

print(f"1a chamada: {ms_1:8.1f} ms  (score {score_1:.6f})")
print(f"2a chamada: {ms_2:8.1f} ms  (score {score_2:.6f})")
print(f"\nDiferenca: {ms_1 - ms_2:.1f} ms — esse e o custo do cold start")

# COMMAND ----------

# MAGIC %md
# MAGIC ### **TO-DO 3** — Validar contra o batch (*training/serving skew*)
# MAGIC
# MAGIC O mesmo pedido, com as mesmas features, **tem que** produzir o mesmo score no
# MAGIC endpoint e no batch do Lab 01. Se divergir, ha *training/serving skew* — o bug mais
# MAGIC traicoeiro de ML em producao, porque nada quebra: o modelo so passa a errar.
# MAGIC
# MAGIC Compare `score_2` com `score_batch` e verifique se a diferenca e desprezivel.

# COMMAND ----------

diferenca = pyabs(score_2 - score_batch)

print(f"pedido_id      : {pedido_id}")
print(f"score batch    : {score_batch:.8f}")
print(f"score online   : {score_2:.8f}")
print(f"diferenca      : {diferenca:.2e}")
print()
if diferenca < 1e-6:
    print("OK — online e batch produzem o MESMO score. Sem skew.")
else:
    print("ATENCAO: divergencia. Confira se a ORDEM das features do payload")
    print("bate com a ordem da signature do modelo.")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Quanto do valor em risco o endpoint consegue barrar em tempo real?
# MAGIC
# MAGIC Vamos scorar um lote pequeno e traduzir a latencia em capacidade operacional.

# COMMAND ----------

lote = (
    spark.table(f"{CATALOGO}.{SCHEMA}.gold_fila_auditoria")
    .select("pedido_id", "score_fraude", "valor_solicitado", *FEATURES)
    .limit(50)
    .toPandas()
)

payload_lote = {"dataframe_records": lote[FEATURES].to_dict(orient="records")}

inicio = time.perf_counter()
headers = cfg.authenticate()
headers["Content-Type"] = "application/json"
r = requests.post(url, headers=headers, json=payload_lote, timeout=120)
r.raise_for_status()
ms_lote = (time.perf_counter() - inicio) * 1000

scores_online = r.json()["predictions"]
lote["score_online"] = scores_online
erro_max = (lote.score_online - lote.score_fraude).abs().max()

print(f"50 pedidos em {ms_lote:.0f} ms  ->  {ms_lote / 50:.1f} ms por pedido")
print(f"Erro maximo vs batch: {erro_max:.2e}")
print()
acima = lote[lote.score_online >= THRESHOLD]
print(f"Acima do threshold ({THRESHOLD}): {len(acima)} de {len(lote)} pedidos")
print(f"Valor que seria retido para analise: R$ {acima.valor_solicitado.sum():,.2f}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Resumo
# MAGIC
# MAGIC | Conceito | O que voce praticou |
# MAGIC |----------|--------------------|
# MAGIC | **Alias nao serve endpoint** | `@champion` e governanca; o endpoint exige numero de versao |
# MAGIC | **Deploy e deliberado** | promover champion **nao** atualiza o endpoint sozinho |
# MAGIC | **Signature** | contrato de entrada: nome, ordem e tipo das 20 features |
# MAGIC | **Scale to zero** | economia vs cold start — e por que producao antifraude nao usa |
# MAGIC | **Latencia** | cold vs warm medidos, nao supostos |
# MAGIC | **Training/serving skew** | mesmo pedido, mesmo score: online == batch |
# MAGIC | **Threshold viaja com o modelo** | lido da tag da versao, nunca hardcoded no app |
# MAGIC
# MAGIC ### O que ainda falta para um app real
# MAGIC
# MAGIC Repare no que fizemos no Passo 3: para montar o payload, **lemos as 20 features de
# MAGIC uma tabela que ja existia**. Um app de verdade nao tem isso — quando o pedido
# MAGIC chega, ele conhece o pedido, e nada mais.
# MAGIC
# MAGIC Sete daquelas features (`prest_*`, `benef_*`) sao **agregados de 2 anos de
# MAGIC historico** por prestador e por beneficiario. Nao da para calcular isso em 50 ms.
# MAGIC
# MAGIC > **Proximo:** Lab 03 — usar o **Lakebase** como store de baixa latencia para
# MAGIC > buscar essas features por chave primaria.
