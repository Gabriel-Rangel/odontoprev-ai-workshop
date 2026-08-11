# Databricks notebook source
# MAGIC %md
# MAGIC # Lab 06 — Agente de Busca de Dentistas
# MAGIC ## Parte 1 de 3 — Preparar os dados de geolocalizacao
# MAGIC
# MAGIC **Objetivo:** dar aos prestadores e beneficiarios do workshop **endereco,
# MAGIC cidade, bairro, latitude e longitude** — sem isso nao existe "dentista mais
# MAGIC proximo".
# MAGIC
# MAGIC ### Onde este lab se encaixa
# MAGIC
# MAGIC Os labs anteriores respondem perguntas da **operadora**:
# MAGIC
# MAGIC | Lab | Pergunta | Quem pergunta |
# MAGIC |---|---|---|
# MAGIC | 01–04 | "este pedido e suspeito?" | auditoria |
# MAGIC | 05 | "e a norma, o que diz?" | auditoria |
# MAGIC | **06** | **"onde tem um dentista pra mim?"** | **o beneficiario** |
# MAGIC
# MAGIC E o outro lado do mesmo problema de negocio. Repare no encadeamento:
# MAGIC
# MAGIC ```
# MAGIC   Lab 01: reembolso e o canal de maior risco de fraude
# MAGIC        |
# MAGIC        v
# MAGIC   Por que o beneficiario usa reembolso?  Muitas vezes porque NAO ACHOU
# MAGIC   um credenciado perto — nas UFs de rede rala (PA, AM, MT) isso e
# MAGIC   legitimo, e foi plantado nos dados de proposito.
# MAGIC        |
# MAGIC        v
# MAGIC   Lab 06: um agente que acha o credenciado reduz reembolso na origem.
# MAGIC           Nao e so conveniencia — e prevencao de sinistro.
# MAGIC ```
# MAGIC
# MAGIC ### O que este lab ensina (e nao e "chatbot")
# MAGIC
# MAGIC | Conceito | Por que importa |
# MAGIC |---|---|
# MAGIC | **Tool calling** | o LLM nao sabe onde tem dentista — ele **chama uma funcao** que sabe |
# MAGIC | **UC Function como ferramenta** | a ferramenta e um objeto governado: tem `GRANT`, lineage e versao |
# MAGIC | **O `COMMENT` e o contrato** | o LLM escolhe a ferramenta lendo a descricao. Comentario ruim = ferramenta nao usada |
# MAGIC | **Encadeamento de ferramentas** | CPF -> endereco/plano -> lat/long -> busca. O agente decide a ordem |
# MAGIC | **Guardrail no system prompt** | "nunca infira especialidade a partir de sintoma" — e requisito clinico, nao capricho |
# MAGIC | **O dado nao vaza para o LLM** | o filtro por plano acontece **no SQL**; o modelo nunca ve a carteira |
# MAGIC
# MAGIC ### As 3 partes
# MAGIC
# MAGIC | # | Notebook | O que faz |
# MAGIC |---|---|---|
# MAGIC | 1 | **este** | geo para prestadores e beneficiarios |
# MAGIC | 2 | `07b_criar_ferramentas` | as 3 UC Functions |
# MAGIC | 3 | `07c_agente_playground` | montar o agente na **UI** do Playground |
# MAGIC
# MAGIC > **Pre-requisito:** `00_Setup/00_configuracao_catalogo` e
# MAGIC > `00_Setup/01_dados_sinteticos`. Os Labs 01–05 **nao** sao necessarios.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuracao

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()

CATALOGO = "workshop_databricks"
SCHEMA = nome

spark.sql(f"USE CATALOG {CATALOGO}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Catalogo: {CATALOGO}")
print(f"Schema:   {SCHEMA}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 1 — Por que precisamos gerar as coordenadas
# MAGIC
# MAGIC O `silver_prestadores` do Setup tem `uf`, mas **nao tem endereco nem coordenada**:
# MAGIC
# MAGIC ```
# MAGIC prestador_id | nome_prestador | cro | especialidade | uf | credenciado | nota_avaliacao
# MAGIC ```
# MAGIC
# MAGIC "UF = SP" nao serve: SP tem 645 municipios. Para responder *"qual o mais
# MAGIC proximo"* precisamos de **latitude e longitude**.
# MAGIC
# MAGIC ### ⚠️ Onde o geocoding funciona — e onde nao
# MAGIC
# MAGIC O caminho intuitivo seria uma **UC Function em Python** chamando uma API de
# MAGIC geocoding (Nominatim/OpenStreetMap). Testamos os dois ambientes, com o **mesmo
# MAGIC codigo**, e o resultado e diferente:
# MAGIC
# MAGIC | Ambiente | Resultado medido |
# MAGIC |---|---|
# MAGIC | **Notebook serverless** (esta celula) | ✅ `200` — devolveu `lat=-23.5648865, lon=-46.6519180` |
# MAGIC | **Dentro de UC Function Python** | ❌ `TimeoutError: timed out` |
# MAGIC
# MAGIC Ou seja: **nao e o serverless que bloqueia** — o notebook tem saida para a
# MAGIC internet normalmente. O que nao tem egress e o **sandbox de execucao da UC
# MAGIC Function**.
# MAGIC
# MAGIC E um isolamento proposital: uma UC Function e um objeto que qualquer um com
# MAGIC `EXECUTE` pode chamar — inclusive um **agente autonomo**. Se ela pudesse abrir
# MAGIC conexao para qualquer host, seria um vetor de exfiltracao com dado governado
# MAGIC dentro. O bloqueio esta exatamente onde deveria estar.
# MAGIC
# MAGIC | Caminho | Funciona como ferramenta de agente? | Observacao |
# MAGIC |---|---|---|
# MAGIC | `requests` / `http.client` em UC Function | ❌ timeout | sandbox sem egress |
# MAGIC | `requests` em **notebook** | ✅ | mas notebook nao e ferramenta de agente |
# MAGIC | **Coordenada pre-calculada na tabela** | ✅ | **o que faremos** |
# MAGIC | `http_request()` + UC Connection HTTP | ⚠️ | caminho governado, com ressalvas — ver Apendice |
# MAGIC
# MAGIC > 💡 **Por que isso importa para o desenho do agente:** a ferramenta que o LLM
# MAGIC > chama roda no sandbox restrito. Enriquecimento que precisa de internet pertence
# MAGIC > ao **pipeline** (notebook/job), nao a ferramenta. E o mesmo principio do Lab 03:
# MAGIC > separar **calcular** de **servir**.
# MAGIC
# MAGIC ### O que faremos
# MAGIC
# MAGIC Geramos endereco e coordenada **na tabela**, a partir de um dicionario de
# MAGIC bairros reais por capital. Isso e mais realista do que parece: numa operadora,
# MAGIC a rede credenciada **ja tem** endereco e georreferenciamento no cadastro — o
# MAGIC geocoding acontece no cadastro, uma vez, nao a cada consulta.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Bairros reais por capital, com coordenadas
# MAGIC
# MAGIC Coordenadas aproximadas de bairros existentes — para a distancia entre eles ter
# MAGIC significado. Os **nomes de rua e numeros sao ficticios**, como todo o dataset.

# COMMAND ----------

# UF -> (cidade, [(bairro, lat, long), ...])
# Coordenadas aproximadas do centroide de cada bairro.
GEO = {
    "SP": ("Sao Paulo", [
        ("Bela Vista", -23.5614, -46.6558), ("Pinheiros", -23.5665, -46.6926),
        ("Moema", -23.6018, -46.6653), ("Tatuape", -23.5401, -46.5766),
        ("Santana", -23.5027, -46.6250), ("Itaim Bibi", -23.5859, -46.6795),
        ("Lapa", -23.5280, -46.7050), ("Ipiranga", -23.5928, -46.6100),
        ("Santo Amaro", -23.6543, -46.7069), ("Perdizes", -23.5375, -46.6800),
    ]),
    "RJ": ("Rio de Janeiro", [
        ("Copacabana", -22.9711, -43.1822), ("Tijuca", -22.9249, -43.2277),
        ("Barra da Tijuca", -23.0045, -43.3650), ("Botafogo", -22.9519, -43.1846),
        ("Meier", -22.9019, -43.2789), ("Ipanema", -22.9838, -43.2044),
    ]),
    "MG": ("Belo Horizonte", [
        ("Savassi", -19.9386, -43.9350), ("Pampulha", -19.8517, -43.9739),
        ("Barreiro", -19.9744, -44.0264), ("Centro", -19.9191, -43.9386),
        ("Buritis", -19.9722, -43.9917),
    ]),
    "PR": ("Curitiba", [
        ("Batel", -25.4408, -49.2908), ("Centro", -25.4290, -49.2671),
        ("Portao", -25.4708, -49.2942), ("Cabral", -25.4067, -49.2611),
    ]),
    "RS": ("Porto Alegre", [
        ("Moinhos de Vento", -30.0248, -51.2044), ("Centro Historico", -30.0330, -51.2300),
        ("Petropolis", -30.0439, -51.1858), ("Partenon", -30.0603, -51.1636),
    ]),
    "BA": ("Salvador", [
        ("Barra", -13.0100, -38.5322), ("Pituba", -12.9938, -38.4569),
        ("Itaigara", -12.9908, -38.4661), ("Federacao", -13.0011, -38.5081),
    ]),
    "SC": ("Florianopolis", [
        ("Centro", -27.5954, -48.5480), ("Trindade", -27.5869, -48.5222),
        ("Estreito", -27.5919, -48.5828),
    ]),
    "PE": ("Recife", [
        ("Boa Viagem", -8.1236, -34.9017), ("Espinheiro", -8.0389, -34.8964),
        ("Graças", -8.0453, -34.8975),
    ]),
    "GO": ("Goiania", [
        ("Setor Bueno", -16.7050, -49.2769), ("Setor Oeste", -16.6800, -49.2694),
        ("Setor Marista", -16.6975, -49.2664),
    ]),
    "CE": ("Fortaleza", [
        ("Aldeota", -3.7386, -38.4989), ("Meireles", -3.7275, -38.4931),
        ("Centro", -3.7275, -38.5267),
    ]),
    "DF": ("Brasilia", [
        ("Asa Sul", -15.8267, -47.9218), ("Asa Norte", -15.7642, -47.8822),
        ("Aguas Claras", -15.8344, -48.0292),
    ]),
    "ES": ("Vitoria", [
        ("Praia do Canto", -20.2989, -40.2925), ("Jardim da Penha", -20.2764, -40.2989),
    ]),
    # UFs de rede rala — poucos bairros de proposito (ver Passo 4)
    "PA": ("Belem", [("Nazare", -1.4497, -48.4783), ("Umarizal", -1.4408, -48.4867)]),
    "AM": ("Manaus", [("Adrianopolis", -3.1019, -60.0114)]),
    "MT": ("Cuiaba", [("Centro Norte", -15.5989, -56.0949)]),
}

RUAS = ["Rua das Acacias", "Avenida Central", "Rua Sete de Setembro", "Alameda dos Ipes",
        "Rua Marechal Deodoro", "Avenida Brasil", "Rua Sao Joao", "Travessa das Flores",
        "Rua Barao do Rio Branco", "Avenida Getulio Vargas", "Rua XV de Novembro",
        "Rua Dom Pedro II", "Avenida Rio Branco", "Rua da Consolacao"]

total_bairros = sum(len(v[1]) for v in GEO.values())
print(f"UFs mapeadas:  {len(GEO)}")
print(f"Bairros:       {total_bairros}")
print(f"Ruas ficticias: {len(RUAS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 2 — Enriquecer os prestadores
# MAGIC
# MAGIC Criamos `gold_rede_credenciada`: a tabela que o agente vai consultar.
# MAGIC
# MAGIC ### Duas decisoes de modelagem que valem discussao em sala
# MAGIC
# MAGIC **1. `planos_aceitos` como `ARRAY<STRING>`.** Um dentista aceita varios planos.
# MAGIC Poderia ser uma tabela de relacionamento, mas `array_contains` num array e mais
# MAGIC rapido para lookup — e este dado e **servido**, nao transacionado.
# MAGIC
# MAGIC **2. Coordenada com ruido.** Somamos um deslocamento aleatorio de ate ~1 km ao
# MAGIC centroide do bairro. Sem isso, todos os dentistas do mesmo bairro ficariam na
# MAGIC **mesma** coordenada e o "mais proximo" seria arbitrario — um empate de 0,00 km.

# COMMAND ----------

import random

import pandas as pd

SEED = 42  # mesma seed do Setup — reprodutivel
random.seed(SEED)

df_prest = spark.table(f"{CATALOGO}.{SCHEMA}.silver_prestadores").toPandas()
df_planos = spark.table(f"{CATALOGO}.{SCHEMA}.silver_planos").toPandas()

NOMES_PLANOS = df_planos.nome_plano.tolist()
print(f"Prestadores: {len(df_prest):,}")
print(f"Planos:      {len(NOMES_PLANOS)} -> {NOMES_PLANOS[:4]} ...")

# COMMAND ----------

# 1 grau de latitude ~ 111 km. 0.009 grau ~ 1 km.
RUIDO = 0.009

linhas = []
for r in df_prest.itertuples():
    cidade, bairros = GEO[r.uf]
    bairro, lat0, lon0 = random.choice(bairros)

    linhas.append({
        "prestador_id": int(r.prestador_id),
        "nome_prestador": r.nome_prestador,
        "cro": r.cro,
        "especialidade": r.especialidade,
        "credenciado": bool(r.credenciado),
        "nota_avaliacao": float(r.nota_avaliacao),
        "endereco": f"{random.choice(RUAS)}, {random.randint(10, 2500)}",
        "bairro": bairro,
        "cidade": cidade,
        "uf": r.uf,
        "latitude": round(lat0 + random.uniform(-RUIDO, RUIDO), 6),
        "longitude": round(lon0 + random.uniform(-RUIDO, RUIDO), 6),
        # Credenciado aceita 3-7 planos; nao credenciado aceita 0 (atende por reembolso)
        "planos_aceitos": (
            sorted(random.sample(NOMES_PLANOS, random.randint(3, 7)))
            if r.credenciado else []
        ),
        "aceita_urgencia": random.random() < 0.35,
        "telefone": f"({random.randint(11, 98)}) {random.randint(3000, 3999)}-{random.randint(1000, 9999)}",
    })

df_rede = pd.DataFrame(linhas)

(spark.createDataFrame(df_rede)
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOGO}.{SCHEMA}.gold_rede_credenciada"))

print(f"gold_rede_credenciada: {len(df_rede):,} prestadores")
print(f"  credenciados:   {int(df_rede.credenciado.sum()):,}")
print(f"  fora da rede:   {int((~df_rede.credenciado).sum()):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Enriquecer os beneficiarios
# MAGIC
# MAGIC O agente vai identificar o beneficiario pelo **CPF** e precisa do endereco dele
# MAGIC (de onde medir a distancia) e do **plano** (para filtrar quem aceita).

# COMMAND ----------

df_benef = spark.table(f"{CATALOGO}.{SCHEMA}.silver_beneficiarios").toPandas()
mapa_plano = dict(zip(df_planos.plano_id, df_planos.nome_plano))

linhas_b = []
for r in df_benef.itertuples():
    cidade, bairros = GEO[r.uf]
    bairro, lat0, lon0 = random.choice(bairros)
    linhas_b.append({
        "beneficiario_id": int(r.beneficiario_id),
        "nome": r.nome,
        "cpf": str(r.cpf),
        "plano": mapa_plano[r.plano_id],
        "segmento": r.segmento,
        "endereco": f"{random.choice(RUAS)}, {random.randint(10, 2500)}",
        "bairro": bairro,
        "cidade": cidade,
        "uf": r.uf,
        "latitude": round(lat0 + random.uniform(-RUIDO, RUIDO), 6),
        "longitude": round(lon0 + random.uniform(-RUIDO, RUIDO), 6),
        "status": r.status,
    })

df_bg = pd.DataFrame(linhas_b)

(spark.createDataFrame(df_bg)
 .write.mode("overwrite").option("overwriteSchema", "true")
 .saveAsTable(f"{CATALOGO}.{SCHEMA}.gold_beneficiarios_geo"))

print(f"gold_beneficiarios_geo: {len(df_bg):,} beneficiarios")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 3 — Um beneficiario conhecido para a demonstracao
# MAGIC
# MAGIC Em sala, ninguem quer decorar um CPF aleatorio de 11 digitos. Criamos **dois
# MAGIC beneficiarios de teste com CPF facil**, em enderecos escolhidos:
# MAGIC
# MAGIC | CPF | Quem | Onde | Plano | Serve para mostrar |
# MAGIC |---|---|---|---|---|
# MAGIC | `11111111111` | Ana (voce) | Bela Vista, Sao Paulo | Premium | caso ideal: rede densa |
# MAGIC | `22222222222` | Carlos | Adrianopolis, Manaus | Essencial | **rede rala** — o caso que gera reembolso |
# MAGIC
# MAGIC O segundo e o mais interessante do lab: e o beneficiario que **legitimamente**
# MAGIC vai pedir reembolso, porque nao ha credenciado perto. Foi plantado nos dados do
# MAGIC Setup (AM tem densidade de rede 0,50) e agora aparece na busca.

# COMMAND ----------

PLANO_DENSO = NOMES_PLANOS[0]
PLANO_RALO = NOMES_PLANOS[-1]

teste = pd.DataFrame([
    {
        "beneficiario_id": 900001, "nome": "Ana Beatriz Souza", "cpf": "11111111111",
        "plano": PLANO_DENSO, "segmento": "individual",
        "endereco": "Avenida Paulista, 1000", "bairro": "Bela Vista",
        "cidade": "Sao Paulo", "uf": "SP",
        "latitude": -23.5614, "longitude": -46.6558, "status": "ativo",
    },
    {
        "beneficiario_id": 900002, "nome": "Carlos Eduardo Lima", "cpf": "22222222222",
        "plano": PLANO_RALO, "segmento": "individual",
        "endereco": "Rua Recife, 250", "bairro": "Adrianopolis",
        "cidade": "Manaus", "uf": "AM",
        "latitude": -3.1019, "longitude": -60.0114, "status": "ativo",
    },
])

(spark.createDataFrame(teste)
 .write.mode("append").saveAsTable(f"{CATALOGO}.{SCHEMA}.gold_beneficiarios_geo"))

print("Beneficiarios de teste inseridos:\n")
print(f"  CPF 11111111111 -> Ana Beatriz Souza | {PLANO_DENSO:12s} | Bela Vista, Sao Paulo")
print(f"  CPF 22222222222 -> Carlos Eduardo Lima | {PLANO_RALO:12s} | Adrianopolis, Manaus")
print("\n👉 Anote estes dois CPFs: sao eles que voce vai usar no Playground (Parte 3).")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Garantir que ha credenciados para a Ana encontrar
# MAGIC
# MAGIC Os planos foram sorteados aleatoriamente entre os prestadores. Para o roteiro de
# MAGIC demonstracao nao depender de sorte, garantimos que **alguns credenciados em Sao
# MAGIC Paulo aceitam o plano da Ana**, em cada especialidade.
# MAGIC
# MAGIC > 💡 Isto e uma decisao consciente de material didatico: um lab que "as vezes nao
# MAGIC > acha nada" queima tempo de sala. Em producao voce **nao** faria isso.

# COMMAND ----------

from pyspark.sql import functions as F

esp_lista = sorted(df_rede.especialidade.unique().tolist())

alvo = []
for esp in esp_lista:
    cand = df_rede[
        (df_rede.credenciado)
        & (df_rede.uf == "SP")
        & (df_rede.especialidade == esp)
    ].prestador_id.head(4).tolist()
    alvo.extend(cand)

print(f"Especialidades: {len(esp_lista)}")
print(f"Prestadores que passam a aceitar '{PLANO_DENSO}': {len(alvo)}")

if alvo:
    spark.sql(f"""
        UPDATE {CATALOGO}.{SCHEMA}.gold_rede_credenciada
        SET planos_aceitos = array_union(planos_aceitos, array('{PLANO_DENSO}'))
        WHERE prestador_id IN ({','.join(str(p) for p in alvo)})
    """)
    print("OK — atualizado.")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 4 — Conferir e entender a assimetria da rede
# MAGIC
# MAGIC Esta e a celula que da sentido ao lab. Ela mostra **quantos credenciados existem
# MAGIC por UF** — e a desigualdade e o ponto.

# COMMAND ----------

spark.sql(f"""
    SELECT
      uf,
      cidade,
      COUNT(*)                                        AS prestadores,
      SUM(CASE WHEN credenciado THEN 1 ELSE 0 END)    AS credenciados,
      ROUND(100.0 * SUM(CASE WHEN credenciado THEN 1 ELSE 0 END) / COUNT(*), 1) AS pct_credenciado,
      COUNT(DISTINCT especialidade)                   AS especialidades
    FROM {CATALOGO}.{SCHEMA}.gold_rede_credenciada
    GROUP BY uf, cidade
    ORDER BY prestadores DESC
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### O que ler nessa tabela
# MAGIC
# MAGIC **SP tem centenas de prestadores. AM e MT tem punhados.** Nao e acidente: o
# MAGIC Setup do workshop define uma densidade de rede por UF, e as UFs do Norte/
# MAGIC Centro-Oeste receberam densidade baixa **de proposito** — para que o
# MAGIC beneficiario de la use reembolso **legitimamente**.
# MAGIC
# MAGIC No Lab 01 isso e um *hard negative*: o modelo de fraude precisa aprender que
# MAGIC "muito reembolso" em Manaus **nao** e suspeito.
# MAGIC
# MAGIC Aqui no Lab 06, a mesma assimetria vira **o caso de uso**:
# MAGIC
# MAGIC | Pergunta | Ana (SP) | Carlos (AM) |
# MAGIC |---|---|---|
# MAGIC | Acha credenciado perto? | sim, varios | provavelmente nao |
# MAGIC | Resultado esperado | lista de opcoes | **lista vazia** |
# MAGIC | O que o agente deve fazer | listar por distancia | explicar que nao ha rede e orientar sobre reembolso |
# MAGIC
# MAGIC > ⭐ **A busca que retorna vazio e a mais valiosa do lab.** Ela prova que a
# MAGIC > ferramenta esta filtrando de verdade — e conecta o agente ao problema de
# MAGIC > sinistro dos Labs 01–04. Um agente que "sempre acha alguem" esta mentindo.

# COMMAND ----------

# MAGIC %md
# MAGIC ### Amostra dos dados

# COMMAND ----------

spark.sql(f"""
    SELECT prestador_id, nome_prestador, especialidade, credenciado,
           endereco, bairro, cidade, uf, latitude, longitude,
           planos_aceitos, nota_avaliacao
    FROM {CATALOGO}.{SCHEMA}.gold_rede_credenciada
    WHERE credenciado AND uf = 'SP'
    ORDER BY nota_avaliacao DESC
    LIMIT 10
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC ### Validar a distancia antes de criar as ferramentas
# MAGIC
# MAGIC Antes de embalar a formula numa UC Function, confirmamos que ela devolve numero
# MAGIC plausivel. Bela Vista -> Pinheiros e ~5 km na vida real.

# COMMAND ----------

spark.sql("""
    SELECT ROUND(6371 * acos(
             cos(radians(-23.5614)) * cos(radians(-23.5665)) *
             cos(radians(-46.6926) - radians(-46.6558)) +
             sin(radians(-23.5614)) * sin(radians(-23.5665))
           ), 2) AS bela_vista_para_pinheiros_km
""").display()

# COMMAND ----------

# MAGIC %md
# MAGIC > ✅ Deu **~5,05 km**. A formula e a **haversine** (distancia em linha reta sobre
# MAGIC > a esfera), com raio da Terra de 6.371 km. E "distancia de passaro", nao de
# MAGIC > carro — suficiente para ordenar por proximidade, e e o que a maioria dos apps
# MAGIC > de busca de rede usa.
# MAGIC
# MAGIC ---
# MAGIC ## Resumo desta parte
# MAGIC
# MAGIC | Tabela | Linhas | Papel |
# MAGIC |---|---|---|
# MAGIC | `gold_rede_credenciada` | ~700 | quem o agente busca (com geo e planos) |
# MAGIC | `gold_beneficiarios_geo` | ~8.002 | quem pergunta (com CPF, endereco, plano) |
# MAGIC
# MAGIC | Aprendizado | Detalhe |
# MAGIC |---|---|
# MAGIC | **UC Function nao acessa internet** | geocoding externo da `TimeoutError` — isolamento proposital |
# MAGIC | **Geocode pertence ao cadastro** | calcula-se uma vez, nao a cada consulta |
# MAGIC | **Ruido na coordenada** | sem ele, todo dentista do bairro empata em 0,00 km |
# MAGIC | **A assimetria da rede e o caso de uso** | SP acha, AM nao — e isso conecta com fraude |
# MAGIC
# MAGIC > **Proximo:** `07b_criar_ferramentas` — transformar isso em 3 ferramentas que o
# MAGIC > LLM sabe chamar.
