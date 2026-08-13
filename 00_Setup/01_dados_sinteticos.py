# Databricks notebook source
# MAGIC %md
# MAGIC # Workshop Hands-On Databricks — Odontoprev
# MAGIC ## Notebook 01 — Geracao dos Dados Sinteticos
# MAGIC
# MAGIC **Objetivo:** simular o ambiente transacional de uma operadora odontologica —
# MAGIC beneficiarios, rede credenciada, procedimentos executados na rede e, o mais
# MAGIC importante para este workshop, os **pedidos de reembolso** (livre escolha).
# MAGIC
# MAGIC | Tabela | Registros | Simula |
# MAGIC |--------|-----------|--------|
# MAGIC | `silver_planos` | 12 | Cadastro de produtos |
# MAGIC | `silver_beneficiarios` | ~8.000 | Base de vidas (titulares) |
# MAGIC | `silver_prestadores` | ~700 | Dentistas/clinicas — credenciados e nao credenciados |
# MAGIC | `silver_procedimentos_rede` | ~45.000 | Procedimentos executados **dentro** da rede |
# MAGIC | `silver_pedidos_reembolso` | ~30.000 | **Tabela alvo do Lab de ML** — livre escolha |
# MAGIC
# MAGIC ### Por que reembolso?
# MAGIC No reembolso (livre escolha) o beneficiario paga o prestador e pede o dinheiro
# MAGIC de volta apresentando **nota fiscal**. A operadora nao controla quem executou,
# MAGIC nem o que foi executado — so ve o documento. Por isso e o canal de **maior risco
# MAGIC de fraude** em saude e odontologia, e o foco da auditoria.
# MAGIC
# MAGIC > Os dados sao **100% sinteticos**. Nomes, CPFs e CROs sao gerados aleatoriamente
# MAGIC > e nao correspondem a pessoas reais.
# MAGIC
# MAGIC ---

# COMMAND ----------

# MAGIC %md
# MAGIC ### Configuracao

# COMMAND ----------

# MAGIC %run ./00_configuracao_catalogo

# COMMAND ----------

usuario = spark.sql("SELECT current_user()").collect()[0][0]
nome = usuario.split("@")[0].replace(".", "_").replace("-", "_").lower()
CATALOGO = "workshop_databricks"
SCHEMA = nome
VOLUME_BASE = f"/Volumes/{CATALOGO}/{SCHEMA}/landing_reembolsos"

spark.sql(f"USE CATALOG {CATALOGO}")
spark.sql(f"USE SCHEMA {SCHEMA}")

print(f"Ambiente: {CATALOGO}.{SCHEMA}")
print(f"Landing:  {VOLUME_BASE}/")

# COMMAND ----------

import random
from datetime import date, timedelta

import pandas as pd

SEED = 42
random.seed(SEED)

# Janela temporal da simulacao: 24 meses fechados
DATA_FIM = date(2026, 6, 30)
DATA_INICIO = DATA_FIM - timedelta(days=730)


def data_aleatoria(inicio: date, fim: date) -> date:
    """Data uniformemente distribuida entre inicio e fim (inclusive).

    Se a janela for vazia ou invertida (acontece com quem aderiu no fim da
    simulacao), devolve `inicio` em vez de estourar.
    """
    dias = (fim - inicio).days
    if dias <= 0:
        return inicio
    return inicio + timedelta(days=random.randint(0, dias))


print(f"Janela simulada: {DATA_INICIO} -> {DATA_FIM}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 1: Planos e tabela de procedimentos (TUSS)
# MAGIC
# MAGIC O **teto de auditoria automatica** e a variavel de negocio mais importante aqui:
# MAGIC pedidos **acima de R$ 1.500** caem em analise manual. Guarde esse numero — varias
# MAGIC das fraudes do dataset foram desenhadas para passar **logo abaixo** dele.

# COMMAND ----------

TETO_AUDITORIA = 1500.00

planos = [
    (1, "Odonto Essencial", "individual", 39.90, 0.60),
    (2, "Odonto Essencial Familia", "individual", 89.90, 0.60),
    (3, "Odonto Plus", "individual", 69.90, 0.70),
    (4, "Odonto Plus Familia", "individual", 149.90, 0.70),
    (5, "Odonto Premium", "individual", 129.90, 0.80),
    (6, "Odonto Premium Familia", "individual", 259.90, 0.80),
    (7, "Empresarial PME Basico", "pme", 29.90, 0.60),
    (8, "Empresarial PME Completo", "pme", 54.90, 0.75),
    (9, "Corporativo Nacional", "corporativo", 24.90, 0.70),
    (10, "Corporativo Nacional Plus", "corporativo", 44.90, 0.80),
    (11, "Corporativo Executivo", "corporativo", 89.90, 0.90),
    (12, "Odonto Ortodontia", "individual", 179.90, 0.75),
]

df_planos = pd.DataFrame(
    planos,
    columns=["plano_id", "nome_plano", "segmento", "mensalidade", "percentual_reembolso"],
)

# codigo TUSS -> (descricao, especialidade, valor de referencia, exige_dente)
TUSS = {
    "81000103": ("Consulta odontologica inicial", "Clinica Geral", 90.0, False),
    "81000018": ("Radiografia panoramica", "Radiologia", 130.0, False),
    "81000212": ("Restauracao em resina composta", "Dentistica", 190.0, True),
    "81000476": ("Tratamento endodontico (canal)", "Endodontia", 680.0, True),
    "81000600": ("Exodontia (extracao) simples", "Cirurgia", 210.0, True),
    "82000167": ("Raspagem e alisamento radicular", "Periodontia", 320.0, False),
    "81000280": ("Coroa unitaria em ceramica", "Protese", 1250.0, True),
    "87000030": ("Protese total (dentadura)", "Protese", 1900.0, False),
    "85100048": ("Implante osseointegrado", "Implantodontia", 3400.0, True),
    "83000160": ("Manutencao de aparelho ortodontico", "Ortodontia", 260.0, False),
    "85200035": ("Clareamento dental (consultorio)", "Dentistica", 720.0, False),
    "81000090": ("Aplicacao de selante / profilaxia", "Odontopediatria", 110.0, True),
}

# Procedimentos "de rotina" — a maior parte do volume legitimo
TUSS_ROTINA = ["81000103", "81000018", "81000212", "81000090", "82000167", "81000600"]
# Procedimentos de alto valor — legitimamente caros (hard negatives)
TUSS_ALTO_VALOR = ["81000280", "87000030", "85100048", "85200035"]

print(f"Teto de auditoria automatica: R$ {TETO_AUDITORIA:,.2f}")
print(f"Procedimentos na tabela TUSS: {len(TUSS)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 2: Beneficiarios
# MAGIC
# MAGIC 8.000 titulares distribuidos por UF (concentrados em SP/RJ/MG, como a carteira
# MAGIC real de uma operadora nacional), com data de adesao e status.

# COMMAND ----------

NOMES = ["Ana", "Bruno", "Carla", "Daniel", "Eduarda", "Felipe", "Gabriela", "Henrique",
         "Isabela", "Joao", "Karina", "Lucas", "Mariana", "Nelson", "Olivia", "Paulo",
         "Queila", "Rafael", "Sabrina", "Thiago", "Ursula", "Vitor", "Wanda", "Yuri",
         "Beatriz", "Caio", "Debora", "Emerson", "Fernanda", "Gustavo", "Helena", "Igor",
         "Juliana", "Kaique", "Larissa", "Marcelo", "Natalia", "Otavio", "Priscila", "Renata"]
SOBRENOMES = ["Silva", "Santos", "Oliveira", "Souza", "Rodrigues", "Ferreira", "Alves",
              "Pereira", "Lima", "Gomes", "Costa", "Ribeiro", "Martins", "Carvalho",
              "Almeida", "Lopes", "Soares", "Fernandes", "Vieira", "Barbosa", "Rocha",
              "Dias", "Nascimento", "Andrade", "Moreira", "Nunes", "Marques", "Machado"]

# UF -> (peso na carteira, densidade da rede credenciada)
# Densidade baixa = beneficiario legitimamente usa mais reembolso (hard negative)
UFS = {
    "SP": (0.34, 0.95), "RJ": (0.14, 0.90), "MG": (0.11, 0.85), "PR": (0.07, 0.85),
    "RS": (0.07, 0.85), "BA": (0.06, 0.70), "SC": (0.05, 0.85), "PE": (0.04, 0.70),
    "GO": (0.03, 0.75), "CE": (0.03, 0.70), "DF": (0.03, 0.90), "ES": (0.02, 0.80),
    "PA": (0.01, 0.55), "AM": (0.005, 0.50), "MT": (0.005, 0.60),
}
uf_lista = list(UFS.keys())
uf_pesos = [UFS[u][0] for u in uf_lista]


def gerar_cpf() -> str:
    return f"{random.randint(100,999)}.{random.randint(100,999)}.{random.randint(100,999)}-{random.randint(10,99)}"


N_BENEFICIARIOS = 8000
beneficiarios = []

for bid in range(1, N_BENEFICIARIOS + 1):
    plano_id = random.choices(range(1, 13), weights=[8, 5, 10, 6, 7, 4, 12, 14, 15, 9, 5, 5])[0]
    segmento = df_planos.loc[df_planos.plano_id == plano_id, "segmento"].iloc[0]
    uf = random.choices(uf_lista, weights=uf_pesos)[0]

    # Adesao pode ser anterior a janela (carteira antiga) ou dentro dela
    if random.random() < 0.55:
        data_adesao = data_aleatoria(date(2019, 1, 1), DATA_INICIO)
    else:
        data_adesao = data_aleatoria(DATA_INICIO, DATA_FIM - timedelta(days=30))

    beneficiarios.append({
        "beneficiario_id": bid,
        "nome": f"{random.choice(NOMES)} {random.choice(SOBRENOMES)} {random.choice(SOBRENOMES)}",
        "cpf": gerar_cpf(),
        "data_nascimento": data_aleatoria(date(1950, 1, 1), date(2015, 12, 31)),
        "genero": random.choice(["F", "M"]),
        "uf": uf,
        "plano_id": plano_id,
        "segmento": segmento,
        "data_adesao": data_adesao,
        "status": "ativo",  # ajustado adiante para os cancelamentos
    })

df_beneficiarios = pd.DataFrame(beneficiarios)
print(f"Beneficiarios gerados: {len(df_beneficiarios):,}")
df_beneficiarios.head(5)

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 3: Prestadores (rede credenciada e fora da rede)
# MAGIC
# MAGIC Note a coluna `credenciado`: prestadores **nao credenciados** so aparecem via
# MAGIC reembolso. Entre eles plantamos algumas **clinicas de fachada** — mas o dataset
# MAGIC nao entrega quais sao. Descobrir isso e trabalho do modelo.

# COMMAND ----------

N_PRESTADORES = 700
ESPECIALIDADES = list({v[1] for v in TUSS.values()})

prestadores = []
for pid in range(1, N_PRESTADORES + 1):
    uf = random.choices(uf_lista, weights=uf_pesos)[0]
    credenciado = random.random() < 0.62  # 62% da rede e credenciada
    prestadores.append({
        "prestador_id": pid,
        "nome_prestador": f"Dr(a). {random.choice(NOMES)} {random.choice(SOBRENOMES)}",
        "cro": f"CRO-{uf} {random.randint(10000, 99999)}",
        "especialidade": random.choice(ESPECIALIDADES),
        "uf": uf,
        "credenciado": credenciado,
        "data_credenciamento": data_aleatoria(date(2015, 1, 1), DATA_FIM) if credenciado else None,
        "nota_avaliacao": round(random.uniform(3.0, 5.0), 1),
    })

df_prestadores = pd.DataFrame(prestadores)

credenciados = df_prestadores[df_prestadores.credenciado].prestador_id.tolist()
nao_credenciados = df_prestadores[~df_prestadores.credenciado].prestador_id.tolist()

# 12 clinicas de fachada, sorteadas entre os NAO credenciados
CLINICAS_FACHADA = random.sample(nao_credenciados, 12)

print(f"Prestadores: {len(df_prestadores):,}  "
      f"(credenciados: {len(credenciados):,} | fora da rede: {len(nao_credenciados):,})")
print(f"Clinicas de fachada plantadas (nao reveladas no dataset): {len(CLINICAS_FACHADA)}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 4: Procedimentos executados na rede
# MAGIC
# MAGIC Esta tabela existe por um motivo especifico: uma das fraudes e cobrar o **mesmo
# MAGIC procedimento duas vezes** — uma pela rede (a operadora paga o dentista) e outra
# MAGIC como reembolso (a operadora paga o beneficiario). Sem cruzar as duas tabelas,
# MAGIC essa fraude e invisivel.

# COMMAND ----------

DENTES = ([str(d) for d in range(11, 19)] + [str(d) for d in range(21, 29)]
          + [str(d) for d in range(31, 39)] + [str(d) for d in range(41, 49)])

proc_rede = []
proc_id = 1

for b in beneficiarios:
    bid = b["beneficiario_id"]
    inicio_uso = max(b["data_adesao"], DATA_INICIO)
    if inicio_uso >= DATA_FIM:
        continue

    densidade = UFS[b["uf"]][1]
    # Quem tem rede densa por perto usa mais a rede
    n_proc = max(0, int(random.gauss(6 * densidade, 3)))

    for _ in range(n_proc):
        codigo = random.choices(TUSS_ROTINA + TUSS_ALTO_VALOR,
                                weights=[22, 12, 26, 8, 10, 8, 4, 2, 2, 4])[0]
        desc, esp, valor_ref, exige_dente = TUSS[codigo]
        valor = round(valor_ref * random.uniform(0.85, 1.15), 2)
        proc_rede.append({
            "procedimento_id": proc_id,
            "beneficiario_id": bid,
            "prestador_id": random.choice(credenciados),
            "codigo_tuss": codigo,
            "descricao": desc,
            "dente": random.choice(DENTES) if exige_dente else None,
            "data_realizacao": data_aleatoria(inicio_uso, DATA_FIM),
            "valor_procedimento": valor,
        })
        proc_id += 1

df_proc_rede = pd.DataFrame(proc_rede)
print(f"Procedimentos na rede: {len(df_proc_rede):,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ---
# MAGIC ## Passo 5: Pedidos de reembolso — a tabela alvo
# MAGIC
# MAGIC Aqui esta o coracao do dataset. Geramos duas populacoes:
# MAGIC
# MAGIC **Legitima (~93%)** — inclui *hard negatives* de proposito:
# MAGIC - ortodontia, que legitimamente gera **muitos pedidos do mesmo prestador** por meses;
# MAGIC - protese e implante, que legitimamente custam **milhares de reais**;
# MAGIC - beneficiarios em UF com rede rala, que legitimamente usam **muito reembolso**.
# MAGIC
# MAGIC **Fraudulenta (~7%)** — 6 padroes que auditores de saude conhecem bem:
# MAGIC
# MAGIC | Persona | Padrao |
# MAGIC |---|---|
# MAGIC | Reembolso assistido | clinica de fachada concentra dezenas de beneficiarios, valores logo abaixo do teto |
# MAGIC | Fracionamento | um tratamento quebrado em 3-6 pedidos menores para escapar da auditoria |
# MAGIC | Duplicidade | mesmo procedimento cobrado na rede **e** como reembolso |
# MAGIC | Adesao oportunista | uso intenso nos primeiros 120 dias e cancelamento em seguida |
# MAGIC | Documento reciclado | mesma nota fiscal reapresentada (`hash_documento` repetido) |
# MAGIC | Implausibilidade clinica | procedimento em dente que ja foi extraido |
# MAGIC
# MAGIC > **Importante:** nenhuma dessas personas aparece como coluna. O sinal esta na
# MAGIC > **relacao entre linhas** — e por isso que o feature engineering do Lab 01 e
# MAGIC > o que decide o resultado do modelo.

# COMMAND ----------

pedidos = []
ped_id = 1


def novo_pedido(bid, prestador_id, codigo, data_proc, valor, dente=None,
                hash_doc=None, fraude=0, persona=None):
    """Cria um pedido de reembolso. `persona` fica fora da tabela final —
    e usada apenas para o gabarito do instrutor."""
    global ped_id
    # O beneficiario costuma pedir reembolso poucos dias depois do procedimento
    dias_ate_pedido = random.choices([1, 3, 7, 15, 30, 60], weights=[15, 25, 25, 20, 10, 5])[0]
    data_sol = min(data_proc + timedelta(days=dias_ate_pedido), DATA_FIM)
    p = {
        "pedido_id": ped_id,
        "beneficiario_id": bid,
        "prestador_id": prestador_id,
        "codigo_tuss": codigo,
        "descricao": TUSS[codigo][0],
        "dente": dente,
        "data_procedimento": data_proc,
        "data_solicitacao": data_sol,
        "valor_solicitado": round(valor, 2),
        "hash_documento": hash_doc or f"NF{random.randint(10**9, 10**10 - 1)}",
        "canal_solicitacao": random.choices(["app", "portal", "central_telefonica", "presencial"],
                                           weights=[55, 28, 12, 5])[0],
        "_fraude": fraude,
        "_persona": persona,
    }
    pedidos.append(p)
    ped_id += 1
    return p


# ---------------------------------------------------------------
# 5.1 Populacao LEGITIMA
# ---------------------------------------------------------------
beneficiarios_ortodontia = set(
    df_beneficiarios[df_beneficiarios.plano_id == 12].beneficiario_id.tolist()
)

for b in beneficiarios:
    bid = b["beneficiario_id"]
    inicio_uso = max(b["data_adesao"], DATA_INICIO)
    if inicio_uso >= DATA_FIM:
        continue

    densidade = UFS[b["uf"]][1]
    # Rede rala -> mais reembolso (legitimo)
    taxa_reembolso = 3.5 + (1 - densidade) * 10
    n_ped = max(0, int(random.gauss(taxa_reembolso, 2.2)))

    # Hard negative 1: ortodontia legitima = serie longa com o MESMO prestador
    if bid in beneficiarios_ortodontia and random.random() < 0.7:
        prestador_orto = random.choice(nao_credenciados)
        data_ini = data_aleatoria(inicio_uso, DATA_FIM - timedelta(days=200))
        n_manutencoes = random.randint(5, 11)
        for m in range(n_manutencoes):
            d = data_ini + timedelta(days=30 * m + random.randint(-4, 4))
            if d > DATA_FIM:
                break
            novo_pedido(bid, prestador_orto, "83000160", d,
                        TUSS["83000160"][2] * random.uniform(0.9, 1.1))

    for _ in range(n_ped):
        # Hard negative 2: alto valor legitimo (protese, implante, coroa, clareamento).
        # Peso alto de proposito: um quarto dos pedidos legitimos e caro, entao
        # "valor alto" isoladamente NAO e sinal de fraude — o modelo precisa de contexto.
        if random.random() < 0.26:
            codigo = random.choice(TUSS_ALTO_VALOR)
        else:
            codigo = random.choices(TUSS_ROTINA, weights=[24, 14, 28, 9, 13, 12])[0]
        desc, esp, valor_ref, exige_dente = TUSS[codigo]
        novo_pedido(
            bid,
            random.choice(nao_credenciados) if random.random() < 0.8 else random.choice(credenciados),
            codigo,
            data_aleatoria(inicio_uso, DATA_FIM),
            valor_ref * random.uniform(0.75, 1.30),
            dente=random.choice(DENTES) if exige_dente else None,
        )

# --- Casos LEGITIMOS que acionam as mesmas "flags" das fraudes ----------------
# Sem estes, features como `hash_repetido` viram preditor quase perfeito (precision
# ~0,99) e o modelo aprende um atalho: os scores colapsam em 0 ou 1 e o exercicio
# de threshold do Lab 01 perde sentido. No mundo real existem explicacoes inocentes
# para cada flag — e e isso que obriga o modelo a pesar o CONTEXTO.

# (a) Documento reapresentado de boa-fe: pedido negado por falta de dados e
#     reenviado com a mesma nota fiscal.
for _ in range(240):
    bid = random.randint(1, N_BENEFICIARIOS)
    b = beneficiarios[bid - 1]
    inicio_uso = max(b["data_adesao"], DATA_INICIO)
    if inicio_uso >= DATA_FIM - timedelta(days=30):
        continue
    prestador = random.choice(nao_credenciados)
    codigo = random.choices(TUSS_ROTINA, weights=[24, 14, 28, 9, 13, 12])[0]
    exige_dente = TUSS[codigo][3]
    dente = random.choice(DENTES) if exige_dente else None
    hash_doc = f"NF{random.randint(10**9, 10**10 - 1)}"
    valor = TUSS[codigo][2] * random.uniform(0.85, 1.15)
    data_proc = data_aleatoria(inicio_uso, DATA_FIM - timedelta(days=30))
    for k in range(2):  # original + reenvio
        novo_pedido(bid, prestador, codigo,
                    data_proc + timedelta(days=k * random.randint(20, 45)),
                    valor, dente=dente, hash_doc=hash_doc)

# (b) Tratamento longo legitimo: varias sessoes do mesmo procedimento no mesmo
#     prestador em poucas semanas (canal em multiplas sessoes, periodontia por
#     quadrante). Parece "fracionamento", mas nao e.
for _ in range(190):
    bid = random.randint(1, N_BENEFICIARIOS)
    b = beneficiarios[bid - 1]
    inicio_uso = max(b["data_adesao"], DATA_INICIO)
    if inicio_uso >= DATA_FIM - timedelta(days=25):
        continue
    prestador = random.choice(nao_credenciados)
    codigo = random.choice(["82000167", "81000476"])
    data_base = data_aleatoria(inicio_uso, DATA_FIM - timedelta(days=25))
    for sessao in range(random.randint(3, 4)):
        novo_pedido(bid, prestador, codigo,
                    data_base + timedelta(days=sessao * random.randint(5, 8)),
                    TUSS[codigo][2] * random.uniform(0.30, 0.45),
                    dente=random.choice(DENTES) if TUSS[codigo][3] else None)

# (c) Retratamento legitimo em dente com exodontia previa: o implante e a protese
#     ACONTECEM justamente onde o dente foi extraido. Sem estes casos,
#     `dente_ja_extraido` seria sinal automatico de fraude.
extracoes_leg = df_proc_rede[df_proc_rede.codigo_tuss == "81000600"]
if len(extracoes_leg) > 0:
    for _, r in extracoes_leg.sample(n=min(260, len(extracoes_leg)),
                                     random_state=SEED + 1).iterrows():
        data_depois = r.data_realizacao + timedelta(days=random.randint(60, 400))
        if data_depois > DATA_FIM:
            continue
        codigo = random.choice(["85100048", "81000280"])  # implante ou coroa
        novo_pedido(int(r.beneficiario_id), random.choice(nao_credenciados), codigo,
                    data_depois, TUSS[codigo][2] * random.uniform(0.85, 1.2),
                    dente=r.dente)

n_legitimos = len(pedidos)
print(f"Pedidos legitimos: {n_legitimos:,}")

# COMMAND ----------

# MAGIC %md
# MAGIC ### 5.2 Personas de fraude

# COMMAND ----------

# --- Persona A: Reembolso assistido (clinica de fachada) --------------------
# A clinica "ajuda" o beneficiario a pedir reembolso, emite nota inflada e fica
# com parte do valor. Assinatura: muitos beneficiarios distintos no mesmo
# prestador fora da rede + valores logo abaixo do teto de auditoria.
benef_fachada = random.sample(range(1, N_BENEFICIARIOS + 1), 330)
for bid in benef_fachada:
    b = beneficiarios[bid - 1]
    inicio_uso = max(b["data_adesao"], DATA_INICIO)
    if inicio_uso >= DATA_FIM:
        continue
    clinica = random.choice(CLINICAS_FACHADA)
    for _ in range(random.randint(1, 3)):
        codigo = random.choice(["81000212", "82000167", "81000476", "85200035"])
        exige_dente = TUSS[codigo][3]
        # Maioria "colada" no teto (mas abaixo); parte com valor banal, para que o
        # modelo nao consiga resolver a persona apenas olhando o valor.
        if random.random() < 0.65:
            valor = TETO_AUDITORIA * random.uniform(0.88, 0.985)
        else:
            valor = TUSS[codigo][2] * random.uniform(0.85, 1.2)
        novo_pedido(bid, clinica, codigo, data_aleatoria(inicio_uso, DATA_FIM), valor,
                    dente=random.choice(DENTES) if exige_dente else None,
                    fraude=1, persona="reembolso_assistido")

# --- Persona B: Fracionamento -----------------------------------------------
# Um tratamento caro e quebrado em varios pedidos menores, no mesmo prestador,
# em poucos dias. Assinatura: rajada de pedidos + soma alta + ticket baixo.
benef_fracionamento = random.sample(range(1, N_BENEFICIARIOS + 1), 110)
for bid in benef_fracionamento:
    b = beneficiarios[bid - 1]
    inicio_uso = max(b["data_adesao"], DATA_INICIO)
    if inicio_uso >= DATA_FIM - timedelta(days=20):
        continue
    prestador = random.choice(nao_credenciados)
    data_base = data_aleatoria(inicio_uso, DATA_FIM - timedelta(days=20))
    n_partes = random.randint(3, 6)
    for parte in range(n_partes):
        codigo = random.choice(["81000212", "81000476", "81000280"])
        exige_dente = TUSS[codigo][3]
        novo_pedido(bid, prestador, codigo,
                    data_base + timedelta(days=parte * random.randint(1, 5)),
                    random.uniform(380, 780),
                    dente=random.choice(DENTES) if exige_dente else None,
                    fraude=1, persona="fracionamento")

# --- Persona C: Duplicidade rede x reembolso --------------------------------
# O procedimento foi feito e pago pela rede; o beneficiario pede reembolso do
# mesmo procedimento. Assinatura: casamento (beneficiario, TUSS, dente, data).
amostra_rede = df_proc_rede.sample(n=300, random_state=SEED)
for _, r in amostra_rede.iterrows():
    novo_pedido(int(r.beneficiario_id), random.choice(nao_credenciados), r.codigo_tuss,
                r.data_realizacao + timedelta(days=random.randint(0, 5)),
                float(r.valor_procedimento) * random.uniform(1.0, 1.4),
                dente=r.dente, fraude=1, persona="duplicidade_rede")

# --- Persona D: Adesao oportunista ------------------------------------------
# Entra no plano, consome intensamente nos primeiros 120 dias (procedimentos
# caros) e cancela. Assinatura: alto valor concentrado logo apos a adesao.
candidatos_recentes = df_beneficiarios[
    df_beneficiarios.data_adesao > DATA_INICIO
].beneficiario_id.tolist()
benef_oportunista = random.sample(candidatos_recentes, min(95, len(candidatos_recentes)))
for bid in benef_oportunista:
    b = beneficiarios[bid - 1]
    for _ in range(random.randint(2, 5)):
        codigo = random.choice(TUSS_ALTO_VALOR + ["81000476", "81000280"])
        exige_dente = TUSS[codigo][3]
        dias = random.randint(5, 120)
        data_proc = b["data_adesao"] + timedelta(days=dias)
        if data_proc > DATA_FIM:
            continue
        novo_pedido(bid, random.choice(nao_credenciados), codigo, data_proc,
                    TUSS[codigo][2] * random.uniform(0.9, 1.3),
                    dente=random.choice(DENTES) if exige_dente else None,
                    fraude=1, persona="adesao_oportunista")
    # Cancela pouco depois
    df_beneficiarios.loc[df_beneficiarios.beneficiario_id == bid, "status"] = "cancelado"

# --- Persona E: Documento reciclado -----------------------------------------
# A mesma nota fiscal e reapresentada (as vezes por beneficiarios diferentes
# do mesmo grupo). Assinatura: hash_documento repetido.
for _ in range(110):
    hash_reciclado = f"NF{random.randint(10**9, 10**10 - 1)}"
    bid_base = random.randint(1, N_BENEFICIARIOS)
    prestador = random.choice(nao_credenciados)
    codigo = random.choice(["81000476", "81000280", "85200035", "82000167"])
    exige_dente = TUSS[codigo][3]
    valor = TUSS[codigo][2] * random.uniform(0.95, 1.15)
    data_base = data_aleatoria(DATA_INICIO, DATA_FIM - timedelta(days=60))
    for k in range(random.randint(2, 3)):
        bid = bid_base if random.random() < 0.6 else random.randint(1, N_BENEFICIARIOS)
        novo_pedido(bid, prestador, codigo,
                    data_base + timedelta(days=k * random.randint(10, 40)), valor,
                    dente=random.choice(DENTES) if exige_dente else None,
                    hash_doc=hash_reciclado, fraude=1, persona="documento_reciclado")

# --- Persona F: Implausibilidade clinica ------------------------------------
# Restauracao ou canal em dente que já foi extraido (exodontia anterior).
extracoes = df_proc_rede[df_proc_rede.codigo_tuss == "81000600"]
if len(extracoes) > 0:
    amostra_ext = extracoes.sample(n=min(190, len(extracoes)), random_state=SEED)
    for _, r in amostra_ext.iterrows():
        data_depois = r.data_realizacao + timedelta(days=random.randint(20, 300))
        if data_depois > DATA_FIM:
            continue
        codigo = random.choice(["81000212", "81000476"])
        novo_pedido(int(r.beneficiario_id), random.choice(nao_credenciados), codigo,
                    data_depois, TUSS[codigo][2] * random.uniform(0.9, 1.2),
                    dente=r.dente, fraude=1, persona="implausibilidade_clinica")

df_pedidos = pd.DataFrame(pedidos)
print(f"Total de pedidos: {len(df_pedidos):,}")
print(f"Fraudes:          {int(df_pedidos._fraude.sum()):,} "
      f"({df_pedidos._fraude.mean()*100:.2f}%)")
print()
print(df_pedidos[df_pedidos._fraude == 1]._persona.value_counts().to_string())

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 6: Desfecho da auditoria — e as colunas que causam vazamento
# MAGIC
# MAGIC Agora simulamos o que a operadora **registrou depois de analisar** cada pedido:
# MAGIC `status`, `valor_reembolsado`, `motivo_glosa`, `data_analise`.
# MAGIC
# MAGIC Essas colunas sao a realidade de qualquer tabela de sinistro — e sao exatamente
# MAGIC as que **nao podem entrar no modelo**, porque so existem depois da decisao que
# MAGIC estamos tentando prever. No Lab 01 voce vai ter que identifica-las.

# COMMAND ----------

status_list, valor_reemb, motivo, data_analise, fraude_conf = [], [], [], [], []

for p in pedidos:
    is_fraude = p["_fraude"] == 1
    # A auditoria nao e perfeita: pega ~72% das fraudes e nega ~4% dos legitimos
    if is_fraude:
        detectado = random.random() < 0.72
    else:
        detectado = random.random() < 0.04

    perc = df_planos.loc[
        df_planos.plano_id == df_beneficiarios.loc[
            df_beneficiarios.beneficiario_id == p["beneficiario_id"], "plano_id"
        ].iloc[0], "percentual_reembolso"
    ].iloc[0]

    if detectado:
        status_list.append("negado")
        valor_reemb.append(0.0)
        motivo.append(random.choice([
            "Documentacao insuficiente", "Divergencia na nota fiscal",
            "Procedimento nao coberto", "Indicio de irregularidade",
            "Duplicidade identificada",
        ]))
    else:
        status_list.append("aprovado")
        valor_reemb.append(round(p["valor_solicitado"] * float(perc), 2))
        motivo.append(None)

    data_analise.append(p["data_solicitacao"] + timedelta(days=random.randint(2, 25)))
    # Rotulo do Lab: fraude confirmada pela auditoria retrospectiva
    fraude_conf.append(p["_fraude"])

df_pedidos["status"] = status_list
df_pedidos["valor_reembolsado"] = valor_reemb
df_pedidos["motivo_glosa"] = motivo
df_pedidos["data_analise"] = data_analise
df_pedidos["fraude_confirmada"] = fraude_conf

# O gabarito de personas nao vai para a tabela publicada
df_gabarito = df_pedidos[["pedido_id", "_persona"]].rename(columns={"_persona": "persona_fraude"})
df_pedidos = df_pedidos.drop(columns=["_fraude", "_persona"])

print(f"Aprovados: {(df_pedidos.status == 'aprovado').sum():,}  |  "
      f"Negados: {(df_pedidos.status == 'negado').sum():,}")
print(f"Fraude confirmada: {df_pedidos.fraude_confirmada.mean()*100:.2f}%")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Passo 7: Gravar no Unity Catalog
# MAGIC
# MAGIC Gravamos as tabelas Delta (usadas pelo Lab) e tambem um CSV dos pedidos no
# MAGIC Volume, representando o arquivo que chegaria do sistema transacional.

# COMMAND ----------

TABELAS = {
    "silver_planos": df_planos,
    "silver_beneficiarios": df_beneficiarios,
    "silver_prestadores": df_prestadores,
    "silver_procedimentos_rede": df_proc_rede,
    "silver_pedidos_reembolso": df_pedidos,
    "_gabarito_personas_fraude": df_gabarito,
}

for tabela, pdf in TABELAS.items():
    (spark.createDataFrame(pdf)
        .write.mode("overwrite")
        .option("overwriteSchema", "true")
        .saveAsTable(f"{CATALOGO}.{SCHEMA}.{tabela}"))
    print(f"  {tabela:<32} {len(pdf):>8,} linhas")

# CSV na landing zone (formato de origem)
df_pedidos.to_csv(f"{VOLUME_BASE}/pedidos_reembolso.csv", index=False)
print(f"\nCSV gravado em {VOLUME_BASE}/pedidos_reembolso.csv")

# COMMAND ----------

# MAGIC %md
# MAGIC ### Verificacao final

# COMMAND ----------

resumo = spark.sql(f"""
    SELECT
      COUNT(*)                                             AS pedidos,
      SUM(fraude_confirmada)                               AS fraudes,
      ROUND(100.0 * AVG(fraude_confirmada), 2)             AS pct_fraude,
      ROUND(SUM(valor_solicitado), 2)                      AS valor_solicitado_total,
      ROUND(SUM(CASE WHEN fraude_confirmada = 1 THEN valor_solicitado END), 2) AS valor_em_risco,
      MIN(data_solicitacao)                                AS primeira_solicitacao,
      MAX(data_solicitacao)                                AS ultima_solicitacao
    FROM {CATALOGO}.{SCHEMA}.silver_pedidos_reembolso
""")
resumo.display()

# COMMAND ----------

print(f"{'='*64}")
print(f"  DADOS GERADOS COM SUCESSO")
print(f"{'='*64}")
print(f"  Schema: {CATALOGO}.{SCHEMA}")
print(f"  Tabela alvo do Lab: silver_pedidos_reembolso")
print(f"{'='*64}")
print()
print("Proximo passo: 01_Lab_ML_Fraude/02a_ml_to_do")
