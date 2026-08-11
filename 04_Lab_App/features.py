"""Montagem do vetor de features do simulador de fraude.

Divisao que estrutura o Lab 03 e este app:

  * 13 features **calculaveis do proprio pedido** (aqui, em memoria)
  *  7 features **agregadas por entidade** (vem do Lakebase — `db.py`)

A ordem final tem que ser EXATAMENTE a da signature do modelo. Por isso a lista
`FEATURES` e a unica fonte de verdade — nunca monte o dict "na mao" na UI.
"""
from datetime import date

# Ordem canonica — a mesma da signature do modelo registrado no Lab 01
FEATURES = [
    "valor_solicitado", "razao_valor_ref", "dias_desde_adesao", "carencia_recente",
    "fora_da_rede", "uf_divergente", "prox_teto", "dias_proc_solic",
    "prest_n_pedidos", "prest_n_benef", "prest_valor_medio", "prest_pct_prox_teto",
    "benef_n_pedidos", "benef_valor_total", "benef_n_prestadores", "pedidos_par_30d",
    "hash_repetido", "existe_na_rede", "dente_ja_extraido", "percentual_reembolso",
]

# As 7 que so existem no store online (usadas para destacar na UI)
FEATURES_LAKEBASE = [
    "prest_n_pedidos", "prest_n_benef", "prest_valor_medio", "prest_pct_prox_teto",
    "benef_n_pedidos", "benef_valor_total", "benef_n_prestadores",
]

TETO_AUDITORIA = 1500.00

# codigo TUSS -> (descricao, valor de referencia, exige dente)
TUSS = {
    "81000103": ("Consulta odontologica inicial", 90.0, False),
    "81000018": ("Radiografia panoramica", 130.0, False),
    "81000212": ("Restauracao em resina composta", 190.0, True),
    "81000476": ("Tratamento endodontico (canal)", 680.0, True),
    "81000600": ("Exodontia (extracao) simples", 210.0, True),
    "82000167": ("Raspagem e alisamento radicular", 320.0, False),
    "81000280": ("Coroa unitaria em ceramica", 1250.0, True),
    "87000030": ("Protese total (dentadura)", 1900.0, False),
    "85100048": ("Implante osseointegrado", 3400.0, True),
    "83000160": ("Manutencao de aparelho ortodontico", 260.0, False),
    "85200035": ("Clareamento dental (consultorio)", 720.0, False),
    "81000090": ("Aplicacao de selante / profilaxia", 110.0, True),
}

# Rotulos amigaveis para a explicacao do score
ROTULOS = {
    "valor_solicitado": "Valor solicitado",
    "razao_valor_ref": "Valor / referencia do procedimento",
    "dias_desde_adesao": "Dias desde a adesao ao plano",
    "carencia_recente": "Uso nos primeiros 120 dias",
    "fora_da_rede": "Prestador fora da rede",
    "uf_divergente": "UF do prestador difere da do beneficiario",
    "prox_teto": "Valor colado no teto de auditoria",
    "dias_proc_solic": "Dias entre procedimento e solicitacao",
    "prest_n_pedidos": "Pedidos totais do prestador",
    "prest_n_benef": "Beneficiarios distintos no prestador",
    "prest_valor_medio": "Ticket medio do prestador",
    "prest_pct_prox_teto": "% dos pedidos do prestador colados no teto",
    "benef_n_pedidos": "Pedidos totais do beneficiario",
    "benef_valor_total": "Valor total pedido pelo beneficiario",
    "benef_n_prestadores": "Prestadores distintos usados pelo beneficiario",
    "pedidos_par_30d": "Pedidos deste par (benef+prestador) em 30 dias",
    "hash_repetido": "Nota fiscal repetida",
    "existe_na_rede": "Mesmo procedimento tambem pago pela rede",
    "dente_ja_extraido": "Procedimento em dente ja extraido",
    "percentual_reembolso": "% de reembolso do plano",
}


def features_do_pedido(
    *,
    valor_solicitado: float,
    codigo_tuss: str,
    data_procedimento: date,
    data_solicitacao: date,
    data_adesao: date,
    fora_da_rede: bool,
    uf_divergente: bool,
    percentual_reembolso: float,
    pedidos_par_30d: int = 1,
    hash_repetido: bool = False,
    existe_na_rede: bool = False,
    dente_ja_extraido: bool = False,
) -> dict:
    """As 13 features que dao para calcular no momento em que o pedido chega."""
    valor_ref = TUSS[codigo_tuss][1]
    dias_desde_adesao = (data_procedimento - data_adesao).days
    return {
        "valor_solicitado": float(valor_solicitado),
        "razao_valor_ref": round(float(valor_solicitado) / valor_ref, 4),
        "dias_desde_adesao": int(dias_desde_adesao),
        "carencia_recente": int(dias_desde_adesao <= 120),
        "fora_da_rede": int(fora_da_rede),
        "uf_divergente": int(uf_divergente),
        "prox_teto": int(TETO_AUDITORIA * 0.85 < valor_solicitado <= TETO_AUDITORIA),
        "dias_proc_solic": int((data_solicitacao - data_procedimento).days),
        "pedidos_par_30d": int(pedidos_par_30d),
        "hash_repetido": int(hash_repetido),
        "existe_na_rede": int(existe_na_rede),
        "dente_ja_extraido": int(dente_ja_extraido),
        "percentual_reembolso": float(percentual_reembolso),
    }


def montar_vetor(do_pedido: dict, do_lakebase: dict) -> dict:
    """Junta as duas metades na ORDEM da signature.

    Falta de feature vira -1 (o mesmo `fillna(-1)` usado no treino do Lab 01):
    um prestador novo, sem historico, e um caso legitimo — nao um erro.
    """
    completo = {**do_pedido, **do_lakebase}
    return {c: float(completo.get(c, -1)) for c in FEATURES}


def faixa_prioridade(score: float, threshold: float) -> tuple[str, str]:
    """Devolve (prioridade, cor). O threshold vem da tag do modelo, nao hardcoded."""
    if score >= 0.80:
        return "CRITICA", "#c0392b"
    if score >= threshold:
        return "ALTA", "#e67e22"
    if score >= threshold * 0.5:
        return "MEDIA", "#f1c40f"
    return "BAIXA", "#27ae60"


# ---------------------------------------------------------------------------
# Presets para demonstrar em sala sem digitar
# ---------------------------------------------------------------------------
PRESETS = {
    "Pedido rotineiro": dict(
        prestador_id=3, beneficiario_id=1000,
        valor_solicitado=210.0, codigo_tuss="81000212", fora_da_rede=True,
        uf_divergente=False, dias_desde_adesao=900, pedidos_par_30d=1,
        hash_repetido=False, existe_na_rede=False, dente_ja_extraido=False,
        ajuda="Restauracao de valor normal, prestador pequeno, beneficiario antigo. "
              "Deve dar score BAIXO.",
    ),
    "Valor colado no teto": dict(
        prestador_id=182, beneficiario_id=1000,
        valor_solicitado=1450.0, codigo_tuss="81000212", fora_da_rede=True,
        uf_divergente=True, dias_desde_adesao=800, pedidos_par_30d=1,
        hash_repetido=False, existe_na_rede=False, dente_ja_extraido=False,
        ajuda="R$ 1.450 numa restauracao de R$ 190 — 7x a referencia, e logo ABAIXO do "
              "teto de R$ 1.500. A regra do teto nao pegaria este pedido.",
    ),
    "Clinica de fachada": dict(
        prestador_id=2, beneficiario_id=1000,
        valor_solicitado=1430.0, codigo_tuss="82000167", fora_da_rede=True,
        uf_divergente=True, dias_desde_adesao=650, pedidos_par_30d=1,
        hash_repetido=False, existe_na_rede=False, dente_ja_extraido=False,
        ajuda="Prestador 2 tem 153 beneficiarios distintos e 22% dos pedidos colados no "
              "teto. O sinal vem do LAKEBASE, nao do pedido.",
    ),
    "Fracionamento": dict(
        prestador_id=175, beneficiario_id=1000,
        valor_solicitado=580.0, codigo_tuss="81000476", fora_da_rede=True,
        uf_divergente=False, dias_desde_adesao=700, pedidos_par_30d=5,
        hash_repetido=False, existe_na_rede=False, dente_ja_extraido=False,
        ajuda="5 pedidos do mesmo par em 30 dias — tratamento quebrado para escapar do "
              "teto, num prestador concentrado.",
    ),
    "Dente ja extraido": dict(
        prestador_id=349, beneficiario_id=1000,
        valor_solicitado=640.0, codigo_tuss="81000476", fora_da_rede=True,
        uf_divergente=False, dias_desde_adesao=1000, pedidos_par_30d=1,
        hash_repetido=False, existe_na_rede=False, dente_ja_extraido=True,
        ajuda="Canal em dente com exodontia anterior: impossibilidade clinica. "
              "Repare que, sozinha, esta flag NAO garante score alto — existem "
              "implantes legitimos no lugar do dente extraido.",
    ),
    "Adesao oportunista": dict(
        prestador_id=301, beneficiario_id=1000,
        valor_solicitado=3600.0, codigo_tuss="85100048", fora_da_rede=True,
        uf_divergente=True, dias_desde_adesao=45, pedidos_par_30d=2,
        hash_repetido=False, existe_na_rede=False, dente_ja_extraido=False,
        ajuda="Implante caro 45 dias apos entrar no plano. Caso genuinamente ambiguo — "
              "um beneficiario novo PODE precisar de tratamento caro.",
    ),
    "Nota fiscal repetida": dict(
        prestador_id=35, beneficiario_id=1000,
        valor_solicitado=700.0, codigo_tuss="81000476", fora_da_rede=True,
        uf_divergente=False, dias_desde_adesao=850, pedidos_par_30d=1,
        hash_repetido=True, existe_na_rede=False, dente_ja_extraido=False,
        ajuda="Mesma nota fiscal reapresentada. Tambem tem explicacao inocente "
              "(reenvio depois de glosa) — por isso o modelo pesa o contexto.",
    ),
}
