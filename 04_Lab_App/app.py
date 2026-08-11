"""Simulador de Fraude em Reembolsos — Databricks App (Streamlit).

Junta os tres labs anteriores num fluxo unico de inferencia em tempo real:

    [pedido novo]
        -> 13 features calculadas do proprio pedido        (features.py)
        -> 7 agregados buscados no Lakebase por chave      (db.py, Lab 03)
        -> vetor de 20 features na ordem da signature
        -> POST /serving-endpoints/<nome>/invocations       (Lab 02)
    [score + prioridade + explicacao + latencia]
"""
import os
import time
from datetime import date, timedelta

import pandas as pd
import plotly.graph_objects as go
import requests
import streamlit as st
from databricks.sdk.core import Config

import features as F

# ---------------------------------------------------------------------------
# Configuracao da pagina — precisa ser a primeira chamada do Streamlit
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Simulador de Fraude — Reembolsos",
    page_icon="🦷",
    layout="wide",
)

SERVING_ENDPOINT = os.getenv("SERVING_ENDPOINT_NAME", "")
USE_LAKEBASE = os.getenv("USE_LAKEBASE", "true").lower() == "true"
WAREHOUSE_ID = os.getenv("DATABRICKS_WAREHOUSE_ID", "")
CATALOGO = os.getenv("CATALOGO", "workshop_databricks")
SCHEMA = os.getenv("SCHEMA_UC", "")
THRESHOLD_PADRAO = float(os.getenv("THRESHOLD", "0.30"))

cfg = Config()


# ---------------------------------------------------------------------------
# Helpers de conexao
# ---------------------------------------------------------------------------
def _host() -> str:
    h = cfg.host or ""
    if h and not h.startswith("http"):
        h = f"https://{h}"
    return h.rstrip("/")


def _headers() -> dict:
    h = cfg.authenticate()
    h["Content-Type"] = "application/json"
    return h


# Um endpoint com `scale to zero` desliga quando fica sem trafego. A primeira
# chamada depois disso precisa subir o container — o famoso **cold start**, que
# pode levar minutos. Por isso o timeout aqui e generoso e ha uma tentativa extra.
TIMEOUT_INFERENCIA = int(os.getenv("TIMEOUT_INFERENCIA", "300"))


def chamar_endpoint(vetor: dict, tentativas: int = 2) -> tuple[float, float]:
    """Chama o Model Serving. Devolve (score, latencia_ms).

    Tolera cold start: se o endpoint estava escalado a zero, a 1a chamada pode
    demorar. Tentamos de novo antes de desistir.
    """
    if not SERVING_ENDPOINT:
        raise RuntimeError("SERVING_ENDPOINT_NAME nao configurado no app.yaml")

    url = f"{_host()}/serving-endpoints/{SERVING_ENDPOINT}/invocations"
    ultimo_erro: Exception | None = None

    for tentativa in range(tentativas):
        inicio = time.perf_counter()
        try:
            resp = requests.post(
                url, headers=_headers(),
                json={"dataframe_records": [vetor]}, timeout=TIMEOUT_INFERENCIA,
            )
            latencia_ms = (time.perf_counter() - inicio) * 1000
            resp.raise_for_status()
            pred = resp.json().get("predictions", [])
            return float(pred[0]), latencia_ms
        except (requests.exceptions.Timeout, requests.exceptions.HTTPError) as e:
            ultimo_erro = e
            if tentativa + 1 < tentativas:
                st.info(
                    "O endpoint estava **escalado a zero** e esta subindo (cold start). "
                    "Tentando novamente..."
                )

    raise RuntimeError(
        f"O endpoint nao respondeu em {TIMEOUT_INFERENCIA}s ({tentativas} tentativas). "
        "Provavelmente cold start. Confira em **Serving > seu endpoint**: se o estado "
        "for *Scaling from zero*, espere e tente de novo. Para eliminar o problema, "
        "desmarque **Scale to zero** no endpoint."
    ) from ultimo_erro


def sql_warehouse(query: str) -> pd.DataFrame:
    """Consulta via SQL Warehouse — usada no fallback e nas listas de apoio."""
    if not WAREHOUSE_ID:
        return pd.DataFrame()
    r = requests.post(
        f"{_host()}/api/2.0/sql/statements",
        headers=_headers(),
        json={"warehouse_id": WAREHOUSE_ID, "statement": query, "wait_timeout": "50s"},
        timeout=90,
    )
    r.raise_for_status()
    d = r.json()
    if d.get("status", {}).get("state") != "SUCCEEDED":
        return pd.DataFrame()
    cols = [c["name"] for c in d["manifest"]["schema"]["columns"]]
    return pd.DataFrame(d.get("result", {}).get("data_array") or [], columns=cols)


@st.cache_data(ttl=300)
def prestadores_suspeitos() -> pd.DataFrame:
    """Prestadores com mais beneficiarios distintos — bons para demonstrar."""
    return sql_warehouse(f"""
        SELECT prestador_id, prest_n_benef, prest_n_pedidos,
               ROUND(prest_valor_medio, 2) AS ticket_medio,
               ROUND(prest_pct_prox_teto, 3) AS pct_colado_teto
        FROM {CATALOGO}.{SCHEMA}.gold_features_prestador
        ORDER BY prest_n_benef DESC
        LIMIT 10
    """)


def buscar_agregados(prestador_id: int, beneficiario_id: int) -> tuple[dict, float, str]:
    """Busca os 7 agregados. Devolve (features, latencia_ms, origem)."""
    if USE_LAKEBASE:
        try:
            import db
            feats, ms = db.buscar_features_online(prestador_id, beneficiario_id)
            if feats:
                return feats, ms, "Lakebase (Postgres, por chave primaria)"
        except Exception as e:  # noqa: BLE001
            st.warning(f"Lakebase indisponivel ({type(e).__name__}) — usando SQL Warehouse.")

    # Fallback: le as mesmas tabelas Delta pelo warehouse (permite rodar sem o Lab 03)
    inicio = time.perf_counter()
    p = sql_warehouse(f"""
        SELECT prest_n_pedidos, prest_n_benef, prest_valor_medio, prest_pct_prox_teto
        FROM {CATALOGO}.{SCHEMA}.gold_features_prestador
        WHERE prestador_id = {int(prestador_id)}
    """)
    b = sql_warehouse(f"""
        SELECT benef_n_pedidos, benef_valor_total, benef_n_prestadores
        FROM {CATALOGO}.{SCHEMA}.gold_features_beneficiario
        WHERE beneficiario_id = {int(beneficiario_id)}
    """)
    ms = (time.perf_counter() - inicio) * 1000

    feats: dict = {}
    for frame in (p, b):
        if not frame.empty:
            feats.update({c: float(frame.iloc[0][c]) for c in frame.columns})
    return feats, ms, "SQL Warehouse (Delta) — fallback"


# ---------------------------------------------------------------------------
# Cabecalho
# ---------------------------------------------------------------------------
st.title("🦷 Simulador de Fraude em Reembolsos")
st.caption(
    "Um pedido de reembolso chega agora. O app calcula as features do pedido, "
    "busca o historico agregado no store online e chama o modelo — em tempo real."
)

if not SERVING_ENDPOINT:
    st.error(
        "**SERVING_ENDPOINT_NAME nao configurado.** Edite o `app.yaml` com o nome do "
        "endpoint criado no Lab 02."
    )
    st.stop()

col_a, col_b, col_c = st.columns(3)
col_a.metric("Endpoint", SERVING_ENDPOINT)
col_b.metric("Features do pedido", "13")
col_c.metric("Features do store online", "7")

st.divider()

# ---------------------------------------------------------------------------
# Formulario
# ---------------------------------------------------------------------------
esq, dir_ = st.columns([3, 2], gap="large")

with esq:
    st.subheader("1. Dados do pedido")

    preset_nome = st.selectbox(
        "Preset (para demonstrar rapido)",
        ["— personalizado —"] + list(F.PRESETS),
        help="Escolha um cenario pronto ou monte o seu.",
    )
    preset = F.PRESETS.get(preset_nome, {})
    if preset.get("ajuda"):
        st.info(preset["ajuda"])

    c1, c2 = st.columns(2)
    with c1:
        prestador_id = st.number_input(
            "ID do prestador", 1, 700,
            value=int(preset.get("prestador_id", 3)),
            help="Prestadores com muitos beneficiarios distintos sao os suspeitos "
                 "(veja a lista ao lado).",
        )
        codigo_tuss = st.selectbox(
            "Procedimento (TUSS)", list(F.TUSS),
            index=list(F.TUSS).index(preset.get("codigo_tuss", "81000212")),
            format_func=lambda c: f"{F.TUSS[c][0]} — ref. R$ {F.TUSS[c][1]:,.2f}",
        )
        valor_solicitado = st.number_input(
            "Valor solicitado (R$)", 0.0, 20000.0,
            value=float(preset.get("valor_solicitado", 210.0)), step=10.0,
        )
    with c2:
        beneficiario_id = st.number_input("ID do beneficiario", 1, 8000,
                                          value=int(preset.get("beneficiario_id", 1000)))
        dias_desde_adesao = st.number_input(
            "Dias desde a adesao ao plano", 0, 5000,
            value=int(preset.get("dias_desde_adesao", 900)),
            help="Abaixo de 120 dias liga a flag de carencia recente.",
        )
        pedidos_par_30d = st.number_input(
            "Pedidos deste par (benef+prestador) em 30 dias", 1, 20,
            value=int(preset.get("pedidos_par_30d", 1)),
            help="Valores altos indicam fracionamento.",
        )

    st.markdown("**Sinais de cruzamento** (o que a auditoria checaria manualmente)")
    s1, s2, s3, s4, s5 = st.columns(5)
    fora_da_rede = s1.checkbox("Fora da rede", value=bool(preset.get("fora_da_rede", True)))
    uf_divergente = s2.checkbox("UF divergente", value=bool(preset.get("uf_divergente", False)))
    hash_repetido = s3.checkbox("NF repetida", value=bool(preset.get("hash_repetido", False)))
    existe_na_rede = s4.checkbox("Ja pago na rede", value=bool(preset.get("existe_na_rede", False)))
    dente_ja_extraido = s5.checkbox("Dente extraido", value=bool(preset.get("dente_ja_extraido", False)))

    percentual_reembolso = st.slider("% de reembolso do plano", 0.5, 1.0, 0.70, 0.05)

    analisar = st.button("Analisar pedido", type="primary", use_container_width=True)

with dir_:
    st.subheader("Prestadores mais concentrados")
    st.caption(
        "Muitos beneficiarios distintos no mesmo prestador fora da rede e a assinatura "
        "de *reembolso assistido*. Use um destes IDs para ver o score subir."
    )
    tabela = prestadores_suspeitos()
    if not tabela.empty:
        st.dataframe(tabela, hide_index=True, use_container_width=True, height=330)
    else:
        st.caption("_(configure `DATABRICKS_WAREHOUSE_ID` para ver esta lista)_")

# ---------------------------------------------------------------------------
# Resultado
# ---------------------------------------------------------------------------
if analisar:
    hoje = date.today()
    do_pedido = F.features_do_pedido(
        valor_solicitado=valor_solicitado,
        codigo_tuss=codigo_tuss,
        data_procedimento=hoje - timedelta(days=3),
        data_solicitacao=hoje,
        data_adesao=hoje - timedelta(days=int(dias_desde_adesao) + 3),
        fora_da_rede=fora_da_rede,
        uf_divergente=uf_divergente,
        percentual_reembolso=percentual_reembolso,
        pedidos_par_30d=pedidos_par_30d,
        hash_repetido=hash_repetido,
        existe_na_rede=existe_na_rede,
        dente_ja_extraido=dente_ja_extraido,
    )

    # Passo 1 — buscar os 7 agregados (Lakebase ou fallback)
    with st.spinner(f"1/2 · Buscando historico do prestador {prestador_id} e do "
                    f"beneficiario {beneficiario_id}..."):
        do_lakebase, ms_lookup, origem = buscar_agregados(prestador_id, beneficiario_id)
        vetor = F.montar_vetor(do_pedido, do_lakebase)

    # Passo 2 — inferencia no endpoint
    with st.spinner(f"2/2 · Chamando o modelo ({SERVING_ENDPOINT})... "
                    "se o endpoint estava escalado a zero, isto pode levar alguns minutos."):
        try:
            score, ms_infer = chamar_endpoint(vetor)
        except Exception as e:  # noqa: BLE001
            st.error(f"**Falha ao chamar o endpoint.**\n\n{e}")
            with st.expander("O que foi calculado antes da falha (as 20 features)"):
                st.json(vetor)
            st.stop()

    prioridade, cor = F.faixa_prioridade(score, THRESHOLD_PADRAO)

    st.divider()
    st.subheader("2. Resultado")

    g1, g2 = st.columns([2, 3], gap="large")

    with g1:
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=score * 100,
            number={"suffix": "%", "font": {"size": 44}},
            title={"text": f"Risco de fraude<br><b style='color:{cor}'>{prioridade}</b>"},
            gauge={
                "axis": {"range": [0, 100]},
                "bar": {"color": cor},
                # Os cortes seguem as faixas de F.faixa_prioridade(). O `sorted`
                # evita faixa invertida quando o threshold homologado passa de 0,80.
                "steps": [
                    {"range": r, "color": c}
                    for r, c in zip(
                        [
                            [0, min(THRESHOLD_PADRAO * 50, 100)],
                            sorted([THRESHOLD_PADRAO * 50, THRESHOLD_PADRAO * 100]),
                            sorted([min(THRESHOLD_PADRAO * 100, 100), 80.0]),
                            [80, 100],
                        ],
                        ["#eafaf1", "#fcf3cf", "#fdebd0", "#f9ebea"],
                    )
                    if r[0] < r[1]
                ],
                "threshold": {
                    "line": {"color": "black", "width": 3},
                    "value": THRESHOLD_PADRAO * 100,
                },
            },
        ))
        fig.update_layout(height=300, margin=dict(t=70, b=10, l=20, r=20))
        st.plotly_chart(fig, use_container_width=True)

        st.caption(f"Linha preta = threshold homologado ({THRESHOLD_PADRAO:.2f})")

        m1, m2 = st.columns(2)
        m1.metric("Lookup de features", f"{ms_lookup:.0f} ms", help=origem)
        m2.metric("Inferencia", f"{ms_infer:.0f} ms", help="Model Serving")
        st.caption(f"Total: **{ms_lookup + ms_infer:.0f} ms** · origem dos agregados: {origem}")

    with g2:
        st.markdown("**Por que este pedido recebeu esse score**")

        if not do_lakebase:
            st.warning(
                f"Nenhum historico encontrado para prestador {prestador_id} / "
                f"beneficiario {beneficiario_id}. As 7 features agregadas foram "
                "preenchidas com -1 (mesmo tratamento do treino)."
            )

        linhas = []
        for c in F.FEATURES:
            v = vetor[c]
            origem_f = "Lakebase" if c in F.FEATURES_LAKEBASE else "pedido"
            linhas.append({
                "Feature": F.ROTULOS.get(c, c),
                "Valor": f"{v:,.4f}".rstrip("0").rstrip(".") if v % 1 else f"{int(v):,}",
                "Origem": origem_f,
            })
        df_expl = pd.DataFrame(linhas)

        st.dataframe(
            df_expl, hide_index=True, use_container_width=True, height=330,
            column_config={"Origem": st.column_config.TextColumn(width="small")},
        )

        alertas = []
        if vetor["razao_valor_ref"] > 3:
            alertas.append(
                f"Valor **{vetor['razao_valor_ref']:.1f}x** a referencia do procedimento"
            )
        if vetor["prox_teto"] == 1:
            alertas.append("Valor **colado no teto** de R$ 1.500 (analise automatica)")
        if vetor["prest_n_benef"] > 50:
            alertas.append(
                f"Prestador com **{int(vetor['prest_n_benef'])} beneficiarios distintos**"
            )
        if vetor["pedidos_par_30d"] >= 3:
            alertas.append(
                f"**{int(vetor['pedidos_par_30d'])} pedidos** do mesmo par em 30 dias"
            )
        if vetor["hash_repetido"] == 1:
            alertas.append("**Nota fiscal repetida**")
        if vetor["existe_na_rede"] == 1:
            alertas.append("Procedimento **tambem pago pela rede** (duplicidade)")
        if vetor["dente_ja_extraido"] == 1:
            alertas.append("Procedimento em **dente ja extraido**")
        if vetor["carencia_recente"] == 1:
            alertas.append("Uso nos **primeiros 120 dias** apos a adesao")

        if alertas:
            st.markdown("**Sinais encontrados:**")
            for a in alertas:
                st.markdown(f"- {a}")
        else:
            st.success("Nenhum sinal relevante — perfil compativel com pedido legitimo.")

    with st.expander("Ver o payload exato enviado ao endpoint"):
        st.json({"dataframe_records": [vetor]})
        st.caption(
            "A ordem das 20 colunas segue a signature do modelo. Trocar a ordem "
            "produziria um score errado — por isso `features.py` e a fonte unica."
        )
