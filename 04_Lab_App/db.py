"""Conexao com o Lakebase (Postgres) para o simulador de fraude.

Funciona nos dois cenarios:
  * **Databricks App**: o resource *Database* injeta `PGHOST`/`PGUSER` (e as vezes
    `PGPASSWORD`). Quando nao ha senha, usamos o token OAuth do service principal
    como senha do Postgres — e assim que o Lakebase autentica.
  * **Local / dev**: nao ha nada injetado, entao resolvemos o host pela REST API e
    geramos uma credencial temporaria.

Baseado no `db.py` do sac-ai-workshop (mesmo padrao, ja validado em producao).
"""
import os
import threading
import time
import uuid

import psycopg2
import requests
from databricks.sdk.core import Config
from psycopg2.extras import RealDictCursor

INSTANCE = os.getenv("LAKEBASE_INSTANCE", "lakebase-CHANGEME")
PG_DB = os.getenv("PGDATABASE", "databricks_postgres")
PG_SCHEMA = os.getenv("PG_SCHEMA", "")  # schema do UC espelhado no Postgres

_cfg = Config()
_BASE = _cfg.host.rstrip("/") if _cfg.host else ""
if _BASE and not _BASE.startswith("http"):
    _BASE = "https://" + _BASE

_lock = threading.Lock()
_state = {"token": None, "exp": 0.0, "host": None, "user": None}


def _auth() -> dict:
    h = _cfg.authenticate()
    h["Content-Type"] = "application/json"
    return h


def _rest(method: str, path: str, **kw):
    r = requests.request(method, f"{_BASE}{path}", headers=_auth(), timeout=30, **kw)
    r.raise_for_status()
    return r.json() if r.text else {}


def _bearer() -> str:
    """Token OAuth do app — usado como SENHA do Postgres no Lakebase."""
    return _cfg.authenticate()["Authorization"].split(" ", 1)[1]


def _resolve_host() -> str:
    if os.getenv("PGHOST"):
        return os.getenv("PGHOST")
    return _rest("GET", f"/api/2.0/database/instances/{INSTANCE}").get("read_write_dns")


def _resolve_user() -> str:
    u = os.getenv("PGUSER") or os.getenv("DATABRICKS_CLIENT_ID")
    if u:
        return u
    return _rest("GET", "/api/2.0/preview/scim/v2/Me").get("userName")


def _refresh() -> None:
    cred = _rest(
        "POST", "/api/2.0/database/credentials",
        json={"request_id": str(uuid.uuid4()), "instance_names": [INSTANCE]},
    )
    _state.update(
        token=cred["token"], exp=time.time() + 50 * 60,
        host=_resolve_host(), user=_resolve_user(),
    )


def get_conn():
    # 1) Dentro do Databricks App: host/user injetados pelo resource Database
    host = os.getenv("PGHOST")
    user = os.getenv("PGUSER") or os.getenv("DATABRICKS_CLIENT_ID")
    if host and user:
        pwd = os.getenv("PGPASSWORD") or _bearer()
        return psycopg2.connect(
            host=host, port=int(os.getenv("PGPORT", "5432")), dbname=PG_DB,
            user=user, password=pwd, sslmode="require",
            cursor_factory=RealDictCursor, connect_timeout=10,
        )

    # 2) Fallback: gera credencial temporaria via REST
    with _lock:
        if not _state["token"] or time.time() > _state["exp"]:
            _refresh()
    return psycopg2.connect(
        host=_state["host"], port=int(os.getenv("PGPORT", "5432")), dbname=PG_DB,
        user=_state["user"], password=_state["token"], sslmode="require",
        cursor_factory=RealDictCursor, connect_timeout=10,
    )


def query_one(sql: str, params: tuple | None = None) -> dict | None:
    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(sql, params or ())
            row = cur.fetchone()
            return dict(row) if row else None
    finally:
        conn.close()


def buscar_features_online(prestador_id: int, beneficiario_id: int) -> tuple[dict, float]:
    """Busca os 7 agregados no Lakebase e devolve (features, latencia_ms).

    E o coracao do Lab 03 aplicado: o app conhece o pedido, mas nao a historia
    do prestador nem do beneficiario. Essas 7 colunas vem daqui.
    """
    esquema = f'"{PG_SCHEMA}".' if PG_SCHEMA else ""
    inicio = time.perf_counter()

    conn = get_conn()
    try:
        with conn.cursor() as cur:
            cur.execute(
                f"SELECT prest_n_pedidos, prest_n_benef, prest_valor_medio, "
                f"prest_pct_prox_teto FROM {esquema}online_features_prestador "
                f"WHERE prestador_id = %s",
                (prestador_id,),
            )
            prest = cur.fetchone()

            cur.execute(
                f"SELECT benef_n_pedidos, benef_valor_total, benef_n_prestadores "
                f"FROM {esquema}online_features_beneficiario "
                f"WHERE beneficiario_id = %s",
                (beneficiario_id,),
            )
            benef = cur.fetchone()
    finally:
        conn.close()

    latencia_ms = (time.perf_counter() - inicio) * 1000

    feats: dict = {}
    feats.update(dict(prest) if prest else {})
    feats.update(dict(benef) if benef else {})
    return feats, latencia_ms
