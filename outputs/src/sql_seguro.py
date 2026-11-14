"""Camada SEGURA de acesso a dados para o text-to-query (sem LangChain; testavel offline).

Defesas em camadas contra SQL gerado por LLM:
  1. validar_sql: so UM comando SELECT/WITH, sem comentarios, sem palavras de escrita, so tabelas permitidas
  2. banco real aberto em modo SOMENTE LEITURA e COPIADO para a memoria apenas com as linhas DO CLIENTE
     (isolamento por linha: mesmo um SQL "malicioso" nao enxerga dados de outros clientes)
  3. authorizer do SQLite na copia: so leitura das tabelas permitidas
  4. limite de linhas devolvidas
"""
import re
import sqlite3
from pathlib import Path
from typing import Tuple

ROOT = Path(__file__).resolve().parent.parent
DB = ROOT / "data" / "loja.db"
TABELAS_PERMITIDAS = {"pedidos", "itens_pedido"}
_PROIBIDAS = re.compile(
    r"\b(insert|update|delete|drop|alter|create|replace|attach|detach|pragma|vacuum|reindex|truncate|begin|commit|rollback)\b",
    re.IGNORECASE,
)

SCHEMA_VISIVEL = """pedidos(id INTEGER, cliente_id INTEGER, data_pedido TEXT 'AAAA-MM-DD', status TEXT, valor_total REAL, data_entrega TEXT, codigo_rastreio TEXT)
  status possiveis: aguardando_pagamento, em_separacao, enviado, entregue, cancelado, devolvido
itens_pedido(id INTEGER, pedido_id INTEGER, produto TEXT, quantidade INTEGER, preco_unit REAL)"""

_SCHEMA_COPIA = """
CREATE TABLE pedidos (id INTEGER PRIMARY KEY, cliente_id INTEGER, data_pedido TEXT, status TEXT,
                      valor_total REAL, data_entrega TEXT, codigo_rastreio TEXT);
CREATE TABLE itens_pedido (id INTEGER PRIMARY KEY, pedido_id INTEGER, produto TEXT, quantidade INTEGER, preco_unit REAL);
"""


def limpar_sql(texto: str) -> str:
    """Remove cercas de markdown e espacos; o LLM as vezes responde ```sql ... ```."""
    t = re.sub(r"```(?:sql)?", "", texto or "", flags=re.IGNORECASE).strip()
    return t.strip()


def validar_sql(sql: str) -> Tuple[bool, str]:
    """Devolve (True, sql_limpo) ou (False, motivo)."""
    s = (sql or "").strip().rstrip(";").strip()
    if not s:
        return False, "consulta vazia"
    if ";" in s:
        return False, "somente um comando por consulta"
    if "--" in s or "/*" in s:
        return False, "comentarios nao sao permitidos"
    if not re.match(r"^(select|with)\b", s, re.IGNORECASE):
        return False, "somente consultas SELECT sao permitidas"
    if _PROIBIDAS.search(s):
        return False, "a consulta contem uma operacao proibida"
    ctes = {n.lower() for n in re.findall(r"(?:\bwith|,)\s*([a-zA-Z_]\w*)\s+as\s*\(", s, re.IGNORECASE)}
    for t in re.findall(r"\b(?:from|join)\s+([a-zA-Z_]\w*)", s, re.IGNORECASE):
        if t.lower() not in TABELAS_PERMITIDAS and t.lower() not in ctes:
            return False, f"tabela nao permitida: {t}"
    return True, s


def _abrir_somente_leitura(db: Path) -> sqlite3.Connection:
    return sqlite3.connect(f"{Path(db).resolve().as_uri()}?mode=ro", uri=True)


def copia_isolada(cliente_id: int, db: Path = DB) -> sqlite3.Connection:
    """Copia em memoria com SO os pedidos (e itens) do cliente. A copia e protegida por authorizer."""
    origem = _abrir_somente_leitura(db)
    try:
        mem = sqlite3.connect(":memory:")
        mem.executescript(_SCHEMA_COPIA)
        pedidos = origem.execute(
            "SELECT id, cliente_id, data_pedido, status, valor_total, data_entrega, codigo_rastreio "
            "FROM pedidos WHERE cliente_id = ?", (cliente_id,)).fetchall()
        mem.executemany("INSERT INTO pedidos VALUES (?,?,?,?,?,?,?)", pedidos)
        ids = [p[0] for p in pedidos]
        if ids:
            marcas = ",".join("?" * len(ids))
            itens = origem.execute(
                f"SELECT id, pedido_id, produto, quantidade, preco_unit FROM itens_pedido WHERE pedido_id IN ({marcas})", ids
            ).fetchall()
            mem.executemany("INSERT INTO itens_pedido VALUES (?,?,?,?,?)", itens)
        mem.commit()
    finally:
        origem.close()

    SELECT = getattr(sqlite3, "SQLITE_SELECT", 21)
    READ = getattr(sqlite3, "SQLITE_READ", 20)
    FUNCTION = getattr(sqlite3, "SQLITE_FUNCTION", 31)
    RECURSIVE = getattr(sqlite3, "SQLITE_RECURSIVE", 33)

    def autorizar(acao, arg1, arg2, banco, origem_):
        if acao in (SELECT, FUNCTION, RECURSIVE):
            return sqlite3.SQLITE_OK
        if acao == READ and (arg1 or "").lower() in TABELAS_PERMITIDAS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY

    mem.set_authorizer(autorizar)
    return mem


def executar_isolado(sql: str, cliente_id: int, db: Path = DB, max_linhas: int = 20):
    """Executa um SELECT JA VALIDADO sobre a copia isolada. Devolve (colunas, linhas)."""
    con = copia_isolada(cliente_id, db)
    try:
        cur = con.execute(sql)
        colunas = [d[0] for d in cur.description] if cur.description else []
        return colunas, cur.fetchmany(max_linhas)
    finally:
        con.close()


def pedido_do_cliente_por_codigo(cliente_id: int, codigo: str, db: Path = DB) -> bool:
    con = copia_isolada(cliente_id, db)
    try:
        return con.execute("SELECT 1 FROM pedidos WHERE codigo_rastreio = ?", (codigo,)).fetchone() is not None
    finally:
        con.close()


def formatar(colunas, linhas) -> str:
    if not linhas:
        return ""
    cab = " | ".join(colunas)
    corpo = "\n".join(" | ".join("" if v is None else str(v) for v in l) for l in linhas)
    return f"{cab}\n{corpo}"
