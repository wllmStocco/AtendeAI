"""Ferramenta TEXT-TO-QUERY: pergunta em linguagem natural -> SQL (gerado pelo LLM) -> execucao segura."""
import sqlite3

from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.tools import tool

from src.sql_seguro import SCHEMA_VISIVEL, executar_isolado, formatar, limpar_sql, validar_sql

SYSTEM_SQL = """Gere UMA consulta SQL (SQLite, somente SELECT) que responda à pergunta do cliente.
Tabelas disponíveis:
{schema}

Regras:
- Use apenas essas duas tabelas e apenas SELECT. Não use comentários nem ponto e vírgula no meio.
- O banco já contém somente os pedidos DESTE cliente: não filtre por cliente_id.
- Para datas relativas use date('now'). Datas estão no formato AAAA-MM-DD.
- Prefira listar poucas colunas relevantes e use LIMIT 10 quando listar vários pedidos.
- Responda SOMENTE com o SQL, sem explicações e sem markdown."""

# v2 (Fase 4.1): regra contra filtros nao pedidos. ADOTADA como padrao em 09/11 (v1: 4/9 acertos em agregacoes; v2: 9/9; ver notebook 04b).
REGRA_V2 = ("- NÃO adicione filtros que a pergunta não pediu (por exemplo, excluir pedidos cancelados ou devolvidos). "
            "Em contagens, somas e médias, considere TODOS os pedidos, a menos que a pergunta peça um filtro.")
SYSTEM_SQL_V2 = SYSTEM_SQL.replace("- Prefira listar", REGRA_V2 + "\n- Prefira listar")

PROMPTS_SQL = {
    "v1": ChatPromptTemplate.from_messages([("system", SYSTEM_SQL), ("human", "{pergunta}")]),
    "v2": ChatPromptTemplate.from_messages([("system", SYSTEM_SQL_V2), ("human", "{pergunta}")]),
}
PROMPT_SQL = PROMPTS_SQL["v2"]


def criar_tool_sql(llm, cliente_id: int, versao: str = "v2"):
    chain = PROMPTS_SQL[versao].partial(schema=SCHEMA_VISIVEL) | llm | StrOutputParser()

    @tool
    def consultar_pedidos(pergunta: str) -> str:
        """Consulta os pedidos do cliente atual no banco da loja (status, datas, valores, itens e código de rastreio).
        Entrada: a pergunta em linguagem natural, ex.: 'qual o status do pedido 10005?'. Só enxerga pedidos deste cliente."""
        bruto = chain.invoke({"pergunta": pergunta})
        ok, sql = validar_sql(limpar_sql(bruto))
        if not ok:
            return f"Consulta recusada por segurança ({sql})."
        try:
            colunas, linhas = executar_isolado(sql, cliente_id)
        except sqlite3.Error as e:
            return f"Não consegui executar a consulta ({e}). SQL: {sql}"
        if not linhas:
            return f"Nenhum pedido encontrado para essa pergunta. (SQL executado: {sql})"
        return f"{formatar(colunas, linhas)}\n(SQL executado: {sql})"

    return consultar_pedidos
