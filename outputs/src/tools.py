"""Ferramentas do agente AtendeAI (entrada de texto unica, exigencia do ZERO_SHOT_REACT_DESCRIPTION).

Fases 2-3 (FERRAMENTAS): consultar_cep (API externa), calcular_prazo_frete (hardcoded), buscar_politica (RAG).
Fase 4 (montar_ferramentas): + consultar_pedidos (text-to-query), consultar_rastreio (microsservico interno),
                             buscar_politica com transformacao de consulta.
"""
import os

import requests
from langchain_core.tools import tool

from src import regras

_INDICE = None


@tool
def consultar_cep(cep: str) -> str:
    """Consulta um CEP na API externa ViaCEP e devolve logradouro, bairro, cidade e UF.
    Entrada: o CEP com 8 digitos, ex.: 01310-100. Use para validar ou descobrir o endereco de um CEP."""
    cep_ok = regras.normalizar_cep(cep)
    if not cep_ok:
        return "CEP inválido: informe 8 dígitos (ex.: 01310-100)."
    if os.getenv("VIACEP_OFFLINE") != "1":
        try:
            r = requests.get(f"https://viacep.com.br/ws/{cep_ok.replace('-', '')}/json/", timeout=8)
            if r.status_code == 200:
                d = r.json()
                if d.get("erro"):
                    return f"CEP {cep_ok} não encontrado na base ViaCEP."
                rua = d.get("logradouro") or "(CEP sem logradouro específico)"
                return f"CEP {cep_ok}: {rua}, {d.get('bairro', '')}, {d.get('localidade', '')}/{d.get('uf', '')} (fonte: ViaCEP)"
        except requests.RequestException:
            pass
    if cep_ok in regras.CEP_OFFLINE:
        return f"CEP {cep_ok}: {regras.CEP_OFFLINE[cep_ok]} (fonte: cache offline)"
    return "Não consegui consultar o ViaCEP agora (erro de rede)."


@tool
def calcular_prazo_frete(entrada: str) -> str:
    """Calcula prazo de entrega (dias uteis) e frete pela tabela da loja, a partir do CEP de destino.
    Entrada: 'CEP' ou 'CEP;valor_do_pedido', ex.: 01310-100;150.00. Frete gratis a partir de R$ 199,00."""
    return regras.calcular_prazo_frete(entrada)


@tool
def buscar_politica(consulta: str) -> str:
    """Busca nas politicas da loja (trocas, devolucoes, cancelamento, garantia, prazos, frete, reembolso)
    os trechos mais relevantes para a pergunta. Entrada: a pergunta do cliente em linguagem natural."""
    global _INDICE
    from src.indice import carregar_indice

    try:
        if _INDICE is None:
            _INDICE = carregar_indice()
    except FileNotFoundError as e:
        return str(e)
    achados = _INDICE.similarity_search(consulta, k=3)
    return " | ".join(f"[{d.metadata.get('fonte')}] {' '.join(d.page_content.split())}" for d in achados)


FERRAMENTAS = [consultar_cep, calcular_prazo_frete, buscar_politica]


# ======================= Fase 4 =======================
TIPOS_FERRAMENTA = {
    "consultar_cep": ("Web API externa (ViaCEP)", "hardcoded", "GET HTTP publico; timeout 8 s; trata CEP inexistente e falha de rede (cache offline)"),
    "calcular_prazo_frete": ("Funcao stateless (estilo serverless)", "hardcoded", "Tabela fixa por regiao do CEP; pura, sem I/O, empacotavel como funcao serverless"),
    "consultar_rastreio": ("API interna / microsservico", "hardcoded", "GET HTTP em servico interno com chave X-API-Key; timeout 5 s; 401/404 tratados; valida que o codigo e do cliente"),
    "consultar_pedidos": ("Text-to-query (SQL)", "semantica", "LLM gera SELECT; validado, executado em copia isolada e somente leitura com so os pedidos do cliente"),
    "buscar_politica": ("RAG (indice FAISS existente)", "semantica", "Busca vetorial; com transformacao de consulta opcional (reescrita, multi-consulta, HyDE)"),
}


def criar_tool_politica(llm=None, estrategia: str = "direta"):
    recuperador = {}

    @tool
    def buscar_politica(consulta: str) -> str:
        """Busca nas politicas da loja (trocas, devolucoes, cancelamento, garantia, prazos, frete, reembolso)
        os trechos mais relevantes para a pergunta. Entrada: a pergunta do cliente em linguagem natural."""
        if "r" not in recuperador:
            try:
                recuperador["r"] = __import__("src.recuperacao", fromlist=["Recuperador"]).Recuperador(llm=llm)
            except FileNotFoundError as e:
                return str(e)
        resultados, _ = recuperador["r"].buscar(consulta, estrategia)
        return " | ".join(f"[{r['fonte']}] {r['trecho']}" for r in resultados)

    return buscar_politica


def criar_tool_rastreio(cliente_id: int):
    @tool
    def consultar_rastreio(codigo: str) -> str:
        """Consulta no servico interno de logistica os eventos de rastreio de um pedido a partir do codigo de rastreio
        (formato BR123456789XX, obtido com a ferramenta consultar_pedidos). Entrada: o codigo de rastreio."""
        from src.servicos import rastreio as srv
        from src.sql_seguro import pedido_do_cliente_por_codigo

        codigo = (codigo or "").strip()
        if not pedido_do_cliente_por_codigo(cliente_id, codigo):
            return "Código de rastreio não encontrado entre os pedidos deste cliente."
        url = srv.garantir_servico()
        try:
            r = requests.get(f"{url}/rastreio/{codigo}", headers={"X-API-Key": srv.chave()}, timeout=5)
        except requests.RequestException:
            return "O serviço de rastreio está indisponível no momento. Tente novamente em instantes."
        if r.status_code == 401:
            return "Falha de autenticação no serviço de rastreio (chave interna inválida)."
        if r.status_code == 404:
            return "O serviço de rastreio não encontrou esse código."
        if r.status_code != 200:
            return f"Serviço de rastreio respondeu com erro {r.status_code}."
        d = r.json()
        eventos = "; ".join(f"{e['data']}: {e['descricao']}" for e in d["eventos"])
        return f"Rastreio {d['codigo']} (status {d['status']}): {eventos} (fonte: serviço de rastreio)"

    return consultar_rastreio


def montar_ferramentas(llm, cliente_id: int = 1, estrategia: str = "direta"):
    """Conjunto completo da Fase 4+. `estrategia` define a transformacao de consulta do RAG."""
    from src.sql_agente import criar_tool_sql

    return [consultar_cep, calcular_prazo_frete, criar_tool_politica(llm, estrategia),
            criar_tool_sql(llm, cliente_id), criar_tool_rastreio(cliente_id)]
