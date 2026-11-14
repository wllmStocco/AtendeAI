"""Logica pura (sem LangChain) usada pelo modo DUMMY para simular o LLM.

Nao e um modelo: e um conjunto de regras deterministicas que permite ao professor
executar o projeto sem credenciais. Resultados reais usam o modo `openai`.
"""
import re

_CEP = re.compile(r"\b(\d{5})-?(\d{3})\b")
_PEDIDO = re.compile(r"(?:pedido|ped\.?|#|n[ºo°]\.?)\s*[:#]?\s*(\d{4,7})", re.IGNORECASE)
_PRODUTOS = ["tênis", "tenis", "camiseta", "notebook", "fone", "celular", "mochila", "geladeira", "jaqueta"]

_RESPOSTAS = {
    "status_pedido": "Vou consultar o status do seu pedido agora mesmo.",
    "troca_devolucao": "Vou verificar a nossa política de troca e devolução para o seu caso.",
    "prazo_frete": "Vou calcular o prazo e o frete para o seu CEP.",
    "cancelamento": "Vou verificar se o seu pedido ainda pode ser cancelado.",
    "saudacao": "Olá! Como posso ajudar você hoje?",
    "outro": "Pode me dar mais detalhes sobre o que você precisa?",
}


def classificar_mensagem(texto: str) -> dict:
    t = texto.lower()

    cep = None
    m = _CEP.search(texto)
    if m:
        cep = f"{m.group(1)}-{m.group(2)}"
        texto_sem_cep = _CEP.sub(" ", texto)
    else:
        texto_sem_cep = texto

    pedido = None
    mp = _PEDIDO.search(texto_sem_cep)
    if mp:
        pedido = mp.group(1)

    produto = next((p for p in _PRODUTOS if p in t), None)

    if "cancel" in t:
        intencao = "cancelamento"
    elif any(k in t for k in ("troc", "devol", "arrepend", "reembols", "dinheiro de volta", "defeito")):
        intencao = "troca_devolucao"
    elif any(k in t for k in ("prazo", "frete", "quanto tempo", "demora")) or (cep and "pedido" not in t):
        intencao = "prazo_frete"
    elif any(k in t for k in ("onde est", "status", "rastre", "cadê", "cade", "chegou", "pedido")):
        intencao = "status_pedido"
    elif any(k in t for k in ("oi", "olá", "ola", "bom dia", "boa tarde", "boa noite")) and len(t.split()) <= 4:
        intencao = "saudacao"
    else:
        intencao = "outro"

    return {
        "intencao": intencao,
        "entidades": {"numero_pedido": pedido, "cep": cep, "produto": produto},
        "requer_ferramenta": intencao not in ("saudacao", "outro"),
        "resposta": _RESPOSTAS[intencao],
        "confianca": 0.4 if intencao == "outro" else 0.9,
    }


# ---------------------------------------------------------------------------
# Simulador de ReAct (modo DUMMY). Le o prompt do ZeroShotAgent e devolve o proximo passo.
# ---------------------------------------------------------------------------
_KW_PRAZO = ("prazo", "frete", "entrega", "demora", "quanto tempo", "chega")
_KW_END = ("endereço", "endereco", "rua", "valid", "cep")
_KW_POLITICA = ("troc", "devol", "cancel", "garantia", "defeito", "reembols", "estorno",
                "arrepend", "politica", "política")


def planejar_dummy(pergunta: str, historico: str, tools: list, observacoes: list, feitas: list):
    """Decide o proximo passo do agente DUMMY. Devolve ("acao", ferramenta, entrada, racional) ou ("final", texto).
    Usado pelo ReAct (Fases 2-4) e pelo agente com tool calling (Fase 5)."""
    q = pergunta.lower()
    mc = _CEP.search(pergunta) or _CEP.search(historico)
    cep = f"{mc.group(1)}-{mc.group(2)}" if mc else None
    quer_prazo = any(k in q for k in _KW_PRAZO)
    quer_end = any(k in q for k in _KW_END)

    passos = []
    if cep and "consultar_cep" in tools and (quer_prazo or quer_end):
        passos.append(("consultar_cep", cep, "Preciso validar o CEP informado."))
    if cep and quer_prazo and "calcular_prazo_frete" in tools:
        passos.append(("calcular_prazo_frete", cep, "Agora calculo prazo e frete da região desse CEP."))
    if "buscar_politica" in tools and any(k in q for k in _KW_POLITICA):
        passos.append(("buscar_politica", pergunta, "Preciso consultar a política da loja para responder."))
    if "consultar_pedidos" in tools and any(k in q for k in ("pedido", "rastre", "onde est", "status")) \
            and not any(p[0] == "buscar_politica" for p in passos):
        passos.append(("consultar_pedidos", pergunta, "Preciso consultar os dados do pedido no banco da loja."))

    if len(observacoes) >= len(passos) and "consultar_rastreio" in tools and any(k in q for k in ("rastre", "onde est")) \
            and "consultar_rastreio" not in feitas:
        codigo = re.search(r"BR\d+XX", " ".join(observacoes))
        if codigo:
            return ("acao", "consultar_rastreio", codigo.group(0),
                    "Tenho o código de rastreio; agora consulto os eventos no serviço de rastreio.")
    if len(observacoes) < len(passos):
        ferramenta, entrada, racional = passos[len(observacoes)]
        return ("acao", ferramenta, entrada, racional)

    if observacoes:
        return ("final", " ".join(o[:400] for o in observacoes))
    if quer_prazo and not cep:
        return ("final", "Para calcular prazo e frete, preciso do seu CEP. Pode informar?")
    return ("final", "Não tenho uma ferramenta para isso agora. Posso encaminhar você a um atendente humano.")


def responder_react(prompt: str) -> str:
    corpo = prompt.rsplit("Begin!", 1)[-1]
    idx = corpo.rfind("Question:")
    historico = corpo[:idx]
    pergunta, _, scratch = corpo[idx + len("Question:"):].partition("\nThought:")
    m_tools = re.search(r"one of \[(.*?)\]", prompt)
    tools = [t.strip() for t in m_tools.group(1).split(",")] if m_tools else []
    observacoes = re.findall(r"Observation:\s*(.*?)\s*(?=\nThought:|$)", scratch, re.S)
    feitas = re.findall(r"Action:\s*(\w+)", scratch)

    passo = planejar_dummy(pergunta.strip(), historico, tools, observacoes, feitas)
    if passo[0] == "acao":
        _, ferramenta, entrada, racional = passo
        return f"Thought: {racional}\nAction: {ferramenta}\nAction Input: {entrada}"
    return f"Thought: I now know the final answer\nFinal Answer: {passo[1]}"


# ---------------------------------------------------------------------------
# Fase 4 (modo DUMMY): geracao de SQL por regras
# ---------------------------------------------------------------------------
def gerar_sql_dummy(pergunta: str) -> str:
    q = pergunta.lower()
    m = re.search(r"(?:pedido|ped\.?|#)\s*[:#]?\s*(\d{4,7})", q)
    if m and any(k in q for k in ("iten", "produto", "comprei", "o que")):
        return ("SELECT p.id, i.produto, i.quantidade, i.preco_unit FROM itens_pedido i "
                f"JOIN pedidos p ON p.id = i.pedido_id WHERE p.id = {m.group(1)}")
    if m:
        return ("SELECT id, data_pedido, status, valor_total, data_entrega, codigo_rastreio "
                f"FROM pedidos WHERE id = {m.group(1)}")
    if any(k in q for k in ("quantos", "total", "gastei")):
        return "SELECT COUNT(*) AS pedidos, ROUND(SUM(valor_total), 2) AS valor_total FROM pedidos"
    return ("SELECT id, data_pedido, status, valor_total, codigo_rastreio FROM pedidos "
            "ORDER BY data_pedido DESC LIMIT 5")
