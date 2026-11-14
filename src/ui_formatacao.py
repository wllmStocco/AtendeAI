"""Funcoes PURAS de formatacao usadas pela interface Streamlit (testaveis sem Streamlit nem LangChain)."""
from typing import Dict, List

_EMOJI = {"sustentada": "✅", "sem_afirmacoes": "✅", "parcial": "⚠️", "sem_evidencia": "❌", "contradicao": "❌"}
_LEGENDA_VEREDITO = {
    "sustentada": "sustentada pelas ferramentas",
    "sem_afirmacoes": "sem afirmações a verificar",
    "parcial": "parte não confirmada (apenas registrada)",
    "sem_evidencia": "afirmação sem nenhuma ferramenta consultada",
    "contradicao": "contradiz a evidência",
}
_LEGENDA_ACAO = {
    "entregue": "resposta entregue como gerada",
    "regenerada": "resposta CORRIGIDA pela verificação",
    "resposta_segura": "resposta segura (encaminhamento a humano)",
    "sem_verificacao": "verificação desligada",
}


def entidades_texto(entidades: Dict) -> str:
    pares = [f"{k}={v}" for k, v in (entidades or {}).items() if v]
    return ", ".join(pares) if pares else "nenhuma"


def trajetoria_texto(ferramentas: List[str]) -> str:
    return " → ".join(ferramentas) if ferramentas else "(nenhuma ferramenta)"


def linhas_passos(passos: List[tuple]) -> List[dict]:
    return [{"ferramenta": f, "entrada": e, "resultado (resumo)": o} for f, e, o in passos]


def verificacao_texto(vd: Dict) -> str:
    if not vd:
        return "—"
    v = vd.get("veredito", "")
    texto = f"{_EMOJI.get(v, '•')} {v} ({_LEGENDA_VEREDITO.get(v, '')})"
    if vd.get("coerente") is False:
        texto += " · ⚠️ incoerência interna"
    return texto


def acao_texto(acao: str) -> str:
    return _LEGENDA_ACAO.get(acao, acao or "—")
