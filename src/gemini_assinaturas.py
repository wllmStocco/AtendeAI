"""Logica PURA (sem LangChain) para devolver as 'thought signatures' do Gemini 3 no formato de compatibilidade OpenAI.

O Gemini 3 exige que cada chamada de funcao reenviada no historico leve
    tool_calls[i]["extra_content"] = {"google": {"thought_signature": "<assinatura>"}}
O LangChain descarta esse campo. Aqui: (1) reaproveita a assinatura real quando ela ainda existe nas mensagens de
origem; (2) senao usa a assinatura de contorno documentada pelo Google/LiteLLM ("skip_thought_signature_validator"),
que o Google avisa poder reduzir a qualidade do raciocinio.
"""
from typing import Dict, Iterable, List

ASSINATURA_CONTORNO = "skip_thought_signature_validator"


def coletar_assinaturas(mensagens_origem: Iterable) -> Dict[str, str]:
    """Procura assinaturas reais nos additional_kwargs['tool_calls'] das mensagens de origem (se existirem)."""
    achadas: Dict[str, str] = {}
    for m in mensagens_origem:
        extras = getattr(m, "additional_kwargs", None) or {}
        for tc in extras.get("tool_calls") or []:
            sig = ((tc.get("extra_content") or {}).get("google") or {}).get("thought_signature")
            if sig and tc.get("id"):
                achadas[tc["id"]] = sig
    return achadas


def injetar_assinaturas(mensagens_payload: List[dict], assinaturas: Dict[str, str]) -> int:
    """Completa tool_calls das mensagens 'assistant' do payload. Devolve quantas chamadas foram alteradas."""
    alteradas = 0
    for msg in mensagens_payload:
        if msg.get("role") != "assistant":
            continue
        for tc in msg.get("tool_calls") or []:
            if not (tc.get("extra_content") or {}).get("google", {}).get("thought_signature"):
                sig = assinaturas.get(tc.get("id"), ASSINATURA_CONTORNO)
                tc["extra_content"] = {"google": {"thought_signature": sig}}
                alteradas += 1
    return alteradas
