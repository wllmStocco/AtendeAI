"""Gerenciamento da janela de contexto: buffer completo + sumarizacao das mensagens antigas."""
from typing import Optional

from langchain_core.messages import SystemMessage, get_buffer_string


class GerenciadorContexto:
    """Quando o buffer passa de `max_mensagens`, resume as mais antigas (1 chamada de LLM) e
    mantem literalmente as ultimas `manter` mensagens (janela deslizante sobre o resumo)."""

    def __init__(self, llm, max_mensagens: int = 12, manter: int = 4):
        self.llm = llm
        self.max_mensagens = max_mensagens
        self.manter = manter

    def _resumir(self, mensagens) -> str:
        texto = get_buffer_string(mensagens)
        prompt = (
            "Resuma a conversa de atendimento abaixo em até 3 frases, em português, preservando CEP, número de "
            "pedido, produto e decisões já tomadas. Não invente nada.\n\n" + texto
        )
        try:
            resumo = self.llm.invoke(prompt).content
            if isinstance(resumo, str) and resumo.strip() and not resumo.startswith("[dummy]"):
                return resumo.strip()
        except Exception:
            pass
        return " ".join(texto.split())[:400]  # reserva (modo dummy ou falha do LLM): trecho truncado

    def aplicar(self, memory) -> Optional[str]:
        msgs = list(memory.chat_memory.messages)
        if len(msgs) <= self.max_mensagens:
            return None
        antigas, recentes = msgs[: -self.manter], msgs[-self.manter:]
        resumo = self._resumir(antigas)
        memory.chat_memory.clear()
        memory.chat_memory.add_message(SystemMessage(content=f"Resumo da conversa anterior: {resumo}"))
        for m in recentes:
            memory.chat_memory.add_message(m)
        return resumo

    def aplicar_historico(self, historico) -> Optional[str]:
        """Mesma politica, para um ChatMessageHistory (Fase 5, RunnableWithMessageHistory)."""
        msgs = list(historico.messages)
        if len(msgs) <= self.max_mensagens:
            return None
        antigas, recentes = msgs[: -self.manter], msgs[-self.manter:]
        resumo = self._resumir(antigas)
        historico.clear()
        historico.add_message(SystemMessage(content=f"Resumo da conversa anterior: {resumo}"))
        for m in recentes:
            historico.add_message(m)
        return resumo
