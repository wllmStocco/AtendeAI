"""Callback de logging e depuracao das execucoes do agente."""
import time
from typing import Any

from langchain_core.callbacks import BaseCallbackHandler


class LogHandler(BaseCallbackHandler):
    def __init__(self, imprimir: bool = False):
        self.imprimir = imprimir
        self.eventos: list = []
        self._t0: dict = {}

    def _log(self, tipo: str, **dados):
        ev = {"t": round(time.perf_counter(), 3), "tipo": tipo, **dados}
        self.eventos.append(ev)
        if self.imprimir:
            print(f"[log] {tipo}: {dados}")

    def on_chat_model_start(self, serialized, messages, **kwargs: Any):
        self._log("llm_start", n_mensagens=sum(len(m) for m in messages))

    def on_llm_start(self, serialized, prompts, **kwargs: Any):
        self._log("llm_start", n_prompts=len(prompts))

    def on_llm_end(self, response, **kwargs: Any):
        try:
            g = response.generations[0][0]
            info = getattr(g, "generation_info", None) or {}
            self._log("llm_end", saida=repr((g.text or "")[:80]), finish_reason=info.get("finish_reason"))
        except Exception as e:  # nunca derrubar o agente por causa do log
            self._log("llm_end", saida=f"<erro ao ler resposta: {e}>", finish_reason=None)

    def on_agent_action(self, action, **kwargs: Any):
        self._log("agent_action", ferramenta=action.tool, entrada=str(action.tool_input)[:120])

    def on_tool_start(self, serialized, input_str, **kwargs: Any):
        self._t0[kwargs.get("run_id")] = time.perf_counter()
        self._log("tool_start", ferramenta=(serialized or {}).get("name"), entrada=str(input_str)[:120])

    def on_tool_end(self, output, **kwargs: Any):
        dur = time.perf_counter() - self._t0.pop(kwargs.get("run_id"), time.perf_counter())
        self._log("tool_end", duracao_s=round(dur, 3), saida=str(output)[:120])

    def on_tool_error(self, error, **kwargs: Any):
        self._log("tool_error", erro=str(error)[:200])

    def on_chain_error(self, error, **kwargs: Any):
        self._log("chain_error", erro=str(error)[:200])

    def on_agent_finish(self, finish, **kwargs: Any):
        self._log("agent_finish", resposta=str(finish.return_values.get("output", ""))[:120])

    def ferramentas_usadas(self) -> list:
        """Ferramentas reais chamadas ('_Exception' e o tratamento de erro de formato do executor)."""
        return [e["ferramenta"] for e in self.eventos
                if e["tipo"] == "agent_action" and e["ferramenta"] != "_Exception"]

    def erros_de_formato(self) -> int:
        """Quantas vezes o LLM respondeu fora do formato ReAct e o executor teve que se recuperar."""
        return sum(1 for e in self.eventos if e["tipo"] == "agent_action" and e["ferramenta"] == "_Exception")
