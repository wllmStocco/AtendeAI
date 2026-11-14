"""Chat model DUMMY (sem rede, sem chave) para o professor executar o projeto.

Subclasse de BaseChatModel: funciona com `prompt | llm | parser`, batch, stream, bind_tools
e com o agente ReAct (initialize_agent). A logica de decisao fica em src/heuristicas.py.
"""
import json
from typing import Any, Iterator, List, Optional

from langchain_core.callbacks import CallbackManagerForLLMRun
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_core.messages import AIMessage, AIMessageChunk, BaseMessage, HumanMessage, ToolMessage
from langchain_core.outputs import ChatGeneration, ChatGenerationChunk, ChatResult

from src.heuristicas import _CEP, classificar_mensagem, gerar_sql_dummy, planejar_dummy, responder_react

_MARCADOR_JSON = "esquema JSON"


def _conteudo(m: BaseMessage) -> str:
    return m.content if isinstance(m.content, str) else str(m.content)


def _gerar_texto(messages: List[BaseMessage]) -> str:
    tudo = "\n".join(_conteudo(m) for m in messages)
    ultima = _conteudo(messages[-1])
    if "Final Answer:" in tudo and "Begin!" in tudo:  # prompt do agente ReAct
        return responder_react(tudo)
    if "Gere UMA consulta SQL" in tudo:  # ferramenta text-to-query (Fase 4)
        return gerar_sql_dummy(ultima)
    if _MARCADOR_JSON in tudo:
        return json.dumps(classificar_mensagem(ultima), ensure_ascii=False)
    return "[dummy] Resumo simulado: modo sem LLM real."


def _nomes_e_params(tools) -> dict:
    """Aceita nomes (str) ou dicts no formato OpenAI; devolve {nome: primeiro_parametro}."""
    saida = {}
    for t in tools:
        if isinstance(t, str):
            saida[t] = None
        else:
            f = t.get("function", {})
            props = (f.get("parameters") or {}).get("properties") or {}
            saida[f.get("name")] = next(iter(props), None)
    return saida


def _agente_com_tools(messages: List[BaseMessage], tools) -> AIMessage:
    """Agente DUMMY com tool calling (Fase 5): decide por regras, via planejar_dummy."""
    nomes = _nomes_e_params(tools)
    ult_humana = max((i for i, m in enumerate(messages) if isinstance(m, HumanMessage)), default=0)
    pergunta = _conteudo(messages[ult_humana])
    historico = "\n".join(_conteudo(m) for m in messages[:ult_humana])
    depois = messages[ult_humana + 1:]
    observacoes = [_conteudo(m) for m in depois if isinstance(m, ToolMessage)]
    feitas = [tc["name"] for m in depois if isinstance(m, AIMessage) for tc in (m.tool_calls or [])]
    passo = planejar_dummy(pergunta, historico, list(nomes), observacoes, feitas)
    if passo[0] == "acao":
        _, ferramenta, entrada, _ = passo
        chave = nomes.get(ferramenta) or "entrada"
        return AIMessage(content="", tool_calls=[{"name": ferramenta, "args": {chave: entrada},
                                                  "id": f"call_dummy_{len(feitas) + 1}", "type": "tool_call"}])
    return AIMessage(content=passo[1])


class DummyChatModel(BaseChatModel):
    @property
    def _llm_type(self) -> str:
        return "dummy-atendeai"

    def bind_tools(self, tools, **kwargs):
        return self.bind(tools=[getattr(t, "name", str(t)) for t in tools], **kwargs)

    def _generate(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> ChatResult:
        tools = kwargs.get("tools") or []
        if tools and any(isinstance(t, dict) for t in tools):  # agente com tool calling (create_openai_tools_agent)
            return ChatResult(generations=[ChatGeneration(message=_agente_com_tools(messages, tools))])
        if tools:  # simula function calling (demonstracao da Fase 2)
            m = _CEP.search(_conteudo(messages[-1]))
            if m and "consultar_cep" in tools:
                msg = AIMessage(
                    content="",
                    tool_calls=[{"name": "consultar_cep", "args": {"cep": f"{m.group(1)}-{m.group(2)}"},
                                 "id": "call_dummy_1", "type": "tool_call"}],
                )
                return ChatResult(generations=[ChatGeneration(message=msg)])
        texto = _gerar_texto(messages)
        return ChatResult(generations=[ChatGeneration(message=AIMessage(content=texto))])

    def _stream(
        self,
        messages: List[BaseMessage],
        stop: Optional[List[str]] = None,
        run_manager: Optional[CallbackManagerForLLMRun] = None,
        **kwargs: Any,
    ) -> Iterator[ChatGenerationChunk]:
        tools = kwargs.get("tools") or []
        if tools and any(isinstance(t, dict) for t in tools):
            # Agente com tool calling: o AgentExecutor consome o modelo por .stream(); sem este ramo o dummy ignorava as
            # ferramentas, devolvia o texto "[dummy] ..." como resposta final e a verificacao acionava a resposta segura.
            msg = _agente_com_tools(messages, tools)
            if msg.tool_calls:
                partes = [{"name": tc["name"], "args": json.dumps(tc["args"], ensure_ascii=False), "id": tc["id"], "index": i}
                          for i, tc in enumerate(msg.tool_calls)]
                yield ChatGenerationChunk(message=AIMessageChunk(content="", tool_call_chunks=partes))
                return
            texto = _conteudo(msg)
        else:
            texto = _gerar_texto(messages)
        pedacos = texto.split(" ")
        for i, p in enumerate(pedacos):
            piece = p + (" " if i < len(pedacos) - 1 else "")
            yield ChatGenerationChunk(message=AIMessageChunk(content=piece))
