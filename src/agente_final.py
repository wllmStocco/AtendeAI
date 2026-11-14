"""AGENTE INTEGRADOR (Fase 5): tool calling + historico + memoria de longo prazo + VERIFICACAO de respostas.

  create_openai_tools_agent + AgentExecutor   -> ferramentas de SQL, API (ViaCEP, rastreio) e RAG sob um unico executor
  RunnableWithMessageHistory                  -> historico da conversa por sessao (memoria de curto prazo)
  MemoriaLongoPrazo + slots + GerenciadorContexto -> memoria da Fase 3 mantida durante a sessao inteira
  src/verificacao.py                          -> NLI + checagem numerica + coerencia, DENTRO do fluxo do agente:
      contradicao / incoerencia / afirmacao SEM evidencia -> tenta corrigir 1 vez; se persistir -> resposta segura (humano)
"""
import os
import re
import time
import uuid
from dataclasses import dataclass, field
from datetime import date
from typing import Dict, List, Optional

from langchain.agents import AgentExecutor, create_openai_tools_agent
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.messages import AIMessage
from langchain_core.runnables.history import RunnableWithMessageHistory

from src.callbacks import LogHandler
from src.chains import build_chain_intencao
from src.chatbot import _INTENCOES_COM_EPISODIO, _MEU_CEP, buscar_cliente
from src.contexto import GerenciadorContexto
from src.memoria_longa import MemoriaLongoPrazo
from src.prompts_final import PROMPT_FINAL
from src.tools import montar_ferramentas
from src.verificacao import Verificacao, criar_nli, evidencia_limpa, verificar

RESPOSTA_SEGURA = ("Não consegui confirmar essa informação com segurança nos dados da loja. "
                   "Para evitar qualquer erro, vou encaminhar você a um atendente humano.")

PROMPT_CORRECAO = """Você é a Lia, atendente de uma loja. A resposta anterior ao cliente tem problemas de consistência com os dados
das ferramentas. Escreva uma nova resposta usando SOMENTE os dados abaixo.

Regras:
- Português, no máximo 4 frases, direto ao ponto. Não cumprimente de novo.
- Não mencione "evidências", nomes de arquivo, SQL nem detalhes técnicos; cite a fonte de forma natural (ex.: "segundo a política de frete").
- Se os dados não bastarem para responder, diga apenas que não tem essa informação e ofereça encaminhar a um atendente humano.

Pergunta do cliente: {pergunta}

Dados das ferramentas:
{evidencias}

Resposta anterior (com problemas): {resposta}

Problemas encontrados:
{problemas}

Nova resposta:"""


def _entrada_texto(tool_input) -> str:
    """tool_input pode ser str ou dict ({'entrada': '80010-000'}): devolve so o conteudo."""
    if isinstance(tool_input, dict):
        return " ".join(str(v) for v in tool_input.values())
    return str(tool_input)


@dataclass
class Turno5:
    mensagem: str
    resposta: str = ""
    resposta_rascunho: str = ""
    intencao: str = ""
    intencao_efetiva: str = ""
    continuacao: bool = False
    entidades: dict = field(default_factory=dict)
    slots: dict = field(default_factory=dict)
    memorias: list = field(default_factory=list)
    ferramentas: list = field(default_factory=list)       # trajetoria: ferramentas chamadas, em ordem
    passos: list = field(default_factory=list)            # (ferramenta, entrada, observacao resumida)
    verificacao_rascunho: dict = field(default_factory=dict)
    verificacao_final: dict = field(default_factory=dict)
    acao: str = ""                                        # entregue | regenerada | resposta_segura | sem_verificacao
    resumo_gerado: Optional[str] = None
    tempo_s: float = 0.0
    erro: Optional[str] = None


class AtendeAIAgente:
    def __init__(self, llm, cliente_id: int, memoria_longa: Optional[MemoriaLongoPrazo] = None,
                 handler: Optional[LogHandler] = None, estrategia_rag: str = "direta", verificar: bool = True,
                 max_mensagens: int = 12, manter: int = 4, verbose: bool = False, nli=None):
        self.llm = llm
        self.cliente = buscar_cliente(cliente_id)
        self.mem_longa = memoria_longa or MemoriaLongoPrazo()
        self.handler = handler or LogHandler()
        self.verificar_ativo = verificar
        self.entrada_na_evidencia = os.getenv("VERIFICACAO_COM_ENTRADA", "0") == "1"   # decidido por medicao (scripts/diagnostico_nli.py)
        self._nli = nli                                    # (funcao, descricao) ou None (carrega sob demanda)

        self.ferramentas = montar_ferramentas(llm, cliente_id, estrategia_rag)
        agente = create_openai_tools_agent(llm, self.ferramentas, PROMPT_FINAL)
        self.executor = AgentExecutor(agent=agente, tools=self.ferramentas, verbose=verbose,
                                      handle_parsing_errors=True, max_iterations=8, return_intermediate_steps=True)
        self.sessoes: Dict[str, InMemoryChatMessageHistory] = {}
        self.com_historico = RunnableWithMessageHistory(
            self.executor, self._obter_historico,
            input_messages_key="input", history_messages_key="chat_history", output_messages_key="output")

        self.classificador = build_chain_intencao(llm)
        self.contexto_mgr = GerenciadorContexto(llm, max_mensagens=max_mensagens, manter=manter)
        self.estado: Dict[str, dict] = {}

    # ---------------------------------------------------------------- sessao ----------------------------------
    def _obter_historico(self, session_id: str) -> InMemoryChatMessageHistory:
        if session_id not in self.sessoes:
            self.sessoes[session_id] = InMemoryChatMessageHistory()
        return self.sessoes[session_id]

    def nova_sessao(self) -> str:
        return f"sessao-{uuid.uuid4().hex[:6]}"

    def _cfg(self, session_id: Optional[str] = None) -> dict:
        cfg: dict = {"callbacks": [self.handler]}
        if session_id:
            cfg["configurable"] = {"session_id": session_id}
        return cfg

    # ---------------------------------------------------------------- verificacao ----------------------------
    def _nli_fn(self):
        if self._nli is None:
            self._nli = criar_nli()
        return self._nli

    def verificar_resposta(self, resposta: str, evidencias: List[str], contexto_extra: str = "",
                           entradas: Optional[List[str]] = None) -> Verificacao:
        fn, modo = self._nli_fn()
        return verificar(resposta, evidencias, fn, contexto_extra=contexto_extra, modo=modo,
                         entradas=entradas if self.entrada_na_evidencia else None)

    def _corrigir(self, mensagem: str, rascunho: str, v: Verificacao, evidencias: List[str]) -> Optional[str]:
        prompt = PROMPT_CORRECAO.format(
            pergunta=mensagem, evidencias="\n".join(f"- {evidencia_limpa(e)[:500]}" for e in evidencias) or "(nenhuma ferramenta foi consultada)",
            resposta=rascunho, problemas="\n".join(f"- {p}" for p in v.problemas))
        try:
            texto = self.llm.invoke(prompt, config={"callbacks": [self.handler]}).content
        except Exception:
            return None
        if not isinstance(texto, str) or not texto.strip() or texto.startswith("[dummy]"):
            return None
        return texto.strip()

    def _verificar_e_corrigir(self, t: Turno5, mensagem: str, contexto: str, evidencias: List[str], sid: str,
                              entradas: Optional[List[str]] = None) -> None:
        extra = f"{mensagem} {contexto}"
        v = self.verificar_resposta(t.resposta_rascunho, evidencias, extra, entradas)
        t.verificacao_rascunho = v.resumo()
        t.verificacao_final = t.verificacao_rascunho
        t.acao = "entregue"
        if v.veredito not in ("contradicao", "sem_evidencia") and v.coerente:
            return
        nova = self._corrigir(mensagem, t.resposta_rascunho, v, evidencias)
        final = v
        if nova:
            v2 = self.verificar_resposta(nova, evidencias, extra, entradas)
            if v2.veredito not in ("contradicao", "sem_evidencia") and v2.coerente:
                t.resposta, t.acao, final = nova, "regenerada", v2
            else:
                t.resposta, t.acao, final = RESPOSTA_SEGURA, "resposta_segura", v2
        else:
            t.resposta, t.acao = RESPOSTA_SEGURA, "resposta_segura"
        t.verificacao_final = final.resumo()
        hist = self._obter_historico(sid)           # o historico deve guardar o que o cliente de fato recebeu
        if hist.messages and isinstance(hist.messages[-1], AIMessage):
            hist.messages[-1] = AIMessage(content=t.resposta)

    # ---------------------------------------------------------------- memorias ------------------------------
    def _montar_contexto(self, est: dict, t: Turno5) -> str:
        partes = []
        dados = {k: v for k, v in est["slots"].items() if v}
        if dados:
            partes.append("dados da conversa: " + ", ".join(f"{k}={v}" for k, v in dados.items()))
        if t.continuacao:
            partes.append(f"continuação do assunto: {t.intencao_efetiva}")
        if t.memorias:
            partes.append("memórias do cliente: " + " | ".join(t.memorias))
        return "; ".join(partes) if partes else "nenhum"

    def _gravar_memorias(self, mensagem: str, t: Turno5, est: dict) -> None:
        cid = self.cliente["id"]
        if t.entidades.get("cep") and _MEU_CEP.search(mensagem):
            self.mem_longa.adicionar(cid, "semantica", f"CEP padrão do cliente: {t.entidades['cep']}", chave="cep_padrao")
        if t.intencao_efetiva in _INTENCOES_COM_EPISODIO:
            texto = f"{date.today().isoformat()}: cliente consultou {_INTENCOES_COM_EPISODIO[t.intencao_efetiva]}"
            if est["slots"].get("produto"):
                texto += f" (produto: {est['slots']['produto']})"
            if est["slots"].get("numero_pedido"):
                texto += f" [pedido {est['slots']['numero_pedido']}]"
            self.mem_longa.adicionar(cid, "episodica", texto)

    # ---------------------------------------------------------------- turno ---------------------------------
    def responder(self, mensagem: str, session_id: str = "sessao-1") -> Turno5:
        inicio = time.perf_counter()
        t = Turno5(mensagem=mensagem)
        est = self.estado.setdefault(session_id, {"slots": {"numero_pedido": None, "cep": None, "produto": None}, "ultima": None})
        try:
            analise = self.classificador.invoke({"mensagem": mensagem}, config=self._cfg())
            t.intencao = analise.intencao
            t.entidades = analise.entidades.model_dump()
            t.continuacao = (analise.intencao in ("outro", "saudacao") and bool(est["ultima"]) and len(mensagem.split()) <= 8)
            t.intencao_efetiva = est["ultima"] if t.continuacao else analise.intencao
            if t.intencao_efetiva not in ("outro", "saudacao"):
                est["ultima"] = t.intencao_efetiva
            for k, v in t.entidades.items():
                if v:
                    est["slots"][k] = v
            t.slots = dict(est["slots"])
            t.memorias = [m["texto"] for m in self.mem_longa.recuperar(self.cliente["id"], mensagem)]
            contexto = self._montar_contexto(est, t)

            saida = self.com_historico.invoke({"input": mensagem, "contexto": contexto}, config=self._cfg(session_id))
            rascunho = saida["output"]
            passos = saida.get("intermediate_steps", [])
            t.passos = [(a.tool, str(a.tool_input)[:120], str(o)[:200]) for a, o in passos]
            t.ferramentas = [a.tool for a, _ in passos]
            evidencias = [str(o) for _, o in passos]
            entradas = [_entrada_texto(a.tool_input) for a, _ in passos]
            t.resposta = t.resposta_rascunho = rascunho
            t.acao = "sem_verificacao"
            if self.verificar_ativo:
                self._verificar_e_corrigir(t, mensagem, contexto, evidencias, session_id, entradas)
            self._gravar_memorias(mensagem, t, est)
            t.resumo_gerado = self.contexto_mgr.aplicar_historico(self._obter_historico(session_id))
        except Exception as e:  # o agente nao deve cair; o erro fica no turno
            t.erro = f"{type(e).__name__}: {str(e)[:300]}"
            if not t.resposta:
                t.resposta = "Desculpe, tive uma instabilidade técnica agora. Pode repetir em instantes?"
        t.tempo_s = round(time.perf_counter() - inicio, 2)
        return t
