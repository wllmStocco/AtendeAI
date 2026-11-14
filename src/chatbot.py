"""AtendeAI como CHATBOT: memoria de curto prazo (sessao) + longo prazo (cliente) + intencao/entidades.

Fluxo de cada turno:
  1. reconhece INTENCAO e extrai ENTIDADES (chain da Fase 1/2)                    -> 1 chamada de LLM
  2. atualiza o estado da sessao (slots: pedido, cep, produto) e trata continuacoes ("e o prazo?")
  3. recupera MEMORIAS DE LONGO PRAZO relevantes do cliente (busca vetorial)
  4. executa o agente ReAct (ConversationBufferMemory = curto prazo)              -> 1 a 3 chamadas
  5. grava memorias novas (semantica por regra, episodica) e gerencia a janela de contexto
"""
import re
import sqlite3
import time
from dataclasses import dataclass, field
from datetime import date
from pathlib import Path
from typing import Optional

from src.agente import criar_agente, criar_memoria
from src.callbacks import LogHandler
from src.chains import build_chain_intencao
from src.contexto import GerenciadorContexto
from src.memoria_longa import MemoriaLongoPrazo
from src.tools import FERRAMENTAS

ROOT = Path(__file__).resolve().parent.parent
_INTENCOES_COM_EPISODIO = {
    "status_pedido": "o status de um pedido",
    "troca_devolucao": "troca ou devolução",
    "prazo_frete": "prazo e frete",
    "cancelamento": "o cancelamento de um pedido",
}
_MEU_CEP = re.compile(r"\bmeu cep\b|\bcep (?:é|e)\b", re.IGNORECASE)


def buscar_cliente(cliente_id: int) -> dict:
    db = ROOT / "data" / "loja.db"
    if db.exists():
        con = sqlite3.connect(db)
        try:
            linha = con.execute("SELECT id, nome FROM clientes WHERE id=?", (cliente_id,)).fetchone()
        finally:
            con.close()
        if linha:
            return {"id": linha[0], "nome": linha[1]}
    return {"id": cliente_id, "nome": f"Cliente {cliente_id}"}


@dataclass
class Turno:
    mensagem: str
    resposta: str = ""
    intencao: str = ""
    intencao_efetiva: str = ""
    continuacao: bool = False
    entidades: dict = field(default_factory=dict)
    slots: dict = field(default_factory=dict)
    memorias: list = field(default_factory=list)
    ferramentas: list = field(default_factory=list)
    erros_formato: int = 0
    resumo_gerado: Optional[str] = None
    tempo_s: float = 0.0
    erro: Optional[str] = None


class AtendeAIChatbot:
    def __init__(self, llm, cliente_id: int, memoria_longa: Optional[MemoriaLongoPrazo] = None,
                 handler: Optional[LogHandler] = None, max_mensagens: int = 12, manter: int = 4,
                 verbose: bool = False, injetar_contexto: bool = True):
        self.llm = llm
        self.cliente = buscar_cliente(cliente_id)
        self.mem_longa = memoria_longa or MemoriaLongoPrazo()
        self.handler = handler or LogHandler()
        self.memoria = criar_memoria(tipo="buffer")           # curto prazo: ConversationBufferMemory
        self.agente = criar_agente(llm, FERRAMENTAS, memoria=self.memoria, verbose=verbose)
        self.classificador = build_chain_intencao(llm)
        self.contexto = GerenciadorContexto(llm, max_mensagens=max_mensagens, manter=manter)
        self.injetar_contexto = injetar_contexto
        self.slots = {"numero_pedido": None, "cep": None, "produto": None}
        self.ultima_intencao: Optional[str] = None
        self.turnos: list = []

    # ------------------------------------------------------------------------------
    def _cfg(self):
        return {"callbacks": [self.handler]}

    def _atualizar_slots(self, entidades: dict) -> None:
        for chave, valor in entidades.items():
            if valor:
                self.slots[chave] = valor

    def _montar_entrada(self, mensagem: str, t: Turno) -> str:
        if not self.injetar_contexto:
            return mensagem
        partes = []
        dados = {k: v for k, v in self.slots.items() if v}
        if dados:
            partes.append("dados da conversa: " + ", ".join(f"{k}={v}" for k, v in dados.items()))
        if t.continuacao:
            partes.append(f"continuação do assunto: {t.intencao_efetiva}")
        if t.memorias:
            partes.append("memórias do cliente: " + " | ".join(t.memorias))
        return f"{mensagem} (contexto do sistema: {'; '.join(partes)})" if partes else mensagem

    def _gravar_memorias(self, mensagem: str, t: Turno) -> None:
        cid = self.cliente["id"]
        if t.entidades.get("cep") and _MEU_CEP.search(mensagem):  # fato estavel -> memoria SEMANTICA (upsert)
            self.mem_longa.adicionar(cid, "semantica", f"CEP padrão do cliente: {t.entidades['cep']}", chave="cep_padrao")
        if t.intencao_efetiva in _INTENCOES_COM_EPISODIO:         # evento -> memoria EPISODICA
            texto = f"{date.today().isoformat()}: cliente consultou {_INTENCOES_COM_EPISODIO[t.intencao_efetiva]}"
            if self.slots.get("produto"):
                texto += f" (produto: {self.slots['produto']})"
            if self.slots.get("numero_pedido"):
                texto += f" [pedido {self.slots['numero_pedido']}]"
            self.mem_longa.adicionar(cid, "episodica", texto)

    # ------------------------------------------------------------------------------
    def responder(self, mensagem: str) -> Turno:
        inicio = time.perf_counter()
        n_eventos = len(self.handler.eventos)
        t = Turno(mensagem=mensagem)
        try:
            analise = self.classificador.invoke({"mensagem": mensagem}, config=self._cfg())
            t.intencao = analise.intencao
            t.entidades = analise.entidades.model_dump()
            # continuacao: mensagem curta e sem intencao propria herda o assunto anterior
            t.continuacao = (analise.intencao in ("outro", "saudacao") and bool(self.ultima_intencao)
                             and len(mensagem.split()) <= 8)
            t.intencao_efetiva = self.ultima_intencao if t.continuacao else analise.intencao
            if t.intencao_efetiva not in ("outro", "saudacao"):
                self.ultima_intencao = t.intencao_efetiva
            self._atualizar_slots(t.entidades)
            t.slots = dict(self.slots)
            achadas = self.mem_longa.recuperar(self.cliente["id"], mensagem)
            t.memorias = [m["texto"] for m in achadas]
            entrada = self._montar_entrada(mensagem, t)
            t.resposta = self.agente.invoke({"input": entrada}, config=self._cfg())["output"]
            self._gravar_memorias(mensagem, t)
            t.resumo_gerado = self.contexto.aplicar(self.memoria)
        except Exception as e:  # o chatbot nao deve cair; o erro fica registrado no turno
            t.erro = f"{type(e).__name__}: {str(e)[:300]}"
            t.resposta = "Desculpe, tive uma instabilidade técnica agora. Pode repetir em instantes?"
        novos = self.handler.eventos[n_eventos:]
        t.ferramentas = [e["ferramenta"] for e in novos if e["tipo"] == "agent_action" and e["ferramenta"] != "_Exception"]
        t.erros_formato = sum(1 for e in novos if e["tipo"] == "agent_action" and e["ferramenta"] == "_Exception")
        t.tempo_s = round(time.perf_counter() - inicio, 2)
        self.turnos.append(t)
        return t
