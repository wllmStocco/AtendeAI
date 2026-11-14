"""Interface Streamlit do AtendeAI (agente integrador da Fase 5).

Uso:   streamlit run app_streamlit.py
Mostra, a cada resposta, COMO a Lia chegou a ela: intencao, ferramentas chamadas (trajetoria), memorias usadas e o
resultado da verificacao (NLI + checagem numerica), inclusive quando a resposta foi corrigida.
"""
import os
import sqlite3
import sys
import uuid
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import streamlit as st
from dotenv import load_dotenv

load_dotenv(ROOT / ".env")

from src.ui_formatacao import acao_texto, entidades_texto, linhas_passos, trajetoria_texto, verificacao_texto

st.set_page_config(page_title="AtendeAI", page_icon="🛍️", layout="wide")

EXEMPLOS = [
    "Oi! Meu CEP é 80010-000. Qual o prazo e o frete?",
    "Onde está o meu pedido 10001? Quero ver o rastreio.",
    "E quais itens estavam nesse pedido?",
    "Posso devolver um produto depois de receber?",
    "Vocês vendem cartão presente?",
    "Qual o prazo de entrega para o meu CEP de sempre?",
]


# ----------------------------------------------------------------------------- recursos em cache ----------------
@st.cache_resource(show_spinner="Carregando o modelo de linguagem...")
def carregar_llm():
    from src.llm_factory import get_llm
    return get_llm()


@st.cache_resource(show_spinner="Carregando o verificador de respostas (NLI), pode demorar na primeira vez...")
def carregar_nli():
    from src.verificacao import criar_nli
    return criar_nli()


@st.cache_resource
def carregar_memoria():
    from src.memoria_longa import MemoriaLongoPrazo
    return MemoriaLongoPrazo()


@st.cache_data
def listar_clientes():
    db = ROOT / "data" / "loja.db"
    if not db.exists():
        return {1: "Cliente 1"}
    con = sqlite3.connect(db)
    try:
        return {r[0]: r[1] for r in con.execute("SELECT id, nome FROM clientes ORDER BY id")}
    finally:
        con.close()


# ----------------------------------------------------------------------------- estado da sessao -----------------
def iniciar_sessao(cliente_id: int) -> None:
    st.session_state.cliente_id = cliente_id
    st.session_state.session_id = f"st-{uuid.uuid4().hex[:6]}"
    st.session_state.mensagens = []
    st.session_state.agente = None


def obter_agente(verificar: bool):
    if st.session_state.agente is None:
        from src.agente_final import AtendeAIAgente
        st.session_state.agente = AtendeAIAgente(
            carregar_llm(), st.session_state.cliente_id, memoria_longa=carregar_memoria(),
            nli=carregar_nli() if verificar else None, verificar=verificar)
    st.session_state.agente.verificar_ativo = verificar
    if verificar and st.session_state.agente._nli is None:
        st.session_state.agente._nli = carregar_nli()
    return st.session_state.agente


def desenhar_detalhes(t) -> None:
    st.markdown(f"**Intenção:** `{t.intencao_efetiva}`{' (continuação do assunto)' if t.continuacao else ''}"
                f"  ·  **Entidades:** {entidades_texto(t.entidades)}")
    st.markdown(f"**Trajetória:** {trajetoria_texto(t.ferramentas)}")
    if t.passos:
        st.table(linhas_passos(t.passos))
    if t.memorias:
        st.markdown("**Memórias do cliente usadas:** " + " | ".join(t.memorias))
    st.markdown(f"**Verificação do rascunho:** {verificacao_texto(t.verificacao_rascunho)}")
    if t.acao in ("regenerada", "resposta_segura"):
        st.warning(f"{acao_texto(t.acao)}. Rascunho original: “{t.resposta_rascunho}”")
        for p in (t.verificacao_rascunho or {}).get("problemas", [])[:3]:
            st.caption("• " + p)
    elif t.verificacao_rascunho and t.verificacao_rascunho.get("problemas"):
        for p in t.verificacao_rascunho["problemas"][:3]:
            st.caption("• " + p)
    st.caption(f"Ação: {acao_texto(t.acao)} · {t.tempo_s} s")
    if t.erro:
        st.error(t.erro)


# ----------------------------------------------------------------------------- barra lateral ------------------
clientes = listar_clientes()
if "cliente_id" not in st.session_state:
    iniciar_sessao(next(iter(clientes)))

with st.sidebar:
    st.header("🛍️ AtendeAI")
    escolhido = st.selectbox("Cliente", list(clientes), index=list(clientes).index(st.session_state.cliente_id),
                             format_func=lambda i: f"{i} · {clientes[i]}")
    if escolhido != st.session_state.cliente_id:
        iniciar_sessao(escolhido)
        st.rerun()
    verificar = st.toggle("Verificação de respostas (NLI)", value=True,
                          help="Desligue para ver o agente sem a camada de verificação.")
    detalhes = st.toggle("Mostrar como a Lia chegou à resposta", value=True)
    if st.button("🔄 Nova sessão (mantém a memória de longo prazo)", use_container_width=True):
        iniciar_sessao(st.session_state.cliente_id)
        st.rerun()
    st.divider()
    st.caption("Mensagens de exemplo (clique para enviar)")
    for exemplo in EXEMPLOS:
        if st.button(exemplo, key=f"ex-{exemplo}", use_container_width=True):
            st.session_state.pendente = exemplo
    st.divider()
    modo = os.getenv("LLM_MODE", "dummy")
    st.caption(f"LLM: `{os.getenv('OPENAI_MODEL', '—') if modo == 'openai' else 'dummy (sem LLM real)'}`")
    st.caption(f"Sessão: `{st.session_state.session_id}`")

# ----------------------------------------------------------------------------- conversa ----------------------
st.title("Lia, atendente virtual da Loja Exemplo")
st.caption(f"Atendendo: **{clientes[st.session_state.cliente_id]}**. Peça status de pedido, rastreio, prazo e frete por CEP, ou políticas da loja.")

for m in st.session_state.mensagens:
    with st.chat_message(m["role"]):
        st.markdown(m["content"])
        if m.get("turno") is not None and detalhes:
            with st.expander("Como a Lia chegou a essa resposta"):
                desenhar_detalhes(m["turno"])

entrada = st.chat_input("Escreva sua mensagem...") or st.session_state.pop("pendente", None)
if entrada:
    with st.chat_message("user"):
        st.markdown(entrada)
    with st.chat_message("assistant"):
        try:
            agente = obter_agente(verificar)
            with st.spinner("A Lia está consultando as ferramentas e verificando a resposta..."):
                turno = agente.responder(entrada, st.session_state.session_id)
            st.markdown(turno.resposta)
            if detalhes:
                with st.expander("Como a Lia chegou a essa resposta", expanded=True):
                    desenhar_detalhes(turno)
        except Exception as e:  # mostra o erro na tela em vez de derrubar o app
            turno = None
            st.error(f"Não foi possível iniciar o agente: {type(e).__name__}: {e}")
    st.session_state.mensagens.append({"role": "user", "content": entrada})
    if turno is not None:
        st.session_state.mensagens.append({"role": "assistant", "content": turno.resposta, "turno": turno})

# ----------------------------------------------------------------------------- estado e memorias (barra lateral)
with st.sidebar:
    with st.expander("Estado da sessão e memórias"):
        ag = st.session_state.agente
        est = (ag.estado.get(st.session_state.session_id, {}) if ag else {})
        st.markdown("**Slots:** " + (entidades_texto(est.get("slots", {})) if est else "—"))
        st.markdown(f"**Última intenção:** {est.get('ultima') or '—'}")
        st.markdown("**Memória de longo prazo:**")
        itens = carregar_memoria().listar(st.session_state.cliente_id)
        if itens:
            for i in itens:
                st.caption(f"[{i['tipo']}] {i['texto']}")
        else:
            st.caption("(vazia)")
        if st.button("Esquecer a memória de longo prazo deste cliente"):
            carregar_memoria().limpar(st.session_state.cliente_id)
            st.rerun()
