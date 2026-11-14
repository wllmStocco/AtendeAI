"""Fabrica do agente ReAct classico (initialize_agent + ZERO_SHOT_REACT_DESCRIPTION) com memoria."""
from langchain.agents import AgentType, initialize_agent
from langchain.memory import ConversationBufferMemory, ConversationBufferWindowMemory

from src.prompts_react import PREFIX, SUFFIX


def criar_memoria(k: int = 6, tipo: str = "janela"):
    """tipo='janela' (Fase 2): ultimas k trocas, sem LLM para resumir.
    tipo='buffer'  (Fase 3): ConversationBufferMemory completo; o GerenciadorContexto resume quando crescer."""
    if tipo == "buffer":
        return ConversationBufferMemory(memory_key="chat_history")
    return ConversationBufferWindowMemory(k=k, memory_key="chat_history")


def criar_agente(llm, ferramentas, memoria=None, callbacks=None, verbose=True, prefix=None):
    return initialize_agent(
        ferramentas,
        llm,
        agent=AgentType.ZERO_SHOT_REACT_DESCRIPTION,
        memory=memoria,
        verbose=verbose,
        handle_parsing_errors=True,
        max_iterations=6,
        callbacks=callbacks,
        agent_kwargs={
            "prefix": prefix or PREFIX,
            "suffix": SUFFIX,
            "input_variables": ["input", "chat_history", "agent_scratchpad"],
        },
    )
