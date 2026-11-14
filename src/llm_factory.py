"""Escolhe o LLM conforme LLM_MODE (dummy | openai).

No modo `openai`, qualquer endpoint compativel com a API da OpenAI funciona:
defina OPENAI_BASE_URL (ex.: o endpoint de compatibilidade do Gemini) alem de OPENAI_API_KEY e OPENAI_MODEL.
Se o endpoint for do Google (googleapis.com), usa o adaptador que devolve as thought_signatures do Gemini 3.
"""
import os


def get_llm(temperature: float = 0.0):
    modo = os.getenv("LLM_MODE", "dummy").lower()
    if modo == "openai":
        extra = {}
        base = os.getenv("OPENAI_BASE_URL")
        if base:
            extra["base_url"] = base
        if base and "googleapis.com" in base:
            from src.llm_gemini import ChatOpenAIGemini as Classe
        else:
            from langchain_openai import ChatOpenAI as Classe
        return Classe(
            model=os.getenv("OPENAI_MODEL", "gpt-4o-mini"),
            temperature=temperature,
            max_retries=int(os.getenv("OPENAI_MAX_RETRIES", "8")),  # espera e tenta de novo em 429/5xx
            **extra,
        )
    from src.dummy_llm import DummyChatModel

    return DummyChatModel()
