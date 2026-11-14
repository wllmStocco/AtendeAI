"""Parsing robusto da saida estruturada: tenta o parser, depois extrai o JSON do texto, depois cai em fallback."""
import re
from typing import Optional


def extrair_json(texto: str) -> Optional[str]:
    """Extrai o primeiro bloco {...} balanceado de um texto (aceita cercas ```json e texto extra)."""
    if not texto:
        return None
    sem_cerca = re.sub(r"```(?:json)?", "", texto)
    ini = sem_cerca.find("{")
    if ini < 0:
        return None
    nivel, em_str, esc = 0, False, False
    for i in range(ini, len(sem_cerca)):
        c = sem_cerca[i]
        if em_str:
            if esc:
                esc = False
            elif c == "\\":
                esc = True
            elif c == '"':
                em_str = False
            continue
        if c == '"':
            em_str = True
        elif c == "{":
            nivel += 1
        elif c == "}":
            nivel -= 1
            if nivel == 0:
                return sem_cerca[ini:i + 1]
    return None


def parse_seguro(texto: str, parser):
    """1) parser direto  2) JSON extraido do texto  3) fallback seguro (nunca levanta excecao)."""
    from langchain_core.exceptions import OutputParserException

    from src.schemas import AtendimentoOutput

    try:
        return parser.parse(texto)
    except OutputParserException:
        pass
    bloco = extrair_json(texto)
    if bloco:
        try:
            return parser.parse(bloco)
        except OutputParserException:
            pass
    return AtendimentoOutput(
        intencao="outro", entidades={}, requer_ferramenta=False,
        resposta="Não consegui interpretar a mensagem. Pode reformular?", confianca=0.0,
    )
