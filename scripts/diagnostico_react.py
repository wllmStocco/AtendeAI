"""Diagnostico do formato ReAct com o LLM real: compara o prompt COM e SEM a regra "uma ação por vez".

Uso:  python scripts/diagnostico_react.py        (gasta ~20 chamadas de LLM)
"""
import sys
import warnings
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src import prompts_react  # noqa: E402
from src.agente import criar_agente, criar_memoria  # noqa: E402
from src.callbacks import LogHandler  # noqa: E402
from src.llm_factory import get_llm  # noqa: E402
from src.tools import FERRAMENTAS  # noqa: E402

PERGUNTAS = ["Qual o prazo e o frete para o CEP 80010-000?", "Posso devolver um produto depois de receber?"]
MARCA = "- Escreva UMA ação"
PREFIX_COM = prompts_react.PREFIX
PREFIX_SEM = "\n".join(l for l in PREFIX_COM.splitlines() if not l.startswith(MARCA))


def rodar(nome: str, prefix: str):
    h = LogHandler()
    agente = criar_agente(get_llm(), FERRAMENTAS, memoria=criar_memoria(), verbose=False, prefix=prefix)
    for p in PERGUNTAS:
        r = agente.invoke({"input": p}, config={"callbacks": [h]})
        print(f"  [{nome}] {p}\n      -> {r['output'][:110]!r}")
    chamadas = [e for e in h.eventos if e["tipo"] == "llm_end"]
    vazias = [e for e in chamadas if e["saida"] in ("''", '""')]
    print(f"  [{nome}] chamadas LLM={len(chamadas)} | erros de formato={h.erros_de_formato()} | respostas vazias={len(vazias)}")
    for e in vazias[:2]:
        print(f"      exemplo de resposta vazia: finish_reason={e['finish_reason']}")
    return h


if __name__ == "__main__":
    print("A regra 'uma ação por vez' está no seu prompts_react.py?", MARCA in PREFIX_COM)
    print("\n--- Variante COM a regra (prompt atual) ---")
    rodar("com regra", PREFIX_COM)
    print("\n--- Variante SEM a regra ---")
    rodar("sem regra", PREFIX_SEM)
