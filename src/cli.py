"""AtendeAI no terminal.   Uso:  python -m src.cli --cliente 1 [--trace] [--modo final|chatbot] [--sem-verificacao]

--modo final (padrao): agente integrador da Fase 5 (tool calling + verificacao).  --modo chatbot: Fase 3 (ReAct).
Comandos: /memorias  /estado  /novasessao  /esquecer  /sair
"""
import argparse
import warnings

warnings.filterwarnings("ignore")

from dotenv import load_dotenv  # noqa: E402

load_dotenv()

from src.chatbot import AtendeAIChatbot  # noqa: E402
from src.agente_final import AtendeAIAgente  # noqa: E402
from src.llm_factory import get_llm  # noqa: E402
from src.memoria_longa import MemoriaLongoPrazo  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser(description="Chatbot AtendeAI")
    ap.add_argument("--cliente", type=int, default=1, help="id do cliente (1 a 5)")
    ap.add_argument("--trace", action="store_true", help="mostra o raciocínio ReAct (Thought/Action)")
    ap.add_argument("--modo", choices=["final", "chatbot"], default="final")
    ap.add_argument("--sem-verificacao", action="store_true", help="desliga a verificação de respostas (modo final)")
    ap.add_argument("--quieto", action="store_true", help="esconde a linha de depuração de cada turno")
    args = ap.parse_args()

    llm = get_llm()
    mem = MemoriaLongoPrazo()
    final = args.modo == "final"
    sessao = "sessao-1"

    def criar():
        if final:
            return AtendeAIAgente(llm, args.cliente, memoria_longa=mem, verbose=args.trace, verificar=not args.sem_verificacao)
        return AtendeAIChatbot(llm, args.cliente, memoria_longa=mem, verbose=args.trace)

    bot = criar()
    print(f"\nAtendeAI — Lia, atendente virtual. Cliente: {bot.cliente['nome']}")
    print("Digite sua mensagem. Comandos: /memorias /estado /novasessao /esquecer /sair\n")

    while True:
        try:
            msg = input("Você: ").strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break
        if not msg:
            continue
        if msg == "/sair":
            break
        if msg == "/memorias":
            for m in mem.listar(bot.cliente["id"]):
                print(f"  [{m['tipo']}] {m['texto']}")
            continue
        if msg == "/estado":
            if final:
                est = bot.estado.get(sessao, {})
                print("  slots:", est.get("slots"), "| última intenção:", est.get("ultima"))
            else:
                print("  slots:", bot.slots, "| última intenção:", bot.ultima_intencao)
            continue
        if msg == "/novasessao":
            if final:
                sessao = bot.nova_sessao()
            else:
                bot = criar()
            print("  (nova sessão: memória de curto prazo zerada; memória de longo prazo mantida)")
            continue
        if msg == "/esquecer":
            mem.limpar(bot.cliente["id"])
            print("  (memória de longo prazo deste cliente apagada)")
            continue
        t = bot.responder(msg, sessao) if final else bot.responder(msg)
        print(f"\nLia: {t.resposta}\n")
        if not args.quieto:
            ferr = ", ".join(t.ferramentas) or "nenhuma"
            print(f"  [intenção={t.intencao_efetiva}{' (continuação)' if t.continuacao else ''} | "
                  f"entidades={ {k: v for k, v in t.entidades.items() if v} } | ferramentas={ferr} | "
                  f"memórias usadas={len(t.memorias)} | {t.tempo_s}s]")
            if final and t.verificacao_final:
                v = t.verificacao_final
                print(f"  [verificação: {v['veredito']}{'' if v['coerente'] else ' | INCOERENTE'} | ação: {t.acao} | {v['modo_nli']}]")
                for p in v["problemas"][:3]:
                    print("    -", p)
            if t.resumo_gerado:
                print("  [contexto resumido: conversa antiga virou um resumo]")
            if t.erro:
                print("  [erro técnico]", t.erro)
            print()
    print("Até logo!")


if __name__ == "__main__":
    main()
