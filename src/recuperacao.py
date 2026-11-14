"""Recuperacao sobre o indice FAISS JA EXISTENTE, com transformacao de consulta.

Estrategias:  direta | reescrita | multi_consulta | hyde | fusao (original + reescrita, por RRF)
O indice e apenas CARREGADO (carregar_indice); a indexacao e feita fora, por scripts/build_index.py.
"""
from typing import List, Optional, Tuple

from src.indice import carregar_indice

PROMPT_REESCRITA = (
    "Reescreva a pergunta do cliente como uma pergunta curta que use o vocabulário formal de uma política de loja "
    "(por exemplo: direito de arrependimento, reembolso, estorno, prazo de entrega, garantia, defeito, cancelamento, "
    "troca, frete). Responda SOMENTE com a pergunta reescrita.\n\nPergunta: {pergunta}"
)
PROMPT_MULTI = (
    "Gere 3 reformulações diferentes da pergunta do cliente, usando sinônimos e o vocabulário formal de políticas de "
    "loja. Responda SOMENTE com as 3 reformulações, uma por linha, sem numeração.\n\nPergunta: {pergunta}"
)
PROMPT_HYDE = (
    "Escreva um trecho curto (2 a 3 frases), no estilo de um documento de política de uma loja virtual, que "
    "responderia à pergunta abaixo. Não explique, apenas escreva o trecho.\n\nPergunta: {pergunta}"
)

ESTRATEGIAS = ("direta", "reescrita", "multi_consulta", "hyde", "fusao")


class Recuperador:
    def __init__(self, llm=None, k: int = 3):
        self.indice = carregar_indice()   # carrega do disco; NAO reconstroi
        self.llm = llm
        self.k = k

    # ---- util -----------------------------------------------------------------
    def _gerar(self, prompt: str) -> Optional[str]:
        if self.llm is None:
            return None
        try:
            texto = self.llm.invoke(prompt).content
        except Exception:
            return None
        if not isinstance(texto, str) or not texto.strip() or texto.startswith("[dummy]"):
            return None
        return texto.strip()

    def _busca(self, texto: str, k: Optional[int] = None) -> List[dict]:
        pares = self.indice.similarity_search_with_score(texto, k=k or self.k)
        return [{"fonte": d.metadata.get("fonte"), "trecho": " ".join(d.page_content.split()), "distancia": float(s)}
                for d, s in pares]

    # ---- estrategias ----------------------------------------------------------
    def direta(self, pergunta: str) -> Tuple[List[dict], str]:
        return self._busca(pergunta), pergunta

    def reescrita(self, pergunta: str) -> Tuple[List[dict], str]:
        nova = self._gerar(PROMPT_REESCRITA.format(pergunta=pergunta)) or pergunta
        return self._busca(nova), nova

    def multi_consulta(self, pergunta: str) -> Tuple[List[dict], str]:
        texto = self._gerar(PROMPT_MULTI.format(pergunta=pergunta)) or ""
        variantes = [pergunta] + [l.strip(" -•\t") for l in texto.splitlines() if l.strip()][:3]
        melhores: dict = {}
        for v in variantes:
            for r in self._busca(v, k=self.k + 2):
                chave = (r["fonte"], r["trecho"])
                if chave not in melhores or r["distancia"] < melhores[chave]["distancia"]:
                    melhores[chave] = r
        ordenados = sorted(melhores.values(), key=lambda r: r["distancia"])[: self.k]
        return ordenados, " || ".join(variantes)

    def hyde(self, pergunta: str) -> Tuple[List[dict], str]:
        hipotetico = self._gerar(PROMPT_HYDE.format(pergunta=pergunta)) or pergunta
        return self._busca(hipotetico), hipotetico

    # ---- Fase 4.1: fusao da consulta original com a reescrita (Reciprocal Rank Fusion) -----------------
    def reescrever(self, pergunta: str) -> str:
        return self._gerar(PROMPT_REESCRITA.format(pergunta=pergunta)) or pergunta

    @staticmethod
    def _rrf(listas: List[List[dict]], c: int = 60) -> List[dict]:
        """Une varias listas ranqueadas: cada trecho soma 1/(c+posicao) por lista em que aparece.
        Em empate, vale a ordem de insercao (a lista da consulta ORIGINAL vem primeiro)."""
        pontos: dict = {}
        melhor: dict = {}
        for lista in listas:
            for posicao, r in enumerate(lista, start=1):
                chave = (r["fonte"], r["trecho"])
                pontos[chave] = pontos.get(chave, 0.0) + 1.0 / (c + posicao)
                if chave not in melhor or r["distancia"] < melhor[chave]["distancia"]:
                    melhor[chave] = r
        ordem = sorted(pontos, key=lambda k: pontos[k], reverse=True)
        return [dict(melhor[k], rrf=round(pontos[k], 5)) for k in ordem]

    def fusao_com(self, pergunta: str, reescrita: str) -> List[dict]:
        kb = self.k + 2
        return self._rrf([self._busca(pergunta, kb), self._busca(reescrita, kb)])[: self.k]

    def fusao(self, pergunta: str) -> Tuple[List[dict], str]:
        nova = self.reescrever(pergunta)
        return self.fusao_com(pergunta, nova), f"{pergunta} || {nova}"

    def buscar(self, pergunta: str, estrategia: str = "direta") -> Tuple[List[dict], str]:
        if estrategia not in ESTRATEGIAS:
            raise ValueError(f"estrategia invalida: {estrategia} (use {ESTRATEGIAS})")
        return getattr(self, estrategia)(pergunta)
