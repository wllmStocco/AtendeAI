"""Memoria de LONGO prazo do AtendeAI (taxonomia CoALA): semantica, episodica e procedural.

Armazenamento: SQLite. Recuperacao: similaridade de cosseno entre embeddings (busca vetorial).
Para algumas dezenas de memorias por cliente, o cosseno em memoria basta; se escalar, trocar por FAISS/pgvector.

Politica de recuperacao:
  - semantica  (perfil: fatos estaveis, ex.: CEP padrao) -> SEMPRE carregada (poucos itens)
  - episodica / procedural                              -> top-k por similaridade com a mensagem
Politica de atualizacao:
  - semantica com `chave` -> UPSERT (um CEP novo substitui o antigo)
  - episodica             -> acrescenta (sem duplicar o mesmo evento no mesmo dia)
"""
import json
import math
import sqlite3
from datetime import date
from pathlib import Path
from typing import Optional

ROOT = Path(__file__).resolve().parent.parent
DB_PADRAO = ROOT / "data" / "memoria_longa.db"
TIPOS = ("semantica", "episodica", "procedural")


def _cosseno(a, b) -> float:
    num = sum(x * y for x, y in zip(a, b))
    den = math.sqrt(sum(x * x for x in a)) * math.sqrt(sum(y * y for y in b))
    return num / den if den else 0.0


class MemoriaLongoPrazo:
    def __init__(self, caminho=DB_PADRAO, embeddings=None):
        self.caminho = Path(caminho)
        self.caminho.parent.mkdir(parents=True, exist_ok=True)
        self._emb = embeddings
        self._con = sqlite3.connect(self.caminho, check_same_thread=False)
        self._con.row_factory = sqlite3.Row
        self._con.executescript(
            """
            CREATE TABLE IF NOT EXISTS memorias (
              id INTEGER PRIMARY KEY AUTOINCREMENT,
              cliente_id INTEGER NOT NULL,
              tipo TEXT NOT NULL,
              chave TEXT,
              texto TEXT NOT NULL,
              criado_em TEXT NOT NULL,
              atualizado_em TEXT NOT NULL,
              vetor TEXT
            );
            CREATE UNIQUE INDEX IF NOT EXISTS ux_memoria_chave
              ON memorias(cliente_id, tipo, chave) WHERE chave IS NOT NULL;
            """
        )

    @property
    def embeddings(self):
        if self._emb is None:
            from src.embeddings import get_embeddings

            self._emb = get_embeddings()
        return self._emb

    def _vetor(self, texto: str) -> str:
        return json.dumps(self.embeddings.embed_query(texto))

    # ---- escrita -----------------------------------------------------------------
    def adicionar(self, cliente_id: int, tipo: str, texto: str, chave: Optional[str] = None) -> int:
        if tipo not in TIPOS:
            raise ValueError(f"tipo invalido: {tipo!r} (use {TIPOS})")
        hoje = date.today().isoformat()
        if chave:  # UPSERT: atualiza o fato existente
            linha = self._con.execute(
                "SELECT id FROM memorias WHERE cliente_id=? AND tipo=? AND chave=?", (cliente_id, tipo, chave)
            ).fetchone()
            if linha:
                self._con.execute(
                    "UPDATE memorias SET texto=?, vetor=?, atualizado_em=? WHERE id=?",
                    (texto, self._vetor(texto), hoje, linha["id"]),
                )
                self._con.commit()
                return linha["id"]
        else:  # evita duplicar o mesmo evento
            linha = self._con.execute(
                "SELECT id FROM memorias WHERE cliente_id=? AND tipo=? AND texto=?", (cliente_id, tipo, texto)
            ).fetchone()
            if linha:
                return linha["id"]
        cur = self._con.execute(
            "INSERT INTO memorias (cliente_id, tipo, chave, texto, criado_em, atualizado_em, vetor) VALUES (?,?,?,?,?,?,?)",
            (cliente_id, tipo, chave, texto, hoje, hoje, self._vetor(texto)),
        )
        self._con.commit()
        return cur.lastrowid

    def limpar(self, cliente_id: int) -> None:
        self._con.execute("DELETE FROM memorias WHERE cliente_id=?", (cliente_id,))
        self._con.commit()

    # ---- leitura -----------------------------------------------------------------
    def listar(self, cliente_id: int, tipo: Optional[str] = None) -> list:
        sql, args = "SELECT * FROM memorias WHERE cliente_id=?", [cliente_id]
        if tipo:
            sql, args = sql + " AND tipo=?", args + [tipo]
        return [dict(r) for r in self._con.execute(sql + " ORDER BY id", args)]

    def recuperar(self, cliente_id: int, consulta: str, k: int = 3, limiar: float = 0.1) -> list:
        """Perfil (semantica) sempre; episodica e procedural por similaridade vetorial com a consulta."""
        memorias = self.listar(cliente_id)
        if not memorias:
            return []
        achados = [m for m in memorias if m["tipo"] == "semantica"]
        candidatas = [m for m in memorias if m["tipo"] != "semantica" and m["vetor"]]
        if candidatas:
            q = self.embeddings.embed_query(consulta)
            pontuadas = []
            for m in candidatas:
                v = json.loads(m["vetor"])
                if len(v) == len(q):  # ignora vetores de outro modelo de embedding
                    pontuadas.append((_cosseno(q, v), m))
            pontuadas.sort(key=lambda p: p[0], reverse=True)
            achados += [m for s, m in pontuadas[:k] if s >= limiar]
        return achados

    def fechar(self) -> None:
        self._con.close()
