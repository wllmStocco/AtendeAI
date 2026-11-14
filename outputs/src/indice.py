"""Indice FAISS das politicas da loja (um diretorio por modo, pois a dimensao dos vetores muda)."""
import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
POLITICAS = ROOT / "data" / "politicas"


def indice_dir() -> Path:
    modo = (os.getenv("EMBEDDINGS_MODE") or os.getenv("LLM_MODE", "dummy")).lower()
    return ROOT / "data" / f"indice_faiss_{modo}"


def construir_indice(forcar: bool = False) -> Path:
    """Script de PREPARACAO (fora do agente). O agente apenas carrega o indice (carregar_indice)."""
    from langchain_community.vectorstores import FAISS
    from langchain_core.documents import Document
    from langchain_text_splitters import RecursiveCharacterTextSplitter

    from src.embeddings import get_embeddings

    destino = indice_dir()
    if destino.exists() and not forcar:
        return destino
    docs = [Document(page_content=p.read_text(encoding="utf-8"), metadata={"fonte": p.name})
            for p in sorted(POLITICAS.glob("*.md"))]
    pedacos = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50).split_documents(docs)
    FAISS.from_documents(pedacos, get_embeddings()).save_local(str(destino))
    return destino


def carregar_indice():
    from langchain_community.vectorstores import FAISS

    from src.embeddings import get_embeddings

    destino = indice_dir()
    if not destino.exists():
        raise FileNotFoundError(f"Indice nao encontrado em {destino}. Rode: python scripts/build_index.py")
    return FAISS.load_local(str(destino), get_embeddings(), allow_dangerous_deserialization=True)
