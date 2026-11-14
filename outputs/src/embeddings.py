"""Embeddings: OpenAI (modo openai) ou hashing lexical offline (modo dummy)."""
import os
import re
import unicodedata
import zlib

_STOP = {"para", "que", "por", "com", "uma", "dos", "das", "nao", "sao", "ate", "seu", "sua", "ele",
         "como", "mais", "pode", "podem", "sem", "foi", "ser", "tem", "isso", "esse", "essa", "pelo", "pela"}


def tokenizar(texto: str) -> list:
    t = unicodedata.normalize("NFKD", texto.lower())
    t = "".join(c for c in t if not unicodedata.combining(c))
    toks = [w for w in re.findall(r"[a-z0-9]+", t) if len(w) >= 3 and w not in _STOP]
    return [w[:5] for w in toks]  # "stemming" grosseiro: prefixo de 5 letras


def get_embeddings():
    # EMBEDDINGS_MODE=hashing permite usar LLM real sem gastar com embeddings (ou se o provedor nao oferecer embeddings)
    if (os.getenv("EMBEDDINGS_MODE") or os.getenv("LLM_MODE", "dummy")).lower() == "openai":
        from langchain_openai import OpenAIEmbeddings

        base = os.getenv("OPENAI_BASE_URL")
        extra = {"base_url": base, "check_embedding_ctx_length": False} if base else {}
        return OpenAIEmbeddings(model=os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small"), **extra)
    return _hashing()


def _hashing(dim: int = 512):
    import numpy as np
    from langchain_core.embeddings import Embeddings

    class HashingEmbeddings(Embeddings):
        """Bag-of-words com hashing (deterministico, sem rede). NAO e semantico."""

        def _vec(self, texto: str):
            v = np.zeros(dim, dtype="float32")
            for tok in tokenizar(texto):
                v[zlib.crc32(tok.encode()) % dim] += 1.0
            n = np.linalg.norm(v)
            return (v / n if n else v).tolist()

        def embed_documents(self, texts):
            return [self._vec(t) for t in texts]

        def embed_query(self, text):
            return self._vec(text)

    return HashingEmbeddings()
