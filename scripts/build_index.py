"""Constroi o indice FAISS das politicas.  Uso: python scripts/build_index.py [--forcar]"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from dotenv import load_dotenv

load_dotenv()

from src.indice import construir_indice  # noqa: E402

if __name__ == "__main__":
    print("Indice em:", construir_indice(forcar="--forcar" in sys.argv))
