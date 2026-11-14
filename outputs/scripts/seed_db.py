"""Gera o SQLite sintetico de pedidos (determinístico, SEED 42).

Uso: python scripts/seed_db.py   ->  data/loja.db
"""
import random
import sqlite3
from datetime import date, timedelta
from pathlib import Path

HOJE = date(2026, 11, 7)  # data de referencia fixa, para reprodutibilidade
DB = Path(__file__).resolve().parent.parent / "data" / "loja.db"

SCHEMA = """
DROP TABLE IF EXISTS itens_pedido; DROP TABLE IF EXISTS pedidos; DROP TABLE IF EXISTS clientes;
CREATE TABLE clientes (id INTEGER PRIMARY KEY, nome TEXT NOT NULL, email TEXT NOT NULL, cep TEXT NOT NULL);
CREATE TABLE pedidos (
  id INTEGER PRIMARY KEY, cliente_id INTEGER NOT NULL REFERENCES clientes(id),
  data_pedido TEXT NOT NULL,
  status TEXT NOT NULL CHECK (status IN ('aguardando_pagamento','em_separacao','enviado','entregue','cancelado','devolvido')),
  valor_total REAL NOT NULL, data_entrega TEXT, codigo_rastreio TEXT);
CREATE TABLE itens_pedido (
  id INTEGER PRIMARY KEY AUTOINCREMENT, pedido_id INTEGER NOT NULL REFERENCES pedidos(id),
  produto TEXT NOT NULL, quantidade INTEGER NOT NULL, preco_unit REAL NOT NULL);
"""

CLIENTES = [
    (1, "Ana Souza", "ana@exemplo.com", "01310-100"),
    (2, "Bruno Lima", "bruno@exemplo.com", "30140-071"),
    (3, "Carla Mendes", "carla@exemplo.com", "80010-000"),
    (4, "Diego Alves", "diego@exemplo.com", "40020-000"),
    (5, "Elisa Rocha", "elisa@exemplo.com", "70040-010"),
]
PRODUTOS = [("Tênis Corrida X", 299.9), ("Camiseta Básica", 59.9), ("Fone Bluetooth", 189.0),
            ("Mochila Urbana", 159.9), ("Notebook 14", 3499.0), ("Jaqueta Corta-Vento", 249.9)]


def main():
    rnd = random.Random(42)
    DB.parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(DB)
    con.executescript(SCHEMA)
    con.executemany("INSERT INTO clientes VALUES (?,?,?,?)", CLIENTES)
    for i in range(20):
        pid = 10001 + i
        dias = rnd.randint(1, 45)
        d_ped = HOJE - timedelta(days=dias)
        status = rnd.choice(["em_separacao", "enviado", "entregue", "entregue", "cancelado", "devolvido"])
        if dias < 3:
            status = rnd.choice(["aguardando_pagamento", "em_separacao"])
        elif dias > 7 and status == "em_separacao":
            status = "entregue"  # pedido antigo nao fica parado em separacao
        itens = rnd.sample(PRODUTOS, rnd.randint(1, 2))
        total = round(sum(p for _, p in itens), 2)
        d_ent = rastreio = None
        if status in ("enviado", "entregue", "devolvido"):
            rastreio = f"BR{rnd.randint(10**8, 10**9 - 1)}XX"
        if status in ("entregue", "devolvido"):
            d_ent = (d_ped + timedelta(days=rnd.randint(3, 8))).isoformat()
        con.execute("INSERT INTO pedidos VALUES (?,?,?,?,?,?,?)",
                    (pid, rnd.randint(1, 5), d_ped.isoformat(), status, total, d_ent, rastreio))
        for nome, preco in itens:
            con.execute("INSERT INTO itens_pedido (pedido_id, produto, quantidade, preco_unit) VALUES (?,?,?,?)",
                        (pid, nome, 1, preco))
    con.commit()
    n = con.execute("SELECT COUNT(*) FROM pedidos").fetchone()[0]
    print(f"{DB} criado com {n} pedidos")
    con.close()


if __name__ == "__main__":
    main()
