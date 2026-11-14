"""Microsservico INTERNO de rastreio (HTTP, somente biblioteca padrao).

Simula o servico da transportadora/logistica da loja: o agente o consome por HTTP, com chave interna,
timeout e tratamento de erros, como faria com um microsservico real.

  GET /saude                  -> {"ok": true}
  GET /rastreio/<codigo>      -> eventos do envio (exige o cabecalho X-API-Key)
"""
import json
import os
import sqlite3
import threading
import time
import urllib.request
from datetime import date, timedelta
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent.parent
DB = ROOT / "data" / "loja.db"
HOST = "127.0.0.1"


def porta() -> int:
    return int(os.getenv("RASTREIO_PORTA", "8765"))


def base_url() -> str:
    return os.getenv("RASTREIO_URL", f"http://{HOST}:{porta()}")


def chave() -> str:
    return os.getenv("INTERNAL_API_KEY", "dev-interno")


def _eventos(data_pedido: str, status: str, data_entrega):
    d0 = date.fromisoformat(data_pedido)
    ev = [{"data": (d0 + timedelta(days=1)).isoformat(), "descricao": "Pedido postado na transportadora"},
          {"data": (d0 + timedelta(days=2)).isoformat(), "descricao": "Em trânsito para o centro de distribuição de destino"}]
    if status in ("entregue", "devolvido") and data_entrega:
        ev.append({"data": data_entrega, "descricao": "Entregue ao destinatário"})
    elif status == "enviado":
        ev.append({"data": (d0 + timedelta(days=4)).isoformat(), "descricao": "Saiu para entrega (previsão: em até 2 dias úteis)"})
    return ev


class _Handler(BaseHTTPRequestHandler):
    def log_message(self, *args):  # silencia o log de acesso
        pass

    def _json(self, status: int, corpo: dict):
        dados = json.dumps(corpo, ensure_ascii=False).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", "application/json; charset=utf-8")
        self.send_header("Content-Length", str(len(dados)))
        self.end_headers()
        self.wfile.write(dados)

    def do_GET(self):
        if self.path == "/saude":
            return self._json(200, {"ok": True})
        if self.path.startswith("/rastreio/"):
            if self.headers.get("X-API-Key") != chave():
                return self._json(401, {"erro": "chave de API invalida"})
            codigo = self.path.split("/rastreio/", 1)[1]
            con = sqlite3.connect(f"{DB.resolve().as_uri()}?mode=ro", uri=True)
            try:
                p = con.execute("SELECT data_pedido, status, data_entrega FROM pedidos WHERE codigo_rastreio = ?",
                                (codigo,)).fetchone()
            finally:
                con.close()
            if not p:
                return self._json(404, {"erro": "codigo de rastreio nao encontrado"})
            return self._json(200, {"codigo": codigo, "status": p[1], "eventos": _eventos(p[0], p[1], p[2])})
        return self._json(404, {"erro": "rota inexistente"})


def _saudavel(url: str, espera: float = 0.6) -> bool:
    try:
        with urllib.request.urlopen(url + "/saude", timeout=espera) as r:
            return r.status == 200
    except Exception:
        return False


def garantir_servico() -> str:
    """Sobe o servico numa thread se ainda nao estiver no ar. Devolve a URL base."""
    url = base_url()
    if _saudavel(url):
        return url
    servidor = ThreadingHTTPServer((HOST, porta()), _Handler)
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    for _ in range(20):
        if _saudavel(url):
            break
        time.sleep(0.1)
    return url


if __name__ == "__main__":  # execucao avulsa:  python -m src.servicos.rastreio
    print("Servico de rastreio em", garantir_servico(), "(Ctrl+C para sair)")
    while True:
        time.sleep(3600)
