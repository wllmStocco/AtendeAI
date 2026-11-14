"""Regras de negocio puras (sem LangChain): CEP, prazo e frete (ferramenta HARDCODED)."""
import re
from typing import Optional

FRETE_GRATIS_ACIMA = 199.00

# primeiro digito do CEP -> (regiao, prazo_min, prazo_max, frete)  [ver data/politicas/03_frete_e_cep.md]
_REGIOES = {
    **{d: ("Sudeste", 2, 4, 14.90) for d in "0123"},
    **{d: ("Sul", 3, 6, 19.90) for d in "89"},
    "7": ("Centro-Oeste", 4, 7, 24.90),
    **{d: ("Nordeste", 5, 9, 29.90) for d in "45"},
    "6": ("Norte", 7, 12, 39.90),
}

# usado apenas se a rede/ViaCEP falhar ou VIACEP_OFFLINE=1
CEP_OFFLINE = {"01310-100": "Avenida Paulista, Bela Vista, São Paulo/SP"}


def _brl(v: float) -> str:
    return f"R$ {v:.2f}".replace(".", ",")


def normalizar_cep(texto: str) -> Optional[str]:
    digitos = re.sub(r"\D", "", texto or "")
    if len(digitos) != 8:
        return None
    return f"{digitos[:5]}-{digitos[5:]}"


def calcular_prazo_frete(entrada: str) -> str:
    """Entrada: 'CEP' ou 'CEP;valor_do_pedido'."""
    partes = (entrada or "").split(";")
    cep = normalizar_cep(partes[0])
    if not cep:
        return "CEP inválido: informe 8 dígitos (ex.: 01310-100)."
    valor = None
    if len(partes) > 1 and partes[1].strip():
        try:
            valor = float(partes[1].strip().replace(",", "."))
        except ValueError:
            return "Valor do pedido inválido. Use um número, ex.: 150.00."
    regiao, pmin, pmax, frete = _REGIOES[cep[0]]
    texto = f"Região {regiao}: prazo de {pmin} a {pmax} dias úteis."
    if valor is None:
        texto += f" Frete padrão de {_brl(frete)} (grátis em pedidos a partir de {_brl(FRETE_GRATIS_ACIMA)})."
    elif valor >= FRETE_GRATIS_ACIMA:
        texto += f" Frete grátis (pedido de {_brl(valor)})."
    else:
        texto += f" Frete de {_brl(frete)} (grátis a partir de {_brl(FRETE_GRATIS_ACIMA)})."
    return texto + " (fonte: política de frete)"
