"""Schemas de saida estruturada do AtendeAI (Fase 1)."""
from typing import Literal, Optional

from pydantic import BaseModel, Field

Intencao = Literal[
    "status_pedido",
    "troca_devolucao",
    "prazo_frete",
    "cancelamento",
    "saudacao",
    "outro",
]


class Entidades(BaseModel):
    numero_pedido: Optional[str] = Field(None, description="Numero do pedido citado pelo cliente")
    cep: Optional[str] = Field(None, description="CEP citado pelo cliente, no formato 00000-000")
    produto: Optional[str] = Field(None, description="Produto citado pelo cliente")


class AtendimentoOutput(BaseModel):
    intencao: Intencao = Field(description="Intencao principal do cliente")
    entidades: Entidades = Field(default_factory=Entidades)
    requer_ferramenta: bool = Field(
        description="True se for preciso consultar pedido, politica ou CEP para responder"
    )
    resposta: str = Field(description="Resposta curta e cordial ao cliente; nao inventar politicas")
    confianca: float = Field(ge=0, le=1, description="Confianca da classificacao, de 0 a 1")
