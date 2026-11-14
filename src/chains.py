"""Chain de intencao (MVP da Fase 1, versao reutilizavel) com parsing robusto da Fase 2."""
from langchain_core.output_parsers import PydanticOutputParser, StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.runnables import RunnableLambda

from src.parsing import parse_seguro
from src.schemas import AtendimentoOutput

SYSTEM = """Você é um atendente de pós-venda de e-commerce.
Analise a mensagem do cliente e responda SOMENTE com um objeto no esquema JSON abaixo, sem texto extra.
- intencao: uma de status_pedido, troca_devolucao, prazo_frete, cancelamento, saudacao, outro
- entidades: numero_pedido, cep, produto (use null quando ausente)
- requer_ferramenta: true se for preciso consultar pedido, política ou CEP para responder
- resposta: resposta curta e cordial; não invente políticas, prazos ou dados
- confianca: número de 0 a 1

{format_instructions}"""


def build_chain_intencao(llm):
    parser = PydanticOutputParser(pydantic_object=AtendimentoOutput)
    prompt = ChatPromptTemplate.from_messages([("system", SYSTEM), ("human", "{mensagem}")]).partial(
        format_instructions=parser.get_format_instructions()
    )
    return prompt | llm | StrOutputParser() | RunnableLambda(lambda t: parse_seguro(t, parser))
