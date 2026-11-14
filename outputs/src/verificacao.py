"""Verificacao de respostas: consistencia FACTUAL (NLI + checagem numerica) e COERENCIA logica.

Para cada afirmacao da resposta do agente:
  - NLI (premissa = trecho de evidencia devolvido pelas ferramentas, hipotese = afirmacao) -> sustentacao,
    contradicao ou neutralidade (= "nao sustentada");
  - checagem NUMERICA: todo numero citado precisa existir na evidencia, na pergunta ou no contexto
    (modelos de NLI sao fracos com numeros);
  - COERENCIA: pares de afirmacoes da PROPRIA resposta sao comparados por NLI para achar auto-contradicao.
A logica de decisao e pura (testavel sem modelo); o modelo de NLI e injetado.
"""
import os
import re
import unicodedata
from dataclasses import asdict, dataclass, field
from typing import Callable, List, Optional, Tuple

NLIFn = Callable[[List[str], str], List[dict]]   # (premissas, hipotese) -> [{"entailment":p,"neutral":p,"contradiction":p}]

_CONVERSACIONAL = re.compile(
    r"\b(posso (te |lhe )?(ajudar|verificar|consultar|encaminhar|checar)|caso (precise|queira|deseje)|se (precisar|quiser|desejar|preferir)|"
    r"é só (me )?avis|me avise|estou à disposição|fico (à|a) disposição|qualquer (outra )?dúvida|deseja (prosseguir|continuar)|"
    r"gostaria de|obrigad[oa]|disponha|bom dia|boa tarde|boa noite|olá|sou a lia|sou (uma )?atendente|meu nome é|"
    r"preciso d[oae]|pode (me )?informar|poderia (me )?(informar|passar)|informe|me (diga|informe|passe)|"
    r"não (tenho|possuo|consigo|encontrei|há)\b.{0,40}\b(informaç|ferramenta|permiss|dado)|não (posso|consigo|é possível)|"
    r"encaminh\w+.{0,30}atendente|atendente humano|sinto muito|desculp)",
    re.IGNORECASE)
MODELO_NLI_PADRAO = "MoritzLaurer/mDeBERTa-v3-base-mnli-xnli"


# ------------------------------------------------------------------ texto ----------------------------------------
def dividir_sentencas(texto: str) -> List[str]:
    partes = re.split(r"(?<=[.!?])\s+(?=[A-ZÁÉÍÓÚÂÊÔÃÕÇ0-9\"“(])|\n+", (texto or "").strip())
    return [p.strip() for p in partes if p.strip()]


def numeros(texto: str) -> set:
    """Numeros do texto como floats (aceita 19,90 | 249.9 | 12.105,40 | 12.105 | datas, que viram componentes)."""
    achados = set()
    for m in re.findall(r"\d+(?:[.,]\d+)*", texto or ""):
        s = m
        if "," in s and "." in s:
            s = s.replace(".", "").replace(",", ".")
        elif "," in s:
            s = s.replace(",", ".")
        elif re.fullmatch(r"\d{1,3}(\.\d{3})+", s):      # 12.105 -> milhar
            s = s.replace(".", "")
        try:
            achados.add(round(float(s), 4))
        except ValueError:
            pass
    return achados


def eh_conversacional(sentenca: str) -> bool:
    """Cortesia, oferta de ajuda, pedido de dado ou abstencao ("nao tenho essa informacao"): nao e uma afirmacao
    sobre a loja e portanto nao deve ser verificada. Se a frase traz numero, continua sendo verificada."""
    return bool(_CONVERSACIONAL.search(sentenca)) and not re.search(r"\d", sentenca)


def afirmacoes(texto: str) -> List[str]:
    """Sentencas que afirmam algo sobre a loja (ignora perguntas, frases muito curtas e frases conversacionais)."""
    saida = []
    for s in dividir_sentencas(texto):
        palavras = re.findall(r"\w+", s)
        if s.endswith("?") or len(palavras) < 4 or eh_conversacional(s):
            continue
        saida.append(s)
    return saida


def evidencia_limpa(ev: str) -> str:
    """Remove ruido tecnico (SQL executado, [arquivo.md]) antes de mostrar a evidencia ao LLM corretor."""
    ev = re.sub(r"\(SQL executado:.*?\)\s*$", "", str(ev), flags=re.S)
    return re.sub(r"\[\d{2}_[^\]]+\.md\]\s*", "", ev).strip()


def _verbalizar_tabela(texto: str) -> Optional[List[str]]:
    linhas = [l for l in texto.split("\n") if l.strip() and not l.startswith("(SQL executado")]
    if len(linhas) >= 2 and re.fullmatch(r"[A-Za-z_]\w*", linhas[0].strip()) and all(len(l) < 80 for l in linhas):
        return [f"{linhas[0].strip()}={v.strip()}" for v in linhas[1:]]      # tabela de uma coluna
    if len(linhas) >= 2 and " | " in linhas[0]:
        cols = [c.strip() for c in linhas[0].split(" | ")]
        saida = []
        for l in linhas[1:]:
            vals = [v.strip() for v in l.split(" | ")]
            if len(vals) == len(cols):
                saida.append(", ".join(f"{c}={v}" for c, v in zip(cols, vals) if v not in ("", "None")))
        return saida or None
    return None


def montar_passagens(evidencias: List[str], limite: int = 700, entradas: Optional[List[str]] = None) -> List[str]:
    """Transforma as saidas das ferramentas em passagens curtas para o NLI.

    `entradas` (opcional): o que o agente enviou a cada ferramenta (CEP, pergunta sobre o pedido...). A saida de uma
    ferramenta muitas vezes NAO repete o que foi perguntado ("Regiao Sul: prazo de 3 a 6 dias" nao diz que o CEP e
    80010-000), entao a entrada e acrescentada como prefixo de cada passagem."""
    passagens: List[str] = []
    for i, ev in enumerate(evidencias):
        prefixo = ""
        if entradas and i < len(entradas) and str(entradas[i]).strip():
            prefixo = f"Consulta: {str(entradas[i]).strip()[:150]}. Resultado: "
        ev = re.sub(r"\(SQL executado:.*?\)\s*$", "", str(ev), flags=re.S).strip()
        tabela = _verbalizar_tabela(ev)
        if tabela:
            passagens += [prefixo + t for t in tabela]
            continue
        for seg in re.split(r"\s\|\s(?=\[)", ev):          # trechos de politica: "[fonte] texto | [fonte] texto"
            seg = re.sub(r"^\[[^\]]+\]\s*", "", seg).strip()
            if len(seg) > 250 and "; " in seg:               # rastreio: eventos separados por ';'
                passagens += [prefixo + p.strip() for p in seg.split("; ") if p.strip()]
            elif seg:
                passagens.append(prefixo + seg)
    return [p[:limite] for p in passagens]


# ------------------------------------------------------------------ resultado -------------------------------------
@dataclass
class VereditoAfirmacao:
    afirmacao: str
    rotulo: str                       # sustentada | nao_sustentada | contradicao
    p_entailment: float = 0.0
    p_contradicao: float = 0.0
    passagem: str = ""
    numeros_sem_suporte: List[float] = field(default_factory=list)


@dataclass
class Verificacao:
    veredito: str                     # sustentada | parcial | contradicao | sem_evidencia | sem_afirmacoes
    afirmacoes: List[VereditoAfirmacao] = field(default_factory=list)
    coerente: bool = True
    incoerencias: List[Tuple[str, str]] = field(default_factory=list)
    modo_nli: str = ""

    @property
    def problemas(self) -> List[str]:
        out = []
        for a in self.afirmacoes:
            if a.rotulo == "contradicao":
                extra = f" (números sem suporte: {a.numeros_sem_suporte})" if a.numeros_sem_suporte else ""
                out.append(f"CONTRADIZ a evidência: \"{a.afirmacao}\"{extra}")
            elif a.rotulo == "nao_sustentada":
                out.append(f"NÃO confirmada pela evidência: \"{a.afirmacao}\"" if a.passagem or a.p_entailment
                           else f"SEM evidência de nenhuma ferramenta: \"{a.afirmacao}\"")
        for x, y in self.incoerencias:
            out.append(f"INCOERENTE: \"{x}\" x \"{y}\"")
        return out

    def resumo(self) -> dict:
        return {"veredito": self.veredito, "coerente": self.coerente, "n_afirmacoes": len(self.afirmacoes),
                "problemas": self.problemas, "modo_nli": self.modo_nli}


def _coberto_pelo_contexto(afirmacao: str, contexto: str, minimo: float = 0.6) -> bool:
    from src.embeddings import tokenizar

    alvo = set(tokenizar(afirmacao))
    return bool(alvo) and len(alvo & set(tokenizar(contexto))) / len(alvo) >= minimo


def _rotulo_nli(probs: dict) -> str:
    return max(("entailment", "neutral", "contradiction"), key=lambda k: probs.get(k, 0.0))


def verificar(resposta: str, evidencias: List[str], nli: NLIFn, contexto_extra: str = "", modo: str = "",
              tau_ent: float = 0.5, tau_con: float = 0.5, tau_incoerencia: float = 0.6,
              max_pares: int = 6, entradas: Optional[List[str]] = None) -> Verificacao:
    afs = afirmacoes(resposta)
    passagens = montar_passagens(evidencias, entradas=entradas)
    permitidos = numeros(" ".join(str(e) for e in evidencias) + " " + contexto_extra
                         + " " + " ".join(str(x) for x in (entradas or [])))

    resultados: List[VereditoAfirmacao] = []
    for a in afs:
        sem_suporte = sorted(n for n in numeros(a) if n not in permitidos)
        if passagens:
            probs = nli(passagens, a)
            i_ent = max(range(len(probs)), key=lambda i: probs[i].get("entailment", 0.0))
            i_con = max(range(len(probs)), key=lambda i: probs[i].get("contradiction", 0.0))
            p_ent, p_con = probs[i_ent].get("entailment", 0.0), probs[i_con].get("contradiction", 0.0)
            melhor = passagens[i_con] if (p_con >= tau_con and p_con > p_ent) else passagens[i_ent]
        else:
            p_ent = p_con = 0.0
            melhor = ""
        if sem_suporte:
            rotulo = "contradicao"
        elif not passagens:
            # sem ferramenta: so vale o que vem do proprio contexto da conversa (mensagem, slots, memorias do cliente)
            rotulo = "sustentada" if _coberto_pelo_contexto(a, contexto_extra) else "nao_sustentada"
        elif p_con >= tau_con and p_con > p_ent:
            rotulo = "contradicao"
        elif p_ent >= tau_ent:
            rotulo = "sustentada"
        else:
            rotulo = "nao_sustentada"
        resultados.append(VereditoAfirmacao(a, rotulo, round(p_ent, 3), round(p_con, 3), melhor, sem_suporte))

    # coerencia logica interna: afirmacoes da propria resposta nao podem se contradizer
    incoerencias: List[Tuple[str, str]] = []
    base = afirmacoes(resposta)[:max_pares]
    for i in range(len(base)):
        for j in range(i + 1, len(base)):
            if nli([base[i]], base[j])[0].get("contradiction", 0.0) >= tau_incoerencia:
                incoerencias.append((base[i], base[j]))

    if not resultados:
        veredito = "sem_afirmacoes"
    elif any(r.rotulo == "contradicao" for r in resultados):
        veredito = "contradicao"
    elif any(r.rotulo == "nao_sustentada" for r in resultados):
        veredito = "sem_evidencia" if not passagens else "parcial"
    else:
        veredito = "sustentada"
    return Verificacao(veredito, resultados, not incoerencias, incoerencias, modo)


# ------------------------------------------------------------------ backends de NLI ------------------------------
def nli_lexical(premissas: List[str], hipotese: str) -> List[dict]:
    """Substituto SEM modelo (modo dummy): sobreposicao de termos. Nao detecta contradicao; so p/ reproducao."""
    from src.embeddings import tokenizar

    h = set(tokenizar(hipotese))
    saida = []
    for p in premissas:
        ent = len(h & set(tokenizar(p))) / len(h) if h else 0.0
        saida.append({"entailment": ent, "neutral": 1.0 - ent, "contradiction": 0.0})
    return saida


_CACHE: dict = {}


def _nli_transformers(nome: str) -> NLIFn:
    if nome in _CACHE:
        return _CACHE[nome]
    import torch
    from transformers import AutoModelForSequenceClassification, AutoTokenizer

    tok = AutoTokenizer.from_pretrained(nome)
    modelo = AutoModelForSequenceClassification.from_pretrained(nome).eval()
    mapa = {}
    for i, rot in modelo.config.id2label.items():
        r = rot.lower()
        mapa[int(i)] = "entailment" if r.startswith("entail") else "contradiction" if r.startswith("contra") else "neutral"

    def f(premissas: List[str], hipotese: str) -> List[dict]:
        enc = tok(list(premissas), [hipotese] * len(premissas), truncation=True, max_length=512,
                  padding=True, return_tensors="pt")
        with torch.no_grad():
            probs = torch.softmax(modelo(**enc).logits, dim=-1)
        return [{mapa[i]: float(p[i]) for i in mapa} for p in probs]

    _CACHE[nome] = f
    return f


def criar_nli(modo: Optional[str] = None) -> Tuple[NLIFn, str]:
    """NLI_MODE=auto (padrao: modelo se houver transformers/torch, senao lexical) | modelo | lexical."""
    modo = (modo or os.getenv("NLI_MODE", "auto")).lower()
    nome = os.getenv("NLI_MODEL", MODELO_NLI_PADRAO)
    if modo in ("auto", "modelo"):
        try:
            return _nli_transformers(nome), f"modelo NLI: {nome}"
        except Exception as e:
            if modo == "modelo":
                raise
            print(f"[verificacao] modelo NLI indisponivel ({type(e).__name__}); usando o substituto lexical.")
    return nli_lexical, "lexical (sem modelo NLI; so para reproducao)"
