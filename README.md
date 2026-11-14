# AtendeAI — Projeto de Disciplina (AI Agents com LangChain)

Atendente virtual de pós-venda de e-commerce. Evolui por fases até um agente único com SQL, RAG (FAISS), API externa (ViaCEP), memória e avaliação no LangSmith.

## Como rodar (modo dummy, sem credenciais)
```
python -m venv .venv && source .venv/bin/activate     # Windows: .venv\Scripts\activate
pip install -r requirements.txt
cp .env.example .env                                   # LLM_MODE=dummy já é o padrão
python scripts/seed_db.py                              # cria data/loja.db
python scripts/build_index.py                          # cria o índice FAISS das políticas (data/indice_faiss_<modo>)
jupyter notebook notebooks/01_fase1_mvp_chain.ipynb
```
Para usar um LLM real: `LLM_MODE=openai`, `OPENAI_API_KEY=...` no `.env`.

## Interface web (Streamlit)
```
pip install streamlit
streamlit run app_streamlit.py
```
Mostra, a cada resposta, a intenção, as ferramentas chamadas, as memórias usadas e o veredito da verificação.

## Estrutura
- `notebooks/` — um notebook por fase
- `src/` — schemas, LLM (real e dummy), ferramentas, prompt PTCF, agente, callbacks, parsing, índice FAISS
- `scripts/seed_db.py` — gera o SQLite sintético (SEED 42)
- `src/servicos/rastreio.py` — microsserviço interno de rastreio (HTTP); `src/sql_seguro.py` — camada segura do text-to-query
- `app_streamlit.py` — interface web; `src/ui_formatacao.py` — formatação pura para a interface
- `src/agente_final.py` — agente integrador (tool calling + histórico + verificação); `src/verificacao.py` — NLI, checagem numérica e coerência
- `data/politicas/` — 6 documentos de política (base do índice FAISS, Fases 2 e 4)