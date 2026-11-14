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
- `docs/uso_de_ia.md` — citação do uso de IA (exigida pelo enunciado)

## Status
- [x] Fase 1 — MVP (chain com saída estruturada)
- [x] Fase 2 — Agente ReAct (notebooks/02_fase2_agente_react.ipynb)
- [x] Fase 3 — Memória e chatbot (notebooks/03_fase3_memoria_chatbot.ipynb; terminal: `python -m src.cli --cliente 1`)
- [x] Fase 4 — Ferramentas externas e RAG (notebooks/04_fase4_ferramentas_rag.ipynb)
- [x] Fase 4.1 — ajustes medidos (notebooks/04b_fase4_1_ajustes_medidos.ipynb)
- [x] Fase 5 — Integrador, verificação e LangSmith (notebooks/05_fase5_agente_integrador.ipynb; terminal: `python -m src.cli --cliente 1 --trace`)
