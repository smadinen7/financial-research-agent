# Financial Research Agent — Task Tracker

## Phase 1: Foundation ✅
- [x] Create `requirements.txt` (all deps: langgraph, langchain, faiss, edgartools, ragas, deepeval, streamlit)
- [x] Create `.env.example` (Gemini default, OpenAI + Claude optional, eval LLM separate)
- [x] Create `src/llm.py` — multi-provider LLM factory (gemini | openai | claude)
- [x] Verify: import, bad-provider error, missing-key error all behave correctly
- [x] Commit + push `feature/foundation`

## Phase 2: Ingest ✅
- [x] `src/ingest.py` — download 10-K/10-Q for AAPL/MSFT/NVDA via `edgartools`
- [x] Section-aware chunking (Item 1A, Item 7, Item 8 for 10-K; Part I/II items for 10-Q)
- [x] Prepend metadata header to each chunk: `[Company | Form | Period | Section]`
- [x] Save per-ticker JSON + consolidated `index.json` to `data/filings/`
- [x] `.gitignore` added (excludes data/filings/, .env, evals/results/)
- [x] Structural tests: chunk IDs unique, metadata headers correct, error paths clean
- [ ] Live smoke test: `python src/ingest.py --tickers AAPL --filing-types 10-K` (needs packages installed + SEC_EDGAR_USER_AGENT set)
- [ ] Commit + push + merge `feature/ingest` PR

## Phase 3: Retriever [ ]
- [ ] `src/retriever.py` — FAISS dense index (`bge-large-en-v1.5` embeddings)
- [ ] BM25 sparse index (`rank-bm25`)
- [ ] RRF (Reciprocal Rank Fusion) hybrid merge
- [ ] Cross-encoder rerank with `bge-reranker-v2-gemma`
- [ ] Smoke test: query returns ranked chunks with scores
- [ ] Commit + push `feature/retriever`

## Phase 4: Agent Graph [ ]
- [ ] `src/nodes.py` — Planner, RetrievalEvaluator (CRAG), Analyst, Synthesizer node implementations
- [ ] `src/graph.py` — TypedDict `AgentState`, LangGraph wiring, conditional re-retrieval loop
- [ ] Conditional edge: confidence < 0.7 && iterations < 3 → back to Retriever
- [ ] Human-in-the-loop checkpoint before Synthesizer
- [ ] Smoke test: full trace visible in LangSmith
- [ ] Commit + push `feature/agent-graph`

## Phase 5: Evaluation [ ]
- [ ] `evals/golden_dataset.json` — 15 curated Q&A pairs (5 per ticker; factoid + comparative + multi-hop)
- [ ] `src/evaluate.py` — RAGAS runner (context_recall, faithfulness, answer_relevancy)
- [ ] DeepEval: reasoning quality + factual correctness with explanations
- [ ] Target: context_recall > 0.75, faithfulness > 0.80
- [ ] Output results to `evals/results/` as JSON + console summary
- [ ] Commit + push `feature/evaluation`

## Phase 6: UI [ ]
- [ ] `app/main.py` — Streamlit: query input, ticker filter sidebar
- [ ] Streaming agent reasoning display (node-by-node progress)
- [ ] Investment thesis cards (structured JSON rendered)
- [ ] Source citations panel (chunk text + filing metadata)
- [ ] Eval scores display
- [ ] Smoke test: `streamlit run app/main.py`
- [ ] Commit + push `feature/ui`

---

## Review Notes

### Phase 1 — Foundation (complete)
- LLM factory refactored to data-driven `_PROVIDERS` registry after /simplify review
- `load_dotenv()` moved to lazy `_load_env()` guard to avoid disk I/O on every import
- Temperature now configurable via `LLM_TEMPERATURE` env var
- PR #1 merged to main

### Phase 2 — Ingest (branch ready, pending live test)
- `src/ingest.py` uses `edgartools` Company API; section-aware extraction via `doc[section_key]`
- `_extract_section_text` handles edgartools returning Section objects, strings, or None
- All structural unit tests pass on Python 3.9
- Pending: install `requirements.txt` and run live smoke test against real EDGAR
