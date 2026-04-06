# Financial Research Agent — Task Tracker

## Phase 1: Foundation ✅
- [x] Create `requirements.txt` (all deps: langgraph, langchain, faiss, edgartools, ragas, deepeval, streamlit)
- [x] Create `.env.example` (Gemini default, OpenAI + Claude optional, eval LLM separate)
- [x] Create `src/llm.py` — multi-provider LLM factory (gemini | openai | claude)
- [x] Verify: import, bad-provider error, missing-key error all behave correctly
- [x] Commit + push `feature/foundation`

## Phase 2: Ingest [ ]
- [ ] `src/ingest.py` — download 10-K/10-Q for AAPL/MSFT/NVDA via `edgartools`
- [ ] Section-aware chunking (Item 1A, Item 7, Item 8)
- [ ] Prepend metadata header to each chunk: `[Company | Form | Period | Section]`
- [ ] Save chunks to `data/filings/` as JSON
- [ ] Smoke test: `python src/ingest.py --tickers AAPL --filing-types 10-K`
- [ ] Commit + push `feature/ingest`

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
_Append findings after each phase completes._
