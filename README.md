# Financial Research Agent

A multi-step AI research agent built with LangGraph that retrieves SEC filings and financial news via a RAG pipeline, reasons over them, and produces structured investment theses — with a full evaluation suite.

![Python](https://img.shields.io/badge/Python-3.11-blue) ![License](https://img.shields.io/badge/license-MIT-green) ![LangGraph](https://img.shields.io/badge/built%20with-LangGraph-1C3C3C) ![Eval](https://img.shields.io/badge/evaluated%20with-RAGAS%20%2B%20DeepEval-purple)

---

## Overview

Most LLM demos stop at "ask a question, get an answer." This project builds a proper agentic system: the agent plans, retrieves, reasons across multiple steps, and produces structured output. Every component — retrieval quality, reasoning accuracy, and output faithfulness — is evaluated with real metrics.

**Scope:** Apple (AAPL), Microsoft (MSFT), and Nvidia (NVDA) — 10-K and 10-Q filings + earnings call transcripts (8-K exhibits). 15 hand-curated golden Q&A pairs for regression testing.

---

## What This Project Does

**Agent Architecture (LangGraph)**
- Multi-node graph: Planner → Retriever → Analyst → Synthesizer → Evaluator
- Conditional edges: re-retrieval triggered when Analyst confidence score falls below threshold (max 3 iterations)
- Human-in-the-loop checkpoint at Synthesizer node
- Full state tracking across the graph via typed `AgentState`

**RAG Pipeline**
- Document ingestion: SEC 10-K/10-Q filings via `sec-edgar-downloader`; earnings call transcripts via EDGAR 8-K exhibits
- Section-aware chunking (splits by SEC filing section headers, not arbitrary character count)
- Dense index: FAISS with `BAAI/bge-large-en-v1.5` embeddings
- Sparse index: BM25 via `rank-bm25`
- Reranking: `BAAI/bge-reranker-large` cross-encoder (local, no API cost)
- Source citations included in all outputs

**Agent Nodes**
| Node | Input | Output |
|------|-------|--------|
| Planner | User question | Decomposed sub-queries + target sections |
| Retriever | Sub-queries | Retrieved chunks with scores |
| Analyst | Chunks | Structured analysis + confidence score |
| Synthesizer | Analysis | Investment thesis (JSON) |
| Evaluator | Thesis + ground truth | RAGAS + DeepEval scores |

**Evaluation**
- **RAGAS** — measures retrieval quality: context recall, faithfulness, answer relevance
- **DeepEval LLM-as-Judge** — measures reasoning quality: multi-step coherence, factual grounding
- Golden dataset: 15 hand-curated Q&A pairs in `evals/golden_dataset.json`

**Cost note:** RAGAS + DeepEval make LLM calls per metric per question. Use `gpt-4o-mini` for evaluation, `gpt-4o` for the agent's Synthesizer node only.

---

## Stack

| Component | Tool |
|-----------|------|
| Agent Orchestration | LangGraph |
| LLM (default) | OpenAI GPT-4o — swap via `LLM_PROVIDER=gemini` env var |
| RAG | LangChain, FAISS, BM25 (`rank-bm25`) |
| Embeddings | `BAAI/bge-large-en-v1.5` (HuggingFace, local) |
| Reranker | `BAAI/bge-reranker-large` (cross-encoder, local) |
| Evaluation | RAGAS, DeepEval |
| Observability | LangSmith |
| UI | Streamlit |
| Environment | Python 3.11 |

---

## API Keys & Environment

```bash
# .env.example
OPENAI_API_KEY=                        # required — GPT-4o for synthesis
LANGCHAIN_API_KEY=                     # required — LangSmith tracing
LANGCHAIN_TRACING_V2=true
LANGCHAIN_PROJECT=financial-research-agent
SEC_EDGAR_USER_AGENT="YourName your@email.com"   # required — EDGAR 403s without this
GEMINI_API_KEY=                        # optional — alternative LLM backend
```

| Credential | Cost | Free Tier |
|-----------|------|-----------|
| OpenAI | Pay-per-token | $5 credit on signup |
| LangSmith | Free tier available | 5K traces/month |
| SEC EDGAR | Free | No key needed, only User-Agent header |
| Embeddings/Reranker | Free | Local models, no API |

---

## Quickstart

```bash
git clone https://github.com/smadinen7/financial-research-agent
cd financial-research-agent
cp .env.example .env          # fill in your keys
pip install -r requirements.txt

# Ingest filings for target companies
python src/ingest.py --tickers AAPL MSFT NVDA --filing-types 10-K 10-Q

# Run the agent
streamlit run app/main.py

# Run evaluation suite
python src/evaluate.py --golden-dataset evals/golden_dataset.json
```

---

## Project Structure

```
financial-research-agent/
├── data/
│   └── filings/            # Ingested SEC documents (gitignored)
├── notebooks/
│   ├── 01_rag_pipeline.ipynb
│   ├── 02_agent_graph.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── ingest.py           # SEC EDGAR download + section-aware chunking
│   ├── retriever.py        # FAISS + BM25 + cross-encoder reranking
│   ├── graph.py            # LangGraph agent graph definition
│   ├── nodes.py            # Planner, Analyst, Synthesizer node logic
│   └── evaluate.py         # RAGAS + DeepEval eval suite
├── app/
│   └── main.py             # Streamlit interface
├── evals/
│   └── golden_dataset.json # 15 hand-curated Q&A pairs
├── .env.example
├── requirements.txt
└── README.md
```

---

## Agent Graph

```
[Planner] → [Retriever] → [Analyst] → [Synthesizer]
                ↑               |
                └── (conf < 0.7, max 3 iterations)
```

---

## Golden Dataset Format

```json
{
  "question": "What were Apple's primary risk factors related to supply chain in their 2023 10-K?",
  "ground_truth": "Apple cited concentration risk in Asia-Pacific manufacturing...",
  "source_filing": "AAPL-10K-2023",
  "section": "Risk Factors",
  "question_type": "factual_retrieval"
}
```

---

## Key Results

*In progress — target metrics below. Results will be updated as experiments complete.*

| Metric | Target | Actual |
|--------|--------|--------|
| Context Recall (RAGAS) | > 0.75 | — |
| Faithfulness (RAGAS) | > 0.80 | — |
| Answer Relevance (RAGAS) | > 0.78 | — |
| Reasoning Quality (DeepEval) | > 0.75 | — |

---

## Limitations

- SEC EDGAR 10-K HTML parsing can produce noisy chunks around financial tables — section-aware chunking mitigates but does not eliminate this
- Evaluation costs accumulate quickly; full eval suite (~15 questions × 4 metrics) triggers ~60 LLM calls
- Agent is scoped to 3 tickers and 2 filing types; generalization to other companies requires re-ingestion
