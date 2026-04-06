# Financial Research Agent

A multi-step AI research agent built with LangGraph that retrieves SEC filings and financial news via a RAG pipeline, reasons over them, and produces structured investment theses — with a full evaluation suite.

---

## Overview

Most LLM demos stop at "ask a question, get an answer." This project builds a proper agentic system: the agent plans, retrieves, reasons across multiple steps, and produces structured output. Every component — retrieval quality, reasoning accuracy, and output faithfulness — is evaluated with real metrics.

---

## What This Project Does

**Agent Architecture (LangGraph)**
- Multi-node graph: Planner → Retriever → Analyst → Synthesizer → Evaluator
- Conditional edges for iterative retrieval when confidence is low
- Human-in-the-loop checkpoint support
- Full state tracking across the graph

**RAG Pipeline**
- Document ingestion: SEC 10-K/10-Q filings + financial news
- Chunking, embedding, and storage in a vector database (FAISS / ChromaDB)
- BM25 hybrid retrieval + semantic reranking
- Source citations in all outputs

**Output**
- Structured investment thesis: company overview, financials, risks, recommendation
- Confidence scores per section

**Evaluation**
- RAGAS — context recall, faithfulness, answer relevance
- DeepEval — LLM-as-Judge for reasoning quality
- Golden dataset for regression testing

---

## Stack

| Component | Tool |
|-----------|------|
| Agent Orchestration | LangGraph |
| LLM | OpenAI GPT-4o / Google Gemini |
| RAG | LangChain, FAISS, BM25 |
| Embeddings | OpenAI / HuggingFace |
| Evaluation | RAGAS, DeepEval |
| Observability | LangSmith |
| UI | Streamlit |
| Environment | Python 3.11 |

---

## Project Structure

```
financial-research-agent/
├── data/
│   └── filings/            # Ingested SEC documents
├── notebooks/
│   ├── 01_rag_pipeline.ipynb
│   ├── 02_agent_graph.ipynb
│   └── 03_evaluation.ipynb
├── src/
│   ├── ingest.py           # Document ingestion + indexing
│   ├── retriever.py        # Hybrid retrieval
│   ├── graph.py            # LangGraph agent definition
│   ├── nodes.py            # Individual agent node logic
│   └── evaluate.py         # RAGAS + DeepEval eval suite
├── app/
│   └── main.py             # Streamlit interface
├── evals/
│   └── golden_dataset.json # Ground truth Q&A pairs
├── requirements.txt
└── README.md
```

---

## Agent Graph

```
[Planner] → [Retriever] → [Analyst] → [Synthesizer]
                ↑               |
                └── (low conf.) ┘
```

---

## Key Results

*In progress — RAGAS scores and sample outputs will be added here.*
