# AI Research Assistant

A production-grade RAG (Retrieval Augmented Generation) system for querying and synthesizing information from research documents.

## Overview

Upload PDF documents and ask questions against them. The system retrieves relevant context using hybrid search and generates grounded, cited answers using an LLM.

## Architecture

```
┌─────────────┐    ┌──────────────┐    ┌──────────┐    ┌───────────┐    ┌────────────────┐
│  PDF Upload │───▶│ Text Extract │───▶│ Chunking │───▶│ Embedding │───▶│ Vector Storage │
└─────────────┘    └──────────────┘    └──────────┘    └───────────┘    └────────────────┘
                                                                                  │
                                                                                  ▼
┌─────────────┐    ┌───────────────┐    ┌───────────────┐    ┌──────────┐    ┌──────────┐
│ Cited Answer│◀───│      LLM      │◀───│   Reranking   │◀───│  Hybrid  │◀───│  Query   │
└─────────────┘    └───────────────┘    └───────────────┘    │  Search  │    │ Rewrite  │
                                                              └──────────┘    └──────────┘
```

## Features

- **Hybrid Search** — combines BM25 keyword search and semantic embeddings for superior retrieval
- **Cross-encoder Reranking** — reranks top 10 candidates to return the 3 most relevant chunks
- **Query Rewriting** — rewrites ambiguous queries using conversation history before retrieval
- **Confidence Scoring** — sigmoid-normalized relevance score for every answer
- **Streaming Responses** — token-by-token streaming via `/ask/stream`
- **Async PDF Processing** — background processing with job status tracking
- **Chat History** — session-based conversation memory
- **Answer Caching** — MD5-keyed cache to avoid redundant LLM calls
- **Evaluation System** — Precision@K, MRR, and NDCG metrics via `evaluate.py`
- **Rate Limiting** — 10 req/min on `/ask`, 5 req/min on `/upload`
- **Persistent Storage** — ChromaDB with disk persistence across restarts
- **Auto Document Summary** — LLM-generated summary on every upload

## Tech Stack

| Component | Technology |
|-----------|------------|
| API Framework | FastAPI |
| Vector Database | ChromaDB |
| Embeddings | sentence-transformers (all-MiniLM-L6-v2) |
| Reranker | CrossEncoder (ms-marco-MiniLM-L-6-v2) |
| Keyword Search | BM25Okapi (rank-bm25) |
| LLM | Groq (llama-3.3-70b-versatile) |
| Text Splitting | LangChain RecursiveCharacterTextSplitter |

## Setup

```bash
git clone https://github.com/rishicodes11/ai-research-assistant
cd ai-research-assistant
python3 -m venv venv
source venv/bin/activate
pip install -r requirements.txt
```

Create a `.env` file:
```
GROQ_API_KEY=your_groq_key_here
```

Run the server:
```bash
uvicorn main:app --reload
```

API docs available at `http://localhost:8000/docs`

## API Endpoints

| Method | Endpoint | Description |
|--------|----------|-------------|
| GET | `/health` | Server health and stats |
| POST | `/upload` | Upload and process PDF |
| POST | `/upload/async` | Upload PDF with background processing |
| GET | `/status/{job_id}` | Check async job status |
| GET | `/documents` | List all uploaded documents |
| DELETE | `/documents/{filename}` | Delete a document |
| POST | `/ask` | Ask a question with citations |
| POST | `/ask/stream` | Streaming question answering |
| POST | `/synthesize` | Generate structured research synthesis |

## Evaluation

Run the evaluation suite:
```bash
python3 evaluate.py
```

Current scores on test suite:
- Mean Precision@3: 0.67
- Mean MRR: 0.70
- Mean NDCG@3: 0.65

## Project Structure

```
research-assistant/
├── main.py                  # FastAPI app entry point
├── requirements.txt         # Dependencies
├── Procfile                 # Railway deployment config
├── .env.example             # Environment variable reference
├── evaluate.py              # IR evaluation metrics
└── app/
    ├── api/
    │   └── routes.py        # All API endpoints
    ├── services/
    │   ├── embedding_service.py   # Sentence transformer embeddings
    │   ├── retrieval_service.py   # Hybrid search (BM25 + semantic)
    │   ├── rerank_service.py      # Cross-encoder reranking
    │   └── llm_service.py         # Groq LLM calls
    ├── db/
    │   └── chroma_manager.py      # ChromaDB operations
    ├── models/
    │   └── schemas.py             # Pydantic request/response models
    ├── utils/
    │   └── chunking.py            # Text chunking
    └── evaluation/
        └── metrics.py             # Precision@K, MRR, NDCG
```