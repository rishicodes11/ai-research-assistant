# AI Research Assistant

A production-grade RAG system that lets you upload research papers and ask questions about them.

Built with FastAPI, ChromaDB, and LLaMA 3.3 70B via Groq.

---

## Features

- JWT authentication with multi-user document isolation
- Hybrid search (BM25 + semantic) with cross-encoder reranking
- Streaming responses with citation support
- Hallucination detection with faithfulness scoring
- Async PDF processing with job status tracking
- Docker support for single command deployment

---

## Quick Start

### With Docker

```bash
git clone https://github.com/rishicodes11/ai-research-assistant
cd ai-research-assistant
cp .env.example .env
docker compose up
```

### Without Docker

```bash
git clone https://github.com/rishicodes11/ai-research-assistant
cd ai-research-assistant
pip install -r requirements.txt
cp .env.example .env
uvicorn main:app --reload
```

---

## Environment Variables

```
GROQ_API_KEY=    # required — get from console.groq.com
SECRET_KEY=      # required — any random string for JWT signing
FRONTEND_URL=    # optional — your frontend URL for CORS
HF_TOKEN=        # optional — HuggingFace token for faster downloads
```

---

## API Endpoints

| Method | Endpoint | Auth | Description |
|---|---|---|---|
| POST | /register | No | Create account |
| POST | /login | No | Get JWT token |
| POST | /upload | Yes | Upload PDF |
| POST | /upload/async | Yes | Upload PDF in background |
| GET | /status/{job_id} | No | Check upload status |
| POST | /ask | Yes | Ask a question |
| POST | /ask/stream | Yes | Streaming answer |
| POST | /synthesize | Yes | Research synthesis |
| GET | /documents | Yes | List your documents |
| DELETE | /documents/{filename} | Yes | Delete a document |
| GET | /health | No | Server health |

---

## Tech Stack

| Component | Technology |
|---|---|
| Backend | FastAPI |
| LLM | LLaMA 3.3 70B via Groq |
| Embeddings | BAAI/bge-base-en-v1.5 (768-dim) |
| Vector DB | ChromaDB |
| Keyword Search | BM25 (rank-bm25) |
| Reranker | CrossEncoder ms-marco-MiniLM-L-6-v2 |
| Auth | JWT (python-jose) + bcrypt |
| User DB | SQLite |
| Container | Docker + Docker Compose |

---

## Project Structure

```
app/
├── api/routes.py              # All API endpoints
├── services/
│   ├── embedding_service.py   # Text embeddings
│   ├── retrieval_service.py   # Hybrid search
│   ├── rerank_service.py      # Cross-encoder reranking
│   └── llm_service.py         # LLM calls + faithfulness check
├── db/
│   ├── chroma_manager.py      # ChromaDB vector database
│   └── user_db.py             # SQLite user management
├── auth/
│   ├── auth_handler.py        # JWT token creation/verification
│   └── auth_bearer.py         # FastAPI route protection
├── models/schemas.py          # Pydantic request/response models
├── utils/chunking.py          # Recursive text chunking
└── evaluation/metrics.py      # Precision@K, MRR, NDCG
```
