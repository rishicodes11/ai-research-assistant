import uuid
import hashlib
from collections import defaultdict
from fastapi import APIRouter, UploadFile, File, HTTPException, Request, BackgroundTasks
from fastapi.responses import StreamingResponse
from slowapi import Limiter
from slowapi.util import get_remote_address

from app.services.embedding_service import embedding_service
from app.services.retrieval_service import retrieval_service
from app.services.llm_service import llm_service
from app.db.chroma_manager import chroma_manager
from app.utils.chunking import chunking_service
from app.models.schemas import QuestionRequest, TopicRequest
from app.services.retrieval_service import retrieval_service

import PyPDF2
import io
import logging

logger = logging.getLogger(__name__)
limiter = Limiter(key_func=get_remote_address)
router = APIRouter()

# In-memory state
chat_histories = defaultdict(list)
answer_cache = {}
processing_jobs = {}
CACHE_MAX_SIZE = 100

# --- Helpers ---
def extract_text(file_bytes: bytes) -> str:
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        text += page.extract_text()
    return text

def get_cache_key(question: str, session_id: str) -> str:
    content = f"{question.strip().lower()}_{session_id}"
    return hashlib.md5(content.encode()).hexdigest()

def process_pdf_background(job_id: str, file_bytes: bytes, filename: str):
    try:
        processing_jobs[job_id] = {"status": "processing", "filename": filename}
        text = extract_text(file_bytes)

        if not text or len(text.strip()) < 50:
            processing_jobs[job_id] = {"status": "failed", "error": "Could not extract text"}
            return

        chunks = chunking_service.chunk_text(text)
        embeddings = embedding_service.get_embeddings_batch(chunks)
        chroma_manager.add_chunks(chunks, embeddings, filename)
        retrieval_service.invalidate_cache()
        summary = llm_service.summarize(text)

        processing_jobs[job_id] = {
            "status": "complete",
            "filename": filename,
            "chunks": len(chunks),
            "summary": summary
        }
        logger.info(f"Background processing complete: {filename}")

    except Exception as e:
        processing_jobs[job_id] = {"status": "failed", "error": str(e)}
        logger.error(f"Background processing failed: {filename} | {str(e)}")

# --- Routes ---
@router.get("/")
def home():
    return {"message": "AI Research Assistant is running!"}

@router.get("/health")
def health_check():
    try:
        return {
            "status": "healthy",
            "documents_loaded": chroma_manager.count(),
            "cache_size": len(answer_cache),
            "model": llm_service.model,
            "embedding_model": embedding_service.model_name
        }
    except Exception as e:
        return {"status": "unhealthy", "error": str(e)}

@router.post("/upload")
@limiter.limit("5/minute")
async def upload_pdf(request: Request, file: UploadFile = File(...)):
    logger.info(f"Upload requested: {file.filename}")

    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    try:
        existing = chroma_manager.get_by_source(file.filename)
        if existing and len(existing["ids"]) > 0:
            return {"message": f"{file.filename} already exists!", "chunks": len(existing["ids"])}

        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max size is 10MB")

        text = extract_text(contents)
        if not text or len(text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF")

        chunks = chunking_service.chunk_text(text)
        embeddings = embedding_service.get_embeddings_batch(chunks)
        chroma_manager.add_chunks(chunks, embeddings, file.filename)
        retrieval_service.invalidate_cache()
        summary = llm_service.summarize(text)

        logger.info(f"Successfully loaded: {file.filename} | chunks: {len(chunks)}")
        return {"message": f"Successfully loaded {file.filename}", "chunks": len(chunks), "summary": summary}

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@router.post("/upload/async")
@limiter.limit("5/minute")
async def upload_pdf_async(request: Request, background_tasks: BackgroundTasks, file: UploadFile = File(...)):
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")

    existing = chroma_manager.get_by_source(file.filename)
    if existing and len(existing["ids"]) > 0:
        return {"message": f"{file.filename} already exists!", "status": "duplicate"}

    contents = await file.read()
    if len(contents) > 10 * 1024 * 1024:
        raise HTTPException(status_code=400, detail="File too large. Max size is 10MB")

    job_id = str(uuid.uuid4())
    processing_jobs[job_id] = {"status": "queued", "filename": file.filename}
    background_tasks.add_task(process_pdf_background, job_id, contents, file.filename)

    return {"message": "PDF queued for processing!", "job_id": job_id, "status": "queued"}

@router.get("/status/{job_id}")
def get_job_status(job_id: str):
    if job_id not in processing_jobs:
        raise HTTPException(status_code=404, detail="Job not found")
    return processing_jobs[job_id]

@router.get("/documents")
def list_documents():
    results = chroma_manager.get_all()
    if not results["ids"]:
        return {"documents": []}
    sources = list(set([meta["source"] for meta in results["metadatas"]]))
    return {"documents": sources, "total": len(sources)}

@router.delete("/documents/{filename}")
def delete_document(filename: str):
    existing = chroma_manager.get_by_source(filename)
    if not existing or len(existing["ids"]) == 0:
        return {"message": f"{filename} not found!"}
    chroma_manager.delete_by_source(filename)
    return {"message": f"{filename} deleted successfully!"}

@router.post("/ask")
@limiter.limit("10/minute")
def ask(request_data: QuestionRequest, request: Request):
    if not request_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    cache_key = get_cache_key(request_data.question, request_data.session_id)
    if cache_key in answer_cache:
        logger.info(f"Cache hit | session: {request_data.session_id}")
        cached_response = answer_cache[cache_key].copy()
        cached_response["cached"] = True
        return cached_response

    try:
        rewritten = llm_service.rewrite_query(request_data.question, chat_histories[request_data.session_id])
        results, confidence, confidence_score = retrieval_service.hybrid_search(rewritten)

        if not results:
            raise HTTPException(status_code=404, detail="No documents found. Please upload a PDF first")

        context = ""
        for i, (doc, meta) in enumerate(results):
            context += f"[{i+1}] Source: {meta['source']}\n{doc}\n\n"

        history_text = ""
        for msg in chat_histories[request_data.session_id][-6:]:
            history_text += f"{msg['role'].upper()}: {msg['content']}\n"

        answer = llm_service.answer(request_data.question, context, history_text)

        chat_histories[request_data.session_id].append({"role": "user", "content": request_data.question})
        chat_histories[request_data.session_id].append({"role": "assistant", "content": answer})

        sources = [{"index": i+1, "source": meta['source'], "chunk": meta['chunk_index'], "preview": doc[:100]}
                   for i, (doc, meta) in enumerate(results)]

        if len(answer_cache) >= CACHE_MAX_SIZE:
            oldest_key = next(iter(answer_cache))
            del answer_cache[oldest_key]

        response = {"answer": answer, "confidence": confidence, "confidence_score": round(confidence_score, 3),
                    "sources": sources, "session_id": request_data.session_id,
                    "rewritten_query": rewritten, "cached": False}
        answer_cache[cache_key] = response
        return response

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

@router.post("/ask/stream")
@limiter.limit("10/minute")
async def ask_stream(request_data: QuestionRequest, request: Request):
    if not request_data.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")

    try:
        rewritten = llm_service.rewrite_query(request_data.question, chat_histories[request_data.session_id])
        results, confidence, confidence_score = retrieval_service.hybrid_search(rewritten)

        if not results:
            raise HTTPException(status_code=404, detail="No documents found")

        context = ""
        for i, (doc, meta) in enumerate(results):
            context += f"[{i+1}] Source: {meta['source']}\n{doc}\n\n"

        history_text = ""
        for msg in chat_histories[request_data.session_id][-6:]:
            history_text += f"{msg['role'].upper()}: {msg['content']}\n"

        def generate():
            full_answer = ""
            for chunk in llm_service.generate_stream(
                f"""You are a research assistant. Answer using ONLY the context below. Add citations [1],[2],[3].

Context:
{context}

Previous conversation:
{history_text}

Question: {request_data.question}

Answer:"""
            ):
                full_answer += chunk
                yield chunk

            chat_histories[request_data.session_id].append({"role": "user", "content": request_data.question})
            chat_histories[request_data.session_id].append({"role": "assistant", "content": full_answer})

        return StreamingResponse(generate(), media_type="text/plain")

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error streaming: {str(e)}")

@router.post("/synthesize")
def synthesize(request: TopicRequest):
    results, _, _ = retrieval_service.hybrid_search(request.topic, n_results=5)
    context = "\n\n".join([f"[From: {meta['source']}]\n{doc}" for doc, meta in results])

    prompt = f"""You are a research assistant. Generate a structured synthesis.

Context:
{context}

Topic: {request.topic}

## Background
## Key Findings
## Comparison Between Documents
## Research Gaps
## Conclusion"""

    return {"topic": request.topic, "synthesis": llm_service.generate(prompt)}