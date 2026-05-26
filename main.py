from rank_bm25 import BM25Okapi
import math
import logging
import time
from datetime import datetime
from fastapi import FastAPI, UploadFile, File, HTTPException
from sentence_transformers import CrossEncoder
from collections import defaultdict
from fastapi import FastAPI, UploadFile, File
from pydantic import BaseModel
from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import chromadb
import os
import io

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)s | %(message)s",
    handlers=[
        logging.FileHandler("app.log"),  # saves to file
        logging.StreamHandler()           # shows in terminal
    ]
)
logger = logging.getLogger(__name__)

load_dotenv()

app = FastAPI(title="AI Research Assistant")

embedder = SentenceTransformer("all-MiniLM-L6-v2")
reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.PersistentClient(path="chroma_db")
collection = chroma_client.get_or_create_collection(name="research_papers")
chat_histories = defaultdict(list)

# --- Helpers ---
def extract_text(file_bytes):
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=100,
        separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
    )
    return splitter.split_text(text)
def summarize_document(text):
    prompt = f"""Read this document and write a clear 2-3 sentence summary of what it's about.
Be specific about the main topics covered.

Document (first 3000 chars):
{text[:3000]}

Summary:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()
def get_embedding(text):
    return embedder.encode(text).tolist()

def store_chunks(chunks, source_name):
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{source_name}_chunk_{i}"],
            metadatas=[{"source": source_name, "chunk_index": i}]
        )
def rewrite_query(question, chat_history):
    if not chat_history:
        return question
    
    history_text = ""
    for msg in chat_history[-4:]:
        history_text += f"{msg['role'].upper()}: {msg['content']}\n"
    
    prompt = f"""Given this conversation history and the user's latest question, rewrite the question to be more specific and self-contained for document search. Return ONLY the rewritten question, nothing else.

Conversation history:
{history_text}

Latest question: {question}

Rewritten question:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content.strip()

def search(query, n_results=3):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "metadatas"]
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]

    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs, metas), key=lambda x: x[0], reverse=True)

    top = ranked[:n_results]
    top_score = float(top[0][0])

    # Convert to 0-100% using sigmoid
    confidence_pct = round(1 / (1 + math.exp(-top_score / 3)) * 100, 1)

    if confidence_pct > 70:
     confidence = "High"
    elif confidence_pct > 45:
     confidence = "Medium"
    else:
     confidence = "Low"

    
    return [(doc, meta) for _, doc, meta in top], confidence, confidence_pct

def hybrid_search(query, n_results=3):
    # Step 1 — Get ALL documents from ChromaDB
    all_docs = collection.get(include=["documents", "metadatas"])
    if not all_docs["ids"]:
        return [], "Low", 0.0

    docs = all_docs["documents"]
    metas = all_docs["metadatas"]

    # Step 2 — Semantic search scores
    query_embedding = get_embedding(query)
    semantic_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=min(10, len(docs)),
        include=["documents", "metadatas"]
    )
    semantic_docs = semantic_results["documents"][0]

    # Build semantic score map
    semantic_scores = {}
    for i, doc in enumerate(semantic_docs):
        semantic_scores[doc] = 1 - (i / len(semantic_docs))

    # Step 3 — BM25 keyword search scores
    tokenized_docs = [doc.lower().split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)

    # Normalize BM25 scores to 0-1
    max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
    normalized_bm25 = [score / max_bm25 for score in bm25_scores]

    # Step 4 — Combine scores (50% semantic + 50% keyword)
    combined = []
    for i, doc in enumerate(docs):
        semantic = semantic_scores.get(doc, 0)
        keyword = normalized_bm25[i]
        hybrid_score = 0.5 * semantic + 0.5 * keyword
        combined.append((hybrid_score, doc, metas[i]))

    # Step 5 — Sort by hybrid score
    combined = sorted(combined, key=lambda x: x[0], reverse=True)
    top = combined[:min(10, len(combined))]

    # Step 6 — Rerank with cross encoder
    pairs = [[query, doc] for _, doc, _ in top]
    rerank_scores = reranker.predict(pairs)
    reranked = sorted(zip(rerank_scores, [d for _, d, _ in top], [m for _, _, m in top]),
                      key=lambda x: x[0], reverse=True)

    final = reranked[:n_results]
    top_score = float(final[0][0])

    confidence_pct = round(1 / (1 + math.exp(-top_score / 3)) * 100, 1)
    if confidence_pct > 70:
        confidence = "High"
    elif confidence_pct > 45:
        confidence = "Medium"
    else:
        confidence = "Low"

    return [(doc, meta) for _, doc, meta in final], confidence, confidence_pct
# --- Routes ---
@app.get("/")
def home():
    return {"message": "AI Research Assistant is running!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    logger.info(f"Upload requested: {file.filename}")
    
    if not file.filename.endswith(".pdf"):
        logger.warning(f"Invalid file type: {file.filename}")
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        existing = collection.get(where={"source": file.filename})
        if existing and len(existing["ids"]) > 0:
            logger.info(f"Duplicate upload attempted: {file.filename}")
            return {
                "message": f"{file.filename} already exists!",
                "chunks": len(existing["ids"])
            }

        contents = await file.read()

        if len(contents) > 10 * 1024 * 1024:
            logger.warning(f"File too large: {file.filename}")
            raise HTTPException(status_code=400, detail="File too large. Max size is 10MB")

        text = extract_text(contents)

        if not text or len(text.strip()) < 50:
            logger.warning(f"No text extracted from: {file.filename}")
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. It may be a scanned image PDF")

        chunks = chunk_text(text)
        store_chunks(chunks, file.filename)
        summary = summarize_document(text)

        logger.info(f"Successfully loaded: {file.filename} | chunks: {len(chunks)}")
        return {
            "message": f"Successfully loaded {file.filename}",
            "chunks": len(chunks),
            "summary": summary
        }

    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Error processing {file.filename}: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing PDF: {str(e)}")

@app.get("/documents")
def list_documents():
    results = collection.get()
    if not results["ids"]:
        return {"documents": []}
    
    sources = list(set([meta["source"] for meta in results["metadatas"]]))
    return {"documents": sources, "total": len(sources)}

@app.delete("/documents/{filename}")
def delete_document(filename: str):
    existing = collection.get(where={"source": filename})
    if not existing or len(existing["ids"]) == 0:
        return {"message": f"{filename} not found!"}
    
    collection.delete(where={"source": filename})
    return {"message": f"{filename} deleted successfully!"}

class QuestionRequest(BaseModel):
    question: str
    session_id: str = "default"

@app.post("/ask")
def ask(request: QuestionRequest):
    logger.info(f"Question received | session: {request.session_id} | question: {request.question}")
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        rewritten = rewrite_query(request.question, chat_histories[request.session_id])
        results, confidence, confidence_score = hybrid_search(rewritten)
        
        if not results:
            raise HTTPException(status_code=404, detail="No documents found. Please upload a PDF first")

        context = ""
        for i, (doc, meta) in enumerate(results):
            context += f"[{i+1}] Source: {meta['source']}\n{doc}\n\n"

        history = chat_histories[request.session_id]
        history_text = ""
        for msg in history[-6:]:
            history_text += f"{msg['role'].upper()}: {msg['content']}\n"

        prompt = f"""You are a research assistant.
Answer the question using ONLY the context below.
Add citations like [1], [2], [3] after each claim.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Previous conversation:
{history_text}

Question: {request.question}

Answer:"""

        response = groq_client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            messages=[{"role": "user", "content": prompt}]
        )

        answer = response.choices[0].message.content

        chat_histories[request.session_id].append({"role": "user", "content": request.question})
        chat_histories[request.session_id].append({"role": "assistant", "content": answer})

        sources = [{"index": i+1, "source": meta['source'], "chunk": meta['chunk_index'], "preview": doc[:100]}
                   for i, (doc, meta) in enumerate(results)]

        logger.info(f"Answer generated | session: {request.session_id} | rewritten: {rewritten}")
        return {"answer": answer, "confidence": confidence,"confidence_score": round(confidence_score, 3), "sources": sources, "session_id": request.session_id, "rewritten_query": rewritten}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

class TopicRequest(BaseModel):
    topic: str

@app.post("/synthesize")
def synthesize(request: TopicRequest):
    results, _, _ = hybrid_search(request.topic, n_results=5)
    context = "\n\n".join([f"[From: {meta['source']}]\n{doc}" for doc, meta in results])

    prompt = f"""You are a research assistant. Generate a structured synthesis on the topic.

Context:
{context}

Topic: {request.topic}

## Background
## Key Findings
## Comparison Between Documents
## Research Gaps
## Conclusion"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    return {"topic": request.topic, "synthesis": response.choices[0].message.content}