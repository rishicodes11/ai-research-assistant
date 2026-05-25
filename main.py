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
    # Retrieve more candidates first
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=10,
        include=["documents", "metadatas"]
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]

    # Rerank
    pairs = [[query, doc] for doc in docs]
    scores = reranker.predict(pairs)
    ranked = sorted(zip(scores, docs, metas), reverse=True)

    # Return top 3
    top = ranked[:n_results]
    return [(doc, meta) for _, doc, meta in top]

# --- Routes ---
@app.get("/")
def home():
    return {"message": "AI Research Assistant is running!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    # Check file type
    if not file.filename.endswith(".pdf"):
        raise HTTPException(status_code=400, detail="Only PDF files are allowed")
    
    try:
        # Check duplicate
        existing = collection.get(where={"source": file.filename})
        if existing and len(existing["ids"]) > 0:
            return {
                "message": f"{file.filename} already exists!",
                "chunks": len(existing["ids"])
            }

        contents = await file.read()

        # Check file size (max 10MB)
        if len(contents) > 10 * 1024 * 1024:
            raise HTTPException(status_code=400, detail="File too large. Max size is 10MB")

        text = extract_text(contents)

        # Check if text was extracted
        if not text or len(text.strip()) < 50:
            raise HTTPException(status_code=400, detail="Could not extract text from PDF. It may be a scanned image PDF")

        chunks = chunk_text(text)
        store_chunks(chunks, file.filename)
        summary = summarize_document(text)

        return {
            "message": f"Successfully loaded {file.filename}",
            "chunks": len(chunks),
            "summary": summary
        }

    except HTTPException:
        raise
    except Exception as e:
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
    if not request.question.strip():
        raise HTTPException(status_code=400, detail="Question cannot be empty")
    
    try:
        rewritten = rewrite_query(request.question, chat_histories[request.session_id])
        results = search(rewritten)
        
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

        return {"answer": answer, "sources": sources, "session_id": request.session_id, "rewritten_query": rewritten}

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error generating answer: {str(e)}")

class TopicRequest(BaseModel):
    topic: str

@app.post("/synthesize")
def synthesize(request: TopicRequest):
    results = search(request.topic, n_results=5)
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