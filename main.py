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
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="research_papers")

# --- Helpers ---
def extract_text(file_bytes):
    text = ""
    reader = PyPDF2.PdfReader(io.BytesIO(file_bytes))
    for page in reader.pages:
        text += page.extract_text()
    return text

def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

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

def search(query, n_results=3):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    return list(zip(results['documents'][0], results['metadatas'][0]))

# --- Routes ---
@app.get("/")
def home():
    return {"message": "AI Research Assistant is running!"}

@app.post("/upload")
async def upload_pdf(file: UploadFile = File(...)):
    contents = await file.read()
    text = extract_text(contents)
    chunks = chunk_text(text)
    store_chunks(chunks, file.filename)
    return {
        "message": f"Successfully loaded {file.filename}",
        "chunks": len(chunks)
    }

class QuestionRequest(BaseModel):
    question: str

@app.post("/ask")
def ask(request: QuestionRequest):
    results = search(request.question)
    context = ""
    for i, (doc, meta) in enumerate(results):
        context += f"[{i+1}] Source: {meta['source']}\n{doc}\n\n"

    prompt = f"""You are a research assistant.
Answer the question using ONLY the context below.
Add citations like [1], [2], [3] after each claim.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {request.question}

Answer:"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content
    sources = [{"index": i+1, "source": meta['source'], "chunk": meta['chunk_index'], "preview": doc[:100]} 
               for i, (doc, meta) in enumerate(results)]

    return {"answer": answer, "sources": sources}

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