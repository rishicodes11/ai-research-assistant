from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import PyPDF2
import chromadb
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))
chroma_client = chromadb.Client()
collection = chroma_client.create_collection(name="research_papers")

# Extract text
def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Chunk text
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,
        chunk_overlap=50
    )
    return splitter.split_text(text)

# Get embedding for one chunk
def get_embedding(text):
    result = client.models.embed_content(
        model="gemini-embedding-001",
        contents=text
    )
    return result.embeddings[0].values

# Store all chunks in ChromaDB
def store_chunks(chunks):
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"chunk_{i}"]
        )
    print(f"Stored {len(chunks)} chunks in vector database!")

# Search for relevant chunks
def search(query, n_results=3):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results
    )
    return results['documents'][0]

# Run it
text = extract_text("papers/sample.pdf")
chunks = chunk_text(text)
store_chunks(chunks)

# Test search
query = "what is this document about?"
relevant_chunks = search(query)

print(f"\n--- Top 3 Relevant Chunks for: '{query}' ---")
for i, chunk in enumerate(relevant_chunks):
    print(f"\n[{i+1}] {chunk[:200]}...")