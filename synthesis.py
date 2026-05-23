from sentence_transformers import SentenceTransformer
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
from groq import Groq
import PyPDF2
import chromadb
import os
import glob

load_dotenv()

embedder = SentenceTransformer("all-MiniLM-L6-v2")
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
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
    splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50)
    return splitter.split_text(text)

# Get embedding
def get_embedding(text):
    return embedder.encode(text).tolist()

# Store chunks
def store_chunks(chunks, source_name):
    for i, chunk in enumerate(chunks):
        embedding = get_embedding(chunk)
        collection.add(
            documents=[chunk],
            embeddings=[embedding],
            ids=[f"{source_name}_chunk_{i}"],
            metadatas=[{"source": source_name}]
        )

# Load all PDFs
def load_all_papers():
    pdf_files = glob.glob("papers/*.pdf")
    for pdf_path in pdf_files:
        source_name = os.path.basename(pdf_path)
        print(f"Loading: {source_name}")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        store_chunks(chunks, source_name)
    print(f"\nLoaded {len(pdf_files)} documents!\n")

# Search
def search(query, n_results=5):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    docs = results['documents'][0]
    sources = [m["source"] for m in results['metadatas'][0]]
    return list(zip(docs, sources))

# Generate structured synthesis
def synthesize(topic):
    results = search(topic)
    context = "\n\n".join([f"[From: {src}]\n{doc}" for doc, src in results])

    prompt = f"""You are an expert research assistant.
Using ONLY the context below, generate a structured research synthesis on the topic.

Context:
{context}

Topic: {topic}

Generate a structured report with these exact sections:

## Background
(What is the context or background of this topic based on the documents?)

## Key Findings
(What are the most important points found in the documents?)

## Comparison Between Documents
(How do different documents approach or discuss this topic differently?)

## Research Gaps
(What questions remain unanswered based on what's in the documents?)

## Conclusion
(Summarize the key takeaways)

Be specific and cite which document each point comes from."""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )
    return response.choices[0].message.content

# Run
load_all_papers()

while True:
    topic = input("Enter a research topic to synthesize (or 'quit'): ")
    if topic.lower() == "quit":
        break
    print("\nGenerating synthesis...\n")
    report = synthesize(topic)
    print(report)
    print("\n" + "="*50 + "\n")