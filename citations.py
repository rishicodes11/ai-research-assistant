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

def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
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

def load_all_papers():
    pdf_files = glob.glob("papers/*.pdf")
    for pdf_path in pdf_files:
        source_name = os.path.basename(pdf_path)
        print(f"Loading: {source_name}")
        text = extract_text(pdf_path)
        chunks = chunk_text(text)
        store_chunks(chunks, source_name)
    print(f"\nLoaded {len(pdf_files)} documents!\n")

def search(query, n_results=3):
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas"]
    )
    docs = results['documents'][0]
    metas = results['metadatas'][0]
    return list(zip(docs, metas))

def ask_with_citations(question):
    results = search(question)

    # Build numbered context
    context = ""
    for i, (doc, meta) in enumerate(results):
        context += f"[{i+1}] Source: {meta['source']} (chunk {meta['chunk_index']})\n{doc}\n\n"

    prompt = f"""You are a research assistant.
Answer the question using ONLY the context below.
After each claim, add a citation like [1], [2], or [3] referring to the source number.
If the answer is not in the context, say "I don't know based on the documents."

Context:
{context}

Question: {question}

Answer (with citations):"""

    response = groq_client.chat.completions.create(
        model="llama-3.3-70b-versatile",
        messages=[{"role": "user", "content": prompt}]
    )

    answer = response.choices[0].message.content

    # Print answer
    print(f"\nAnswer: {answer}")

    # Print citation legend
    print("\n--- Sources ---")
    for i, (doc, meta) in enumerate(results):
        print(f"[{i+1}] {meta['source']} — chunk {meta['chunk_index']}")
        print(f"    \"{doc[:100]}...\"")

# Run
load_all_papers()

while True:
    question = input("\nAsk a question (or 'quit'): ")
    if question.lower() == "quit":
        break
    ask_with_citations(question)