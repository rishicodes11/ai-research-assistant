from google import genai
from langchain_text_splitters import RecursiveCharacterTextSplitter
from dotenv import load_dotenv
import PyPDF2
import os

load_dotenv()

client = genai.Client(api_key=os.getenv("GOOGLE_API_KEY"))

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

# Generate embeddings
def get_embeddings(chunks):
    embeddings = []
    for chunk in chunks:
        result = client.models.embed_content(
            model="models/gemini-embedding-001",
            contents=chunk
        )
        embeddings.append(result.embeddings[0].values)
    return embeddings

# Run it
text = extract_text("papers/sample.pdf")
chunks = chunk_text(text)
embeddings = get_embeddings(chunks)

print(f"Total chunks: {len(chunks)}")
print(f"Embedding dimensions: {len(embeddings[0])}")
print(f"Sample embedding (first 5 numbers): {embeddings[0][:5]}")