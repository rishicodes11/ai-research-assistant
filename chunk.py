from langchain_text_splitters import RecursiveCharacterTextSplitter
import PyPDF2

# Extract text (same as before)
def extract_text(pdf_path):
    text = ""
    with open(pdf_path, "rb") as f:
        reader = PyPDF2.PdfReader(f)
        for page in reader.pages:
            text += page.extract_text()
    return text

# Chunk the text
def chunk_text(text):
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=500,      # each chunk = 500 characters
        chunk_overlap=50     # chunks share 50 characters with next chunk
    )
    chunks = splitter.split_text(text)
    return chunks

# Run it
text = extract_text("papers/sample.pdf")
chunks = chunk_text(text)

print(f"Total chunks: {len(chunks)}")
print(f"\n--- Sample Chunk 1 ---\n{chunks[0]}")
print(f"\n--- Sample Chunk 2 ---\n{chunks[1]}")