from langchain_text_splitters import RecursiveCharacterTextSplitter

class ChunkingService:
    def __init__(self, chunk_size: int = 500, chunk_overlap: int = 100):
        self.splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            separators=["\n\n", "\n", ". ", "! ", "? ", ", ", " ", ""]
        )

    def chunk_text(self, text: str) -> list:
        return self.splitter.split_text(text)

# Singleton instance
chunking_service = ChunkingService()
