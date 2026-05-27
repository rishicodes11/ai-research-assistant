import chromadb

class ChromaManager:
    def __init__(self, path: str = "chroma_db", collection_name: str = "research_papers"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list, embeddings: list, source_name: str):
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{source_name}_chunk_{i}"],
                metadatas=[{"source": source_name, "chunk_index": i}]
            )

    def search(self, query_embedding: list, n_results: int = 10):
        return self.collection.query(
            query_embeddings=[query_embedding],
            n_results=n_results,
            include=["documents", "metadatas"]
        )

    def get_all(self):
        return self.collection.get(include=["documents", "metadatas"])

    def get_by_source(self, source: str):
        return self.collection.get(where={"source": source})

    def delete_by_source(self, source: str):
        self.collection.delete(where={"source": source})

    def count(self):
        return len(self.collection.get()["ids"])

# Singleton instance
chroma_manager = ChromaManager()