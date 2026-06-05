import chromadb

class ChromaManager:
    def __init__(self, path: str = "chroma_db", collection_name: str = "research_papers"):
        self.client = chromadb.PersistentClient(path=path)
        self.collection = self.client.get_or_create_collection(name=collection_name)

    def add_chunks(self, chunks: list, embeddings: list, source_name: str, user_id: str):
        for i, (chunk, embedding) in enumerate(zip(chunks, embeddings)):
            self.collection.add(
                documents=[chunk],
                embeddings=[embedding],
                ids=[f"{user_id}_{source_name}_chunk_{i}"],
                metadatas=[{"source": source_name, "chunk_index": i, "user_id": user_id}]
            )

    def search(self, query_embedding: list, n_results: int = 10, user_id: str = None):
        where = {"user_id": user_id} if user_id else None
        kwargs = {
            "query_embeddings": [query_embedding],
            "n_results": n_results,
            "include": ["documents", "metadatas"]
        }
        if where:
            kwargs["where"] = where
        return self.collection.query(**kwargs)

    def get_all(self, user_id: str = None):
        if user_id:
            return self.collection.get(
                where={"user_id": user_id},
                include=["documents", "metadatas"]
            )
        return self.collection.get(include=["documents", "metadatas"])

    def get_by_source(self, source: str, user_id: str = None):
        if user_id:
            where = {"$and": [{"source": source}, {"user_id": user_id}]}
        else:
            where = {"source": source}
        return self.collection.get(where=where)

    def delete_by_source(self, source: str, user_id: str = None):
        if user_id:
            where = {"$and": [{"source": source}, {"user_id": user_id}]}
        else:
            where = {"source": source}
        self.collection.delete(where=where)

    def count(self, user_id: str = None):
        if user_id:
            return len(self.collection.get(where={"user_id": user_id})["ids"])
        return len(self.collection.get()["ids"])

# Singleton instance
chroma_manager = ChromaManager()