from sentence_transformers import SentenceTransformer

class EmbeddingService:
    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.model_name = model_name

    def get_embedding(self, text: str) -> list:
        return self.model.encode(text).tolist()

    def get_embeddings_batch(self, texts: list) -> list:
        return self.model.encode(texts).tolist()

# Singleton instance
embedding_service = EmbeddingService()
