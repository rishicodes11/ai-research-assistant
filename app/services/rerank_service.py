from sentence_transformers import CrossEncoder
import math

class RerankService:
    def __init__(self, model_name: str = "cross-encoder/ms-marco-MiniLM-L-6-v2"):
        self.model = CrossEncoder(model_name)

    def rerank(self, query: str, docs: list, top_n: int = 3) -> tuple:
        pairs = [[query, doc] for doc in docs]
        scores = self.model.predict(pairs)
        ranked = sorted(zip(scores, docs), key=lambda x: x[0], reverse=True)

        top = ranked[:top_n]
        top_score = float(top[0][0])

        confidence_pct = round(1 / (1 + math.exp(-top_score / 3)) * 100, 1)

        if confidence_pct > 70:
            confidence = "High"
        elif confidence_pct > 45:
            confidence = "Medium"
        else:
            confidence = "Low"

        return [doc for _, doc in top], confidence, confidence_pct

# Singleton instance
rerank_service = RerankService()
