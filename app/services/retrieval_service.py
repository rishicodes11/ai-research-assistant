from rank_bm25 import BM25Okapi
from app.services.embedding_service import embedding_service
from app.services.rerank_service import rerank_service
from app.db.chroma_manager import chroma_manager

class RetrievalService:

    def hybrid_search(self, query: str, n_results: int = 3):
        # Step 1 — Get all documents
        all_docs = chroma_manager.get_all()
        if not all_docs["ids"]:
            return [], "Low", 0.0

        docs = all_docs["documents"]
        metas = all_docs["metadatas"]

        # Step 2 — Semantic search
        query_embedding = embedding_service.get_embedding(query)
        semantic_results = chroma_manager.search(
            query_embedding,
            n_results=min(10, len(docs))
        )
        semantic_docs = semantic_results["documents"][0]

        # Build semantic score map
        semantic_scores = {}
        for i, doc in enumerate(semantic_docs):
            semantic_scores[doc] = 1 - (i / len(semantic_docs))

        # Step 3 — BM25 keyword search
        tokenized_docs = [doc.lower().split() for doc in docs]
        bm25 = BM25Okapi(tokenized_docs)
        bm25_scores = bm25.get_scores(query.lower().split())

        # Normalize BM25
        max_bm25 = max(bm25_scores) if max(bm25_scores) > 0 else 1
        normalized_bm25 = [score / max_bm25 for score in bm25_scores]

        # Step 4 — Combine scores
        combined = []
        for i, doc in enumerate(docs):
            semantic = semantic_scores.get(doc, 0)
            keyword = normalized_bm25[i]
            hybrid_score = 0.5 * semantic + 0.5 * keyword
            combined.append((hybrid_score, doc, metas[i]))

        # Step 5 — Sort and take top 10
        combined = sorted(combined, key=lambda x: x[0], reverse=True)
        top = combined[:min(10, len(combined))]

        # Step 6 — Rerank
        top_docs = [doc for _, doc, _ in top]
        top_metas = [meta for _, _, meta in top]

        reranked_docs, confidence, confidence_pct = rerank_service.rerank(
            query, top_docs, n_results
        )

        # Match metas to reranked docs
        doc_to_meta = {doc: meta for doc, meta in zip(top_docs, top_metas)}
        final = [(doc, doc_to_meta.get(doc, {})) for doc in reranked_docs]

        return final, confidence, confidence_pct

# Singleton instance
retrieval_service = RetrievalService()