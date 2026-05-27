import math
from app.services.retrieval_service import retrieval_service

def check_relevance(chunk_text: str, expected_keywords: list):
    chunk_lower = chunk_text.lower()
    matches = [kw for kw in expected_keywords if kw.lower() in chunk_lower]
    return len(matches) > 0, matches

def precision_at_k(retrieved_chunks: list, expected_keywords: list, k: int = 3):
    relevant = 0
    for chunk, _ in retrieved_chunks[:k]:
        is_relevant, _ = check_relevance(chunk, expected_keywords)
        if is_relevant:
            relevant += 1
    return relevant / k

def mrr(retrieved_chunks: list, expected_keywords: list):
    for i, (chunk, _) in enumerate(retrieved_chunks):
        is_relevant, _ = check_relevance(chunk, expected_keywords)
        if is_relevant:
            return 1 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_chunks: list, expected_keywords: list, k: int = 3):
    dcg = 0
    for i, (chunk, _) in enumerate(retrieved_chunks[:k]):
        is_relevant, _ = check_relevance(chunk, expected_keywords)
        if is_relevant:
            dcg += 1 / math.log2(i + 2)
    idcg = sum([1 / math.log2(i + 2) for i in range(min(k, len(expected_keywords)))])
    return dcg / idcg if idcg > 0 else 0.0

def run_evaluation(test_cases: list):
    all_precision, all_mrr, all_ndcg = [], [], []

    for i, test in enumerate(test_cases):
        results, confidence, score = retrieval_service.hybrid_search(test["question"])

        if not results:
            continue

        p = precision_at_k(results, test["expected_keywords"])
        m = mrr(results, test["expected_keywords"])
        n = ndcg_at_k(results, test["expected_keywords"])

        all_precision.append(p)
        all_mrr.append(m)
        all_ndcg.append(n)

        print(f"Test {i+1}: {test['question']}")
        print(f"  Precision@3: {p:.2f} | MRR: {m:.2f} | NDCG@3: {n:.2f}")

    if all_precision:
        avg = (sum(all_precision) + sum(all_mrr) + sum(all_ndcg)) / (3 * len(all_precision))
        print(f"\nMean Precision@3: {sum(all_precision)/len(all_precision):.2f}")
        print(f"Mean MRR:         {sum(all_mrr)/len(all_mrr):.2f}")
        print(f"Mean NDCG@3:      {sum(all_ndcg)/len(all_ndcg):.2f}")
        print(f"Overall: {'🟢 EXCELLENT' if avg > 0.7 else '🟡 GOOD' if avg > 0.5 else '🔴 NEEDS IMPROVEMENT'} ({avg:.2f})")