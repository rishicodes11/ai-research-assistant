import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from main import hybrid_search, collection
import json

# ---- Test Cases ----
# Add questions and keywords you expect to find in retrieved chunks
test_cases = [
    {
        "question": "what is the assignment submission deadline?",
        "expected_keywords": ["deadline", "last date", "submission", "exam cycle"]
    },
    {
        "question": "what is the credence of internal assignment?",
        "expected_keywords": ["20%", "credence", "internal assignment"]
    },
    {
        "question": "how many attempts does a student get?",
        "expected_keywords": ["attempt", "three", "submit"]
    },
    {
        "question": "what happens if assignment is not submitted?",
        "expected_keywords": ["hold", "result", "withheld", "pending"]
    },
    {
        "question": "what is the word limit for answers?",
        "expected_keywords": ["word", "limit", "words"]
    }
]

def check_relevance(chunk_text, expected_keywords):
    """Check if chunk contains any expected keywords"""
    chunk_lower = chunk_text.lower()
    matches = [kw for kw in expected_keywords if kw.lower() in chunk_lower]
    return len(matches) > 0, matches

def precision_at_k(retrieved_chunks, expected_keywords, k=3):
    """What fraction of top K chunks are relevant?"""
    relevant = 0
    for chunk, _ in retrieved_chunks[:k]:
        is_relevant, _ = check_relevance(chunk, expected_keywords)
        if is_relevant:
            relevant += 1
    return relevant / k

def mrr(retrieved_chunks, expected_keywords):
    """Where is the first relevant chunk?"""
    for i, (chunk, _) in enumerate(retrieved_chunks):
        is_relevant, _ = check_relevance(chunk, expected_keywords)
        if is_relevant:
            return 1 / (i + 1)
    return 0.0

def ndcg_at_k(retrieved_chunks, expected_keywords, k=3):
    """Rewards finding relevant chunks early"""
    import math
    dcg = 0
    for i, (chunk, _) in enumerate(retrieved_chunks[:k]):
        is_relevant, _ = check_relevance(chunk, expected_keywords)
        if is_relevant:
            dcg += 1 / math.log2(i + 2)
    
    # Ideal DCG (all relevant at top)
    idcg = sum([1 / math.log2(i + 2) for i in range(min(k, len(expected_keywords)))])
    return dcg / idcg if idcg > 0 else 0.0

def evaluate():
    print("=" * 60)
    print("RAG EVALUATION REPORT")
    print("=" * 60)

    all_precision = []
    all_mrr = []
    all_ndcg = []

    for i, test in enumerate(test_cases):
        print(f"\n📝 Test {i+1}: {test['question']}")
        print(f"   Expected keywords: {test['expected_keywords']}")

        # Run hybrid search
        results, confidence, score = hybrid_search(test["question"], n_results=3)

        if not results:
            print("   ❌ No results returned!")
            continue

        # Calculate metrics
        p_at_3 = precision_at_k(results, test["expected_keywords"])
        mrr_score = mrr(results, test["expected_keywords"])
        ndcg_score = ndcg_at_k(results, test["expected_keywords"])

        all_precision.append(p_at_3)
        all_mrr.append(mrr_score)
        all_ndcg.append(ndcg_score)

        print(f"   Confidence: {confidence} ({score}%)")
        print(f"   Precision@3: {p_at_3:.2f}")
        print(f"   MRR: {mrr_score:.2f}")
        print(f"   NDCG@3: {ndcg_score:.2f}")

        # Show retrieved chunks
        for j, (chunk, meta) in enumerate(results):
            is_relevant, matches = check_relevance(chunk, test["expected_keywords"])
            status = "✅" if is_relevant else "❌"
            print(f"   [{j+1}] {status} {meta['source']} — matches: {matches}")
            print(f"       \"{chunk[:100]}...\"")

    # Overall scores
    print("\n" + "=" * 60)
    print("OVERALL SCORES")
    print("=" * 60)
    print(f"Mean Precision@3: {sum(all_precision)/len(all_precision):.2f}")
    print(f"Mean MRR:         {sum(all_mrr)/len(all_mrr):.2f}")
    print(f"Mean NDCG@3:      {sum(all_ndcg)/len(all_ndcg):.2f}")
    print("=" * 60)

    # Grade
    avg = (sum(all_precision) + sum(all_mrr) + sum(all_ndcg)) / (3 * len(test_cases))
    if avg > 0.7:
        grade = "🟢 EXCELLENT"
    elif avg > 0.5:
        grade = "🟡 GOOD"
    else:
        grade = "🔴 NEEDS IMPROVEMENT"

    print(f"Overall Grade: {grade} ({avg:.2f})")

if __name__ == "__main__":
    evaluate()