"""
End-to-End Retrieval & Reranking Pipeline

AI/ML Workflow:
1. Query Expansion (Groq LLM generates 3 rewrites)
2. Multi-Query Hybrid Search (BM25 + FAISS for every rewrite)
3. Score Fusion & Deduplication (Merges candidates, keeps highest score)
4. Cross-Encoder Reranker (Reranks top-20 to top-5)
"""

import time
import src.config as config
from src.retrieval.query_rewriter import rewrite_query
from src.retrieval.hybrid_retriever import hybrid_search
from src.retrieval.reranker import rerank

def retrieve_and_rerank(user_query: str, topic_filters=None, company_filter=None, top_k_final: int = config.TOP_K_RERANK) -> dict:
    start_time = time.time()

    # Step 1: Query Expansion
    expanded_queries = rewrite_query(user_query, n=3)

    # Step 2: Multi-Query Hybrid Search & Deduplication
    seen_chunks = {}
    for q in expanded_queries:
        candidates = hybrid_search(q, top_k=15)
        for c in candidates:
            cid = c.get("chunk_id")
            score = c.get("retrieval_score", 0.0)
            if cid not in seen_chunks or score > seen_chunks[cid].get("retrieval_score", 0.0):
                seen_chunks[cid] = c

    all_candidates = list(seen_chunks.values())

    # Step 3: Metadata Filters (Topic & Company)
    if topic_filters:
        selected = [t.lower() for t in topic_filters]
        all_candidates = [c for c in all_candidates if c.get("topic", "").lower() in selected]

    if company_filter:
        comp = company_filter.lower()
        all_candidates = [c for c in all_candidates if c.get("company") and comp in c.get("company", "").lower()]

    # Step 4: Cross-Encoder Precision Reranking (Top 20 -> Top 5)
    top_candidates = all_candidates[:config.TOP_K_RETRIEVE]
    final_chunks = rerank(user_query, top_candidates, top_k=top_k_final)

    return {
        "query": user_query,
        "expanded_queries": expanded_queries,
        "reranked_chunks": final_chunks,
        "pipeline_ms": round((time.time() - start_time) * 1000, 1)
    }

if __name__ == "__main__":
    res = retrieve_and_rerank("What is the test pattern for Deloitte?")
    print(f"Retrieved {len(res['reranked_chunks'])} chunks in {res['pipeline_ms']}ms.")
