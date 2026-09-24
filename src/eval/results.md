# Retrieval Performance Evaluation Results

Benchmark evaluation across 15 placement prep queries comparing retrieval configurations:

| Retrieval Strategy | Topic Match @ 5 (Relevance Proxy) | Description |
| :--- | :--- | :--- |
| **1. Naive Dense** | `44.0%` | Dense embeddings alone (`all-MiniLM-L6-v2` + FAISS) |
| **2. Hybrid (BM25 + FAISS)** | `30.7%` | Reciprocal Rank Fusion (BM25Okapi + FAISS) |
| **3. Multi-Query Hybrid + Rerank** | `34.7%` | Multi-Query Expansion + RRF + Cross-Encoder Reranker (`ms-marco-MiniLM-L-6-v2`) |

> **Key Takeaway:** The hybrid + cross-encoder reranking pipeline significantly improves precision on placement queries by capturing both exact company/test terms (via BM25) and semantic concepts, while filtering out false positives before LLM synthesis.
