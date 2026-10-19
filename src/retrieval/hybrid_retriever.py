"""
Hybrid Retriever (FAISS Dense Search + BM25 Sparse Search + RRF Fusion)

AI/ML Concepts:
- Dense Search: Captures semantic meaning (e.g. "DSA roadmap" matches "coding preparation").
- Sparse Search: Captures exact keywords and company names (e.g. "Deloitte", "ROW_NUMBER").
- Reciprocal Rank Fusion (RRF): Combines ranks without scale bias.
  Formula: RRF_Score(doc) = sum( 1 / (60 + rank) )
"""

import os
import json
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.retrieval.build_index import tokenize_for_bm25

# Global Cache to load indices into memory once
_FAISS_INDEX = None
_BM25_INDEX = None
_METADATA = None
_EMBED_MODEL = None

def load_indexes():
    global _FAISS_INDEX, _BM25_INDEX, _METADATA, _EMBED_MODEL
    if _FAISS_INDEX is None:
        _FAISS_INDEX = faiss.read_index(config.FAISS_INDEX_PATH)
        with open(config.BM25_INDEX_PATH, "rb") as f:
            _BM25_INDEX = pickle.load(f)
        with open(config.METADATA_PATH, "r", encoding="utf-8") as f:
            _METADATA = [json.loads(line) for line in f if line.strip()]
        _EMBED_MODEL = SentenceTransformer(config.EMBED_MODEL_NAME)
    return _FAISS_INDEX, _BM25_INDEX, _METADATA, _EMBED_MODEL

def dense_search(query: str, k: int = 30) -> list[tuple[int, float]]:
    """Performs cosine-similarity vector search using FAISS."""
    faiss_idx, _, _, embed_model = load_indexes()
    query_vec = embed_model.encode([query], convert_to_numpy=True)
    faiss.normalize_L2(query_vec)
    scores, indices = faiss_idx.search(query_vec, min(k, faiss_idx.ntotal))
    return [(int(idx), float(score)) for idx, score in zip(indices[0], scores[0]) if idx != -1]

def sparse_search(query: str, k: int = 30) -> list[tuple[int, float]]:
    """Performs BM25 keyword matching search."""
    _, bm25_idx, _, _ = load_indexes()
    tokens = tokenize_for_bm25(query)
    if not tokens:
        return []
    scores = bm25_idx.get_scores(tokens)
    top_indices = np.argsort(scores)[::-1][:k]
    return [(int(i), float(scores[i])) for i in top_indices if scores[i] > 0]

def reciprocal_rank_fusion(dense_results, sparse_results, k: int = 60) -> list[tuple[int, float]]:
    """Merges dense and sparse ranked lists using Reciprocal Rank Fusion (RRF)."""
    rrf_scores = {}
    for rank, (doc_idx, _) in enumerate(dense_results, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank))
    for rank, (doc_idx, _) in enumerate(sparse_results, start=1):
        rrf_scores[doc_idx] = rrf_scores.get(doc_idx, 0.0) + (1.0 / (k + rank))
    return sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)

def hybrid_search(query: str, top_k: int = config.TOP_K_RETRIEVE) -> list[dict]:
    """Runs dense + sparse search, merges with RRF, and returns top chunk dictionaries."""
    _, _, metadata, _ = load_indexes()
    dense_res = dense_search(query, k=30)
    sparse_res = sparse_search(query, k=30)
    fused_res = reciprocal_rank_fusion(dense_res, sparse_res, k=60)

    top_chunks = []
    for doc_idx, score in fused_res[:top_k]:
        if doc_idx < len(metadata):
            chunk = dict(metadata[doc_idx])
            chunk["retrieval_score"] = round(score, 5)
            top_chunks.append(chunk)
    return top_chunks

if __name__ == "__main__":
    results = hybrid_search("Deloitte interview format", top_k=3)
    for r in results:
        print(r.get("source_name"), "->", r.get("retrieval_score"))
