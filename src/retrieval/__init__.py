"""
Placement Prep RAG - Retrieval Package
Ordered 5-Step Retrieval Architecture:
1. step1_build_index.py      -> Offline index creation (FAISS + BM25)
2. step2_query_rewriter.py   -> Multi-query expansion with Pydantic structured output
3. step3_hybrid_retriever.py -> FAISS + BM25 + Reciprocal Rank Fusion (RRF)
4. step4_reranker.py         -> Cross-Encoder self-attention precision reranker
5. step5_pipeline.py         -> End-to-end unified retrieval pipeline
"""

from src.retrieval.step1_build_index import tokenize_for_bm25, build_indexes
from src.retrieval.step2_query_rewriter import rewrite_query, QueryVariations
from src.retrieval.step3_hybrid_retriever import hybrid_search, dense_search, sparse_search, load_indexes
from src.retrieval.step4_reranker import rerank
from src.retrieval.step5_pipeline import retrieve_and_rerank
