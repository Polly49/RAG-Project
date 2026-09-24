"""
Hybrid Index Builder (FAISS Dense Vector Index + BM25 Sparse Keyword Index)

AI/ML Concepts:
1. Dense Index: Generates 384-dim semantic embeddings via 'all-MiniLM-L6-v2'.
   Vectors are L2-normalized so Inner Product (IndexFlatIP) equals exact Cosine Similarity.
2. Sparse Index: Builds BM25Okapi inverted index for exact keyword & company name matching.
"""

import os
import json
import pickle
import re
import faiss
from sentence_transformers import SentenceTransformer
from rank_bm25 import BM25Okapi
import src.config as config

def tokenize_for_bm25(text: str) -> list[str]:
    """Tokenizes text into lowercase alphanumeric words for BM25."""
    return re.findall(r"\w+", text.lower())

def build_indexes():
    # 1. Load Chunks
    if not os.path.exists(config.CHUNKS_FILE):
        print(f"Chunks file not found at {config.CHUNKS_FILE}. Run chunker first.")
        return

    with open(config.CHUNKS_FILE, "r", encoding="utf-8") as f:
        chunks = [json.loads(line) for line in f if line.strip()]

    print(f"Loaded {len(chunks)} chunks for indexing.")
    os.makedirs(config.INDEXES_DIR, exist_ok=True)
    texts = [c["text"] for c in chunks]

    # 2. Build Dense FAISS Index (Cosine Similarity)
    print(f"Building Dense FAISS index with {config.EMBED_MODEL_NAME}...")
    embed_model = SentenceTransformer(config.EMBED_MODEL_NAME)
    embeddings = embed_model.encode(texts, batch_size=32, show_progress_bar=True, convert_to_numpy=True)
    
    # L2 normalize so dot product equals cosine similarity
    faiss.normalize_L2(embeddings)
    dim = embeddings.shape[1]
    faiss_index = faiss.IndexFlatIP(dim)
    faiss_index.add(embeddings)
    faiss.write_index(faiss_index, config.FAISS_INDEX_PATH)
    print(f"Saved FAISS index ({dim}-dim, {faiss_index.ntotal} vectors).")

    # 3. Build Sparse BM25 Keyword Index
    print("Building Sparse BM25 index...")
    tokenized_corpus = [tokenize_for_bm25(t) for t in texts]
    bm25 = BM25Okapi(tokenized_corpus)
    with open(config.BM25_INDEX_PATH, "wb") as f:
        pickle.dump(bm25, f)
    print("Saved BM25 index.")

    # 4. Save Metadata Mapping
    with open(config.METADATA_PATH, "w", encoding="utf-8") as f:
        for c in chunks:
            f.write(json.dumps(c) + "\n")
    print("Saved metadata mapping.")

if __name__ == "__main__":
    build_indexes()
