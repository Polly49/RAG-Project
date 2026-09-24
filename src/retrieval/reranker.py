"""
Cross-Encoder Reranker Module

AI/ML Concept:
- Bi-Encoder (FAISS): Encodes query and text separately for fast search (O(N)).
- Cross-Encoder: Passes [Query + Text] together through full transformer self-attention
  for maximum accuracy (O(K)).
We retrieve broad candidates (top-20) via hybrid search, then rerank down to top-5 with the Cross-Encoder.
"""

from sentence_transformers import CrossEncoder
import src.config as config

_RERANKER_MODEL = None

def get_reranker() -> CrossEncoder:
    global _RERANKER_MODEL
    if _RERANKER_MODEL is None:
        _RERANKER_MODEL = CrossEncoder(config.RERANKER_MODEL_NAME)
    return _RERANKER_MODEL

def rerank(query: str, candidates: list[dict], top_k: int = config.TOP_K_RERANK) -> list[dict]:
    """Scores candidate chunks jointly against the query and returns top_k highest scoring chunks."""
    if not candidates or len(candidates) <= top_k:
        return candidates

    model = get_reranker()
    pairs = [[query, chunk.get("text", "")] for chunk in candidates]
    scores = model.predict(pairs)

    # Attach scores and sort
    for chunk, score in zip(candidates, scores):
        chunk["rerank_score"] = float(round(score, 4))

    return sorted(candidates, key=lambda x: x.get("rerank_score", 0), reverse=True)[:top_k]

if __name__ == "__main__":
    test_q = "How is Deloitte interview process?"
    samples = [
        {"text": "InfoEdge test has ML questions.", "source_name": "infoedge.md"},
        {"text": "Deloitte Analyst round has 65 questions in 65 mins.", "source_name": "deloitte.md"}
    ]
    print(rerank(test_q, samples, top_k=1))
