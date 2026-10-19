"""
Retrieval Agent Node
LangGraph wrapper around the hybrid retrieval & cross-encoder reranking pipeline.
"""

import src.config as config
from src.agents.state import GraphState
from src.retrieval.pipeline import retrieve_and_rerank

def retrieval_node(state: GraphState) -> GraphState:
    query = state.get("user_query", "").strip()
    if not query:
        state["retrieved_chunks"] = []
        return state

    results = retrieve_and_rerank(
        user_query=query,
        topic_filters=state.get("topic_filters"),
        company_filter=state.get("company_filter"),
        top_k_final=config.TOP_K_RERANK
    )
    
    state["expanded_queries"] = results.get("expanded_queries", [query])
    state["retrieved_chunks"] = results.get("reranked_chunks", [])
    return state
