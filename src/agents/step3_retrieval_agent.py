"""
Step 3: Retrieval Agent Node with Autonomous Tool Calling & Self-Reflection
Wields the hybrid search & reranking pipeline as a tool.
Evaluates candidate relevance and triggers a cyclic loop if retrieved context is insufficient.
"""

import os
import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.agents.step1_state import GraphState
from src.retrieval.step5_pipeline import retrieve_and_rerank
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Autonomous Query Reformulation Prompt for the Self-Correction Loop
reformulation_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Retrieval Agent's Query Reformulator.
The initial search for the student's question failed to retrieve sufficient high-quality placement notes.
Generate a broader, alternative keyword-focused search query to find relevant interview patterns, senior advice, or syllabus.
Output ONLY the new search query string. Do not include quotes, explanations, or punctuation."""),
    ("user", "Original Query: {query}")
])

def grade_retrieval_relevance(chunks: list[dict], query: str) -> bool:
    """Self-reflection heuristic: Checks if retrieved chunks provide sufficient relevance."""
    if not chunks or len(chunks) == 0:
        return False

    # Check top Cross-Encoder score (negative values < -2.0 indicate irrelevance in ms-marco)
    top_score = chunks[0].get("rerank_score", 0.0)
    if top_score < -2.5:
        return False

    return True

def retrieval_node(state: GraphState) -> GraphState:
    """Autonomous Retrieval Agent: Runs search tool, grades context, and self-corrects if needed."""
    current_retry = state.get("retry_count", 0)
    
    # Use reformulated query if the self-correction loop was triggered, otherwise original query
    active_query = state.get("reformulated_query") or state.get("user_query", "").strip()

    if not active_query:
        state["retrieved_chunks"] = []
        state["is_relevant"] = False
        return state

    # 1. Execute Retrieval Tool
    results = retrieve_and_rerank(
        user_query=active_query,
        topic_filters=state.get("topic_filters"),
        company_filter=state.get("company_filter"),
        top_k_final=config.TOP_K_RERANK
    )

    chunks = results.get("reranked_chunks", [])
    state["expanded_queries"] = results.get("expanded_queries", [active_query])
    state["retrieved_chunks"] = chunks

    # 2. Self-Reflection: Grade Retrieved Context
    is_sufficient = grade_retrieval_relevance(chunks, active_query)

    # 3. Decision: If insufficient and retries remain, trigger Autonomous Feedback Loop
    if not is_sufficient and current_retry < 1:
        state["retry_count"] = current_retry + 1
        state["is_relevant"] = False

        # Autonomously reformulate search query for the loop
        api_key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY
        if api_key:
            try:
                model_name = os.getenv("GROQ_MODEL", config.DEFAULT_GROQ_MODEL)
                llm = ChatGroq(api_key=api_key, model=model_name, temperature=0.3)
                chain = reformulation_prompt | llm | StrOutputParser()
                new_query = chain.invoke({"query": active_query}).strip()
                state["reformulated_query"] = new_query
            except Exception:
                # Fallback reformulation: append general interview preparation keywords
                state["reformulated_query"] = f"{active_query} interview preparation roadmap"
        else:
            state["reformulated_query"] = f"{active_query} interview preparation roadmap"
    else:
        # Context is sufficient OR max retries reached: proceed to synthesis
        state["is_relevant"] = True

    return state

if __name__ == "__main__":
    res = retrieval_node({"user_query": "What is Deloitte test pattern?", "retry_count": 0})
    print(f"Retrieved {len(res['retrieved_chunks'])} chunks. Relevant: {res['is_relevant']}")
