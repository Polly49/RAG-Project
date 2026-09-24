"""
Router Agent Node
Uses LLM zero-shot classification to decide if a query is a:
- SIMPLE lookup (e.g. "What is Deloitte's test pattern?")
- SYNTHESIS query (e.g. "Should I focus on DSA or Machine Learning first?")
"""

from groq import Groq
import src.config as config
from src.agents.state import GraphState

def router_node(state: GraphState) -> GraphState:
    query = state.get("user_query", "").strip()
    if not query:
        state["needs_synthesis"] = False
        return state

    client = Groq(api_key=config.GROQ_API_KEY)
    
    prompt = f"""You are an intent classifier for a campus placement assistant.
Classify this student question into one category:
1. SIMPLE: Direct factual lookup (test pattern, syllabus, eligibility).
2. SYNTHESIS: Strategy, timeline, comparison, or advice question where seniors might differ.

Query: "{query}"
Respond with ONLY the word "SIMPLE" or "SYNTHESIS"."""

    try:
        response = client.chat.completions.create(
            model=config.DEFAULT_GROQ_MODEL,
            messages=[{"role": "user", "content": prompt}],
            temperature=0.0,
            max_tokens=10
        )
        decision = response.choices[0].message.content.strip().upper()
        state["needs_synthesis"] = ("SYNTHESIS" in decision)
    except Exception:
        state["needs_synthesis"] = True  # Safe default

    return state
