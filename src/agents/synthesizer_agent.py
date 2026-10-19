"""
Synthesizer Agent Node using LangChain & LangGraph
Grounds responses strictly on retrieved excerpts and detects conflicting senior guidance.
"""

import os
import sys
from pathlib import Path
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.agents.state import GraphState

def detect_conflicts(text: str) -> bool:
    """Checks if the answer identifies divergent senior advice."""
    signals = ["differ", "conflict", "disagree", "diverge", "on the other hand", "whereas", "varying advice"]
    return any(sig in text.lower() for sig in signals)

synthesizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Placement Prep Knowledge Assistant.
Answer the student's question using ONLY the provided Source Excerpts.

RULES:
1. STRICT GROUNDING: Do not hallucinate outside facts.
2. CONFLICT DETECTION: If seniors give differing advice (e.g. DSA vs Projects), explicitly highlight the difference:
   '⚠️ Note: Sources offer differing guidance: [Senior A] suggests X, while [Senior B] suggests Y.'
3. CITATIONS: Cite the senior or source name for specific tips or patterns.
4. STRUCTURE: Use clean markdown headers and bullet points."""),
    ("user", "STUDENT QUESTION:\n{query}\n\nRETRIEVED EXCERPTS:\n{context}\n\nSynthesize a clear response:")
])

def synthesizer_node(state: GraphState) -> GraphState:
    query = state.get("user_query", "").strip()
    chunks = state.get("retrieved_chunks", [])

    if not chunks:
        state["final_answer"] = "No relevant placement notes found for this query. Please try rephrasing."
        state["has_conflict"] = False
        state["sources_used"] = []
        return state

    # Format context blocks with metadata
    context_blocks = []
    sources = set()
    for idx, c in enumerate(chunks, 1):
        senior = c.get("senior_id") or "Senior Contributor"
        source = c.get("source_name") or "Guide"
        sources.add(f"{senior} ({source})")
        context_blocks.append(f"[Source #{idx} | {senior} | {source}]\n{c.get('text', '')}")

    context_text = "\n\n---\n\n".join(context_blocks)

    api_key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY
    if not api_key:
        state["final_answer"] = "⚠️ **Groq API Key is missing!** Please enter your Groq API key in the sidebar or update the `.env` file to generate synthesized answers."
        state["has_conflict"] = False
        state["sources_used"] = sorted(list(sources))
        return state

    try:
        llm = ChatGroq(api_key=api_key, model=config.DEFAULT_GROQ_MODEL, temperature=0.2, max_tokens=1000)
        chain = synthesizer_prompt | llm | StrOutputParser()
        answer = chain.invoke({"query": query, "context": context_text})

        state["final_answer"] = answer
        state["has_conflict"] = detect_conflicts(answer)
        state["sources_used"] = sorted(list(sources))
    except Exception as e:
        state["final_answer"] = f"Error generating answer: {e}"
        state["has_conflict"] = False
        state["sources_used"] = sorted(list(sources))

    return state
