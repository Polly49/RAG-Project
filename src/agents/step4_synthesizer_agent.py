"""
Step 4: Synthesizer Agent Node using LangChain & LangGraph
Grounds responses strictly on retrieved excerpts, detects conflicting senior guidance,
and provides conversational routing for out-of-scope interactions.
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
from src.agents.step1_state import GraphState

def detect_conflicts(text: str) -> bool:
    """Checks if the answer identifies divergent senior advice."""
    signals = ["differ", "conflict", "disagree", "diverge", "on the other hand", "whereas", "varying advice"]
    return any(sig in text.lower() for sig in signals)

synthesizer_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Placement Prep Knowledge Assistant.
Answer the student's question using ONLY the provided Source Excerpts.

RULES:
1. STRICT GROUNDING: Do not hallucinate outside facts.
2. CONFLICT DETECTION: If seniors give differing advice (e.g. DSA vs Projects, or varying test rounds), explicitly highlight the difference:
   '⚠️ Note: Sources offer differing guidance: [Senior A] suggests X, while [Senior B] suggests Y.'
3. CITATIONS: Cite the senior or source name for specific tips or patterns.
4. STRUCTURE: Use clean markdown headers, bold highlights, and bullet points."""),
    ("user", "STUDENT QUESTION:\n{query}\n\nRETRIEVED EXCERPTS:\n{context}\n\nSynthesize a clear, grounded response:")
])

out_of_scope_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Placement Prep Knowledge Assistant.
The student has asked a general conversational question or an out-of-scope inquiry.
Politely greet them, explain that your expertise is campus placement preparation, and guide them on what they can ask (e.g., test patterns for Deloitte, Google, InfoEdge; DSA roadmaps; HR rounds; compensation records)."""),
    ("user", "{query}")
])

def synthesizer_node(state: GraphState) -> GraphState:
    query = state.get("user_query", "").strip()
    api_key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY

    # 1. Handle Out-Of-Scope Conversational Queries
    if state.get("is_out_of_scope"):
        if api_key:
            try:
                model_name = os.getenv("GROQ_MODEL", config.DEFAULT_GROQ_MODEL)
                llm = ChatGroq(api_key=api_key, model=model_name, temperature=0.3)
                chain = out_of_scope_prompt | llm | StrOutputParser()
                state["final_answer"] = chain.invoke({"query": query})
            except Exception:
                state["final_answer"] = "Hello! I am your AI Campus Placement Assistant. I can help you with company test patterns, technical interview questions, DSA roadmaps, compensation records, and senior preparation notes. What company or topic are you preparing for?"
        else:
            state["final_answer"] = "Hello! I am your AI Campus Placement Assistant. Please enter your Groq API key in the sidebar to begin searching placement notes."
        state["has_conflict"] = False
        state["sources_used"] = []
        return state

    # 2. Handle In-Scope Queries: Check Chunks
    chunks = state.get("retrieved_chunks", [])
    if not chunks:
        state["final_answer"] = "No relevant placement notes found for this query in our knowledge base. Please try searching for specific company names (e.g. Deloitte, InfoEdge) or topics (e.g. SQL, DSA, System Design)."
        state["has_conflict"] = False
        state["sources_used"] = []
        return state

    # 3. Format Context Blocks with Senior Metadata
    context_blocks = []
    sources = set()
    for idx, c in enumerate(chunks, 1):
        senior = c.get("senior_id") or "Senior Contributor"
        source = c.get("source_name") or "Guide"
        sources.add(f"{senior} ({source})")
        context_blocks.append(f"[Source #{idx} | {senior} | {source}]\n{c.get('text', '')}")

    context_text = "\n\n---\n\n".join(context_blocks)

    if not api_key:
        state["final_answer"] = "⚠️ **Groq API Key is missing!** Please enter your Groq API key in the sidebar or update `.env` to generate synthesized answers."
        state["has_conflict"] = False
        state["sources_used"] = sorted(list(sources))
        return state

    # 4. Generate Grounded Answer & Detect Conflicts
    try:
        model_name = os.getenv("GROQ_MODEL", config.DEFAULT_GROQ_MODEL)
        llm = ChatGroq(api_key=api_key, model=model_name, temperature=0.2, max_tokens=1000)
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
