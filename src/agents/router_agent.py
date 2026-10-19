"""
Router Agent Node using LangChain & LangGraph
Classifies user queries into SIMPLE lookup vs SYNTHESIS using ChatGroq.
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

router_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are an intent classifier for a campus placement assistant. Classify the user query into either 'SIMPLE' (direct factual lookup, test pattern, syllabus) or 'SYNTHESIS' (strategy, comparison, roadmap where advice may differ). Respond with ONLY the single word 'SIMPLE' or 'SYNTHESIS'."),
    ("user", "{query}")
])

def router_node(state: GraphState) -> GraphState:
    query = state.get("user_query", "").strip()
    if not query:
        state["needs_synthesis"] = False
        return state

    api_key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY
    if not api_key:
        state["needs_synthesis"] = True
        return state

    try:
        llm = ChatGroq(api_key=api_key, model=config.DEFAULT_GROQ_MODEL, temperature=0.0)
        chain = router_prompt | llm | StrOutputParser()
        decision = chain.invoke({"query": query}).strip().upper()
        state["needs_synthesis"] = ("SYNTHESIS" in decision)
    except Exception:
        state["needs_synthesis"] = True

    return state
