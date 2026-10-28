"""
Step 2: Router Agent Node using LangChain Structured Outputs & LangGraph
Autonomous intent classifier & dispatcher.
Enforces type-safe schema validation via Pydantic Literal["IN_SCOPE", "OUT_OF_SCOPE"].
"""

import os
import sys
from pathlib import Path
from typing import Literal
from pydantic import BaseModel, Field
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import src.config as config
from src.agents.step1_state import GraphState

# --- Pydantic Schema with Literal Enum Validation ---
class RouteDecision(BaseModel):
    """Enforces strict routing classification for the multi-agent graph."""
    intent: Literal["IN_SCOPE", "OUT_OF_SCOPE"] = Field(
        description="Classify as 'IN_SCOPE' for placement/interview questions or 'OUT_OF_SCOPE' for greetings/general chit-chat."
    )

router_prompt = ChatPromptTemplate.from_messages([
    ("system", """You are the Router Agent for a Campus Placement Knowledge Assistant.
Analyze the user query and classify its intent into exactly one of two categories:

1. 'IN_SCOPE': Questions about campus placements, job interviews, company test patterns (e.g. Deloitte, Google, InfoEdge), coding rounds, DSA, system design, HR rounds, resume advice, roadmaps, or compensation.
2. 'OUT_OF_SCOPE': Conversational greetings ('hi', 'hello', 'who are you?'), general chit-chat, or topics completely unrelated to campus placements."""),
    ("user", "{query}")
])

def router_node(state: GraphState) -> GraphState:
    """Classifies user intent using Pydantic structured outputs and sets routing flags."""
    query = state.get("user_query", "").strip()
    if not query:
        state["router_decision"] = "OUT_OF_SCOPE"
        state["is_out_of_scope"] = True
        state["needs_synthesis"] = False
        return state

    api_key = os.getenv("GROQ_API_KEY") or config.GROQ_API_KEY
    if not api_key:
        state["router_decision"] = "IN_SCOPE"
        state["is_out_of_scope"] = False
        state["needs_synthesis"] = True
        return state

    try:
        model_name = os.getenv("GROQ_MODEL", config.DEFAULT_GROQ_MODEL)
        llm = ChatGroq(api_key=api_key, model=model_name, temperature=0.0)
        
        # Enforce strict structured output conforming to RouteDecision
        structured_llm = llm.with_structured_output(RouteDecision)
        chain = router_prompt | structured_llm
        
        decision: RouteDecision = chain.invoke({"query": query})
        
        state["router_decision"] = decision.intent
        state["is_out_of_scope"] = (decision.intent == "OUT_OF_SCOPE")
        state["needs_synthesis"] = (decision.intent == "IN_SCOPE")
    except Exception:
        # Resilient fallback on any network/schema failure
        state["router_decision"] = "IN_SCOPE"
        state["is_out_of_scope"] = False
        state["needs_synthesis"] = True

    return state

if __name__ == "__main__":
    print("Testing 'Hello':", router_node({"user_query": "hello there"})["router_decision"])
    print("Testing 'Deloitte':", router_node({"user_query": "What is Deloitte test pattern?"})["router_decision"])
