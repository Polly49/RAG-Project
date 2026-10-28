"""
Step 5: LangGraph Multi-Agent Orchestrator
Compiles an autonomous, cyclic multi-agent graph with self-correcting feedback loops:

         ┌───────────────────────┐
         │         START         │
         └───────────┬───────────┘
                     │
                     ▼
         ┌───────────────────────┐
         │     Router Agent      │
         └───────────┬───────────┘
                     │ (Conditional Edge)
         ┌───────────┴───────────┐
         │ [IN_SCOPE]            │ [OUT_OF_SCOPE]
         ▼                       │
   ┌───────────┐                 │
┌─►│ Retrieval │                 │
│  │   Agent   │                 │
│  └─────┬─────┘                 │
│        │ (Conditional Edge)    │
│        ├────────────────┐      │
│ [LOOP: Insufficient]    │      │
│ [Retry < 1]             │      │
└────────┘                ▼      ▼
                   ┌───────────────┐
                   │  Synthesizer  │
                   │     Agent     │
                   └───────┬───────┘
                           │
                           ▼
                   ┌───────────────┐
                   │      END      │
                   └───────────────┘
"""

import sys
from pathlib import Path

# Add project root to sys.path
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from langgraph.graph import StateGraph, START, END
from src.agents.step1_state import GraphState
from src.agents.step2_router_agent import router_node
from src.agents.step3_retrieval_agent import retrieval_node
from src.agents.step4_synthesizer_agent import synthesizer_node

# --- Conditional Edge Functions ---

def route_after_router(state: GraphState) -> str:
    """Decides whether to execute retrieval or jump straight to synthesizer for out-of-scope chat."""
    if state.get("is_out_of_scope"):
        return "synthesizer"
    return "retrieval"

def route_after_retrieval(state: GraphState) -> str:
    """Cyclic Edge: Determines whether to trigger the autonomous self-correction loop or synthesize."""
    is_relevant = state.get("is_relevant", True)
    retry_count = state.get("retry_count", 0)

    # If context was evaluated as insufficient and we haven't exceeded max retry cap: LOOP!
    if not is_relevant and retry_count <= 1:
        return "retrieval"  # Autonomous Feedback Loop!

    return "synthesizer"

def create_agent_graph():
    """Builds and compiles the cyclic multi-agent graph with self-correction."""
    workflow = StateGraph(GraphState)

    # 1. Add Agent Nodes
    workflow.add_node("router", router_node)
    workflow.add_node("retrieval", retrieval_node)
    workflow.add_node("synthesizer", synthesizer_node)

    # 2. Add Edges & Conditional Loops
    workflow.add_edge(START, "router")

    # Conditional Branch from Router (In-Scope vs Out-Of-Scope)
    workflow.add_conditional_edges(
        "router",
        route_after_router,
        {
            "retrieval": "retrieval",
            "synthesizer": "synthesizer"
        }
    )

    # Cyclic Feedback Loop from Retrieval (Self-Correction Loop)
    workflow.add_conditional_edges(
        "retrieval",
        route_after_retrieval,
        {
            "retrieval": "retrieval",    # The Self-Reflective Feedback Loop!
            "synthesizer": "synthesizer" # Proceeds once context is satisfactory
        }
    )

    # Synthesizer completes the pipeline
    workflow.add_edge("synthesizer", END)

    return workflow.compile()

# Compile the pipeline once for reuse across Streamlit and FastAPI
pipeline = create_agent_graph()

def run_query(user_query: str, topic_filters=None, company_filter=None) -> dict:
    """Entrypoint to run a student query through the autonomous multi-agent pipeline."""
    initial_state: GraphState = {
        "user_query": user_query,
        "topic_filters": topic_filters,
        "company_filter": company_filter,
        "router_decision": "IN_SCOPE",
        "is_out_of_scope": False,
        "expanded_queries": [],
        "needs_synthesis": False,
        "retrieved_chunks": [],
        "is_relevant": False,
        "retry_count": 0,
        "reformulated_query": None,
        "final_answer": "",
        "has_conflict": False,
        "sources_used": []
    }
    return pipeline.invoke(initial_state)

if __name__ == "__main__":
    print("\n--- Test 1: In-Scope Placement Query ---")
    res1 = run_query("What is the test pattern for Deloitte?")
    print("Answer:\n", res1.get("final_answer")[:200], "...")
    print("Retries used:", res1.get("retry_count"))

    print("\n--- Test 2: Out-Of-Scope Conversational Query ---")
    res2 = run_query("Hello there, how are you?")
    print("Answer:\n", res2.get("final_answer"))
    print("Retrieved chunks count:", len(res2.get("retrieved_chunks", [])))
